"""
Flask 網站：拍照辨識品牌 → 輸入車況 → 估價
用法：python app.py  → 打開 http://127.0.0.1:5000
"""
import io
import json
import os
from functools import lru_cache

import joblib
import keras
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

import car_config as config
from price_utils import make_input

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)          # 不管從哪裡啟動，都用專案資料夾當工作目錄（模型路徑才找得到）

app = Flask(__name__, template_folder=os.path.join(BASE, "templates"),
            static_folder=os.path.join(BASE, "static"))


@lru_cache(maxsize=1)
def price_models():
    d = config.PRICE_MODEL_DIR
    if not os.path.exists(os.path.join(d, "nn.keras")):
        return None
    with open(os.path.join(d, "metrics.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return {"meta": meta,
            "pre": joblib.load(os.path.join(d, "preprocessor.joblib")),
            "LinearRegression": joblib.load(os.path.join(d, "linear_regression.joblib")),
            "RandomForest": joblib.load(os.path.join(d, "random_forest.joblib")),
            "Keras神經網路": keras.models.load_model(os.path.join(d, "nn.keras"))}


@lru_cache(maxsize=1)
def image_model():
    d = config.IMAGE_MODEL_DIR
    if not os.path.exists(os.path.join(d, "car_brand_cnn.keras")):
        return None
    with open(os.path.join(d, "metrics.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return {"meta": meta, "cnn": keras.models.load_model(os.path.join(d, "car_brand_cnn.keras"))}


def predict_price(p, X):
    P = p["pre"].transform(X)
    dense = P.toarray().astype("float32") if hasattr(P, "toarray") else P.astype("float32")
    return {
        "LinearRegression": float(np.exp(p["LinearRegression"].predict(P)[0])),
        "RandomForest": float(np.exp(p["RandomForest"].predict(P)[0])),
        "Keras神經網路": float(np.exp(p["Keras神經網路"].predict(dense, verbose=0).ravel()[0])),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/options")
def options():
    p, im = price_models(), image_model()
    return jsonify({
        "price_ready": p is not None, "image_ready": im is not None,
        "options": p["meta"]["options"] if p else {},
        "year_range": p["meta"]["year_range"] if p else [2005, 2020],
        "price_metrics": p["meta"]["metrics"] if p else {},
        "best_price_model": p["meta"]["best"] if p else None,
        "image_metrics": im["meta"]["results"] if im else {},
        "gbp_to_twd": config.GBP_TO_TWD,
    })


@app.route("/api/recognize", methods=["POST"])
def recognize():
    im = image_model()
    if im is None:
        return jsonify({"error": "還沒有圖片模型，請先執行 python train_image.py"}), 400
    f = request.files.get("photo")
    if not f:
        return jsonify({"error": "請選擇一張照片"}), 400
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(f.read()))).convert("RGB")
    except Exception:
        return jsonify({"error": "無法讀取這個檔案，請上傳 JPG 或 PNG 圖片"}), 400

    # 和訓練時相同：整張縮成 224x224（image_dataset_from_directory 的預設做法）
    x = tf.image.resize(np.asarray(img, dtype="float32"), (config.IMG_SIZE, config.IMG_SIZE))[None]
    prob = im["cnn"].predict(x, verbose=0)[0]
    names = im["meta"]["class_names"]
    order = prob.argsort()[::-1]
    top = [{"brand": names[i], "label": config.display(names[i]), "prob": round(float(prob[i]) * 100, 1)}
           for i in order[:3]]
    best = top[0]
    priceable = best["brand"] in (price_models() or {"meta": {"options": {}}})["meta"]["options"]
    if best["brand"] == config.OTHER:
        note = "看起來不是系統支援的品牌，請手動選擇品牌。"
    elif best["prob"] < 50:
        note = "模型不太確定，請確認品牌是否正確。"
    else:
        note = ""
    return jsonify({"top": top, "brand": best["brand"], "label": best["label"], "priceable": priceable, "note": note})


@app.route("/api/price", methods=["POST"])
def price():
    p = price_models()
    if p is None:
        return jsonify({"error": "還沒有價格模型，請先執行 python train_price.py"}), 400
    d = request.get_json()
    try:
        year, km = int(d["year"]), float(d["mileage_km"])
        X = make_input(d["brand"], d["model"], year, km, d["transmission"], d["fuel"], d["engine"])
    except (KeyError, ValueError):
        return jsonify({"error": "請把資料填完整"}), 400

    preds = predict_price(p, X)
    best = p["meta"]["best"]

    # 折舊試算：如果再開 k 年（年份當作舊 k 年、里程照目前的平均每年里程增加）
    y_max = p["meta"]["year_range"][1]
    per_year_km = km / max(1, y_max - year + 1)
    curve = []
    for k in range(0, 6):
        Xk = make_input(d["brand"], d["model"], year - k, km + k * per_year_km,
                        d["transmission"], d["fuel"], d["engine"])
        curve.append(round(predict_price(p, Xk)[best]))

    r = config.GBP_TO_TWD
    return jsonify({
        "preds": {k: {"gbp": round(v), "twd": round(v * r, -3)} for k, v in preds.items()},
        "best": best, "mape": p["meta"]["metrics"][best]["MAPE_%"],
        "curve": {"labels": ["現在"] + [f"再開 {k} 年" for k in range(1, 6)],
                  "gbp": curve, "twd": [round(c * r, -3) for c in curve]},
    })


if __name__ == "__main__":
    if price_models() is None and image_model() is None:
        raise SystemExit("找不到模型，請先執行 python train_price.py 和 python train_image.py")
    app.run(debug=True)
