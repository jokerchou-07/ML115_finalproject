"""
數值部分：二手車價格預測（回歸）

比較 4 個模型（預測的是 log(價格)，因為價格分布右偏，取 log 後誤差比較平均）：
  0. 基準線：同品牌、同車型、同年份的「平均價格」（最簡單的估價方式）
  1. Scikit-Learn  Linear Regression
  2. Scikit-Learn  Random Forest
  3. Keras         神經網路（Dense）

用法：python train_price.py
輸出：models/price/ 底下的模型、metrics.json、圖表
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
import keras
from keras import layers

import car_config as config
from plot_font import setup_chinese_font
from price_utils import FEATURES, build_preprocessor, load_price_data

setup_chinese_font()
OUT = config.PRICE_MODEL_DIR
out = lambda n: os.path.join(OUT, n)


def evaluate(y_true, y_pred):
    return {"MAE_£": round(float(mean_absolute_error(y_true, y_pred)), 1),
            "MAPE_%": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 2),
            "R2": round(float(r2_score(y_true, y_pred)), 4)}


def build_nn(n_in, y_mean, y_std):
    """
    最後一層 Rescaling 把輸出換算回 log(價格)：輸出 = 網路結果 × 標準差 + 平均。
    第一版沒有這層，網路要從 0 慢慢學到 9~10（log 價格的大小），第 1 輪 loss 高達 4.45、
    驗證 loss 忽高忽低，第 2 輪就被 EarlyStopping 停掉，誤差率 15.3% 比基準線還差。
    加上這層後，網路一開始就從「平均價」附近起步，只需要學「比平均貴多少、便宜多少」。
    （換算寫在模型裡面，所以 app.py 不用改）
    """
    model = keras.Sequential([
        layers.Input((n_in,)),
        layers.Dense(256, activation="relu"), layers.Dropout(0.1),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
        layers.Rescaling(scale=y_std, offset=y_mean),
    ])
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse", metrics=["mae"])
    return model


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_price_data()
    X, y = df[FEATURES], df["price"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42)
    print(f"訓練 {len(X_tr)}｜驗證 {len(X_va)}｜測試 {len(X_te)}")

    pre = build_preprocessor().fit(X_tr)
    P_tr, P_va, P_te = (pre.transform(a) for a in (X_tr, X_va, X_te))
    ly_tr, ly_va = np.log(y_tr), np.log(y_va)
    preds = {}

    # 0. 基準線：同品牌+車型+年份的平均價（查不到就退回同品牌+車型、再退回同品牌）
    tr = X_tr.assign(price=y_tr)
    g3 = tr.groupby(["brand", "model", "year"]).price.mean()
    g2 = tr.groupby(["brand", "model"]).price.mean()
    g1 = tr.groupby("brand").price.mean()
    preds["基準線(查平均價)"] = np.array([
        g3.get((b, m, yr), g2.get((b, m), g1.get(b, tr.price.mean())))
        for b, m, yr in zip(X_te.brand, X_te.model, X_te.year)])

    print("訓練 Linear Regression ...")
    lr = Ridge(alpha=1.0).fit(P_tr, ly_tr)          # 加一點正則化，避免罕見車型係數爆掉
    preds["LinearRegression"] = np.exp(lr.predict(P_te))

    print("訓練 Random Forest ...（只用 CPU，約 3~10 分鐘，請耐心等候）")
    rf = RandomForestRegressor(n_estimators=100, max_depth=25, min_samples_leaf=3, n_jobs=-1,
                               random_state=42).fit(P_tr, ly_tr)
    preds["RandomForest"] = np.exp(rf.predict(P_te))

    print("訓練 Keras 神經網路 ...")
    dense = lambda a: a.toarray().astype("float32") if hasattr(a, "toarray") else a.astype("float32")
    nn = build_nn(P_tr.shape[1], float(ly_tr.mean()), float(ly_tr.std()))
    hist = nn.fit(dense(P_tr), ly_tr, validation_data=(dense(P_va), ly_va), epochs=100, batch_size=256,
                  verbose=2, callbacks=[
                      keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                      # 驗證 loss 停滯時自動把學習率減半，讓後面學得更細
                      keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5, verbose=1)])
    preds["Keras神經網路"] = np.exp(nn.predict(dense(P_te), verbose=0).ravel())

    metrics = {k: evaluate(y_te, v) for k, v in preds.items()}
    print("\n========== 測試集結果 ==========")
    for k, v in metrics.items():
        print(f"  {k:<16} 平均誤差 £{v['MAE_£']:>7,.0f}   誤差率 {v['MAPE_%']:5.1f}%   R² {v['R2']:.3f}")

    # 各品牌的誤差（用最好的模型）
    best = min((k for k in metrics if k != "基準線(查平均價)"), key=lambda k: metrics[k]["MAE_£"])
    by_brand = (pd.DataFrame({"brand": X_te.brand.values, "err": np.abs(preds[best] - y_te) / y_te * 100})
                .groupby("brand").err.mean().round(1).to_dict())
    print(f"\n  最好的模型：{best}；各品牌誤差率(%)：{by_brand}")

    joblib.dump(pre, out("preprocessor.joblib"))
    joblib.dump(lr, out("linear_regression.joblib"))
    joblib.dump(rf, out("random_forest.joblib"), compress=3)
    nn.save(out("nn.keras"))

    # 網站下拉選單需要的選項
    options = {b: {"models": sorted(g.model.unique().tolist(), key=lambda m: -(g.model == m).sum()),
                   "fuel": sorted(g.fuelType.unique().tolist()),
                   "transmission": sorted(g.transmission.unique().tolist()),
                   "engine_median": float(g.engineSize.median())}
               for b, g in df.groupby("brand")}
    with open(out("metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "best": best, "by_brand_mape": by_brand,
                   "year_range": [int(df.year.min()), int(df.year.max())], "options": options},
                  f, ensure_ascii=False, indent=2)

    plots(metrics, preds, y_te, X_te, hist, rf, pre)
    print(f"\n✅ 完成！模型與圖表在 {OUT}/")


def plots(metrics, preds, y_te, X_te, hist, rf, pre):
    names = list(metrics)
    # 圖 1：模型比較
    plt.figure(figsize=(8, 4))
    bars = plt.bar(names, [metrics[n]["MAPE_%"] for n in names],
                   color=["#999", "#4C72B0", "#55A868", "#C44E52"])
    plt.bar_label(bars, fmt="%.1f%%"); plt.ylabel("平均誤差率 %（越低越好）")
    plt.title("二手車價格預測：模型比較"); plt.tight_layout()
    plt.savefig(out("1_model_compare.png"), dpi=150); plt.close()

    # 圖 2：預測 vs 實際
    best = min((k for k in metrics if "基準" not in k), key=lambda k: metrics[k]["MAE_£"])
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_te, preds[best], s=3, alpha=0.3)
    lim = [0, np.percentile(y_te, 99.5)]
    plt.plot(lim, lim, "r--", label="完美預測"); plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("實際價格 (£)"); plt.ylabel("預測價格 (£)"); plt.legend()
    plt.title(f"預測 vs 實際（{best}）"); plt.tight_layout()
    plt.savefig(out("2_pred_vs_actual.png"), dpi=150); plt.close()

    # 圖 3：特徵重要性（One-Hot 的欄位加總回原本的特徵）
    names_oh = pre.get_feature_names_out()
    imp = pd.Series(rf.feature_importances_, index=names_oh)
    groups = imp.groupby(lambda n: n.split("__")[1].split("_")[0] if n.startswith("cat__")
                         else n.split("__")[1]).sum().sort_values()
    label = {"brand": "品牌", "model": "車型", "year": "年份", "mileage": "里程",
             "engineSize": "排氣量", "transmission": "變速箱", "fuelType": "燃料"}
    plt.figure(figsize=(6, 4))
    plt.barh([label.get(i, i) for i in groups.index], groups.values, color="#55A868")
    plt.title("哪些因素最影響二手車價格？（Random Forest）"); plt.tight_layout()
    plt.savefig(out("3_feature_importance.png"), dpi=150); plt.close()

    # 圖 4：神經網路訓練曲線
    plt.figure(figsize=(6, 4))
    plt.plot(hist.history["loss"], label="訓練 loss"); plt.plot(hist.history["val_loss"], label="驗證 loss")
    plt.xlabel("epoch"); plt.legend(); plt.title("Keras 神經網路訓練過程"); plt.tight_layout()
    plt.savefig(out("4_nn_training.png"), dpi=150); plt.close()


if __name__ == "__main__":
    keras.utils.set_random_seed(42)
    main()
