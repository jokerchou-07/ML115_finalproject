"""
挑出 Demo 用的照片 + 列出資料集裡有哪些車款

1. 從「測試集」（訓練時沒看過的照片）裡，每個品牌抽幾張複製到 demo_photos 資料夾
   - 預設「隨機抽」：答對答錯都有，比較真實（檔名標 OK / NG）
   - 加 --best：只挑模型答對、信心最高的（保證 Demo 成功）
2. 列出每個品牌在資料集裡有哪些車款
   → 想自己上網找照片時，搜尋這些車款（例如「2012 Audi A5 coupe」）會比較容易認對

用法：python pick_demo_photos.py                     每個品牌隨機抽 3 張
      python pick_demo_photos.py --per-brand 5       每個品牌隨機抽 5 張
      python pick_demo_photos.py --seed 7            換一批隨機照片
      python pick_demo_photos.py --best              只挑答對、信心最高的
"""
import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict

import numpy as np

import car_config as config

OUT_DIR = "demo_photos"


def model_name_of(path):
    """檔名是 00123_Audi_A5_Coupe_2012.jpg → 取出 Audi A5 Coupe 2012"""
    name = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"^\d+_", "", name).replace("_", " ")


def main(per_brand=3, best_mode=False, seed=None):
    import keras
    import tensorflow as tf

    test_dir = os.path.join(config.IMAGE_DATA_DIR, "test")
    if not os.path.isdir(test_dir):
        raise SystemExit("找不到測試照片，請先執行 python download_data.py --images")
    cnn = keras.models.load_model(os.path.join(config.IMAGE_MODEL_DIR, "car_brand_cnn.keras"))
    with open(os.path.join(config.IMAGE_MODEL_DIR, "metrics.json"), encoding="utf-8") as f:
        names = json.load(f)["class_names"]

    # ===== 2. 列出車款 =====
    print("========== 資料集裡的車款（上網找照片時可以搜尋這些）==========")
    for brand in names:
        if brand == config.OTHER:
            continue
        models = sorted({model_name_of(p) for p in glob.glob(os.path.join(test_dir, brand, "*.jpg"))})
        print(f"\n【{brand}】{len(models)} 款")
        for m in models:
            print(f"   {m}")

    # ===== 1. 挑照片 =====
    mode = "只挑答對、信心最高" if best_mode else "隨機抽（答對答錯都有）"
    print(f"\n========== 挑選 Demo 照片：{mode}（需要 1~3 分鐘）==========")
    rng = np.random.default_rng(seed)
    total = total_ok = 0
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    for idx, brand in enumerate(names):
        paths = sorted(glob.glob(os.path.join(test_dir, brand, "*.jpg")))
        if not paths:
            continue
        ds = keras.utils.image_dataset_from_directory(
            os.path.join(test_dir, brand), labels=None, shuffle=False,
            image_size=(config.IMG_SIZE, config.IMG_SIZE), batch_size=32, verbose=False)
        files = ds.file_paths
        prob = cnn.predict(ds, verbose=0)
        pred = prob.argmax(1)
        right = np.where(pred == idx)[0]
        if best_mode:
            chosen = right[np.argsort(-prob[right, idx])][:per_brand]
        else:
            chosen = rng.choice(len(files), size=min(per_brand, len(files)), replace=False)
        ok = int((pred[chosen] == idx).sum())
        total += len(chosen); total_ok += ok
        print(f"  {config.display(brand):<15} 整個測試集答對 {len(right)}/{len(files)} 張"
              f"（{len(right) / len(files):.0%}）｜這次抽 {len(chosen)} 張，答對 {ok} 張")
        tag = "other" if brand == config.OTHER else brand.lower().replace("mercedes-benz", "benz")
        for n, i in enumerate(chosen, 1):
            model = model_name_of(files[i]).replace(" ", "_").replace("/", "-")
            guess = names[pred[i]].lower().replace("mercedes-benz", "benz")
            result = "OK" if pred[i] == idx else f"NG-guess-{guess}"
            shutil.copy(files[i], os.path.join(OUT_DIR, f"{tag}_{n}_{model}_{result}.jpg"))
    print(f"\n這次抽出的 {total} 張照片，模型答對 {total_ok} 張（{total_ok / max(total, 1):.0%}）")
    print(f"✅ 已複製到 {OUT_DIR}/ 資料夾，Demo 時上傳這些照片")
    print("   檔名格式：實際品牌_編號_車款_結果.jpg（OK = 答對；NG-guess-xxx = 答錯，模型猜成 xxx）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-brand", type=int, default=3)
    ap.add_argument("--best", action="store_true", help="只挑答對、信心最高的")
    ap.add_argument("--seed", type=int, default=None, help="指定數字可以重現同一批隨機照片")
    a, _ = ap.parse_known_args()   # parse_known_args：直接貼到 Colab / Jupyter 執行也不會報錯
    main(a.per_brand, a.best, a.seed)
