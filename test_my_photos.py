"""
用自己拍的照片測試品牌辨識

1. 把照片放進專案裡的 my_photos 資料夾（jpg / png 都可以）
   💡 檔名用車子的實際品牌開頭，例如  toyota_1.jpg、bmw_停車場.jpg、honda_1.jpg、豐田_1.jpg
      程式就會自動幫你算準確率（不是 7 個品牌的，例如 honda，會當作「其他品牌」）
2. 執行：python test_my_photos.py
   測其他資料夾（例如 pick_demo_photos.py 挑出的照片）：python test_my_photos.py demo_photos
3. 結果圖存在 my_photos_result.png
   顏色：綠色 = 答對、紅色 = 答錯、黑色 = 檔名沒標品牌（無法判斷對錯）
"""
import glob
import json
import os

import matplotlib
import numpy as np

import car_config as config

PHOTO_DIR = "my_photos"
ALIASES = {"benz": "Mercedes-Benz", "mercedes": "Mercedes-Benz", "vw": "Volkswagen",
           "豐田": "Toyota", "福特": "Ford", "現代": "Hyundai", "賓士": "Mercedes-Benz",
           "福斯": "Volkswagen", "奧迪": "Audi", "寶馬": "BMW", "其他": "Other", "other": "Other"}


def true_brand_from_name(fname, class_names):
    """從檔名猜實際品牌：toyota_1.jpg → Toyota；honda_1.jpg → Other；看不出來 → None。"""
    head = os.path.basename(fname).split("_")[0].split(".")[0].split("-")[0].lower()
    if not head or head.isdigit() or head.startswith(("img", "dsc", "photo", "pxl")):
        return None
    if head in ALIASES:
        return ALIASES[head]
    for c in class_names:
        if c.lower().startswith(head) or head.startswith(c.lower()[:4]):
            return c
    return config.OTHER


def main(show=False, photo_dir=None):
    global PHOTO_DIR
    PHOTO_DIR = photo_dir or PHOTO_DIR
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import keras
    import tensorflow as tf
    from PIL import Image, ImageOps
    from plot_font import setup_chinese_font
    setup_chinese_font()

    os.makedirs(PHOTO_DIR, exist_ok=True)
    paths = sorted(p for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
                   for p in glob.glob(os.path.join(PHOTO_DIR, f"*.{ext}")))
    if not paths:
        print(f"⚠ {PHOTO_DIR} 資料夾裡沒有照片，請先把照片放進去")
        return
    cnn = keras.models.load_model(os.path.join(config.IMAGE_MODEL_DIR, "car_brand_cnn.keras"))
    with open(os.path.join(config.IMAGE_MODEL_DIR, "metrics.json"), encoding="utf-8") as f:
        names = json.load(f)["class_names"]

    cols = min(len(paths), 4)
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    right = judged = 0
    for ax, p in zip(axes.ravel(), paths):
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")   # 手機照片依 EXIF 轉正
        x = tf.image.resize(np.asarray(img, dtype="float32"), (config.IMG_SIZE, config.IMG_SIZE))[None]
        prob = cnn.predict(x, verbose=0)[0]
        top = prob.argsort()[::-1][:3]
        truth = true_brand_from_name(p, names)
        mark = ""
        if truth is not None:
            judged += 1
            ok = names[top[0]] == truth
            right += ok
            mark = " ✓" if ok else f" ✗（實際：{config.display(truth)}）"
        ax.imshow(img)
        title = "\n".join(f"{config.display(names[i])} {prob[i]:.0%}" for i in top)
        ax.set_title(f"{os.path.basename(p)[:20]}{mark}\n{title}", fontsize=10,
                     color="green" if mark.startswith(" ✓") else ("red" if mark else "black"))
        print(f"{os.path.basename(p):<28} → {config.display(names[top[0]])} ({prob[top[0]]:.0%}){mark}")
    plt.tight_layout()
    fig.suptitle("綠色 = 答對　紅色 = 答錯　黑色 = 檔名沒標品牌", fontsize=12, y=1.0)
    plt.savefig("my_photos_result.png", dpi=110, bbox_inches="tight")
    if judged:
        print(f"\n準確率：{right}/{judged} = {right / judged:.0%}（只計算檔名有標品牌的照片）")
    print("結果圖已存成 my_photos_result.png")
    if show:
        plt.show()


if __name__ == "__main__":
    import sys
    folder = next((a for a in sys.argv[1:] if not a.startswith("-") and os.path.isdir(a)), None)
    main(photo_dir=folder)
