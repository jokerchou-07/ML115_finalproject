"""
接著目前的品牌辨識模型繼續訓練（不用從頭來）

    python continue_train.py                     再練 20 輪（解凍最後 40 層，和原本一樣）
    python continue_train.py --epochs 30         再練 30 輪
    python continue_train.py --unfreeze 80       多解凍一些層（模型能學更多，但也更容易過擬合）
    python continue_train.py --restore           練壞了？還原成繼續訓練前的模型

保護機制：
  1. 開始前先備份目前的模型（car_brand_cnn_backup.keras）
  2. 用「驗證集」判斷有沒有變好：變好才採用，沒變好就保留原本的模型
     （不拿測試集挑模型，否則測試分數就不公平了）
  3. 驗證集和 train_image.py 用同一個 seed 切，所以訓練時沒看過的照片還是沒看過
"""
import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import keras

import car_config as config
import train_image as T

MODEL = T.out("car_brand_cnn.keras")
BACKUP = T.out("car_brand_cnn_backup.keras")
METRICS = T.out("metrics.json")


def set_trainable(model, n_unfreeze):
    """解凍 MobileNetV2 最後 n 層（BatchNorm 保持凍結，小資料微調比較穩定）。"""
    base = next(l for l in model.layers if isinstance(l, keras.Model) and "mobilenet" in l.name.lower())
    base.trainable = True
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def main(args):
    if args.restore:
        if not os.path.exists(BACKUP):
            raise SystemExit("找不到備份檔，沒辦法還原")
        shutil.copy(BACKUP, MODEL)
        raise SystemExit("✅ 已還原成繼續訓練前的模型（測試集分數請重跑 continue_train.py --epochs 0 更新）")
    if not os.path.exists(MODEL):
        raise SystemExit("找不到模型，請先執行 python train_image.py")

    tr, va, te, class_names = T.load_datasets(args.batch)
    labels = [config.display(c) for c in class_names]
    n = len(class_names)
    y_tr = T.labels_of(tr)
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(n), y=y_tr)))
    te_paths = list(te.file_paths)
    tr, va, te = T.compact_cache(tr), T.compact_cache(va), T.compact_cache(te)
    y_te = T.labels_of(te)

    model = keras.models.load_model(MODEL)
    shutil.copy(MODEL, BACKUP)
    print(f"已備份目前的模型 → {BACKUP}")

    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    _, val_before = model.evaluate(va, verbose=0)
    print(f"目前模型的驗證準確率：{val_before:.3f}")

    hist = None
    if args.epochs > 0:
        set_trainable(model, args.unfreeze)
        model.compile(optimizer=keras.optimizers.Adam(args.lr),   # 改了可訓練層之後要重新 compile
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        print(f"\n繼續訓練 {args.epochs} 輪（解凍最後 {args.unfreeze} 層，學習率 {args.lr}）")
        hist = model.fit(tr, validation_data=va, epochs=args.epochs, class_weight=cw, verbose=2, callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1)])
        _, val_after = model.evaluate(va, verbose=0)
        if val_after > val_before:
            model.save(MODEL)
            print(f"\n✔ 驗證準確率 {val_before:.3f} → {val_after:.3f}，有進步，採用新模型")
        else:
            model = keras.models.load_model(BACKUP)
            print(f"\n✘ 驗證準確率 {val_before:.3f} → {val_after:.3f}，沒有進步，保留原本的模型")

    # ===== 用測試集評估最後採用的模型 =====
    prob = model.predict(te, verbose=0)
    pred = prob.argmax(1)
    res = {"accuracy": round(float(accuracy_score(y_te, pred)), 4),
           "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
           "top2_accuracy": round(float(np.mean([y in p.argsort()[-2:] for y, p in zip(y_te, prob)])), 4)}
    print("\n========== 測試集結果（Stanford Cars 官方測試集）==========")
    print(f"CNN遷移學習：{res}")
    print(classification_report(y_te, pred, labels=range(n), target_names=labels, zero_division=0))

    with open(METRICS, encoding="utf-8") as f:
        meta = json.load(f)
    meta["results"]["CNN遷移學習"] = res
    with open(METRICS, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    T.plot_confusion(y_te, pred, labels)
    T.plot_examples(te_paths, prob, y_te, labels)
    if hist is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(hist.history["accuracy"], label="訓練準確率")
        plt.plot(hist.history["val_accuracy"], label="驗證準確率")
        plt.axhline(val_before, color="gray", ls="--", label="繼續訓練前的驗證準確率")
        plt.xlabel("epoch（繼續訓練）"); plt.legend(); plt.title("繼續訓練過程")
        plt.tight_layout(); plt.savefig(T.out("4_continue_training.png"), dpi=150); plt.close()
    print(f"\n✅ 完成！metrics.json、混淆矩陣、範例圖都已更新（{config.IMAGE_MODEL_DIR}/）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--unfreeze", type=int, default=40, help="解凍 MobileNetV2 最後幾層")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--restore", action="store_true", help="還原成繼續訓練前的模型")
    a = ap.parse_args()
    keras.utils.set_random_seed(42)
    main(a)
