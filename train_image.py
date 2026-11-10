"""
圖片部分：從照片辨識汽車品牌（7 個品牌 + 其他品牌）

比較兩種方法：
  1. Scikit-Learn：MobileNetV2 抽特徵（不訓練）+ Logistic Regression 分類
  2. Keras CNN  ：MobileNetV2 遷移學習 + 資料增強（DataGen），兩階段訓練

測試集：Stanford Cars 官方切好的測試集（訓練時完全不會用到）

用法：python train_image.py
      python train_image.py --quick     （快速檢查程式能不能跑）
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import keras
from keras import layers

import car_config as config
from plot_font import setup_chinese_font

setup_chinese_font()
S = config.IMG_SIZE
OUT = config.IMAGE_MODEL_DIR
out = lambda n: os.path.join(OUT, n)


def load_datasets(batch):
    train_dir = os.path.join(config.IMAGE_DATA_DIR, "train")
    test_dir = os.path.join(config.IMAGE_DATA_DIR, "test")
    if not os.path.isdir(train_dir):
        raise SystemExit("找不到照片，請先執行 python prepare_car_images.py")
    kw = dict(image_size=(S, S), batch_size=batch, label_mode="int")
    tr, va = keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.15, subset="both", seed=42, **kw)
    class_names = tr.class_names
    te = keras.utils.image_dataset_from_directory(test_dir, shuffle=False, class_names=class_names, **kw)
    return tr, va, te, class_names


def compact_cache(ds):
    """照片存成 uint8 快取在記憶體（只佔 float32 的 1/4），用的時候再轉回 float32。
    第一版直接 cache() float32，訓練+測試快取約 6GB，Colab 記憶體容易爆掉。"""
    return (ds.map(lambda x, y: (tf.cast(tf.clip_by_value(x, 0, 255), tf.uint8), y)).cache()
              .map(lambda x, y: (tf.cast(x, tf.float32), y)).prefetch(tf.data.AUTOTUNE))


def labels_of(ds):
    return np.concatenate([y.numpy() for _, y in ds])


def build_cnn(n_classes, pretrained=True):
    # 資料增強（= 課堂的 ImageDataGenerator）
    # 車子的品牌和顏色、亮度無關 → 這裡可以放心調亮度、對比（和空污主題剛好相反！）
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="data_augmentation")
    base = keras.applications.MobileNetV2(input_shape=(S, S, 3), include_top=False,
                                          weights="imagenet" if pretrained else None)
    base.trainable = False
    inputs = keras.Input((S, S, 3))
    x = augment(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def extract_features(base, ds):
    """
    Scikit-Learn 用：MobileNetV2 當作「特徵抽取器」，每張照片變成 1280 維向量。
    ⚠ 特徵和標籤一定要在「同一次讀取」中一起取出！
      訓練資料每次讀取都會重新打亂順序，第一版分兩次讀，特徵和答案對不起來，準確率只有 16%（≈亂猜）。
    """
    inp = keras.Input((S, S, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inp)
    x = layers.GlobalAveragePooling2D()(base(x, training=False))
    fe = keras.Model(inp, x)
    F, Y = [], []
    for b, y in ds:
        F.append(fe.predict_on_batch(b)); Y.append(y.numpy())
    return np.concatenate(F), np.concatenate(Y)


def main(args):
    os.makedirs(OUT, exist_ok=True)
    tr, va, te, class_names = load_datasets(args.batch)
    if args.quick:
        tr, va = tr.take(5), va.take(2)
    n = len(class_names)
    labels = [config.display(c) for c in class_names]      # 畫面/圖表顯示用（Other → 其他品牌）
    print("類別：", labels)

    y_tr = labels_of(tr)
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(n), y=y_tr)))
    te_paths = list(te.file_paths)          # 記下測試照片路徑，畫範例圖時只讀需要的幾張
    tr, va, te = compact_cache(tr), compact_cache(va), compact_cache(te)
    y_te = labels_of(te)

    model, base = build_cnn(n, pretrained=not args.no_pretrained)

    # ===== 1. Scikit-Learn：CNN 特徵 + Logistic Regression =====
    print("\n[1/2] Scikit-Learn：用 MobileNetV2 抽特徵 + Logistic Regression ...")
    F_tr, yf_tr = extract_features(base, tr)
    F_te, _ = extract_features(base, te)
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(F_tr, yf_tr)
    pred_lr = clf.predict(F_te)
    joblib.dump(clf, out("logreg_on_cnn_features.joblib"))

    # ===== 2. Keras：遷移學習 =====
    print("\n[2/2] Keras：MobileNetV2 遷移學習")
    ckpt = out("car_brand_cnn.keras")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(tr, validation_data=va, epochs=args.epochs, class_weight=cw, verbose=2,
                   callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    stage1_val = max(h1.history["val_accuracy"])
    model.save(ckpt)

    # 第二階段微調：只有在驗證準確率「真的變好」時才採用（上一個主題學到的教訓）
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(tr, validation_data=va, epochs=args.finetune_epochs, class_weight=cw, verbose=2,
                   callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    stage2_val = max(h2.history["val_accuracy"]) if h2.history else 0
    if stage2_val > stage1_val:
        model.save(ckpt)
        print(f"  ✔ 微調有效（驗證 {stage1_val:.3f} → {stage2_val:.3f}），採用微調後的模型")
    else:
        model = keras.models.load_model(ckpt)
        print(f"  ✘ 微調沒有變好（{stage1_val:.3f} → {stage2_val:.3f}），保留第一階段的模型")

    prob = model.predict(te, verbose=0)
    pred_cnn = prob.argmax(1)

    # ===== 評估 =====
    results = {}
    print("\n========== 測試集結果（Stanford Cars 官方測試集）==========")
    for name, pred in [("LogReg(CNN特徵)", pred_lr), ("CNN遷移學習", pred_cnn)]:
        results[name] = {
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4)}
        print(f"\n{name}：{results[name]}")
        print(classification_report(y_te, pred, labels=range(n), target_names=labels, zero_division=0))
    top2 = float(np.mean([y in p.argsort()[-2:] for y, p in zip(y_te, prob)]))
    results["CNN遷移學習"]["top2_accuracy"] = round(top2, 4)
    print(f"CNN 前兩名包含正確答案的比例（Top-2）：{top2:.1%}")

    with open(out("metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names, "results": results, "n_test": int(len(y_te))},
                  f, ensure_ascii=False, indent=2)
    plot_confusion(y_te, pred_cnn, labels)
    plot_history(h1, h2)
    plot_examples(te_paths, prob, y_te, labels)
    print(f"\n✅ 完成！模型與圖表在 {OUT}/")


def plot_confusion(y, pred, names):
    cm = confusion_matrix(y, pred, labels=range(len(names)))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("模型預測"); ax.set_ylabel("實際品牌"); ax.set_title("品牌辨識混淆矩陣（CNN）")
    fig.tight_layout(); fig.savefig(out("1_confusion_matrix.png"), dpi=150); plt.close(fig)


def plot_history(h1, h2):
    acc = h1.history["accuracy"] + h2.history.get("accuracy", [])
    val = h1.history["val_accuracy"] + h2.history.get("val_accuracy", [])
    plt.figure(figsize=(7, 4))
    plt.plot(acc, label="訓練準確率"); plt.plot(val, label="驗證準確率")
    plt.axvline(len(h1.history["accuracy"]) - 0.5, color="gray", ls="--", label="開始微調")
    plt.xlabel("epoch"); plt.legend(); plt.title("CNN 訓練過程")
    plt.tight_layout(); plt.savefig(out("2_cnn_training.png"), dpi=150); plt.close()


def plot_examples(paths, prob, y, names, k=10):
    """只讀抽到的 k 張照片（第一版把整個測試集讀進記憶體，Colab 會爆掉）。"""
    idx = np.random.default_rng(0).choice(len(y), size=min(k, len(y)), replace=False)
    fig, axes = plt.subplots(2, (len(idx) + 1) // 2, figsize=(15, 6.5))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(plt.imread(paths[i])); ax.axis("off")
        p = prob[i].argmax()
        ax.set_title(f"實際：{names[y[i]]}\n預測：{names[p]}（{prob[i][p]:.0%}）", fontsize=9,
                     color="green" if p == y[i] else "red")
    fig.tight_layout(); fig.savefig(out("3_examples.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--finetune-epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="不下載 ImageNet 權重（只用於離線測試）")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.finetune_epochs = 1, 1
    keras.utils.set_random_seed(42)
    main(a)
