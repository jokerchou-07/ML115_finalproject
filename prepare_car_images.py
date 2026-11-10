"""
步驟：下載 Stanford Cars（Hugging Face），整理成「一個品牌一個資料夾」

    data/car_images/train/Audi/xxx.jpg
    data/car_images/train/BMW/...
    data/car_images/train/Other/...     ← 其他品牌（資料夾用英文名稱）
    data/car_images/test/...        ← 官方切好的測試集，訓練時完全不會用到

原始資料有 196 類（品牌+車型+年份），這裡只取「品牌」：
7 個價格資料也有的品牌 + 其他品牌（Honda、Nissan…等，用來讓系統知道「這不是能估價的品牌」）

用法：python prepare_car_images.py
"""
import os
import random
import shutil

import car_config as config

# 196 類名稱的第一個字通常是品牌，但有些品牌是兩個字
MULTI_WORD = ["AM General", "Aston Martin", "Land Rover", "Rolls-Royce"]


def brand_of(class_name):
    for b in MULTI_WORD:
        if class_name.startswith(b):
            return b
    return class_name.split()[0]


def target_of(class_name):
    b = brand_of(class_name)
    return b if b in config.IMAGE_BRANDS else config.OTHER


def export_split(ds_split, names, split, rng):
    out_root = os.path.join(config.IMAGE_DATA_DIR, split)
    rows = list(range(len(ds_split)))
    labels = list(ds_split["label"])
    targets = [target_of(names[l]) for l in labels]

    # 「其他品牌」只抽一部分
    other_idx = [i for i in rows if targets[i] == config.OTHER]
    keep_other = set(rng.sample(other_idx, min(len(other_idx), config.OTHER_MAX_PER_SPLIT)))
    keep = [i for i in rows if targets[i] != config.OTHER or i in keep_other]

    counts = {}
    for n, i in enumerate(keep, 1):
        item = ds_split[i]
        t = targets[i]
        d = os.path.join(out_root, t)
        os.makedirs(d, exist_ok=True)
        model_name = names[labels[i]].replace(" ", "_").replace("/", "-")
        item["image"].convert("RGB").save(os.path.join(d, f"{i:05d}_{model_name}.jpg"), quality=92)
        counts[t] = counts.get(t, 0) + 1
        if n % 1000 == 0:
            print(f"  {split}: {n}/{len(keep)}")
    return counts


def main(ds=None):
    if ds is None:
        from datasets import load_dataset, load_dataset_builder
        print(f"從 Hugging Face 下載 {config.HF_DATASET}（只下載 train / test，第一次需要幾分鐘）...")
        try:
            # 這份資料集另外附了加雜訊的版本（contrast、gaussian_noise…），只抓需要的兩份
            ds = load_dataset(config.HF_DATASET,
                              data_files={"train": "data/train-*", "test": "data/test-*"})
        except Exception as e:
            print(f"  只下載部分檔案失敗（{e}），改成下載完整資料集")
            ds = load_dataset(config.HF_DATASET)
    names = getattr(ds["train"].features["label"], "names", None)
    if not names:
        from datasets import load_dataset_builder
        names = load_dataset_builder(config.HF_DATASET).info.features["label"].names
    if os.path.exists(config.IMAGE_DATA_DIR):
        shutil.rmtree(config.IMAGE_DATA_DIR)
    rng = random.Random(42)
    for split in ["train", "test"]:
        counts = export_split(ds[split], names, split, rng)
        print(f"\n{split} 各品牌張數：")
        for b in config.IMAGE_BRANDS + [config.OTHER]:
            print(f"  {config.display(b):<15}{counts.get(b, 0):>6}")
    print(f"\n✅ 照片已整理到 {config.IMAGE_DATA_DIR}/")


if __name__ == "__main__":
    main()
