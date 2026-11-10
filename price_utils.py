"""二手車價格資料：讀取、清理、特徵（訓練與網站共用）。"""
import os

import numpy as np
import pandas as pd

import car_config as config

CAT_COLS = ["brand", "model", "transmission", "fuelType"]
NUM_COLS = ["year", "mileage", "engineSize"]
FEATURES = CAT_COLS + NUM_COLS
# 不使用 tax（牌照稅）和 mpg（油耗）：一般使用者估價時不會知道這些數字


def load_price_data(data_dir=config.PRICE_DATA_DIR):
    """讀取各品牌 CSV，合併成一張表並清理。"""
    frames = []
    for fname, brand in config.PRICE_FILES.items():
        path = None
        for root, _, files in os.walk(data_dir):
            if fname in files:
                path = os.path.join(root, fname)
                break
        if path is None:
            print(f"  ⚠ 找不到 {fname}，略過 {brand}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={c: "tax" for c in df.columns if c.startswith("tax")})
        df["brand"] = brand
        frames.append(df)
    if not frames:
        raise SystemExit(f"在 {data_dir} 找不到任何價格資料，請先下載 Kaggle 資料集（見 README）")

    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    for c in ["model", "transmission", "fuelType"]:
        df[c] = df[c].astype(str).str.strip()          # 原始資料的車型前面有空白，例如 " A3"
    for c in ["year", "price", "mileage", "engineSize"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 清理明顯錯誤的資料
    df = df.dropna(subset=["year", "price", "mileage", "engineSize"])
    df = df[(df.year >= 1995) & (df.year <= 2020)]        # 資料裡有 2060 年這種錯字
    df = df[(df.engineSize > 0) & (df.engineSize < 7)]     # 排氣量 0 是缺值
    df = df[(df.price >= 500) & (df.price <= 150000)]
    df = df[df.mileage < 300000]
    df = df.drop_duplicates()
    print(f"讀取價格資料：{n0} 筆 → 清理後 {len(df)} 筆，{df.brand.nunique()} 個品牌")
    return df.reset_index(drop=True)


def build_preprocessor():
    """類別欄位 One-Hot、數值欄位標準化。"""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])


def make_input(brand, model, year, mileage_km, transmission, fuel, engine):
    """網站輸入 → 模型輸入（公里轉英里）。"""
    return pd.DataFrame([{
        "brand": brand, "model": model, "year": float(year),
        "mileage": float(mileage_km) / config.KM_PER_MILE,
        "transmission": transmission, "fuelType": fuel, "engineSize": float(engine),
    }])[FEATURES]
