"""專案設定。"""
import os

DATA_DIR = "data"
MODEL_DIR = "models"

# ===== 二手車價格資料（Kaggle：100,000 UK Used Car Data set）=====
PRICE_DATA_DIR = os.path.join(DATA_DIR, "used_cars")
PRICE_MODEL_DIR = os.path.join(MODEL_DIR, "price")

# 檔名 → 品牌（注意：原始資料把 Hyundai 拼成 hyundi）
PRICE_FILES = {
    "audi.csv": "Audi", "bmw.csv": "BMW", "ford.csv": "Ford", "hyundi.csv": "Hyundai",
    "merc.csv": "Mercedes-Benz", "skoda.csv": "Skoda", "toyota.csv": "Toyota",
    "vauxhall.csv": "Vauxhall", "vw.csv": "Volkswagen",
}

# ===== 車子照片（Stanford Cars，從 Hugging Face 下載）=====
HF_DATASET = "tanganke/stanford_cars"
IMAGE_DATA_DIR = os.path.join(DATA_DIR, "car_images")
IMAGE_MODEL_DIR = os.path.join(MODEL_DIR, "image")
IMG_SIZE = 224

# 圖片要辨識的品牌：兩份資料都有的 7 個品牌 + 「其他品牌」
IMAGE_BRANDS = ["Audi", "BMW", "Ford", "Hyundai", "Mercedes-Benz", "Toyota", "Volkswagen"]
OTHER = "Other"              # 資料夾用英文（Windows + TensorFlow 讀中文路徑可能出問題），畫面上顯示成中文


def display(name):
    """類別名稱 → 畫面上顯示的名稱。"""
    return "其他品牌" if name == OTHER else name
OTHER_MAX_PER_SPLIT = 1200   # 「其他品牌」照片很多，抽一部分就好，避免類別嚴重不平衡

# ===== 顯示用 =====
GBP_TO_TWD = 44.0            # 英鎊兌台幣（約略值，2026/11 約 44，可自行更新）
KM_PER_MILE = 1.609
