"""
下載全部資料（在自己電腦用；Windows / Mac 都可以）

    python download_data.py            兩份都下載
    python download_data.py --price    只下載價格資料（Kaggle，需要 Token）
    python download_data.py --images   只下載車子照片（Hugging Face，不需要 Token）

Kaggle Token 的設定方式（擇一）：
  1. 什麼都不設定：執行時會跳出提示，貼上 Token 即可（這次有效）
  2. PowerShell：$env:KAGGLE_API_TOKEN="你的Token"  然後在同一個視窗執行
  3. 存成檔案（以後都不用再輸入）：把 Token 存到  C:\\Users\\你的帳號\\.kaggle\\access_token
"""
import argparse
import os
from getpass import getpass

import car_config as config

KAGGLE_SLUG = "adityadesai13/used-car-dataset-ford-and-mercedes"
KAGGLE_URL = "https://www.kaggle.com/datasets/" + KAGGLE_SLUG


def has_price_data():
    d = config.PRICE_DATA_DIR
    if not os.path.isdir(d):
        return False
    found = {f for _, _, fs in os.walk(d) for f in fs}
    return all(f in found for f in config.PRICE_FILES)


def ensure_token():
    home = os.path.expanduser("~")
    files = [os.path.join(home, ".kaggle", n) for n in ("access_token", "kaggle.json")]
    if os.environ.get("KAGGLE_API_TOKEN") or any(os.path.exists(f) for f in files):
        return
    token = getpass("請貼上 Kaggle API Token 後按 Enter（輸入時看不到字是正常的）：").strip()
    os.environ["KAGGLE_API_TOKEN"] = token
    if input("要把 Token 存起來，下次就不用再輸入嗎？(y/N) ").strip().lower() == "y":
        os.makedirs(os.path.join(home, ".kaggle"), exist_ok=True)
        with open(files[0], "w") as f:
            f.write(token)
        print(f"  已存到 {files[0]}")


def download_price():
    if has_price_data():
        print(f"✅ 價格資料已經在 {config.PRICE_DATA_DIR}，略過下載")
        return
    ensure_token()
    print(f"從 Kaggle 下載價格資料 ...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi   # 要在設定 Token 之後才 import
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_SLUG, path=config.PRICE_DATA_DIR, unzip=True, quiet=False)
    except Exception as e:
        raise SystemExit(
            f"\n❌ 下載失敗：{e}\n"
            "可能是 Token 錯誤或過期（401/403）。也可以手動下載：\n"
            f"  1. 打開 {KAGGLE_URL}\n"
            "  2. 按 Download 下載 zip\n"
            f"  3. 解壓縮到專案的 {config.PRICE_DATA_DIR} 資料夾（裡面要看到 audi.csv、bmw.csv…）")
    print(f"✅ 價格資料下載完成：{config.PRICE_DATA_DIR}")


def download_images():
    import prepare_car_images
    prepare_car_images.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", action="store_true")
    ap.add_argument("--images", action="store_true")
    a = ap.parse_args()
    both = not (a.price or a.images)
    if a.price or both:
        download_price()
    if a.images or both:
        download_images()
