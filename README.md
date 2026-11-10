# 🚗 拍照估車價：二手車 AI 估價系統

**拍一張車子的照片 → AI 辨識品牌 → 輸入年份、里程等車況 → 預測二手價格**

| | 圖片辨識 | 數值預測 |
|---|---|---|
| 問題 | 這是什麼品牌的車？ | 這台車二手大概值多少？ |
| 資料 | Stanford Cars（16,185 張，取 7 個品牌＋其他品牌） | UK Used Car（約 10 萬筆，9 個品牌） |
| Scikit-Learn | Logistic Regression（用 CNN 抽出的特徵） | Linear Regression、Random Forest |
| Keras | MobileNetV2 遷移學習 + 資料增強 | Dense 神經網路 |
| 串接 | 辨識出的品牌 → 自動填入估價表單、成為價格模型的輸入特徵 ||

## 專案結構
```
used-car-price/
├── used_car_local.ipynb    ← ⭐ VS Code 本地版筆記本（從這裡開始）
├── car_config.py           ← 設定（品牌、路徑、匯率）。不叫 config.py：會和 Colab 內建的同名模組撞名
├── download_data.py        ← 下載價格資料（Kaggle）和車子照片（Hugging Face）
├── continue_train.py       ← 接著目前的品牌辨識模型繼續訓練（有備份、沒進步不採用）
├── test_my_photos.py       ← 用 my_photos 資料夾裡的照片測試品牌辨識
├── pick_demo_photos.py     ← 從測試集挑出 Demo 用的照片、列出資料集有哪些車款
├── price_utils.py          ← 價格資料讀取、清理、特徵
├── train_price.py          ← 步驟 1：訓練價格模型
├── prepare_car_images.py   ← 步驟 2：下載並整理車子照片
├── train_image.py          ← 步驟 3：訓練品牌辨識模型
├── app.py                  ← 步驟 4：Flask 網站
├── templates/index.html
├── static/chart.umd.min.js ← 圖表套件已內建，Demo 現場沒網路也能跑
└── plot_font.py            ← 圖表中文字型
```

---

## 方法一：在自己電腦用 VS Code 跑（Windows）

### 第一次設定
1. VS Code 安裝擴充套件 **Python**、**Jupyter**
2.「檔案」→「開啟資料夾」→ 選 **`used-car-price`** 資料夾
3. `Ctrl+Shift+P` → **Python: Create Environment** → **Venv** → 選 Python → 勾選 `requirements.txt`
4. 打開 **`used_car_local.ipynb`**，右上角「選取核心」→ 選 `.venv`
5. 從上到下一格一格執行

### 不用筆記本，用終端機也可以
VS Code「終端機」→「新增終端機」（PowerShell），依序執行：
```powershell
.venv\Scripts\activate
python download_data.py          # 下載兩份資料（會要求輸入 Kaggle Token）
python train_price.py            # 價格模型，約 5～15 分鐘
python train_image.py --quick    # 品牌辨識快速測試
python train_image.py            # 品牌辨識正式訓練，CPU 約 1～2 小時
python test_my_photos.py         # 測試 my_photos 資料夾裡的照片
python app.py                    # 打開網站 http://127.0.0.1:5000
```

### Kaggle Token
- 第一次執行 `download_data.py` 會要求貼上 Token，並詢問要不要存起來（存在 `C:\Users\你的帳號\.kaggle\access_token`）
- 也可以先在 PowerShell 設定：`$env:KAGGLE_API_TOKEN="你的Token"`
- 下載失敗時，可以手動到 Kaggle 下載 zip，解壓縮到 `data/used_cars`

### 本地跑要注意
- **Windows 沒有 GPU 加速**（新版 TensorFlow 不支援 Windows 原生 GPU），圖片訓練比 Colab 慢很多
- 時間不夠：`python train_image.py --epochs 8 --finetune-epochs 10`，或只把圖片訓練拿到 Colab 跑（見方法二），再把 `models/image` 資料夾放回本地
- 車子照片約 2GB，下載到使用者資料夾的 Hugging Face 快取，再整理到 `data/car_images`

---

## 方法二：在 Google Colab 跑（有免費 GPU）
打開 **`used_car_colab.ipynb`**（另外提供），上傳到 Colab，從上到下執行即可。

---

## 訓練產生的圖表（簡報用）

**`models/price/`**

| 檔案 | 簡報用途 |
|---|---|
| `1_model_compare.png` | 4 個價格模型誤差率比較 |
| `2_pred_vs_actual.png` | 預測 vs 實際散佈圖 |
| `3_feature_importance.png` | 哪些因素最影響車價 |
| `4_nn_training.png` | 神經網路訓練曲線 |

**`models/image/`**

| 檔案 | 簡報用途 |
|---|---|
| `1_confusion_matrix.png` | 品牌辨識混淆矩陣（哪些品牌最容易搞混） |
| `2_cnn_training.png` | CNN 訓練曲線 |
| `3_examples.png` | 測試照片的辨識範例（對/錯） |

---

## 方法說明

### 數值：二手車價格預測
- **資料清理**：去除年份錯誤（資料裡有 2060 年）、排氣量 0、極端價格、重複資料；車型名稱去除前後空白
- **特徵**：品牌、車型、年份、里程、變速箱、燃料、排氣量。不用牌照稅和油耗，因為一般人估價時不會知道
- **前處理**：類別欄位 One-Hot、數值欄位標準化
- **預測 log(價格)**：價格分布右偏（少數高價車），取 log 後誤差比較平均，最後再轉回來
- **基準線**：「同品牌、同車型、同年份的平均價」，也就是最直覺的估價方式。機器學習模型要贏過它才有意義
- **評估**：平均誤差（£）、平均誤差率 MAPE、R²

### 圖片：品牌辨識
- **資料**：Stanford Cars 原本有 196 類（品牌＋車型＋年份），整理成 8 類：Audi、BMW、Ford、Hyundai、Mercedes-Benz、Toyota、Volkswagen（價格資料也有的品牌）＋**其他品牌**
- **為什麼要「其他品牌」**：如果拍到 Honda，模型才能回答「不支援」，而不是硬猜成 Toyota
- **測試集**：Stanford Cars **官方切好的測試集**，訓練時完全不會用到
- **資料增強**：翻轉、旋轉、縮放、對比、亮度。車子品牌和亮度、顏色無關，所以可以放心調（和空污主題剛好相反）
- **兩階段訓練**：先凍結 MobileNetV2 只訓練分類層，再解凍後 40 層微調；**微調後驗證準確率沒變好就保留第一階段**
- **Scikit-Learn 對照**：MobileNetV2 只當「特徵抽取器」，每張照片變成 1280 維向量，再交給 Logistic Regression 分類

## ⚠ 限制（簡報「討論」頁可用）
1. **價格是英國市場、約 2020 年的資料**，台幣依匯率換算，和台灣行情一定有差異。台灣沒有公開的大型二手車成交資料
2. **Stanford Cars 多是 2012 年以前的美國車款**，較新的車款（例如新一代 Toyota）外觀可能差很多，辨識準確度會下降。**請提早用手機拍幾台車測試**
3. 圖片只辨識「品牌」，「車型」需要使用者自己選。可以延伸成辨識車型
4. 「其他品牌」只抽樣部分照片，遇到少見品牌仍可能誤判

## 資料來源
- Krause, J. et al. (2013). *3D Object Representations for Fine-Grained Categorization* (Stanford Cars). 僅供非商業研究使用
- Aditya Desai. *100,000 UK Used Car Data set*. Kaggle
