"""讓 matplotlib 圖表能顯示中文（Windows / Mac / Colab 都適用）。"""
import glob

import matplotlib.pyplot as plt
from matplotlib import font_manager

CANDIDATES = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "Noto Sans CJK JP",
              "Noto Sans TC", "Heiti TC", "Arial Unicode MS"]


def setup_chinese_font():
    # Colab / Linux：裝了 fonts-noto-cjk 後，字型檔不一定被 matplotlib 自動登記，手動加入
    for path in glob.glob("/usr/share/fonts/**/*CJK*", recursive=True):
        try:
            font_manager.fontManager.addfont(path)
        except Exception:
            pass
    available = {f.name for f in font_manager.fontManager.ttflist}
    found = [c for c in CANDIDATES if c in available]
    if not found:
        print("⚠ 找不到中文字型，圖表中文會變方框。"
              "Colab 請先執行：!apt-get -qq install -y fonts-noto-cjk")
    plt.rcParams["font.sans-serif"] = found + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
