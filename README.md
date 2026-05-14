# 太陽能安裝潛力地區預測模型

> 國立臺灣大學 資訊管理學系｜製造數據科學｜第六組

---

## 專案簡介

太陽能安裝普及受環境、社經及政策等多重因素影響，識別高潛力推廣地區為能源轉型之關鍵決策議題。本專案以 Stanford DeepSolar 資料集為基礎，涵蓋全美逾 73,000 筆人口普查區資料與 169 個欄位，分三階段建構太陽能安裝潛力預測模型：

1. **美國建模**：以原論文 SolarForest 為基準，透過特徵篩選與機器學習方法建立具可解釋性之預測模型
2. **跨國遷移**：篩除美國特有地域性欄位，保留可泛化特徵，結合臺灣社經、氣象與政策資料進行潛力推論
3. **驗證與決策**：串接台電太陽光電發電量統計及各科學園區光電設備資料進行合理性驗證，並以多屬性決策分析（MADA）框架排序各地區推廣優先序

**應用場景**：太陽能業者廣告投放目標篩選、政府補助資源精準分配

---

## 成員

| 姓名 | 學號 |
|------|------|
| 李倢安 | B12705041|
| 林湘庭 | B12705008|
| 張佑丞 | B12705055 |
| 藍柏婷 | B12705030|
| 柯絲昀 | B12705066|

---

## 資料來源

| 資料集 | 來源 | 用途 |
|--------|------|------|
| DeepSolar Dataset | [Kaggle](https://www.kaggle.com/datasets/tunguz/deep-solar-dataset) | 美國建模主資料 |
| 家庭收支調查 | 行政院主計總處 | 台灣社經特徵 |
| 氣象觀測資料 | 中央氣象署 | 台灣日照輻射量 |
| 太陽光電發電量統計 | 台灣電力公司 | 外部驗證 |
| 科學園區光電設備資料 | 國科會 | 外部驗證 |

---

## 專案結構

```
project/
├── README.md
├── CLAUDE.md                  # Claude Code 工作指引
├── requirements.txt
├── deepsolar_tract.csv        # 原始資料（請自行下載）
├── data/
│   ├── processed/             # 處理後資料
│   └── taiwan/                # 台灣對應資料
├── notebooks/
│   ├── 01_eda.ipynb           # 探索性資料分析
│   ├── 02_feature_selection.ipynb   # 共線性篩選
│   ├── 03_model_us.ipynb      # 美國模型訓練
│   ├── 04_transfer_taiwan.ipynb     # 台灣遷移推論
│   └── 05_mada.ipynb          # MADA 優先序排名
├── src/
│   ├── features.py            # 特徵工程
│   ├── model.py               # 模型訓練與評估
│   └── mada.py                # MADA 排序（TOPSIS）
└── outputs/
    ├── figures/               # 圖表輸出
    └── results/               # 模型結果與排名
```

---

## 環境安裝

Python 3.10 以上

```bash
git clone https://github.com/your-repo/deepsolar-taiwan.git
cd deepsolar-taiwan
pip install -r requirements.txt
```

`requirements.txt`：
```
pandas
numpy
scikit-learn
xgboost
statsmodels
scipy
matplotlib
seaborn
jupyter
kaggle
```

下載資料集：
```bash
kaggle datasets download -d tunguz/deep-solar-dataset
unzip deep-solar-dataset.zip
```

---

## 執行流程

```bash
# 依序執行各階段 notebook
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_selection.ipynb
jupyter notebook notebooks/03_model_us.ipynb
jupyter notebook notebooks/04_transfer_taiwan.ipynb
jupyter notebook notebooks/05_mada.ipynb
```

---

## 方法摘要

### 特徵工程
依欄位名稱 prefix 將 169 個變數分群（環境、社經、政策、住宅等），並以 Pearson 相關係數（閾值 |r| > 0.85）與變異數膨脹因子（VIF > 10）篩除共線性特徵。

### 模型
以梯度提升分類器（GBM）為主模型，RandomForest 對應原論文 SolarForest 作為基準比較，StratifiedKFold 五折交叉驗證評估 F1、AUC-ROC。

### 跨國遷移
移除美國特有欄位（補貼制度、族裔組成），保留氣候、電力、社經等可泛化特徵，對齊台灣行政區資料後進行潛力推論。

### 決策排序（MADA）
採用 TOPSIS 方法，整合模型潛力分數、日照輻射量、電價、家戶收入等屬性，輸出各地區推廣優先排名。

### 驗證
以 Spearman 相關係數衡量預測排名與台電實際裝設量之一致性，驗證跨國遷移模型的有效性。

---

## 指導資訊

- **課程**：製造數據科學
- **系所**：國立臺灣大學 資訊管理學系
- **學期**：2025–2026 學年度

---

## 參考文獻

Yu, J., Wang, Z., Majumdar, A., & Rajagopal, R. (2018). DeepSolar: A machine learning framework to efficiently construct a solar deployment database in the United States. *Joule*, 2(12), 2605–2617. https://doi.org/10.1016/j.joule.2018.11.021