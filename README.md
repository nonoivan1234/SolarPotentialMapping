# 太陽能安裝潛力地區預測模型

> 國立臺灣大學 資訊管理學系｜製造數據科學｜第六組

---

## 專案簡介

太陽能安裝普及受環境、社經及政策等多重因素影響，識別高潛力推廣地區為能源轉型之關鍵決策議題。本專案以 Stanford DeepSolar 資料集為基礎，涵蓋全美逾 73,000 筆人口普查區資料與 169 個欄位，分三階段建構太陽能安裝潛力預測模型：

1. **美國建模**：以原論文 SolarForest 為基準，透過特徵篩選與機器學習方法建立具可解釋性之預測模型
2. **跨國遷移**：移除無法對應至台灣制度之美國地域性欄位，保留具一般能源經濟意義之政策、電價、氣候與社經特徵，結合臺灣資料建立相對部署傾向與推廣優先度排序
3. **驗證與決策**：串接台電太陽光電發電量統計及各科學園區光電設備資料進行合理性驗證，並以多屬性決策分析（MADA）框架排序各地區推廣優先序

**應用場景**：太陽能業者廣告投放目標篩選、政府補助資源精準分配

> 本研究不直接預測台灣真實裝設密度，而是利用美國 DeepSolar 訓練出的 adoption/deployment pattern，建立台灣各地區的相對太陽能部署傾向與推廣優先度排序。

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
│   │   ├── legacy/             # 舊版 baseline 輸出
│   │   └── transfer/           # 台灣遷移導向新版輸出
│   └── taiwan/                # 台灣對應資料
├── notebooks/
│   ├── 01_eda.ipynb           # 探索性資料分析
│   ├── 02_feature_selection.ipynb   # VIF 與 tree/transfer 特徵集
│   ├── 03_model_us.ipynb      # 美國三分類 baseline 模型
│   ├── 03_model_us_twostage.ipynb   # Two-stage / SolarForest-style 正式模型
│   ├── 03b_model_explainability.ipynb   # SHAP 模型解釋（三分類）
│   ├── 03b_model_explainability_twostage.ipynb  # SHAP 模型解釋（two-stage）
│   ├── legacy/                 # 舊版 notebook 備份
│   ├── 04_transfer_taiwan.ipynb     # 台灣遷移推論
│   └── 05_mada.ipynb          # MADA 優先序排名
├── src/
│   ├── features.py            # 特徵工程
│   ├── model.py               # 模型訓練與評估
│   └── mada.py                # MADA 排序（TOPSIS）
└── outputs/
    ├── figures/
    │   ├── legacy/             # 舊版圖表
    │   └── transfer/           # 新版圖表
    └── results/
        ├── legacy/             # 舊版分析結果
        └── transfer/           # 新版分析結果
```

---

## 環境安裝

Python 3.10 以上

```powershell
git clone https://github.com/your-repo/deepsolar-taiwan.git
cd deepsolar-taiwan
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Windows 建議用 `python -m pip`，不要直接用 `pip`，可避免 PowerShell 找不到 `pip` 指令。

若 PowerShell 阻擋虛擬環境啟動，可先執行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
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
```powershell
kaggle datasets download -d tunguz/deep-solar-dataset
unzip deep-solar-dataset.zip
# Windows 的解壓縮請改用：Expand-Archive .\deep-solar-dataset.zip -DestinationPath .
```

解壓縮後，專案根目錄應出現：

```text
deepsolar_tract.csv
```

第一次執行 notebook 前，請先建立輸出資料夾，否則 `plt.savefig()` 或 `to_csv()` 可能因目標路徑不存在而失敗：

```powershell
New-Item -ItemType Directory -Force -Path data\processed\legacy, data\processed\transfer, data\taiwan, outputs\figures\legacy, outputs\figures\transfer, outputs\results\legacy, outputs\results\transfer
```

---

## 執行流程

```bash
# 依序執行各階段 notebook
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_selection.ipynb
# jupyter notebook notebooks/03_model_us.ipynb    # 三分類 baseline（可選，供比較用）
jupyter notebook notebooks/03_model_us_twostage.ipynb
# jupyter notebook notebooks/03b_model_explainability.ipynb    # 三分類 SHAP（可選，供比較用）
jupyter notebook notebooks/03b_model_explainability_twostage.ipynb
jupyter notebook notebooks/04_transfer_taiwan.ipynb    # 尚未實作
jupyter notebook notebooks/05_mada.ipynb    # 尚未實作
```

---

## 方法摘要

### 特徵工程
依欄位名稱 prefix 將 169 個變數分群（環境、社經、政策、住宅等）。新版流程區分不可轉移欄位（如族裔、投票、行政代碼）與可重新 mapping 的政策/能源經濟概念，並同時輸出 VIF 篩選版與 tree/transfer 版特徵集。

### 目標變數定義

本專案目前保留兩種美國 baseline：

1. **三分類 baseline**：用 `tile_count` 的分位數切成低/中/高部署程度，方便快速比較特徵集與分類模型。
2. **Two-stage / SolarForest-style model**：更接近 DeepSolar 原論文邏輯，先判斷是否已有太陽能部署，再針對已部署地區預測部署強度。

#### 三分類 baseline

以 `tile_count`（普查區內現有太陽能板數量）的三分位數切分預測目標：

| Class | 條件 | 意義 | 業務優先度 |
|-------|------|------|-----------|
| **2（高潛力）** | `tile_count ≤ q33` | 裝設量低，開發空間大 | 主要廣告投放 / 補助目標 |
| **1（中潛力）** | `q33 < tile_count ≤ q66` | 部分滲透，次要推廣 | 次要目標 |
| **0（已飽和）** | `tile_count > q66` | 普及率高，邊際效益低 | 排除 |

> 裝設量**低**代表潛力**高**是業務推廣角度的 proxy，不代表物理條件一定更適合。  
> 因此三分類結果應解讀為 deployment/adoption level proxy，而不是真正的物理可行性等級。

#### Two-stage model

Two-stage model 將問題拆成兩步：

```text
Stage 1:
tile_count > 0 嗎？
判斷一個 census tract 是否已有太陽能部署。

Stage 2:
若 tile_count > 0，預測 log1p(tile_count / household_count × 1000)。
也就是在已部署地區中，以每千戶部署密度估計部署強度。
```

這比三分類更接近原論文 SolarForest 的精神，也更適合作為正式方法主線。三分類 baseline 仍保留，用來做快速比較與消融實驗。

### 模型

`03_model_us.ipynb` 是三分類 baseline，會比較 Logistic Regression、RandomForest、GradientBoosting 與 XGBoost，並輸出 `model_comparison_*`、`feature_importance_*` 與模型檔。

`03_model_us_twostage.ipynb` 是 two-stage formal baseline：

- Stage 1 使用 binary classifier 預測 `tile_count > 0`
- Stage 2 使用 regressor 只針對 `tile_count > 0` 的地區預測 `log1p(tile_count / household_count × 1000)`（每千戶部署密度）
- 預設模型為 RandomForest，對應原論文 SolarForest 思路；也可切換為 XGBoost

兩者輸出分開命名，避免混淆：

- 三分類：`best_us_model_tree.pkl`
- Two-stage：`best_us_twostage_model_tree.pkl`

### 跨國遷移
移除無法對應至台灣制度之美國地域性欄位，保留具一般能源經濟意義之政策、電價與氣候特徵，並於台灣遷移階段重新建立對應 mapping。

本研究不直接預測台灣真實裝設密度，而是利用美國 DeepSolar 訓練出的 adoption/deployment pattern，建立台灣各地區的相對太陽能部署傾向與推廣優先度排序。因此，台灣端模型輸出應解讀為 relative score / ranking，而不是絕對安裝密度、真實發電量或 ROI。

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
