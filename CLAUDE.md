# CLAUDE.md — 太陽能安裝潛力預測專案

## 專案概述

以 Stanford DeepSolar 資料集建構太陽能安裝潛力預測模型，分三階段進行：
1. 以原論文 SolarForest 為基準，建立可解釋性預測模型
2. 篩除美國地域性欄位，保留可跨國泛化特徵，推論台灣各地區潛力
3. 串接台電資料驗證合理性，以 MADA 排序各地區推廣優先序

**商業目的**：廣告投放目標地區篩選、政府補助資源分配

---

## 資料集

- **來源**：https://www.kaggle.com/datasets/tunguz/deep-solar-dataset
- **檔名**：`deepsolar_tract.csv`（下載後放在專案根目錄）
- **規模**：73,000+ 筆人口普查區 × 169 欄
- **目標變數**：`tile_count`（現有太陽能板數量）

下載指令：
```bash
kaggle datasets download -d tunguz/deep-solar-dataset
unzip deep-solar-dataset.zip
```

---

## 專案結構

```
project/
├── CLAUDE.md
├── deepsolar_tract.csv
├── data/
│   └── taiwan/              # 台灣社經、氣象、政策資料（第二階段用）
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_selection.ipynb
│   ├── 03_model_us.ipynb
│   ├── 04_transfer_taiwan.ipynb
│   └── 05_mada.ipynb
├── src/
│   ├── features.py          # 特徵工程函式
│   ├── model.py             # 模型訓練與評估
│   └── mada.py              # MADA 排序邏輯
├── outputs/
│   ├── figures/
│   └── results/
└── requirements.txt
```

---

## 執行順序

### STEP 1｜EDA & 欄位 Prefix 分群

```python
# 依欄位名稱底線前的第一個詞分群
def get_prefix(col):
    return col.split('_')[0] if '_' in col else 'single'
```

預期的 prefix 群組（共約 15 群）：
| Prefix | 內容 | 是否可跨國泛化 |
|--------|------|----------------|
| `solar` | 太陽能裝設量（**目標變數群**） | — |
| `avg` / `median` | 收入、房價等平均/中位數 | ✓ |
| `gini` | 收入不平等指標 | ✓ |
| `electricity` | 電力消費與電價 | ✓ |
| `heating` / `cooling` | 冷暖度日數（氣候） | ✓ |
| `incentive` | 補貼政策年限 | ✗ 美國特有 |
| `housing` | 住宅特徵 | ✓ |
| `population` | 人口統計 | ✓ |
| `education` | 教育程度 | ✓ |
| `poverty` | 貧窮率 | ✓ |
| `age` | 年齡分布 | ✓ |
| `employment` | 就業率 | ✓ |
| `race` / `white` / `black` | 族裔組成 | ✗ 美國特有 |
| `fips` / `state` / `county` | 地理識別碼 | ✗ 僅識別用，不入模型 |

---

### STEP 2｜共線性篩選

執行順序：
1. 對所有數值欄位計算 **Pearson 相關係數矩陣**
2. 標記 `|r| > 0.85` 的變數對
3. 每對中保留與目標變數相關性較高者，刪除另一個
4. 以 **VIF（變異數膨脹因子）** 二次確認，移除 VIF > 10 的欄位

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data['feature'] = features
    vif_data['VIF'] = [variance_inflation_factor(df[features].values, i)
                       for i in range(len(features))]
    return vif_data.sort_values('VIF', ascending=False)
```

**輸出**：`outputs/results/selected_features.csv`（保留欄位清單）

---

### STEP 3｜目標變數定義（三分法）

```python
q33 = df['tile_count'].quantile(0.33)
q66 = df['tile_count'].quantile(0.66)

def label_potential(n):
    if n <= q33:   return 2   # 高潛力：尚未普及
    elif n <= q66: return 1   # 中潛力
    else:          return 0   # 已飽和，廣告價值低
```

三類標籤的商業意義：
- **Class 2（高潛力）**：主要廣告投放與政府補助目標
- **Class 1（中潛力）**：次要推廣目標
- **Class 0（已飽和）**：排除，資源轉移至其他地區

---

### STEP 4｜模型訓練（美國資料）

使用以下模型並比較：
- `RandomForestClassifier`（對應原論文 SolarForest）
- `GradientBoostingClassifier` / `XGBClassifier`
- `LogisticRegression`（可解釋性 baseline）

評估指標：
```python
metrics = {
    'accuracy', 'f1_macro', 'roc_auc_ovr',
    'precision_macro', 'recall_macro'
}
# 使用 StratifiedKFold(n_splits=5) 交叉驗證
```

**輸出**：`outputs/results/model_comparison.csv`

---

### STEP 5｜跨國遷移推論（台灣）

移除步驟：
1. 刪除所有 `incentive_*` 欄位（美國補貼制度）
2. 刪除所有族裔相關欄位（`race_*`, `white_*`, `black_*`, `hispanic_*`）
3. 刪除地理識別碼（`fips`, `state`, `county`）
4. 保留氣候、電力、社經、住宅類特徵

台灣對應資料來源建議：
| 特徵類型 | 台灣資料來源 |
|----------|-------------|
| 日照輻射量 | 中央氣象署、NASA POWER API |
| 電力零售價 | 台電電價查詢 |
| 家戶收入 | 主計總處家庭收支調查 |
| 人口密度 | 內政部戶政司 |
| 政策補貼 | 能源局太陽光電政策 |

---

### STEP 6｜MADA 排序

框架說明：
- **MADA（Multi-Attribute Decision Analysis）**：備選方案已知（各行政區），根據多維屬性排出推廣優先序
- 不是 MODA（不做 Pareto 最佳化）

推薦方法：**TOPSIS**（Technique for Order of Preference by Similarity to Ideal Solution）

```python
# 屬性權重（可依政策需求調整）
weights = {
    'predicted_potential_score': 0.40,  # 模型輸出潛力分數
    'solar_irradiance':          0.25,  # 日照輻射量
    'electricity_price':         0.20,  # 電價（越高越有誘因）
    'median_household_income':   0.15,  # 收入（支付能力）
}
```

**輸出**：`outputs/results/taiwan_priority_ranking.csv`

---

### STEP 7｜驗證

以台電與科學園區資料做外部驗證：
- 計算預測排名與實際裝設量的 **Spearman 相關係數**
- 比原論文直接套用的相關係數更高 → 證明跨國篩選有效

```python
from scipy.stats import spearmanr
rho, p = spearmanr(predicted_rank, actual_installation)
print(f"Spearman ρ = {rho:.3f}, p = {p:.4f}")
```

---

## 對比原論文的貢獻說明

| 面向 | 原論文 SolarForest | 本研究 |
|------|-------------------|--------|
| 適用範圍 | 美國 | 可跨國遷移 |
| 特徵集 | 全部 94 欄（含美國特有） | 系統篩除地域性欄位 |
| 決策輸出 | 密度預測值 | 可操作的優先排序清單 |
| 驗證方式 | 美國內部交叉驗證 | 台灣實際裝設量外部驗證 |

---

## 注意事項

- 所有圖表存至 `outputs/figures/`，格式用 `.png`（300 dpi）
- 每個 notebook 開頭 `set_seed(42)` 確保可重現
- 不要直接修改原始 `deepsolar_tract.csv`，所有處理後的資料存 `data/processed/`
- 台灣資料若格式不同，在 `src/features.py` 裡寫對應的 `align_taiwan_features()` 函式統一對齊欄位