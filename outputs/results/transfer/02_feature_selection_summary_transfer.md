# 02 Feature Selection Summary - Transfer Pipeline

## 中文版

### 這一步在做什麼

`02_feature_selection` 是新版 transfer pipeline 的特徵篩選。它的目標不是只產生一組特徵，而是同時產出兩種用途不同的 feature set：

- `tree` 版：給 XGBoost / RandomForest / LightGBM 這類 tree-based model 使用，不做 VIF 移除。
- `vif` 版：保留給需要較低共線性的解釋型模型或比較實驗使用。

這樣做的原因是：VIF 對線性模型很重要，但對 XGBoost 不一定必要。XGBoost 可以處理高度相關的特徵；如果硬做 VIF，反而可能把 domain-critical features 刪掉，例如 `daily_solar_radiation`, `relative_humidity`, `electricity_price_*`。

### 前處理結果

新版流程先移除：

- target / leakage 欄位
- ID/helper 欄位
- 真正不適合轉移的美國特有欄位，例如 race、voting、state/county/fips

但保留政策與能源經濟概念：

- `incentive_count_residential`
- `incentive_count_nonresidential`
- `incentive_residential_state_level`
- `incentive_nonresidential_state_level`
- `net_metering`
- `feedin_tariff`
- `cooperate_tax`
- `property_tax`
- `sales_tax`
- `rebate`
- `avg_electricity_retail_rate`

結果：

- 初始 feature columns：`132`
- numeric columns：`130`
- target 不缺失的資料筆數：`72,537`
- median imputation 後無剩餘缺失值

### 分類標籤

這一步把 `tile_count` 切成三類 `potential_label`：

- `0`：高部署 / 低推廣潛力（已飽和區，tile_count > q66）
- `1`：中等部署 / 中推廣潛力
- `2`：低部署 / 高推廣潛力（未開發區，tile_count ≤ q33）

切點：

- 33% quantile：`1`
- 66% quantile：`11`

三類樣本數接近平衡：

- label `0`：`24,279`
- label `1`：`24,174`
- label `2`：`24,084`

這代表後續分類模型不會嚴重受類別不平衡影響。

### Tree Feature Set

Correlation screening 後：

- 原本 `130` 個 numeric features
- 篩掉 `28` 個高度相關欄位
- 保留 `102` 個 tree features

輸出：

- `outputs/results/transfer/selected_features_tree.csv`
- `data/processed/transfer/us_modeling_ready_tree.csv`

建模資料大小：

- `(72,537, 103)`
- 其中 `102` 個 features，加上 `potential_label`

tree 版 top features by absolute target correlation：

| feature | target_corr | 解讀 |
|---|---:|---|
| `incentive_count_residential` | `0.361` | 住宅補助/誘因越多，太陽能部署通常越高 |
| `relative_humidity` | `-0.353` | 濕度越高，部署量傾向越低，可能 proxy 雲量/氣候條件 |
| `electricity_price_industrial` | `0.336` | 工業電價越高，太陽能投資誘因可能越強 |
| `daily_solar_radiation` | `0.325` | 日照越高，部署量傾向越高 |
| `feedin_tariff` | `0.322` | FIT 政策與部署量正相關 |
| `electricity_price_commercial` | `0.294` | 商業電價與太陽能部署正相關 |
| `housing_unit_median_gross_rent` | `0.291` | 租金/地區經濟條件可能與部署能力有關 |
| `rebate` | `0.280` | rebate 與部署量正相關 |
| `incentive_residential_state_level` | `0.269` | 州級住宅誘因與部署量正相關 |
| `education_college` | `0.246` | 教育程度相關社經條件與部署量有關 |

tree 版保留了幾個對台灣遷移很重要的欄位：

- `daily_solar_radiation`
- `relative_humidity`
- `electricity_price_industrial`
- `electricity_price_commercial`
- `feedin_tariff`
- `rebate`
- `incentive_*`
- `housing_*`
- `education_*`

這是新版流程最重要的結果。

### VIF Feature Set

VIF 版從 tree feature set 繼續做 iterative VIF removal，直到最大 VIF 小於等於 `10`，或達到迭代上限。

結果：

- VIF feature count：`55`
- 建模資料大小：`(72,537, 56)`
- 其中 `55` 個 features，加上 `potential_label`

輸出：

- `outputs/results/transfer/selected_features_vif.csv`
- `data/processed/transfer/us_modeling_ready_vif.csv`

VIF 版 top features by absolute target correlation：

| feature | target_corr | 解讀 |
|---|---:|---|
| `feedin_tariff` | `0.322` | FIT 是保留下來的重要政策變數 |
| `rebate` | `0.280` | rebate 與部署量正相關 |
| `heating_fuel_solar` | `0.245` | 住宅能源使用型態中的 solar proxy |
| `frost_days` | `-0.230` | 氣候冷/嚴苛程度 proxy，目前先保留 |
| `housing_unit_median_value` | `0.222` | 房價/資產條件與部署能力有關 |
| `education_bachelor` | `0.216` | 教育程度與部署量正相關 |
| `property_tax` | `0.179` | 稅制政策概念仍被保留 |
| `heating_fuel_solar_rate` | `0.152` | solar heating fuel rate 與部署量相關 |
| `unemployed` | `0.146` | 社經結構 proxy |
| `net_metering` | `0.138` | net metering 政策概念保留 |

### 重要觀察

VIF 版雖然更乾淨，但它刪掉了不少 domain-critical features：

- `daily_solar_radiation`
- `relative_humidity`
- `population`
- `electricity_price_commercial`
- `median_household_income`
- `electricity_consume_residential`
- `wind_speed`

這些欄位被刪不是因為「不重要」，而是因為它們與其他欄位高度共線。對 XGBoost 來說，這不一定是問題，所以新版主線建議以 `tree` 版作為 03 model 的預設輸入。

### 建議怎麼用

目前建議：

- 主線模型：使用 `us_modeling_ready_tree.csv`
- 比較/解釋模型：可使用 `us_modeling_ready_vif.csv`
- 台灣遷移：優先參考 `selected_features_tree.csv`，再逐欄建立台灣對應資料

也就是：

```text
01 EDA
-> 02 feature selection
-> selected_features_tree.csv
-> 03_model_us.ipynb
-> 03b_model_explainability.ipynb
-> Taiwan feature mapping
```

### 這一步的產出

- `outputs/results/transfer/selected_features_tree.csv`
- `outputs/results/transfer/selected_features_vif.csv`
- `data/processed/transfer/us_modeling_ready_tree.csv`
- `data/processed/transfer/us_modeling_ready_vif.csv`
- `data/processed/transfer/column_metadata.json`
- `outputs/figures/transfer/02_correlation_heatmap_transfer.png`
- `outputs/figures/transfer/02_vif_final_transfer.png`

---

## English Summary

`02_feature_selection` now produces two transfer-oriented feature sets. The `tree` feature set keeps `102` features after correlation screening and is the recommended input for XGBoost/RandomForest/LightGBM models. The `vif` feature set keeps `55` features after iterative VIF removal and is useful for lower-collinearity comparison or more linear interpretability workflows.

The redesigned workflow keeps policy and energy-economic concepts such as incentives, electricity prices, FIT, rebates, taxes, and net metering. This is a major change from the earlier US-only logic: these fields are not directly transferable as US encodings, but their concepts can be mapped to Taiwan later.

The main recommendation is to use `data/processed/transfer/us_modeling_ready_tree.csv` for `03_model_us.ipynb`, because VIF removal can drop domain-critical features that tree-based models can still use effectively.
