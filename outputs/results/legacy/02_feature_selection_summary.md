# 02 Feature Selection Summary

## Purpose

This notebook prepares the DeepSolar data for modeling by selecting a cleaner and less redundant feature set.

Main steps:

1. Drop US-specific columns and target leakage columns.
2. Keep `tile_count` temporarily so it can be converted into a classification target.
3. Impute missing numeric values.
4. Convert `tile_count` into three potential classes.
5. Remove highly correlated feature pairs.
6. Apply iterative VIF screening to reduce multicollinearity.
7. Export selected features and the modeling-ready dataset.

## Load and Prepare

```text
Feature columns after dropping US-only/targets: 121
```

The notebook removes US-specific fields such as voting, race, incentive, and location identifiers. It also removes solar outcome fields that would leak the answer into the model, while keeping `tile_count` temporarily as the target.

After this step, 121 columns remain before numeric filtering and feature selection.

## Missing Value Imputation

```text
Numeric columns: 119
Rows after dropping missing target: 72537
Any remaining missing: 0
```

There are 119 numeric candidate feature columns after removing the target. No rows were dropped for missing `tile_count`, so the full 72,537 observations are retained.

Remaining missing values were filled with each column's median. This is a simple and stable baseline imputation strategy, especially for skewed socioeconomic and climate variables.

## Target Definition

The notebook converts `tile_count` into a three-class label using the 33rd and 66th percentiles.

```text
Quantiles: 33%=1.00, 66%=11.00

potential_label
0    24279
1    24174
2    24084
```

The labels are defined as:

- `2`: `tile_count <= q33`, interpreted as high future potential or low current deployment.
- `1`: `q33 < tile_count <= q66`, interpreted as medium potential.
- `0`: `tile_count > q66`, interpreted as saturated or already high-deployment areas.

The three classes are almost perfectly balanced, with about 24,000 observations per class. This is useful for classification because the model will not be dominated by one class.

One conceptual note: the class names should be explained clearly in the final report. Since `tile_count` measures existing deployment, low `tile_count` does not automatically mean high solar potential. It means low current deployment, which the project interprets as possible opportunity when combined with other features.

## Correlation Screening

```text
Features after correlation screening: 94 (dropped 25)
```

The notebook removes one feature from each pair of highly correlated variables where absolute Pearson correlation exceeds 0.85.

This reduces the feature set from 119 to 94 candidate features. The goal is to remove duplicated information before the VIF step. Pairwise correlation screening catches simple cases such as two variables that are almost direct substitutes.

Generated figure:

```text
outputs/figures/02_correlation_heatmap.png
```

## VIF Screening

The notebook then applies iterative VIF screening with:

```text
VIF threshold = 10
Max iterations = 50
```

VIF stands for Variance Inflation Factor. It measures how strongly one feature can be explained by the other features. A high VIF means the feature is highly redundant with the rest of the feature set.

The iterative process is:

1. Compute VIF for all remaining features.
2. Find the feature with the highest VIF.
3. If the highest VIF is less than or equal to 10, stop.
4. Otherwise, drop the feature with the highest VIF.
5. Recompute VIF and repeat.

Important dropped features include:

```text
atmospheric_pressure          VIF=4316.6
transportation_car_alone_rate VIF=2125.7
heating_fuel_gas_rate         VIF=1972.3
employ_rate                   VIF=1161.6
cooling_design_temperature    VIF=869.9
daily_solar_radiation         VIF=319.8
relative_humidity             VIF=230.3
median_household_income       VIF=45.8
electricity_price_industrial  VIF=19.9
```

The VIF values are extremely high at the beginning, which means many features are combinations or close substitutes of other features. This is common in census-style datasets, where rates, counts, income measures, age shares, education shares, and climate variables often overlap.

Final result:

```text
Final valid feature count: 49
```

The VIF step reduces the feature set from 94 to 49 valid features. These 49 features are used for downstream modeling.

Generated figure:

```text
outputs/figures/02_vif_final.png
```

## Selected Feature Dictionary

The current selected feature list contains 49 valid modeling features.

| Original column | Category | 中文欄位意義 |
|---|---|---|
| `heating_fuel_solar` | Energy / Housing | 使用太陽能作為住宅取暖能源的戶數 |
| `frost_days` | Climate | 一年中霜凍天數，反映寒冷氣候程度 |
| `housing_unit_median_value` | Housing / Income | 住宅單位房價中位數 |
| `education_bachelor` | Education | 最高學歷為學士的人數 |
| `heating_fuel_solar_rate` | Energy / Housing | 使用太陽能取暖的住宅比例 |
| `unemployed` | Labor / Economy | 失業人口數 |
| `education_doctoral` | Education | 最高學歷為博士的人數 |
| `poverty_family_below_poverty_level_rate` | Income / Poverty | 低於貧窮線家庭比例 |
| `diversity` | Demographics | 人口多樣性指標 |
| `occupancy_vacant_rate` | Housing | 空屋比例 |
| `transportation_home_rate` | Transportation | 在家工作或不通勤比例 |
| `occupation_administrative_rate` | Occupation | 行政支援相關職業比例 |
| `health_insurance_none_rate` | Health / Socioeconomic | 無健康保險人口比例 |
| `occupation_manufacturing_rate` | Occupation | 製造業職業比例 |
| `transportation_motorcycle_rate` | Transportation | 摩托車通勤比例 |
| `heating_fuel_coal_coke_rate` | Energy / Housing | 使用煤或焦炭取暖的住宅比例 |
| `travel_time_less_than_10_rate` | Transportation | 通勤時間少於 10 分鐘比例 |
| `occupation_finance_rate` | Occupation | 金融、保險、房地產相關職業比例 |
| `education_professional_school_rate` | Education | 專業學位人口比例 |
| `heating_fuel_none` | Energy / Housing | 無取暖燃料的住宅戶數 |
| `transportation_public_rate` | Transportation | 大眾運輸通勤比例 |
| `transportation_walk_rate` | Transportation | 步行通勤比例 |
| `education_doctoral_rate` | Education | 博士學歷人口比例 |
| `occupation_information_rate` | Occupation | 資訊產業相關職業比例 |
| `occupation_public_rate` | Occupation | 公共行政相關職業比例 |
| `heating_fuel_other_rate` | Energy / Housing | 使用其他取暖燃料的住宅比例 |
| `travel_time_60_89_rate` | Transportation | 通勤時間 60 到 89 分鐘比例 |
| `population_density` | Demographics / Geography | 人口密度 |
| `heating_fuel_electricity` | Energy / Housing | 使用電力取暖的住宅戶數 |
| `education_less_than_high_school` | Education | 低於高中學歷的人數 |
| `electricity_consume_total` | Electricity | 總用電量 |
| `travel_time_40_59_rate` | Transportation | 通勤時間 40 到 59 分鐘比例 |
| `age_15_17_rate` | Age | 15 到 17 歲人口比例 |
| `occupation_wholesale_rate` | Occupation | 批發業職業比例 |
| `total_area` | Geography | 地區總面積 |
| `occupation_transportation_rate` | Occupation | 運輸、倉儲、公用事業相關職業比例 |
| `heating_fuel_other` | Energy / Housing | 使用其他取暖燃料的住宅戶數 |
| `travel_time_30_39_rate` | Transportation | 通勤時間 30 到 39 分鐘比例 |
| `age_18_24_rate` | Age | 18 到 24 歲人口比例 |
| `occupation_construction_rate` | Occupation | 營造業職業比例 |
| `occupation_retail_rate` | Occupation | 零售業職業比例 |
| `occupation_agriculture_rate` | Occupation | 農林漁牧等職業比例 |
| `water_area` | Geography | 水域面積 |
| `heating_fuel_fuel_oil_kerosene_rate` | Energy / Housing | 使用燃油或煤油取暖的住宅比例 |
| `occupation_arts_rate` | Occupation | 藝術、娛樂、餐旅等服務職業比例 |
| `age_75_84_rate` | Age | 75 到 84 歲人口比例 |
| `transportation_bicycle_rate` | Transportation | 自行車通勤比例 |
| `age_more_than_85_rate` | Age | 85 歲以上人口比例 |
| `transportation_carpool_rate` | Transportation | 共乘通勤比例 |

High-level feature groups:

- **Energy / Housing**: heating fuel variables and electricity consumption. These describe household energy structure and building or housing conditions.
- **Socioeconomic / Education**: education, poverty, unemployment, health insurance, and housing value variables. These capture adoption capacity and economic context.
- **Occupation / Transportation**: job sector and commuting variables. These may reflect urban form, lifestyle, and local economic structure.
- **Climate / Geography**: frost days, total area, water area, and population density. These describe environmental and spatial conditions.
- **Age / Demographics**: age shares and diversity. These capture population composition.

## Exported Outputs

Selected feature list:

```text
outputs/results/selected_features.csv
```

Modeling-ready dataset:

```text
data/processed/us_modeling_ready.csv
Shape: (72537, 50)
```

The modeling dataset contains 49 selected features plus the `potential_label` target column.

Final class distribution:

```text
potential_label
0    24279
1    24174
2    24084
```

## Issues and Notes

During review, `target_corr` in `selected_features.csv` was found to be blank for all rows. The correlations were not actually missing; this was caused by pandas index alignment during assignment. The current corrected output now contains valid `target_corr` values.

Current logic:

```python
selected_df = pd.DataFrame({'feature': remaining})
selected_df['target_corr'] = df_clean[remaining].corrwith(df_clean[target_col])
```

`selected_df` has integer indices, while `corrwith()` returns a Series indexed by feature name. Pandas aligns by index labels, so no rows match and all values become `NaN`.

A safer version would be:

```python
corr_series = df_clean[remaining].corrwith(df_clean[target_col])
selected_df['target_corr'] = selected_df['feature'].map(corr_series)
```

An earlier review found that `Unnamed: 0` could remain in the selected features if the CSV index column was not dropped. The current corrected workflow removes it, and `Unnamed: 0` is not present in either `selected_features.csv` or `us_modeling_ready.csv`.

Finally, some meaningful variables such as `daily_solar_radiation`, `relative_humidity`, and `electricity_price_industrial` were removed by VIF. This is statistically reasonable if they are highly redundant, but the project should consider whether domain-critical features should be retained for interpretability or Taiwan transfer.

---

# 02 特徵篩選摘要

## 目的

這本 notebook 的目標是把 DeepSolar 原始資料整理成可以拿去建模的乾淨特徵集。

主要步驟如下：

1. 移除美國專屬欄位與 target leakage 欄位。
2. 暫時保留 `tile_count`，用來建立分類目標。
3. 對數值欄位做缺失值填補。
4. 將 `tile_count` 切成三個潛力類別。
5. 移除高度相關的特徵。
6. 使用 VIF 反覆篩除共線性太高的特徵。
7. 輸出 selected features 和 modeling-ready dataset。

## 載入與初步整理

```text
Feature columns after dropping US-only/targets: 121
```

這一步會移除美國專屬欄位，例如 voting、race、incentive、行政區代碼等，也會移除太陽能結果相關欄位，避免模型直接看到答案。

不過這裡會暫時保留 `tile_count`，因為後面還需要用它建立分類標籤。

整理後剩下 121 個欄位。

## 缺失值填補

```text
Numeric columns: 119
Rows after dropping missing target: 72537
Any remaining missing: 0
```

扣掉目標欄位後，有 119 個數值型候選特徵。因為 `tile_count` 沒有缺失，所以沒有任何 row 被刪掉，仍然保留完整的 72,537 筆資料。

其他數值欄位的缺失值用各欄位的 median 填補。這是一個穩定的 baseline 做法，尤其適合收入、人口、氣候等可能偏態的資料。

## 建立分類目標

Notebook 用 `tile_count` 的第 33 百分位數和第 66 百分位數，把資料切成三類。

```text
Quantiles: 33%=1.00, 66%=11.00

potential_label
0    24279
1    24174
2    24084
```

分類邏輯是：

- `2`：`tile_count <= q33`，目前部署量低，專案中解讀為高潛力或低開發區。
- `1`：`q33 < tile_count <= q66`，中等部署量。
- `0`：`tile_count > q66`，目前部署量較高，解讀為較飽和或已開發區。

三個類別幾乎一樣大，每類大約 24,000 筆。這對分類模型是好事，因為模型不會被某一個類別主導。

但要注意概念解釋：`tile_count` 衡量的是「目前已安裝量」，低安裝量不一定等於高潛力。這個專案是把低安裝量視為可能有推廣空間，但最好要搭配日照、電價、收入、政策等其他條件一起解讀。

## 相關性篩選

```text
Features after correlation screening: 94 (dropped 25)
```

這一步會檢查兩兩特徵之間的 Pearson 相關係數。如果兩個特徵的絕對相關超過 0.85，就刪掉其中一個。

經過這一步，候選特徵從 119 個降到 94 個，刪掉 25 個高度重複的欄位。

這一步主要處理簡單的「兩兩重複」問題，例如兩個欄位幾乎在描述同一件事。

產生的圖檔：

```text
outputs/figures/02_correlation_heatmap.png
```

## VIF 共線性篩選

接著 notebook 做 iterative VIF screening：

```text
VIF threshold = 10
Max iterations = 50
```

VIF 是 Variance Inflation Factor，可以理解成「某個特徵能不能被其他特徵很好地解釋」。VIF 越高，代表這個特徵和其他特徵越重複。

這段流程是：

1. 計算目前所有特徵的 VIF。
2. 找出 VIF 最高的特徵。
3. 如果最高 VIF 已經小於等於 10，就停止。
4. 如果最高 VIF 還大於 10，就刪掉 VIF 最高的特徵。
5. 重新計算 VIF，重複以上步驟。

前幾個被刪掉的欄位與 VIF 如下：

```text
atmospheric_pressure          VIF=4316.6
transportation_car_alone_rate VIF=2125.7
heating_fuel_gas_rate         VIF=1972.3
employ_rate                   VIF=1161.6
cooling_design_temperature    VIF=869.9
daily_solar_radiation         VIF=319.8
relative_humidity             VIF=230.3
median_household_income       VIF=45.8
electricity_price_industrial  VIF=19.9
```

一開始的 VIF 非常高，代表資料裡很多欄位彼此高度重複。這在 census 類資料很常見，因為人口比例、收入、教育、住宅、交通、氣候變數之間常常有強烈關聯。

最後結果：

```text
Final valid feature count: 49
```

VIF 篩選將特徵從 94 個進一步降到 49 個，這 49 個欄位就是後續建模使用的有效特徵。

產生的圖檔：

```text
outputs/figures/02_vif_final.png
```

## 篩選後欄位字典

目前有效的 selected features 共 49 個。下表保留原始欄位名稱，並補上欄位性質與中文解釋。

| 原始欄位名稱 | 欄位性質 | 中文欄位意義 |
|---|---|---|
| `heating_fuel_solar` | 能源 / 住宅 | 使用太陽能作為住宅取暖能源的戶數 |
| `frost_days` | 氣候 | 一年中霜凍天數，反映寒冷氣候程度 |
| `housing_unit_median_value` | 住宅 / 收入 | 住宅單位房價中位數 |
| `education_bachelor` | 教育 | 最高學歷為學士的人數 |
| `heating_fuel_solar_rate` | 能源 / 住宅 | 使用太陽能取暖的住宅比例 |
| `unemployed` | 勞動 / 經濟 | 失業人口數 |
| `education_doctoral` | 教育 | 最高學歷為博士的人數 |
| `poverty_family_below_poverty_level_rate` | 收入 / 貧窮 | 低於貧窮線家庭比例 |
| `diversity` | 人口結構 | 人口多樣性指標 |
| `occupancy_vacant_rate` | 住宅 | 空屋比例 |
| `transportation_home_rate` | 交通 / 通勤 | 在家工作或不通勤比例 |
| `occupation_administrative_rate` | 職業結構 | 行政支援相關職業比例 |
| `health_insurance_none_rate` | 健康 / 社經 | 無健康保險人口比例 |
| `occupation_manufacturing_rate` | 職業結構 | 製造業職業比例 |
| `transportation_motorcycle_rate` | 交通 / 通勤 | 摩托車通勤比例 |
| `heating_fuel_coal_coke_rate` | 能源 / 住宅 | 使用煤或焦炭取暖的住宅比例 |
| `travel_time_less_than_10_rate` | 交通 / 通勤 | 通勤時間少於 10 分鐘比例 |
| `occupation_finance_rate` | 職業結構 | 金融、保險、房地產相關職業比例 |
| `education_professional_school_rate` | 教育 | 專業學位人口比例 |
| `heating_fuel_none` | 能源 / 住宅 | 無取暖燃料的住宅戶數 |
| `transportation_public_rate` | 交通 / 通勤 | 大眾運輸通勤比例 |
| `transportation_walk_rate` | 交通 / 通勤 | 步行通勤比例 |
| `education_doctoral_rate` | 教育 | 博士學歷人口比例 |
| `occupation_information_rate` | 職業結構 | 資訊產業相關職業比例 |
| `occupation_public_rate` | 職業結構 | 公共行政相關職業比例 |
| `heating_fuel_other_rate` | 能源 / 住宅 | 使用其他取暖燃料的住宅比例 |
| `travel_time_60_89_rate` | 交通 / 通勤 | 通勤時間 60 到 89 分鐘比例 |
| `population_density` | 人口 / 地理 | 人口密度 |
| `heating_fuel_electricity` | 能源 / 住宅 | 使用電力取暖的住宅戶數 |
| `education_less_than_high_school` | 教育 | 低於高中學歷的人數 |
| `electricity_consume_total` | 電力 | 總用電量 |
| `travel_time_40_59_rate` | 交通 / 通勤 | 通勤時間 40 到 59 分鐘比例 |
| `age_15_17_rate` | 年齡結構 | 15 到 17 歲人口比例 |
| `occupation_wholesale_rate` | 職業結構 | 批發業職業比例 |
| `total_area` | 地理 | 地區總面積 |
| `occupation_transportation_rate` | 職業結構 | 運輸、倉儲、公用事業相關職業比例 |
| `heating_fuel_other` | 能源 / 住宅 | 使用其他取暖燃料的住宅戶數 |
| `travel_time_30_39_rate` | 交通 / 通勤 | 通勤時間 30 到 39 分鐘比例 |
| `age_18_24_rate` | 年齡結構 | 18 到 24 歲人口比例 |
| `occupation_construction_rate` | 職業結構 | 營造業職業比例 |
| `occupation_retail_rate` | 職業結構 | 零售業職業比例 |
| `occupation_agriculture_rate` | 職業結構 | 農林漁牧等職業比例 |
| `water_area` | 地理 | 水域面積 |
| `heating_fuel_fuel_oil_kerosene_rate` | 能源 / 住宅 | 使用燃油或煤油取暖的住宅比例 |
| `occupation_arts_rate` | 職業結構 | 藝術、娛樂、餐旅等服務職業比例 |
| `age_75_84_rate` | 年齡結構 | 75 到 84 歲人口比例 |
| `transportation_bicycle_rate` | 交通 / 通勤 | 自行車通勤比例 |
| `age_more_than_85_rate` | 年齡結構 | 85 歲以上人口比例 |
| `transportation_carpool_rate` | 交通 / 通勤 | 共乘通勤比例 |

欄位可以大致分成幾類：

- **能源 / 住宅**：取暖燃料、用電量、住宅價值等，反映家庭能源使用與居住條件。
- **社經 / 教育**：教育程度、失業、貧窮、健康保險等，反映地區經濟能力與採用新技術的條件。
- **職業 / 交通**：產業結構與通勤方式，可能反映都市型態、土地使用與生活型態。
- **氣候 / 地理**：霜凍天數、面積、水域面積、人口密度等，描述環境與空間條件。
- **年齡 / 人口結構**：不同年齡層比例與多樣性，反映地區人口組成。

## 輸出結果

特徵清單：

```text
outputs/results/selected_features.csv
```

建模資料：

```text
data/processed/us_modeling_ready.csv
Shape: (72537, 50)
```

`us_modeling_ready.csv` 裡有 49 個特徵，加上 1 個目標欄位 `potential_label`，總共 50 欄。

最終類別分布：

```text
potential_label
0    24279
1    24174
2    24084
```

## 問題與注意事項

檢查時曾發現 `selected_features.csv` 裡的 `target_corr` 全部是空的。這不是因為相關性真的算不出來，而是 pandas index 對齊造成的小 bug。目前修正後的輸出已經有正確的 `target_corr` 數值。

目前寫法：

```python
selected_df = pd.DataFrame({'feature': remaining})
selected_df['target_corr'] = df_clean[remaining].corrwith(df_clean[target_col])
```

`selected_df` 的 index 是 0, 1, 2...，但 `corrwith()` 回傳的 Series index 是 feature name。pandas 指派欄位時會依照 index 對齊，所以對不上，結果全部變成 `NaN`。

可以改成：

```python
corr_series = df_clean[remaining].corrwith(df_clean[target_col])
selected_df['target_corr'] = selected_df['feature'].map(corr_series)
```

先前檢查時曾發現，如果沒有先移除 CSV index 欄位，`Unnamed: 0` 可能會留在 selected features 裡。目前修正後的流程已移除此欄位，`selected_features.csv` 和 `us_modeling_ready.csv` 都不包含 `Unnamed: 0`。

此外，`daily_solar_radiation`、`relative_humidity`、`electricity_price_industrial` 這些有領域意義的欄位在 VIF 篩選中被刪掉。從統計角度來看，這代表它們和其他欄位高度重複；但若專案重視可解釋性或台灣遷移，可能需要討論是否保留某些 domain-critical features。

