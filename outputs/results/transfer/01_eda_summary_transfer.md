# 01 EDA Summary - Transfer Pipeline

## 中文版

### 這一步在做什麼

`01_eda` 是新版 transfer pipeline 的資料體檢。它不是在訓練模型，而是在確認 DeepSolar 原始資料的規模、目標欄位分布、缺失值、欄位類型，以及哪些欄位適合進入「台灣遷移導向」的後續流程。

### 資料規模

- 原始資料共有 `72,537` 筆 census tract。
- 欄位數共有 `169` 欄。
- 主要觀察目標仍是 `tile_count`，也就是每個 census tract 內偵測到的太陽能板 tile 數量。

### tile_count 分布

`tile_count` 的分布非常偏斜：

- 平均值：`30.26`
- 中位數：`4`
- 75% 分位數：`22`
- 最大值：`4,468`
- `tile_count = 0` 的區域有 `16,279` 筆，約 `22.4%`
- 非零資料的中位數是 `8`

這代表多數地區太陽能部署量很低，少數地區部署量非常高。這也是後續不直接做普通 regression，而先把潛力切成分類標籤的原因之一。

### 欄位結構

欄位 prefix 顯示資料主要包含：

- 氣候與環境：`daily_solar_radiation`, `relative_humidity`, `frost_days`, `cooling_degree_days`, `heating_degree_days`
- 能源經濟：`electricity_price_*`, `electricity_consume_*`
- 政策/誘因：`incentive_*`, `feedin_tariff`, `net_metering`, `rebate`, `tax`
- 社經條件：`income`, `education`, `poverty`, `employment`, `occupation`
- 住宅條件：`housing_unit_*`, `occupancy_*`, `mortgage_*`
- 美國特有社會/政治欄位：`race_*`, `voting_*`, `state`, `county`, `fips`

新版流程的重點是不要把「政策/電價」誤當成不可轉移欄位刪掉。它們在台灣有對應概念，只是未來要重新 mapping。

### 缺失值

共有 `103 / 169` 欄有缺失值。缺失最多的欄位包括：

- `voting_2012_dem_percentage`, `voting_2012_gop_percentage`：各缺 `10,554`
- 多數氣候欄位如 `daily_solar_radiation`, `relative_humidity`, `frost_days`, `cooling_degree_days`：各缺 `5,802`
- 部分住宅價值/租金欄位也有少量缺失

在 `02_feature_selection` 中，數值欄位會用 median 補值，避免模型因缺失值無法訓練。

### Transfer 欄位分類結果

新版欄位分類結果：

- 不可轉移的美國特有欄位：`23` 欄
- 保留的政策/能源經濟概念欄位：`11` 欄
- ID / location helper 欄位：`3` 欄
- target / leakage 欄位：`12` 欄
- transfer candidate feature 欄位：`131` 欄

這裡的核心改動是：只移除真正難以轉移的美國地域/政治/族裔欄位，而不是把 incentive、電價、FIT 等能源經濟概念整組刪除。

### 與 tile_count 的初步相關性

與 `tile_count` 正相關較高的非 target 欄位包括：

- `incentive_count_residential`
- `incentive_count_nonresidential`
- `electricity_price_industrial`
- `daily_solar_radiation`
- `electricity_price_commercial`

負相關較明顯的欄位包括：

- `relative_humidity`
- 部分貧窮、交通、年齡與住宅狀態變數

注意：這裡只是單變量相關性，不代表因果，也不代表模型最後一定最依賴這些欄位。

### 這一步的產出

- `data/processed/transfer/column_metadata.json`
- `outputs/figures/transfer/01_tile_count_distribution.png`
- `outputs/figures/transfer/01_missing_values.png`

---

## English Summary

`01_eda` inspects the DeepSolar dataset before modeling. The dataset contains `72,537` census tracts and `169` columns. The target proxy, `tile_count`, is highly skewed: the median is `4`, the mean is `30.26`, and `22.4%` of tracts have zero solar tiles.

The redesigned transfer workflow separates truly non-transferable US-specific fields from transferable policy and energy-economic concepts. It found `23` non-transferable columns, `11` policy/economic concept columns to keep, `3` ID/location helper columns, `12` target/leakage columns, and `131` transfer-candidate feature columns.

This step confirms that the new pipeline should preserve electricity price, incentive, FIT, rebate, climate, housing, and socioeconomic concepts for later Taiwan mapping, while removing race, voting, and US administrative identifiers from the main transfer feature set.
