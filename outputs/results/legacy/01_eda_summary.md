# 01 EDA Summary

## Dataset Overview

The DeepSolar dataset was loaded successfully.

```text
Shape: (72537, 169)
```

This means the dataset contains 72,537 census-tract-level observations and 169 columns. The loaded shape matches the expected DeepSolar dataset scale, so the raw data file appears to be valid.

## Target Distribution

The current target variable is `tile_count`, which is used as the main proxy for solar deployment intensity.

```text
count    72537.000000
mean        30.255787
std         86.337406
min          0.000000
25%          1.000000
50%          4.000000
75%         22.000000
max       4468.000000

Zero count: 16279 (22.4%)
Non-zero median: 8.0
```

The distribution is highly right-skewed. Most areas have very few solar tiles, while a small number of areas have extremely high installation counts. The median is only 4, but the maximum is 4,468.

About 22.4% of areas have `tile_count = 0`, meaning they have no detected solar deployment. This supports treating the task as a classification or ranking problem rather than a direct raw-count regression problem.

Generated figure:

```text
outputs/figures/01_tile_count_distribution.png
```

## Feature Groups

Columns were grouped by prefix to understand the structure of the 169 variables.

Top prefix groups:

```text
heating          17
education        15
race             14
single           12
age              12
occupation       12
electricity       9
transportation    7
travel            7
voting            6
solar             5
```

These groups show that the dataset combines climate, energy, socioeconomic, demographic, housing, transportation, voting, incentive, and solar installation variables.

Important groups:

- `solar_*`: solar deployment outcome variables. These are strongly related to the target and should generally be removed from model inputs to avoid target leakage.
- `electricity_*`: electricity prices and consumption. These may be useful and potentially transferable to Taiwan.
- `heating_*` / `cooling_*`: climate and energy-demand related variables.
- `education_*`, `housing_*`, `occupation_*`, `income` variables: socioeconomic features.
- `incentive_*`, `race_*`, `voting_*`: US-specific or politically/demographically specific features that should be handled carefully before transfer to Taiwan.

## Missing Values

```text
Columns with missing values: 103 / 169
```

Missing values are common in the dataset. The top missing columns include:

```text
voting_2012_dem_percentage        10554
voting_2012_gop_percentage        10554
elevation                          5802
lon                                5802
cooling_design_temperature         5802
lat                                5802
relative_humidity                  5802
earth_temperature_amplitude        5802
earth_temperature                  5802
heating_design_temperature         5802
frost_days                         5802
heating_degree_days                5802
cooling_degree_days                5802
wind_speed                         5802
atmospheric_pressure               5802
daily_solar_radiation              5802
air_temperature                    5802
```

The repeated missing count of 5,802 across many climate and geographic variables suggests that some census tracts may be missing matched environmental data.

This matters because variables such as `daily_solar_radiation`, `air_temperature`, and `relative_humidity` are likely important for solar potential modeling. Later notebooks should apply a consistent imputation strategy before modeling.

Generated figure:

```text
outputs/figures/01_missing_values.png
```

## US-only vs Generalizable Features

The notebook separates columns into US-specific features, target/leakage columns, and potentially generalizable features.

```text
US-only columns found: 34
Target columns found: 12
Generalizable feature columns: 123
Total: 169
```

Interpretation:

- 34 columns are treated as US-specific, including voting, race, policy incentive, and administrative location columns.
- 12 columns are target or solar outcome variables and should be removed from model inputs to avoid target leakage.
- 123 columns are treated as potentially transferable to Taiwan.

One detail to watch: `generalizable_cols` currently includes `Unnamed: 0`, which appears to be a leftover CSV index column. It should probably be removed before modeling. The list also includes `lat` and `lon`, although `prepare_features()` later drops them when `drop_ids=True`.

Saved metadata:

```text
data/processed/column_metadata.json
```

## Correlation Preview

The strongest absolute correlations with `tile_count` are:

```text
solar_system_count                      0.899767
total_panel_area_residential            0.860234
tile_count_residential                  0.859787
solar_system_count_residential          0.857702
solar_system_count_nonresidential       0.836872
total_panel_area                        0.787353
tile_count_nonresidential               0.738657
total_panel_area_nonresidential         0.643371
number_of_solar_system_per_household    0.511970
solar_panel_area_divided_by_area        0.487082
```

Most of these are solar deployment outcome variables. They are highly correlated because they are essentially alternative measurements of the target. These should not be used as predictive features.

More meaningful non-target signals include:

```text
incentive_count_residential       0.361331
incentive_count_nonresidential    0.358120
relative_humidity                -0.356200
electricity_price_industrial      0.336128
daily_solar_radiation             0.327462
```

These features are more interpretable as drivers or contextual factors:

- Incentives may increase solar adoption.
- Higher industrial electricity prices may be associated with stronger solar economics.
- Higher solar radiation is naturally relevant for solar deployment.
- Relative humidity has a negative correlation in this preview, possibly reflecting climate or regional effects.

Note: the notebook label `Top 15 negatively correlated` is misleading. The code sorts by absolute correlation and then prints the last 15 values, so this section actually shows the weakest absolute correlations, not the most negative correlations.

## Notes for Next Step

- `tile_count` is highly right-skewed, so classification, ranking, or log-transformed regression is more appropriate than raw-count regression.
- Solar outcome variables must be removed before modeling to avoid target leakage.
- Missing values affect 103 out of 169 columns and require a clear imputation strategy.
- `Unnamed: 0` should be removed as an index artifact.
- The correlation preview heading should be corrected to avoid confusing weakest correlations with negative correlations.
- The generated metadata file is ready for downstream feature selection.

---

# 01 EDA 摘要

## 資料概覽

DeepSolar 資料已成功讀入。

```text
Shape: (72537, 169)
```

這代表資料共有 72,537 筆 census tract 層級的觀測值，以及 169 個欄位。這個資料規模符合 DeepSolar 主資料的預期，因此原始資料檔看起來是正確的。

## 目標變數分布

目前使用 `tile_count` 作為主要目標變數，用來代表各地區太陽能部署程度。

```text
count    72537.000000
mean        30.255787
std         86.337406
min          0.000000
25%          1.000000
50%          4.000000
75%         22.000000
max       4468.000000

Zero count: 16279 (22.4%)
Non-zero median: 8.0
```

`tile_count` 的分布非常右偏。大多數地區的太陽能 tile 數量很少，中位數只有 4，但最大值高達 4,468。這代表少數地區有非常高的太陽能部署量，而大部分地區集中在低安裝量。

另外，有 16,279 筆資料的 `tile_count` 為 0，占整體 22.4%。也就是說，約五分之一的地區完全沒有偵測到太陽能部署。

這個結果表示，後續如果直接預測原始 `tile_count` 數值，模型可能會受到極端值和偏態分布影響。因此，把問題轉成分類、排序，或對目標做 log transform，會比直接做 raw-count regression 更合理。

產生的圖檔：

```text
outputs/figures/01_tile_count_distribution.png
```

## 特徵群組

Notebook 依照欄位名稱 prefix 將 169 個變數分組，以了解資料欄位的主題結構。

主要 prefix 數量如下：

```text
heating          17
education        15
race             14
single           12
age              12
occupation       12
electricity       9
transportation    7
travel            7
voting            6
solar             5
```

這顯示 DeepSolar 資料不是單純的太陽能資料，而是結合了氣候、能源、社經、人口結構、住宅、交通、投票、補貼政策與太陽能安裝結果等多種資訊。

幾個重要欄位群的意義如下：

- `solar_*`：太陽能安裝結果或衍生指標，和目標變數高度相關，後續建模時通常要移除，避免 target leakage。
- `electricity_*`：電價與用電量，可能對太陽能採用有解釋力，也比較可能轉移到台灣資料。
- `heating_*` / `cooling_*`：氣候與住宅能源需求相關變數。
- `education_*`、`housing_*`、`occupation_*`、income 相關欄位：社經條件。
- `incentive_*`、`race_*`、`voting_*`：較偏美國制度、族裔或政治脈絡，跨國遷移到台灣時需要小心處理，甚至移除。

## 缺失值

```text
Columns with missing values: 103 / 169
```

169 個欄位中，有 103 個欄位存在缺失值，代表缺失值處理會是後續建模的重要步驟。

缺失最多的欄位包括：

```text
voting_2012_dem_percentage        10554
voting_2012_gop_percentage        10554
elevation                          5802
lon                                5802
cooling_design_temperature         5802
lat                                5802
relative_humidity                  5802
earth_temperature_amplitude        5802
earth_temperature                  5802
heating_design_temperature         5802
frost_days                         5802
heating_degree_days                5802
cooling_degree_days                5802
wind_speed                         5802
atmospheric_pressure               5802
daily_solar_radiation              5802
air_temperature                    5802
```

值得注意的是，多個地理與氣候欄位都剛好缺 5,802 筆，這可能代表某些 census tract 沒有成功對應到環境或氣象資料。

這點很重要，因為像 `daily_solar_radiation`、`air_temperature`、`relative_humidity` 這些欄位理論上都會影響太陽能潛力。後續建模前需要有一致的缺值填補策略，例如 median imputation，或針對特定欄位設計更合理的處理方式。

產生的圖檔：

```text
outputs/figures/01_missing_values.png
```

## 美國專屬欄位與可泛化特徵

Notebook 將欄位分成美國專屬欄位、目標或洩漏欄位，以及可能可泛化到台灣的欄位。

```text
US-only columns found: 34
Target columns found: 12
Generalizable feature columns: 123
Total: 169
```

解讀如下：

- 34 個欄位被視為美國專屬欄位，例如投票資料、族裔組成、補貼制度、州與郡代碼。
- 12 個欄位是目標或太陽能安裝結果相關欄位，不能作為模型輸入，否則會造成 target leakage。
- 123 個欄位被暫時視為可能可泛化到台灣的特徵。

不過，這裡有一個需要注意的地方：`generalizable_cols` 目前包含 `Unnamed: 0`，這看起來像 CSV index 殘留欄位，後續應該移除。另外，清單中也包含 `lat` 和 `lon`，但 `prepare_features()` 後續在 `drop_ids=True` 時會將它們移除。

儲存的 metadata：

```text
data/processed/column_metadata.json
```

## 與目標變數的相關性預覽

與 `tile_count` 絕對相關最高的欄位如下：

```text
solar_system_count                      0.899767
total_panel_area_residential            0.860234
tile_count_residential                  0.859787
solar_system_count_residential          0.857702
solar_system_count_nonresidential       0.836872
total_panel_area                        0.787353
tile_count_nonresidential               0.738657
total_panel_area_nonresidential         0.643371
number_of_solar_system_per_household    0.511970
solar_panel_area_divided_by_area        0.487082
```

這些欄位大多是太陽能安裝結果或其衍生指標，因此與 `tile_count` 高度相關並不意外。但它們本質上接近「答案的另一種表達」，後續建模時必須移除，否則模型會發生 target leakage，看起來準確但沒有真正預測能力。

比較有解釋意義、且不是直接結果欄位的訊號包括：

```text
incentive_count_residential       0.361331
incentive_count_nonresidential    0.358120
relative_humidity                -0.356200
electricity_price_industrial      0.336128
daily_solar_radiation             0.327462
```

這些欄位比較像可能影響太陽能部署的外部因素：

- 補貼政策可能提升太陽能採用率。
- 工業電價越高，可能代表太陽能的經濟誘因越強。
- 日照輻射量越高，理論上越適合太陽能發電。
- 相對濕度呈負相關，可能反映氣候或區域差異。

注意：notebook 中 `Top 15 negatively correlated` 這個標題有點誤導。程式實際上是先依照絕對相關排序，再印出最後 15 個，因此那段不是「最負相關」的欄位，而是「絕對相關最弱」的欄位。

## 下一步注意事項

- `tile_count` 高度右偏，後續適合做分類、排序，或使用 log transform 後再做 regression。
- 太陽能結果欄位必須從模型輸入中移除，避免 target leakage。
- 169 個欄位中有 103 個欄位有缺值，需要明確的缺值填補策略。
- `Unnamed: 0` 應視為 index artifact 並移除。
- correlation preview 的 `Top 15 negatively correlated` 標題應修正，避免誤解。
- `column_metadata.json` 已成功產生，可供後續 feature selection notebook 使用。
