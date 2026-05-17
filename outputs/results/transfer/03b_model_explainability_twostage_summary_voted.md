# 03b Two-stage Model Explainability Summary - Transfer Pipeline

## 中文版

### 這份 notebook 在做什麼

`03b_model_explainability_twostage` 是在解釋 two-stage model，而不是重新訓練模型。

它讀取：

- `outputs/results/transfer/best_us_twostage_model_voted.pkl`

並分別對兩個模型做 SHAP 解釋：

```text
Stage 1 classifier:
解釋哪些特徵影響「有沒有太陽能部署」

Stage 2 regressor:
解釋哪些特徵影響「已部署地區的相對部署強度」
```

這兩個階段不能混在一起看，因為它們回答的是不同問題。

### 使用的資料

這次解釋的模型與資料：

- model path：`best_us_twostage_model_voted.pkl`
- classifier：`RandomForestClassifier`
- regressor：`RandomForestRegressor`
- features：`90`
- full data shape：`72,537 x 90`
- installed rows：`56,258`
- installed rate：約 `0.776`
- Stage 1 SHAP sample：`2,000 x 90`
- Stage 2 SHAP sample：`2,000 x 90`

這裡的 SHAP 是抽樣解釋，不是用全部 72,537 筆資料。這樣做是合理的，因為 SHAP 對 tree ensemble 也會有計算成本。

### 一個需要注意的小警訊

notebook 在計算 deployment density 時出現：

```text
RuntimeWarning: divide by zero encountered in divide
RuntimeWarning: invalid value encountered in divide
```

這代表某些列的 household denominator 可能是 `0` 或無效值。這個 warning 不一定會影響這份 SHAP ranking，因為 SHAP 主要解釋已訓練好的模型與輸入特徵，但後續若要正式計算 density，建議先處理：

```text
household_count <= 0 -> 設為 NaN
再做 median imputation 或直接排除
```

這樣可以避免 density 指標出現 infinity 或 NaN。

---

## Stage 1 SHAP：有沒有太陽能部署

Stage 1 解釋的是：

```text
這個 census tract 像不像美國資料中「已經有太陽能部署」的地區？
```

### Stage 1 前 15 名重要特徵

| rank | feature | mean_abs_shap | 解讀 |
|---:|---|---:|---|
| 1 | `total_area` | `0.0356` | 區域尺度/土地面積，影響是否能觀察到部署 |
| 2 | `population_density` | `0.0355` | 都市化與人口密度，是 adoption 有無的重要 proxy |
| 3 | `education_bachelor` | `0.0268` | 教育程度與社經條件相關 |
| 4 | `population` | `0.0211` | 區域人口規模與市場大小 |
| 5 | `heating_fuel_coal_coke_rate` | `0.0198` | 能源使用結構 proxy |
| 6 | `daily_solar_radiation` | `0.0191` | 日照條件影響是否部署 |
| 7 | `housing_unit_median_value` | `0.0142` | 房價/資產條件 |
| 8 | `education_college` | `0.0129` | 教育/社經條件 |
| 9 | `relative_humidity` | `0.0122` | 氣候條件 |
| 10 | `electricity_consume_total` | `0.0120` | 用電規模/需求 proxy |
| 11 | `electricity_price_industrial` | `0.0117` | 電價誘因 |
| 12 | `education_professional_school` | `0.0105` | 高教育/社經 proxy |
| 13 | `housing_unit_median_gross_rent` | `0.0103` | 地區租金/經濟條件 |
| 14 | `heating_fuel_gas` | `0.0099` | 能源使用型態 |
| 15 | `occupancy_vacant_rate` | `0.0089` | 住宅使用狀態 |

### Stage 1 怎麼解讀

Stage 1 的重要特徵比較偏向：

- 區域規模
- 人口密度
- 教育/社經條件
- 住宅價值
- 日照與氣候
- 用電與能源使用結構

這符合 adoption model 的直覺。它不是只看日照，也會看「哪裡比較有市場、住戶與社經條件支持太陽能部署」。

對台灣遷移來說，Stage 1 score 可以解讀成：

```text
台灣某地區像不像美國資料中已經發生 solar adoption 的地區。
```

---

## Stage 2 SHAP：已部署地區的相對部署強度

Stage 2 解釋的是：

```text
在已經有太陽能部署的地區中，
哪些條件會影響部署強度高低？
```

這不是台灣真實安裝密度預測，而是相對 intensity pattern。

### Stage 2 前 15 名重要特徵

| rank | feature | mean_abs_shap | 解讀 |
|---:|---|---:|---|
| 1 | `relative_humidity` | `0.4483` | 最重要的強度解釋因子，可能 proxy 雲量/氣候濕度 |
| 2 | `electricity_price_industrial` | `0.2489` | 工業電價與部署強度有強關聯 |
| 3 | `population_density` | `0.1487` | 人口密度/都市型態與部署強度相關 |
| 4 | `frost_days` | `0.1290` | 美國氣候嚴苛程度 proxy，台灣需找替代概念 |
| 5 | `incentive_residential_state_level` | `0.0879` | 州級住宅補助/政策支持 |
| 6 | `heating_fuel_gas` | `0.0624` | 能源使用結構 |
| 7 | `occupancy_owner_rate` | `0.0594` | 自有住宅率，與安裝決策相關 |
| 8 | `rebate` | `0.0562` | rebate 政策誘因 |
| 9 | `daily_solar_radiation` | `0.0551` | 日照條件 |
| 10 | `housing_unit_median_value` | `0.0542` | 房價/資產條件 |
| 11 | `education_high_school_graduate` | `0.0483` | 教育/人口結構 |
| 12 | `median_household_income` | `0.0426` | 家戶收入 |
| 13 | `sales_tax` | `0.0405` | 稅制政策 |
| 14 | `total_area` | `0.0364` | 區域尺度 |
| 15 | `incentive_count_residential` | `0.0330` | 住宅補助數量 |

### Stage 2 怎麼解讀

Stage 2 的重要特徵比 Stage 1 更集中在：

- 氣候條件：`relative_humidity`, `frost_days`, `daily_solar_radiation`
- 能源經濟：`electricity_price_industrial`
- 政策誘因：`incentive_residential_state_level`, `rebate`, `sales_tax`
- 住宅權屬與資產：`occupancy_owner_rate`, `housing_unit_median_value`

這表示，當一個地區已經具備 adoption 條件後，部署強度更受到氣候、電價、政策與住宅 ownership 的影響。

對台灣遷移來說，Stage 2 score 可以解讀成：

```text
若某台灣地區具備 adoption 條件，
它像不像美國資料中部署強度較高的地區。
```

不應直接解讀成：

```text
台灣真實安裝密度 = 某個絕對數值。
```

---

## Stage 1 vs Stage 2 的差異

combined importance 顯示，有些特徵兩階段都重要，有些只在其中一階段重要。

### 兩階段都重要

| feature | Stage 1 rank | Stage 2 rank | 解讀 |
|---|---:|---:|---|
| `population_density` | 2 | 3 | 同時影響有無部署與部署強度 |
| `daily_solar_radiation` | 6 | 9 | 日照是穩定重要的自然條件 |
| `housing_unit_median_value` | 7 | 10 | 住宅資產條件同時影響 adoption 與 intensity |
| `relative_humidity` | 9 | 1 | Stage 2 特別重要，代表強度階段更依賴氣候條件 |
| `electricity_price_industrial` | 11 | 2 | 電價對部署強度特別重要 |

### 比較偏 Stage 1 的特徵

- `total_area`
- `education_bachelor`
- `population`
- `education_professional_school`

這些比較像「是否出現部署」的 adoption background。

### 比較偏 Stage 2 的特徵

- `relative_humidity`
- `electricity_price_industrial`
- `frost_days`
- `incentive_residential_state_level`
- `rebate`
- `occupancy_owner_rate`

這些比較像「部署後強度高不高」的 intensity drivers。

---

## 對台灣遷移的意義

目前這份 SHAP 結果很有用，因為它可以幫我們決定台灣資料蒐集優先順序。

### 台灣優先 mapping 的特徵類型

1. **人口/都市型態**
   - 人口密度
   - 區域人口
   - 土地面積或行政區面積

2. **氣候/自然條件**
   - 日照量
   - 相對濕度
   - 雲量或降雨日數
   - 台灣版氣候嚴苛度 proxy，用來替代 `frost_days`

3. **能源經濟**
   - 住宅/商業/工業電價
   - 用電量
   - 電價級距或高壓/低壓用戶結構

4. **政策誘因**
   - 中央補助
   - 地方政府補助
   - FIT
   - rebate / tax-like policy proxy

5. **住宅與社經**
   - 房價
   - 家戶所得
   - 自有住宅率
   - 教育程度

### frost_days 怎麼辦

`frost_days` 在 Stage 2 排第 4，但台灣沒有直接對應的霜凍天數。這不代表要硬找 frost，而是要找台灣合理的氣候替代變數，例如：

- rain days
- precipitation
- cloud cover
- typhoon exposure
- high humidity days
- extreme heat days
- seasonal solar radiation variability

這些更能反映台灣的太陽能維運與發電條件。

---

## 結論

這份 SHAP 結果支持目前的 two-stage 解讀：

```text
Stage 1:
比較像 adoption likelihood model。
它回答「哪裡比較可能已經有太陽能部署」。

Stage 2:
比較像 deployment intensity pattern model。
它回答「已部署地區中，哪些條件對部署強度更重要」。
```

放到台灣時，這些輸出應該用來建立：

```text
相對太陽能部署傾向
推廣優先度排序
資料蒐集優先清單
```

而不是直接宣稱可以預測台灣真實安裝密度、真實發電量或 ROI。

---

## English Summary

This notebook explains the two-stage RandomForest model with SHAP. Stage 1 explains the binary adoption classifier, while Stage 2 explains the deployment-intensity regressor for already-installed tracts.

Stage 1 is mainly driven by area, population density, education, population, solar radiation, housing value, humidity, and electricity demand. This stage should be interpreted as an adoption-likelihood model.

Stage 2 is dominated by relative humidity, industrial electricity price, population density, frost days, residential incentives, ownership rate, rebates, solar radiation, housing value, and household income. This stage should be interpreted as a relative deployment-intensity pattern, not a direct prediction of Taiwan's true installed density.

For Taiwan transfer, the SHAP results are useful for prioritizing which Taiwan features to collect and map: population density, solar radiation, humidity, electricity prices, policy incentives, housing value, ownership, income, and Taiwan-specific weather proxies that can replace frost days.
