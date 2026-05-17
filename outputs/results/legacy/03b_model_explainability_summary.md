# 03b Model Explainability Summary

## Purpose

This optional notebook explains the best US model from `03_model_us.ipynb` using SHAP values.

Input files:

```text
outputs/results/best_us_model.pkl
data/processed/us_modeling_ready.csv
```

Output files:

```text
outputs/results/03b_shap_importance.csv
outputs/figures/03b_shap_overall_bar.png
outputs/figures/03b_shap_summary_class_0.png
outputs/figures/03b_shap_summary_class_1.png
outputs/figures/03b_shap_summary_class_2.png
```

The model being explained is:

```text
Model: XGBoost
Data shape: (72537, 50)
Feature count: 49
```

SHAP was computed on a reproducible sample of 2,000 observations:

```text
SHAP sample shape: (2000, 49)
```

## What SHAP Means

SHAP explains how a trained model uses features to make predictions.

In this project, the model predicts three classes:

```text
0 = Saturated
1 = Medium
2 = High
```

Because this is a multiclass model, SHAP produces one explanation group for each class:

```text
Class/output 0: (2000, 49)
Class/output 1: (2000, 49)
Class/output 2: (2000, 49)
```

This means:

- 2,000 sampled observations were explained.
- Each observation has 49 selected features.
- Each class has its own SHAP explanation.

Important caution: SHAP values explain the model, not the real-world causal mechanism. A high SHAP value means the model relies on that feature for prediction. It does not prove that changing the feature would cause solar deployment to change.

## How to Read the SHAP Importance Table

The file `03b_shap_importance.csv` contains:

- `mean_abs_shap_overall`: average absolute SHAP impact across all classes.
- `mean_abs_shap_class_0`: average absolute SHAP impact for class 0.
- `mean_abs_shap_class_1`: average absolute SHAP impact for class 1.
- `mean_abs_shap_class_2`: average absolute SHAP impact for class 2.

These values measure impact size, not positive or negative direction. Larger values mean the model uses that feature more strongly.

Top overall SHAP features:

| Rank | Feature | Overall mean abs SHAP |
|---:|---|---:|
| 1 | `electricity_consume_total` | 0.4839 |
| 2 | `housing_unit_median_value` | 0.2237 |
| 3 | `total_area` | 0.1909 |
| 4 | `population_density` | 0.1348 |
| 5 | `frost_days` | 0.1325 |
| 6 | `education_bachelor` | 0.1234 |
| 7 | `education_less_than_high_school` | 0.0756 |
| 8 | `heating_fuel_coal_coke_rate` | 0.0684 |
| 9 | `unemployed` | 0.0656 |
| 10 | `travel_time_less_than_10_rate` | 0.0569 |

Interpretation:

- `electricity_consume_total` is by far the most influential feature in the SHAP explanation.
- Housing value, total area, population density, frost days, and education variables are also important.
- This broadly matches the earlier XGBoost feature importance result, but SHAP provides class-specific explanations as well.

Generated figure:

```text
outputs/figures/03b_shap_overall_bar.png
```

## Class-Specific SHAP Importance

### Class 0: Saturated

Top features for class 0:

| Rank | Feature | Mean abs SHAP |
|---:|---|---:|
| 1 | `electricity_consume_total` | 0.8268 |
| 2 | `housing_unit_median_value` | 0.3580 |
| 3 | `total_area` | 0.3167 |
| 4 | `frost_days` | 0.1905 |
| 5 | `population_density` | 0.1399 |
| 6 | `unemployed` | 0.1157 |
| 7 | `education_less_than_high_school` | 0.1018 |
| 8 | `heating_fuel_coal_coke_rate` | 0.0926 |
| 9 | `education_bachelor` | 0.0864 |
| 10 | `diversity` | 0.0782 |

For the saturated class, the model relies heavily on electricity consumption, housing value, and area-related features. This suggests the model uses energy demand, housing market context, and spatial scale to identify areas that already resemble high-deployment or saturated regions.

Plot:

```text
outputs/figures/03b_shap_summary_class_0.png
```

### Class 1: Medium

Top features for class 1:

| Rank | Feature | Mean abs SHAP |
|---:|---|---:|
| 1 | `electricity_consume_total` | 0.1903 |
| 2 | `frost_days` | 0.0970 |
| 3 | `travel_time_less_than_10_rate` | 0.0696 |
| 4 | `population_density` | 0.0539 |
| 5 | `total_area` | 0.0464 |
| 6 | `education_bachelor` | 0.0459 |
| 7 | `housing_unit_median_value` | 0.0424 |
| 8 | `occupation_public_rate` | 0.0372 |
| 9 | `heating_fuel_electricity` | 0.0359 |
| 10 | `travel_time_40_59_rate` | 0.0323 |

The medium class has smaller SHAP magnitudes overall. This matches the model evaluation result, where the medium class was the hardest to classify. Middle categories often have less distinct feature patterns than extreme classes.

Plot:

```text
outputs/figures/03b_shap_summary_class_1.png
```

### Class 2: High

Top features for class 2:

| Rank | Feature | Mean abs SHAP |
|---:|---|---:|
| 1 | `electricity_consume_total` | 0.4346 |
| 2 | `housing_unit_median_value` | 0.2706 |
| 3 | `education_bachelor` | 0.2380 |
| 4 | `population_density` | 0.2104 |
| 5 | `total_area` | 0.2096 |
| 6 | `education_less_than_high_school` | 0.1133 |
| 7 | `frost_days` | 0.1099 |
| 8 | `heating_fuel_coal_coke_rate` | 0.0989 |
| 9 | `travel_time_less_than_10_rate` | 0.0658 |
| 10 | `unemployed` | 0.0622 |

For the high class, the model again relies on electricity consumption, housing value, education, population density, and area. Education-related variables are especially prominent for this class.

Plot:

```text
outputs/figures/03b_shap_summary_class_2.png
```

## How to Read SHAP Summary Plots

Each SHAP summary plot explains one class.

In each plot:

- Each row is one feature.
- Each dot is one sampled observation.
- Dot color shows the feature value:
  - Red means higher feature value.
  - Blue means lower feature value.
- The x-axis is the SHAP value for that class:
  - Dots on the right push the model toward that class.
  - Dots on the left push the model away from that class.

Example interpretation:

If red dots for `electricity_consume_total` appear mostly on the right side of the class 0 plot, then high electricity consumption tends to push the model toward class 0. If red dots appear on the left side, then high electricity consumption tends to push the model away from that class.

The bar chart and the summary plots answer different questions:

- The bar chart answers: which features matter most overall?
- The summary plots answer: how do high or low feature values push predictions toward or away from each class?

## Main Takeaways

The SHAP analysis reinforces the importance of:

- Total electricity consumption.
- Housing value.
- Total area.
- Population density.
- Frost days.
- Education-related variables.

The result suggests that the model is not relying on a single type of variable. It combines energy demand, housing conditions, geography, climate, and socioeconomic structure.

However, SHAP is still a model explanation method. The results should be written as model behavior:

```text
The model relies heavily on electricity consumption and housing value when predicting solar deployment class.
```

Not as causal claims:

```text
Increasing electricity consumption causes solar deployment to increase.
```

---

# 03b 模型解釋摘要

## 目的

這本 notebook 使用 SHAP 來解釋 `03_model_us.ipynb` 訓練出的最佳美國模型。

輸入檔案：

```text
outputs/results/best_us_model.pkl
data/processed/us_modeling_ready.csv
```

輸出檔案：

```text
outputs/results/03b_shap_importance.csv
outputs/figures/03b_shap_overall_bar.png
outputs/figures/03b_shap_summary_class_0.png
outputs/figures/03b_shap_summary_class_1.png
outputs/figures/03b_shap_summary_class_2.png
```

被解釋的模型是：

```text
Model: XGBoost
Data shape: (72537, 50)
Feature count: 49
```

SHAP 使用 2,000 筆抽樣資料來計算：

```text
SHAP sample shape: (2000, 49)
```

## SHAP 是什麼意思

SHAP 是用來解釋模型預測的工具。它回答的是：

> 對模型來說，每個欄位如何影響預測結果？

本專案的模型會預測三個類別：

```text
0 = Saturated
1 = Medium
2 = High
```

因為這是三分類模型，所以 SHAP 會分別產生三組解釋：

```text
Class/output 0: (2000, 49)
Class/output 1: (2000, 49)
Class/output 2: (2000, 49)
```

這代表：

- 解釋了 2,000 筆抽樣資料。
- 每筆資料有 49 個特徵。
- 每個類別都有一組獨立的 SHAP 解釋。

重要提醒：SHAP 解釋的是模型，不是現實世界的因果關係。SHAP 值高代表模型很依賴該特徵做預測，但不代表改變該特徵就一定會造成太陽能部署改變。

## 如何看 SHAP importance 表

`03b_shap_importance.csv` 包含：

- `mean_abs_shap_overall`：三個類別平均後的整體 SHAP 影響力。
- `mean_abs_shap_class_0`：對 class 0 的平均 SHAP 影響力。
- `mean_abs_shap_class_1`：對 class 1 的平均 SHAP 影響力。
- `mean_abs_shap_class_2`：對 class 2 的平均 SHAP 影響力。

這些數值代表「影響力大小」，不是正負方向。數字越大，代表模型越依賴該特徵。

整體 SHAP 前 10 名：

| 排名 | 特徵 | Overall mean abs SHAP |
|---:|---|---:|
| 1 | `electricity_consume_total` | 0.4839 |
| 2 | `housing_unit_median_value` | 0.2237 |
| 3 | `total_area` | 0.1909 |
| 4 | `population_density` | 0.1348 |
| 5 | `frost_days` | 0.1325 |
| 6 | `education_bachelor` | 0.1234 |
| 7 | `education_less_than_high_school` | 0.0756 |
| 8 | `heating_fuel_coal_coke_rate` | 0.0684 |
| 9 | `unemployed` | 0.0656 |
| 10 | `travel_time_less_than_10_rate` | 0.0569 |

解讀：

- `electricity_consume_total` 是 SHAP 解釋中最重要的特徵。
- 住宅價值、總面積、人口密度、霜凍天數、教育變數也很重要。
- 這與前面的 XGBoost feature importance 大致一致，但 SHAP 可以進一步看每個類別各自的解釋。

產生的圖檔：

```text
outputs/figures/03b_shap_overall_bar.png
```

## 各類別 SHAP 重要性

### Class 0: Saturated

Class 0 前 10 名特徵：

| 排名 | 特徵 | Mean abs SHAP |
|---:|---|---:|
| 1 | `electricity_consume_total` | 0.8268 |
| 2 | `housing_unit_median_value` | 0.3580 |
| 3 | `total_area` | 0.3167 |
| 4 | `frost_days` | 0.1905 |
| 5 | `population_density` | 0.1399 |
| 6 | `unemployed` | 0.1157 |
| 7 | `education_less_than_high_school` | 0.1018 |
| 8 | `heating_fuel_coal_coke_rate` | 0.0926 |
| 9 | `education_bachelor` | 0.0864 |
| 10 | `diversity` | 0.0782 |

對 class 0 來說，模型最依賴總用電量、住宅價值、總面積等變數。這表示模型會用能源需求、住宅市場條件與空間規模來辨識較像 saturated 的地區。

圖檔：

```text
outputs/figures/03b_shap_summary_class_0.png
```

### Class 1: Medium

Class 1 前 10 名特徵：

| 排名 | 特徵 | Mean abs SHAP |
|---:|---|---:|
| 1 | `electricity_consume_total` | 0.1903 |
| 2 | `frost_days` | 0.0970 |
| 3 | `travel_time_less_than_10_rate` | 0.0696 |
| 4 | `population_density` | 0.0539 |
| 5 | `total_area` | 0.0464 |
| 6 | `education_bachelor` | 0.0459 |
| 7 | `housing_unit_median_value` | 0.0424 |
| 8 | `occupation_public_rate` | 0.0372 |
| 9 | `heating_fuel_electricity` | 0.0359 |
| 10 | `travel_time_40_59_rate` | 0.0323 |

Class 1 的 SHAP 數值整體較小，代表模型對中間類別的判斷比較不明確。這也呼應 `03_model_us_summary.md` 中的結果：Medium class 是最難分類的一類。

圖檔：

```text
outputs/figures/03b_shap_summary_class_1.png
```

### Class 2: High

Class 2 前 10 名特徵：

| 排名 | 特徵 | Mean abs SHAP |
|---:|---|---:|
| 1 | `electricity_consume_total` | 0.4346 |
| 2 | `housing_unit_median_value` | 0.2706 |
| 3 | `education_bachelor` | 0.2380 |
| 4 | `population_density` | 0.2104 |
| 5 | `total_area` | 0.2096 |
| 6 | `education_less_than_high_school` | 0.1133 |
| 7 | `frost_days` | 0.1099 |
| 8 | `heating_fuel_coal_coke_rate` | 0.0989 |
| 9 | `travel_time_less_than_10_rate` | 0.0658 |
| 10 | `unemployed` | 0.0622 |

對 class 2 來說，模型主要依賴總用電量、住宅價值、教育程度、人口密度與總面積。教育相關變數在這一類中特別明顯。

圖檔：

```text
outputs/figures/03b_shap_summary_class_2.png
```

## 如何看 SHAP summary plot

每一張 SHAP summary plot 都是在解釋一個類別。

圖上的元素可以這樣看：

- 每一列是一個特徵。
- 每一個點是一筆抽樣資料。
- 顏色代表該特徵值大小：
  - 紅色代表該特徵值較高。
  - 藍色代表該特徵值較低。
- X 軸是該類別的 SHAP value：
  - 點在右邊，代表把模型預測推向該類別。
  - 點在左邊，代表把模型預測推離該類別。

舉例來說，如果在 class 0 的圖中，`electricity_consume_total` 的紅點大多在右邊，表示高總用電量會讓模型更傾向判為 class 0。若紅點大多在左邊，則表示高總用電量會讓模型遠離 class 0。

bar chart 和 summary plot 回答的問題不同：

- bar chart：哪些特徵整體最重要？
- summary plot：高或低的特徵值，會把預測往哪個類別推？

## 主要結論

SHAP 分析強化了幾個重要訊號：

- 總用電量是模型最重要的解釋特徵。
- 住宅價值、總面積、人口密度也很重要。
- 氣候變數如霜凍天數仍有影響。
- 教育與社經結構也參與模型判斷。

整體來看，模型不是只依賴單一類型變數，而是同時結合能源需求、住宅條件、地理空間、氣候與社經特徵。

但寫報告時要使用「模型行為」的語氣：

```text
模型在預測太陽能部署類別時，高度依賴總用電量與住宅價值等特徵。
```

不要寫成因果推論：

```text
提高總用電量會造成太陽能部署增加。
```
