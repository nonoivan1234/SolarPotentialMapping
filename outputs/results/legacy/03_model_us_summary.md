# 03 US Model Training Summary

## Purpose

This notebook trains and compares several classification models using the feature-selected DeepSolar dataset.

Input dataset:

```text
data/processed/us_modeling_ready.csv
Shape: (72537, 50)
Features: 49
Target: potential_label
```

The target distribution is balanced:

```text
potential_label
0    24279
1    24174
2    24084
```

This means the dataset contains 49 selected features plus one target column. The three classes are nearly equal in size, which is useful for model comparison because no class dominates the training process.

## Models Compared

Four models were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Evaluation was performed with 5-fold stratified cross-validation using:

- Accuracy
- Macro F1
- One-vs-rest ROC AUC
- Macro precision
- Macro recall

Macro-averaged metrics are appropriate here because this is a multiclass classification task and each class should be treated with similar importance.

## Cross-Validation Results

Saved result:

```text
outputs/results/model_comparison.csv
```

| Model | Accuracy | F1 macro | ROC AUC OVR | Precision macro | Recall macro |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.6001 | 0.5976 | 0.7818 | 0.5961 | 0.6000 |
| RandomForest | 0.6857 | 0.6874 | 0.8519 | 0.6899 | 0.6856 |
| GradientBoosting | 0.6851 | 0.6860 | 0.8514 | 0.6872 | 0.6849 |
| XGBoost | 0.6999 | 0.7012 | 0.8641 | 0.7031 | 0.6998 |

XGBoost performs best across all reported metrics. Its macro F1 is 0.7012 and ROC AUC OVR is 0.8641, indicating the strongest overall classification performance among the tested models.

The tree-based models clearly outperform Logistic Regression. This suggests that nonlinear relationships and feature interactions matter for this problem.

Generated figure:

```text
outputs/figures/03_model_comparison.png
```

## Best Model

The best model was selected by macro F1:

```text
Best model by F1-macro: XGBoost
```

The trained model bundle was saved to:

```text
outputs/results/best_us_model.pkl
```

## Feature Importance

Saved result:

```text
outputs/results/feature_importance.csv
```

Top important features from the best model:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `electricity_consume_total` | 0.1388 |
| 2 | `population_density` | 0.0726 |
| 3 | `housing_unit_median_value` | 0.0671 |
| 4 | `education_bachelor` | 0.0656 |
| 5 | `total_area` | 0.0512 |
| 6 | `frost_days` | 0.0371 |
| 7 | `heating_fuel_coal_coke_rate` | 0.0315 |
| 8 | `unemployed` | 0.0221 |
| 9 | `heating_fuel_solar_rate` | 0.0214 |
| 10 | `travel_time_less_than_10_rate` | 0.0200 |

Interpretation:

- `electricity_consume_total` is the most important feature, suggesting that energy demand is strongly related to solar deployment class.
- `population_density`, `total_area`, and `housing_unit_median_value` indicate that spatial and housing market characteristics matter.
- Education and socioeconomic indicators also contribute meaningfully.
- Climate or energy-related variables such as `frost_days` and heating fuel rates remain relevant.

Generated figure:

```text
outputs/figures/03_feature_importance.png
```

## Hold-out Validation

The notebook also performs an 80/20 stratified train-test split for interpretability and reports classification performance on the hold-out test set.

```text
               precision    recall  f1-score   support

Saturated (0)       0.83      0.79      0.81      4856
   Medium (1)       0.58      0.58      0.58      4835
     High (2)       0.69      0.71      0.70      4817

     accuracy                           0.70     14508
    macro avg       0.70      0.70      0.70     14508
 weighted avg       0.70      0.70      0.70     14508
```

The best model performs well on the saturated class, with F1 = 0.81. The medium class is the hardest to classify, with F1 = 0.58. This is expected because middle categories often have less distinct boundaries than extreme classes.

Generated figure:

```text
outputs/figures/03_confusion_matrix.png
```

## Notes and Issues

The current model is a solid baseline for US data. XGBoost reaches about 0.70 macro F1 and 0.864 ROC AUC OVR.

The medium class remains the main weakness. Future improvements could include:

- Testing alternative target definitions.
- Keeping or reintroducing domain-critical features such as solar radiation if interpretability or transferability is important.
- Performing hyperparameter tuning for XGBoost.
- Checking whether class labels should represent current deployment, future potential, or deployment saturation more explicitly.

One implementation note: Logistic Regression emits a FutureWarning because `n_jobs=-1` is no longer useful in newer scikit-learn versions. This does not affect current results, but `n_jobs=-1` can be removed from the Logistic Regression definition.

---

# 03 美國模型訓練摘要

## 目的

這本 notebook 使用 `02_feature_selection` 產生的特徵篩選資料，訓練並比較多個分類模型。

輸入資料：

```text
data/processed/us_modeling_ready.csv
Shape: (72537, 50)
Features: 49
Target: potential_label
```

目標類別分布如下：

```text
potential_label
0    24279
1    24174
2    24084
```

也就是資料共有 72,537 筆，包含 49 個特徵與 1 個目標欄位。三個類別數量幾乎相同，因此模型訓練不會明顯偏向某一個類別。

## 比較的模型

本階段比較四個模型：

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

評估方式是 5-fold stratified cross-validation，使用的指標包括：

- Accuracy
- Macro F1
- One-vs-rest ROC AUC
- Macro precision
- Macro recall

因為這是三分類問題，而且三個類別都重要，所以使用 macro-averaged metrics 是合理的。

## 交叉驗證結果

儲存結果：

```text
outputs/results/model_comparison.csv
```

| 模型 | Accuracy | F1 macro | ROC AUC OVR | Precision macro | Recall macro |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.6001 | 0.5976 | 0.7818 | 0.5961 | 0.6000 |
| RandomForest | 0.6857 | 0.6874 | 0.8519 | 0.6899 | 0.6856 |
| GradientBoosting | 0.6851 | 0.6860 | 0.8514 | 0.6872 | 0.6849 |
| XGBoost | 0.6999 | 0.7012 | 0.8641 | 0.7031 | 0.6998 |

XGBoost 在所有主要指標上表現最好，macro F1 為 0.7012，ROC AUC OVR 為 0.8641。

可以看到 tree-based models 明顯優於 Logistic Regression，代表這個任務可能存在非線性關係與特徵交互作用。

產生的圖檔：

```text
outputs/figures/03_model_comparison.png
```

## 最佳模型

依照 macro F1 選出的最佳模型是：

```text
Best model by F1-macro: XGBoost
```

模型已儲存至：

```text
outputs/results/best_us_model.pkl
```

## 特徵重要性

儲存結果：

```text
outputs/results/feature_importance.csv
```

前 10 個重要特徵如下：

| 排名 | 特徵 | Importance |
|---:|---|---:|
| 1 | `electricity_consume_total` | 0.1388 |
| 2 | `population_density` | 0.0726 |
| 3 | `housing_unit_median_value` | 0.0671 |
| 4 | `education_bachelor` | 0.0656 |
| 5 | `total_area` | 0.0512 |
| 6 | `frost_days` | 0.0371 |
| 7 | `heating_fuel_coal_coke_rate` | 0.0315 |
| 8 | `unemployed` | 0.0221 |
| 9 | `heating_fuel_solar_rate` | 0.0214 |
| 10 | `travel_time_less_than_10_rate` | 0.0200 |

解讀如下：

- `electricity_consume_total` 是最重要的特徵，表示總用電量與太陽能部署類別有明顯關聯。
- `population_density`、`total_area`、`housing_unit_median_value` 顯示空間條件與住宅市場特徵很重要。
- 教育與社經變數也有貢獻。
- `frost_days`、取暖燃料比例等氣候與能源使用變數仍然有解釋力。

產生的圖檔：

```text
outputs/figures/03_feature_importance.png
```

## Hold-out 驗證

Notebook 另外做了 80/20 stratified train-test split，用來看最佳模型在測試集上的表現。

```text
               precision    recall  f1-score   support

Saturated (0)       0.83      0.79      0.81      4856
   Medium (1)       0.58      0.58      0.58      4835
     High (2)       0.69      0.71      0.70      4817

     accuracy                           0.70     14508
    macro avg       0.70      0.70      0.70     14508
 weighted avg       0.70      0.70      0.70     14508
```

模型對 `Saturated (0)` 的辨識最好，F1-score 為 0.81。`Medium (1)` 是最難分類的類別，F1-score 只有 0.58。這是可以理解的，因為中間類別通常不像兩端類別那麼明確。

產生的圖檔：

```text
outputs/figures/03_confusion_matrix.png
```

## 注意事項

目前模型可以視為美國資料上的一版可靠 baseline。XGBoost 達到約 0.70 macro F1 與 0.864 ROC AUC OVR。

主要弱點是中間類別較難分類。後續可以考慮：

- 測試不同的目標定義方式。
- 若重視可解釋性或台灣遷移，可討論是否保留或重新加入日照等領域重要特徵。
- 對 XGBoost 做 hyperparameter tuning。
- 更清楚定義 label 代表的是目前部署程度、未來潛力，還是市場飽和程度。

實作上還有一個小提醒：Logistic Regression 會出現 `n_jobs=-1` 的 FutureWarning。這不影響目前結果，但之後可以從 Logistic Regression 設定中移除 `n_jobs=-1`。
