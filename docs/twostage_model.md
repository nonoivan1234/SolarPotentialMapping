# Two-Stage Model 設計文件與修改紀錄

> 本文件記錄 `03_model_us_twostage.ipynb` 與 `src/model.py` 的設計決策、對應課程依據，以及各項修改的技術說明。

---

## 模型架構說明

### 為什麼用 Two-Stage，不用單一分類模型？

原論文 DeepSolar（Yu et al., 2018）的 SolarForest 就是兩階段架構。`tile_count` 的分布有嚴重的**零膨脹（zero-inflation）**問題：約 70% 的 census tract 根本沒有任何太陽能裝設。若直接用一個模型預測，零的地方太多，模型難以同時學好「有沒有裝」和「裝了多少」這兩個性質完全不同的問題。

| 階段 | 問題 | 目標變數 | 訓練資料 |
|------|------|----------|----------|
| Stage 1 | 有沒有太陽能裝設？ | `tile_count > 0`（binary） | 全部 tracts |
| Stage 2 | 裝了多少？ | `log1p(tile_count / household_count × 1000)` | 只有 `tile_count > 0` 的 tracts |

### 為什麼 Stage 2 用 density 而不是 raw tile_count？

用每千戶裝設量（solar systems per thousand households）而非 `tile_count` 原始值：

- `tile_count` 受 census tract 人口規模影響（人多的地方裝設數量天然偏高，不代表採用率高）
- density 去除人口規模的干擾，才能跨地區比較（e.g., 台灣的縣市人口差異很大）
- 這與原論文 SolarForest 的目標變數定義一致

### 為什麼對 density 取 log1p？

即使只看已裝設的 tracts，密度分布仍然高度右偏（少數地區密度極高）。Ch5 建議對右偏分布做對數轉換，降低極端值對模型的影響。

---

## src/model.py 修改內容

### 新增 import
```python
from sklearn.impute import SimpleImputer
```
`cross_validate_twostage()` 內部需要對每個 fold 獨立補值，需要這個類別。

### TWOSTAGE_MODELS — 加入 `oob_score=True`

```python
'classifier': RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1,
    class_weight='balanced',
    oob_score=True,   # ← 新增
),
'regressor': RandomForestRegressor(
    n_estimators=200, random_state=42, n_jobs=-1,
    oob_score=True,   # ← 新增
),
```

**依據**：Ch8 OOB 誤差 — Random Forest 訓練時自動保留未被抽到的樣本（Out-of-Bag），可作為免費的泛化估計，不需要額外的 CV。與 10-fold CV 一起使用，可以互相佐證。

### 新增兩個模型家族至 TWOSTAGE_MODELS

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

'GradientBoosting': {
    'classifier': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    'regressor':  GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
},
'Linear': {
    'classifier': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))]),
    'regressor':  Pipeline([('scaler', StandardScaler()), ('reg', Ridge())]),
},
```

**設計說明**：
- `GradientBoosting`：與 `XGBoost` 並列為 Boosting 方法，作為原生 sklearn 版本的對比
- `Linear`：可解釋性 baseline（對應 CLAUDE.md STEP 4 的 LogisticRegression + 線性迴歸）；Linear model 需要 StandardScaler，以 Pipeline 包裝確保 scaler 在每次 fit 內部自動執行，不需要在外部處理
- `oob_score` 只對 RandomForest 有效；GradientBoosting 和 Linear 無此屬性，notebook 中的 `hasattr` 判斷會自動跳過

依據：CLAUDE.md STEP 4 要求比較 RandomForest、GradientBoosting/XGBoost、LogisticRegression 三類模型。

### 新增 `cross_validate_twostage()` 函式

```python
def cross_validate_twostage(clf_model, reg_model, X_raw, y_installed, y_density, cv=10):
```

**設計原則**：

1. **StratifiedKFold**（不是 KFold）：Stage 1 的目標 `y_installed` 是不均衡的（約 30% 正類）。`StratifiedKFold` 保證每個 fold 的正/負比例一致，避免某個 fold 正樣本極少而指標失真。依據：Ch12（數據平衡）。

2. **Per-fold imputation**：`SimpleImputer` 在每個 fold 的訓練集上獨立 `fit`，再 `transform` 驗證集。若在全資料上 fit 再切分，驗證集的統計量（中位數）就洩漏進去了。依據：Ch3 資料洩漏（Data Leakage）。

3. **End-to-end 評估**：Stage 1 預測為「無裝設」的 tracts，density 預測為 0；Stage 1 預測為「有裝設」的 tracts，再由 Stage 2 預測密度。R² 算在全部測試 tracts 上，反映完整 pipeline 的實際表現。

4. **回傳指標**：`cv_r2_mean/std`（端對端 R²）、`cv_stage1_f1_mean/std`（Stage 1 F1）。

---

## notebooks/03_model_us_twostage.ipynb 修改內容

### cell-1：新增 import
```python
from src.model import TWOSTAGE_MODELS, cross_validate_twostage, ...
```

### cell-3：目標變數計算 + NaN 修正

**原始問題**：`household_count` 的 NaN 直接填 1，會讓少數地區的 density 暴漲（例如 1000 片 ÷ 1 戶 × 1000 = 1,000,000）。

**修法**：改用欄位中位數填補，並輸出 NaN 數量供確認：

```python
household_nan_count = model_df['household_count'].isna().sum()
household_median = model_df['household_count'].median()
y_households = model_df['household_count'].fillna(household_median).clip(lower=1).values
y_density = y_tile / y_households * 1000
```

依據：Ch5 遺漏值處理——中位數填補比固定值填補更合理，不引入不切實際的極端值。

### cell-4：train_test_split 加入 y_density

同時切分 `X_raw`、`y_installed`、`y_tile`、`y_density` 四個陣列，保證所有資料對齊。

### Comprehensive Comparison 3-layer loop（cad31c93 + 4b98f1fd）

對 **3 個特徵集 × 4 個模型家族 = 12 組組合**，每組跑：
- 10-fold full-pipeline CV（端對端 R² + Stage 1 F1）
- Hold-out Stage 1 分類指標（ROC-AUC / F1 / Precision / Recall）
- Hold-out Stage 2 迴歸指標（log1p R² / density R² / MAE / RMSE）

```python
for fs in ['voted', 'tree', 'vif']:
    for family_name, family_models in TWOSTAGE_MODELS.items():
        cv_res = cross_validate_twostage(...)         # 10-fold CV
        clf_fitted, clf_m = evaluate_binary_classifier(...)   # Stage 1 hold-out
        reg_fitted, reg_m_log = evaluate_regressor(...)       # Stage 2 hold-out (log1p)
        reg_m_raw = { ... }                          # Stage 2 hold-out (density scale)
        full_results.append({...})

full_df.to_csv('twostage_full_comparison.csv')
# 印出 CV R² pivot table 與 Stage 1 ROC-AUC pivot table
```

**設計原則**：
- 每個特徵集都重新取 X（y 只依賴 `tile_count` / `household_count`，不受特徵集影響）
- 每個特徵集有獨立的 imputer（fit on 訓練集，transform 測試集）
- `cross_validate_twostage` 每 fold 內部自己 fit imputer，與 hold-out imputer 各自獨立
- 結果存成 12 行的 `twostage_full_comparison.csv`，方便橫向比較

後面的 Stage 1 混淆矩陣與 Stage 2 特徵重要性詳細分析，使用 config cell 指定的 `MODEL_FAMILY` + `FEATURE_SET`（預設 `'RandomForest'` + `'voted'`）。

### cell-2：config 說明

`FEATURE_SET = 'voted'` 設為預設（voted 90 特徵，Ch9 最嚴謹）。  
`MODEL_FAMILY = 'RandomForest'` 用於後段詳細分析（混淆矩陣、特徵重要性）。  
執行時直接印出 `list(TWOSTAGE_MODELS.keys())`，讓使用者一眼看到有哪些選項。

### cell-6：Stage 1 OOB 輸出

```python
if hasattr(classifier, 'oob_score_'):
    print(f'Stage 1 OOB Accuracy: {classifier.oob_score_:.4f}')
```

依據：Ch8 — OOB 提供免費的 hold-out 估計，訓練完就能讀取，不需要額外計算。

### cell-8 markdown：更新說明

Stage 2 目標變數說明改為 density（每千戶裝設量），並說明為何使用密度而非 raw count。

### cell-9：Stage 2 使用 density + OOB 輸出

- 訓練目標：`log1p(y_dens_train[train_positive])`
- 指標名稱：`_density`（原為 `_raw_tile_count`）
- 訓練後輸出 `regressor.oob_score_`（log1p 尺度的 OOB R²）

### cell-10：圖表軸標籤更新

```python
plt.xlabel('Actual density (solar / 1000 HH)')
plt.ylabel('Predicted density (solar / 1000 HH)')
```

### cell-13：bundle 說明字串更新

```python
'stage2_target': 'log1p(tile_count / household_count * 1000), trained only where tile_count > 0',
```

---

## MDS 課程對應總表

| 修改 | 對應章節 | 內容 |
|------|----------|------|
| StratifiedKFold | Ch12 | 類別不均衡 → CV 要保持各 fold 比例一致 |
| Per-fold imputation | Ch3 | 資料洩漏 → 補值只能在訓練集上 fit |
| oob_score=True | Ch8 | OOB 誤差作為 RF 的免費泛化估計 |
| log1p(density) | Ch5 | 右偏分布的對數轉換 |
| class_weight='balanced' | Ch12 | Cost-sensitive learning 處理不均衡 |
| 10-fold CV | Ch3 | K-Fold 提供比 hold-out 更穩健的估計 |
| household_count 中位數補值 | Ch5 | 遺漏值補值策略選擇 |
| 四模型比較（RF / XGB / GB / Linear） | CLAUDE.md STEP 4 | RandomForest、GradientBoosting/XGBoost、LogisticRegression 全部比較 |
| Linear 用 Pipeline 包裝 StandardScaler | Ch5 / Ch6 | 線性模型需要標準化；Pipeline 確保 scaler 在 CV 各 fold 內部重新 fit，無洩漏 |
| tune_twostage()（Sequential Tuning） | 附錄D | RandomizedSearchCV 高效搜尋超參數；Stage 1 scoring=f1，Stage 2 scoring=r2 |

---

## src/model.py — 新增 TWOSTAGE_PARAM_GRIDS 與 tune_twostage()

### TWOSTAGE_PARAM_GRIDS

```python
TWOSTAGE_PARAM_GRIDS = {
    'RandomForest': {
        'classifier': {'n_estimators': [100,200,300], 'max_depth': [None,10,20], 'max_features': ['sqrt','log2'], 'min_samples_leaf': [1,2,5]},
        'regressor':  {'n_estimators': [100,200,300], 'max_depth': [None,10,20], 'max_features': ['sqrt','log2',0.3], 'min_samples_leaf': [1,2,5]},
    },
    'XGBoost': {
        'classifier': {'n_estimators': [100,200,300], 'max_depth': [4,6,8], 'learning_rate': [0.05,0.1,0.2], 'subsample': [0.7,0.8,1.0], 'colsample_bytree': [0.7,0.8,1.0]},
        'regressor':  {# 同上},
    },
    'GradientBoosting': {
        'classifier': {'n_estimators': [100,200,300], 'max_depth': [3,5,7], 'learning_rate': [0.05,0.1,0.2], 'subsample': [0.7,0.8,1.0]},
        'regressor':  {# 同上},
    },
}
```

**為什麼不含 Linear？** Linear 在 `TWOSTAGE_MODELS` 中包在 `Pipeline` 裡，RandomizedSearchCV 的 param_distributions 需加前綴（`clf__C`、`reg__alpha`），設計上與 tree-based 不同。Linear 為可解釋性 baseline，不是主要調整對象。

### tune_twostage()

```python
def tune_twostage(model_name, X, y_installed, y_density, n_iter=20, cv=5):
```

**設計原則（Sequential Tuning）**：

1. **Stage 1 獨立調整**：全資料跑 `RandomizedSearchCV`，scoring=`f1`，使用 `StratifiedKFold` 保持類別比例（Ch12）。
2. **Stage 2 獨立調整**：只取 `y_installed == 1` 的正類樣本，scoring=`r2`（log1p 尺度），使用 `KFold`（迴歸無需 stratify）。
3. **Pre-impute once**：在進入 search 前用 `SimpleImputer(median)` 處理遺漏值，兩個 search 共用同一份 imputed X。此處與 `cross_validate_twostage` 略有不同（那裡每 fold 重新 fit imputer），因為 RandomizedSearchCV 內部的 CV 已在 imputed 資料上進行，目的是參數搜尋效率，不是最終泛化評估。
4. **回傳值**：`best_clf`、`best_reg`、metrics dict（含 best F1 與 best R²）。

**依據**：附錄D 超參數最佳化 — RandomizedSearchCV 比 GridSearchCV 在高維參數空間更有效率（n_iter=20 等同隨機採樣 20 組參數組合）。

---

## notebooks/03_model_us_twostage.ipynb — 新增調整區塊

### 新增 cell：超參數調整 markdown + code（在 Model Comparison 之後）

```python
# 若 MODEL_FAMILY 在 TWOSTAGE_PARAM_GRIDS 中，執行 tune_twostage()
tuned_clf, tuned_reg, tune_metrics = tune_twostage(
    MODEL_FAMILY, X_raw, y_installed, y_density, n_iter=20, cv=5
)
# 存入 best_us_twostage_model_{FEATURE_SET}_tuned.pkl（供台灣遷移推論使用）
```

若 MODEL_FAMILY 為 Linear，自動跳過（印出提示），不影響後續流程。

---

## 目前評估指標說明

執行完整 notebook 後會產生以下結果：

| 檔案 | 內容 |
|------|------|
| `twostage_full_comparison.csv` | **主要比較表**：3 特徵集 × 4 模型家族 = 12 行；含 10-fold CV R² ± std、Stage 1 F1 ± std、Stage 1 ROC-AUC/Precision/Recall、Stage 2 log1p R²/MAE/RMSE、density 尺度 R²/MAE/RMSE |
| `twostage_classifier_metrics_{FEATURE_SET}.csv` | Hold-out：主線 MODEL_FAMILY 的 Stage 1 詳細指標 |
| `twostage_regressor_metrics_{FEATURE_SET}.csv` | Hold-out：主線 MODEL_FAMILY 的 Stage 2 詳細指標（log1p 與 density 兩種尺度） |
| `best_us_twostage_model_{FEATURE_SET}.pkl` | 主線模型 bundle（classifier + regressor + imputer + feature_names） |
| `best_us_twostage_model_{FEATURE_SET}_tuned.pkl` | 超參數調整後的 bundle（含 tune_metrics） |

另有 OOB 輸出（Stage 1 OOB Accuracy、Stage 2 OOB R²，僅 RandomForest）直接印在 cell 輸出，不存檔。  
執行時也會印出 CV R² pivot table 與 Stage 1 ROC-AUC pivot table，方便快速比較 12 組結果。

---

## 已知缺口（低優先，待補充）

| 缺口 | 對應章節 | 說明 | 建議補法 |
|------|---------|------|---------|
| Wrapper 特徵選擇 | Ch9 §2 | Forward / Backward Stepwise 未做；現有三方法投票（Lasso + ElasticNet + RF）在實務上已覆蓋此用途 | 在 `02_feature_selection.ipynb` 加 `SequentialFeatureSelector` 區塊 |
| SMOTE / Undersampling | Ch12 | Stage 1 以 `class_weight='balanced'`（Ch12 Cost-sensitive）處理不均衡；三分類本身類別平衡（各約 33%），不需要 | 若要補，在 Stage 1 訓練前加 `imbalanced-learn` 的 SMOTE，限用於 two-stage Stage 1 |
