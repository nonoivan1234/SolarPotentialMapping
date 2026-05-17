# Code Review Issues — Solar Potential Mapping

> 審閱日期：2026-05-16
> 審閱者：Claude（對照 MDS 課程知識庫）
> 對照章節：Ch3（模型評估）、Ch5（數據預處理）、Ch9（特徵挑選）、Ch12（數據平衡）

---

## 審閱摘要

| # | 嚴重度 | 類別 | 位置 | 狀態 |
|---|--------|------|------|------|
| 1 | 嚴重 | Data Leakage | `src/model.py` | 已修正 |
| 2 | 中等 | 方法論不完整 | `notebooks/02_feature_selection.ipynb` | 已修正 |
| 3 | 中等 | 統計方法選擇 | `notebooks/02_feature_selection.ipynb` | 已修正 |
| 4 | 輕微 | PerformanceWarning | `notebooks/02_feature_selection.ipynb` | 已修正 |
| 5 | 輕微 | 超參數調優缺失 | `src/model.py` | 已修正 |
| 6 | 輕微 | 業務指標未對齊 | `src/model.py`, `notebooks/03_model_us.ipynb` | 已修正 |
| 7 | 補充 | FutureWarning | `src/model.py` | 已修正（順帶） |

---

## Issue #1：Data Leakage — LogisticRegression 的 StandardScaler 放在 CV 外

**嚴重度**：嚴重（方法論錯誤，造成評估分數虛高）

**位置**：[src/model.py](../src/model.py) → `train_and_evaluate()`

**問題描述**

原始程式碼在呼叫 `cross_validate` 之前，先對整個 `X` 做 `scaler.fit_transform(X)`。
這導致「驗證折（validation fold）的統計資訊（均值、標準差）」洩漏至 scaler，
使 CV 分數過於樂觀，無法真實反映模型對未見資料的泛化能力。

```python
# 修正前（data leakage）
scaler = StandardScaler()
X = scaler.fit_transform(X)          # 整個 X 含驗證折資訊 → 洩漏
scores = cross_validate(model, X, y, cv=skf, ...)
```

**對應 MDS 課程**：Ch3 §交叉驗證 — 驗證集必須完全模擬未見資料，不得使用任何來自驗證集的統計資訊。

**修正方式**

將 `StandardScaler` 包進 `sklearn.pipeline.Pipeline`，讓 scaler 在每折內部獨立 fit：

```python
# 修正後（正確）
eval_model = Pipeline([('scaler', StandardScaler()), ('clf', clone(model))])
scores = cross_validate(eval_model, X, y, cv=skf, ...)
```

**受影響範圍**：僅 `train_and_evaluate()` 的 CV 評估流程。Notebook 03 中對全資料的最終 fit 不受影響（無 CV 的 leakage）。

---

## Issue #2：特徵挑選方法不完整（缺 Embedded 法）

**嚴重度**：中等（方法論不符合 MDS Ch9 標準）

**位置**：[notebooks/02_feature_selection.ipynb](../notebooks/02_feature_selection.ipynb)

**問題描述**

原始 notebook 只有**過濾法（Filter）**（Pearson 相關係數 + VIF），缺少：
- Embedded：Lasso / ElasticNet 正則化選擇（Ch9 §3 核心方法）
- Embedded：Random Forest 特徵重要性（Ch9 §3）
- 投票彙整機制（Ch9 §4）

MDS Ch9 建議四大方法 + 投票，半導體封裝脫層案例中也用了 Lasso + 逐步迴歸 + 投票法（Q=200 次 Under-sampling）。

**對應 MDS 課程**：Ch9 §3 嵌入法（Embedded）、§4 投票法與工程驗證

**修正方式**

新增以下四個分析段落（插入在 `corr_selected` 篩選之後）：

1. **ANOVA F-test** — 對分類目標做 F 統計排序，同時輸出 Mutual Information
2. **LassoCV** — 5-fold CV 自動選 alpha，L1 壓縮係數至 0 的特徵剔除
3. **ElasticNetCV** — L1+L2 結合，`l1_ratio` 由 CV 自動選，處理高共線性特徵群更穩健
4. **RF importance** — 300 棵樹的平均不純度下降，取 >= 平均重要性的特徵
5. **Voting** — 三種方法中 >= 2 方法選到 → `voted_features`

**新增輸出**

| 檔案 | 內容 |
|------|------|
| `outputs/results/transfer/filter_ranking.csv` | ANOVA F-score + Mutual Information 排名 |
| `outputs/results/transfer/voting_summary.csv` | 各方法投票結果（lasso / enet / rf / vote_count） |
| `data/processed/transfer/us_modeling_ready_voted.csv` | voted 特徵集建模資料 |

**metadata.json 新增欄位**

```json
{
  "lasso_features": [...],
  "enet_features": [...],
  "rf_features": [...],
  "voted_features": [...]
}
```

---

## Issue #3：Filter 法對分類目標使用 Pearson 不理想

**嚴重度**：中等（統計方法不匹配）

**位置**：[notebooks/02_feature_selection.ipynb](../notebooks/02_feature_selection.ipynb) → `remove_high_corr_features()` 的「保留哪個」邏輯

**問題描述**

`remove_high_corr_features()` 使用 Pearson 相關係數（feature vs 連續 `tile_count`）決定當兩特徵高共線時保留哪個。

但本問題是三分類（Class 0/1/2），在分類目標下，衡量特徵鑑別能力應使用：
- **ANOVA F-test**：連續特徵 vs 類別標籤（Ch9 §1 過濾法指標之一）
- **Mutual Information**：非線性關係也能捕捉

Pearson 與 ANOVA F-test 在排序上有可能出現不一致，特別是非線性或異方差的特徵。

**對應 MDS 課程**：Ch9 §1 過濾法 — 相關性指標包括 Pearson、Jaccard、ANOVA 等，需依問題類型選擇

**修正方式**

Issue #2 的修正已同時提供 ANOVA F-score 與 Mutual Information 排名，可與原 Pearson 結果對照，作為特徵保留決策的補充依據。

`remove_high_corr_features()` 本身的邏輯（以 Pearson 決定 tie-breaking）保持不變，但 `filter_ranking.csv` 提供了多指標對照。

---

## Issue #4：DataFrame 碎片化造成 PerformanceWarning

**嚴重度**：輕微（不影響結果，但影響執行效率）

**位置**：[notebooks/02_feature_selection.ipynb](../notebooks/02_feature_selection.ipynb) → cell `dbee6c84`

**問題描述**

```python
for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
```

逐欄賦值（共 130 次）導致 DataFrame 內部記憶體高度碎片化，後續 `df_clean['potential_label'] = ...` 觸發：

```
PerformanceWarning: DataFrame is highly fragmented.
This is usually the result of calling `frame.insert` many times,
which has poor performance. Consider joining all columns at once
using pd.concat(axis=1) instead.
```

**對應 MDS 課程**：Ch5 §數據清理 — 遺漏值填補應注意效能

**修正方式**

在 fillna 迴圈後加入一行：

```python
df_clean = df_clean.copy()  # consolidate fragmented memory
```

`DataFrame.copy()` 會重新分配連續記憶體，消除碎片化。

---

## Issue #5：超參數調優

**嚴重度**：輕微

**位置**：[src/model.py](../src/model.py)、[notebooks/03_model_us.ipynb](../notebooks/03_model_us.ipynb)

**問題描述**

所有模型使用手動設定的固定超參數：

```python
RandomForestClassifier(n_estimators=200, ...)
XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, ...)
```

MDS 附錄D（超參數最佳化）建議系統性搜尋。XGBoost 對 `max_depth`、`min_child_weight` 非常敏感；RandomForest 對 `max_features` 和 `min_samples_leaf` 也有顯著影響。

**對應 MDS 課程**：附錄D 超參數最佳化

**修正方式**

在 `src/model.py` 新增 `PARAM_GRIDS` 與 `tune_model()`：

```python
PARAM_GRIDS = {
    'RandomForest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2', 0.3],
        'min_samples_leaf': [1, 2, 5],
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
    },
}

def tune_model(X, y, model_name, n_iter=30, cv=5, scoring='recall_class2'):
    """RandomizedSearchCV over PARAM_GRIDS[model_name]."""
    _scoring = METRICS.get(scoring, scoring) if isinstance(scoring, str) else scoring
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        clone(MODELS[model_name]),
        param_distributions=PARAM_GRIDS[model_name],
        n_iter=n_iter, cv=skf, scoring=_scoring,
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_
```

在 notebook 03 新增 cell 呼叫（在模型比較視覺化之後）：

```python
tuned_estimator, tuned_params, tuned_score = tune_model(
    X, y, model_name=tune_target, n_iter=30, cv=5, scoring='recall_class2'
)
# → 儲存至 best_us_model_{FEATURE_SET}_tuned.pkl
```

`scoring='recall_class2'` 字串會自動從 `METRICS` dict 取得對應的 `make_scorer` 物件（見 Issue #6 修正）。

---

## Issue #6：Class 2（高潛力）的 Recall 未被優先考慮

**嚴重度**：輕微（業務目標未充分反映在評估指標中）

**位置**：[src/model.py](../src/model.py) → `METRICS`、`MODELS`；[notebooks/03_model_us.ipynb](../notebooks/03_model_us.ipynb)

**問題描述**

本專案商業目的是「廣告投放目標篩選 + 政府補助資源分配」，業務上需優先找出高潛力地區（Class 2）：

| 錯誤類型 | 業務影響 |
|---------|---------|
| FP（把「已飽和」誤判為「高潛力」） | 廣告打在無效益地區，直接浪費預算 |
| FN（把「高潛力」誤判為「已飽和」） | 錯失目標客群，機會成本高 |

依 MDS Ch3 §FP vs FN 的風險權衡 — 「需依問題決定哪種風險代價較高」，本問題應優先最大化 **Class 2 Recall**。但原本 `f1_macro` 等權重三類，Class 2 並未被特別強調。

**對應 MDS 課程**：Ch3 §FP vs FN 的風險權衡

**修正方式 — 三處同步修改**

**① `src/model.py`：新增 `recall_class2` custom scorer，`METRICS` 改為 dict**

```python
from sklearn.metrics import make_scorer, recall_score

def _recall_class2(y_true, y_pred):
    per_class = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return float(per_class[2])

METRICS = {
    'accuracy':        'accuracy',
    'f1_macro':        'f1_macro',
    'roc_auc_ovr':     'roc_auc_ovr',
    'precision_macro': 'precision_macro',
    'recall_macro':    'recall_macro',
    'recall_class2':   make_scorer(_recall_class2),   # ← 新增
}
```

`cross_validate` 接受 dict 型 scoring，`results_df` 會自動多一欄 `recall_class2`。

**② `src/model.py`：LR 與 RF 加入 `class_weight`，訓練時對 Class 2 誤分類代價加倍**

```python
MODELS = {
    'LogisticRegression': LogisticRegression(
        max_iter=1000, class_weight={0: 1, 1: 1, 2: 2}
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1,
        class_weight={0: 1, 1: 1, 2: 2}   # ← 新增
    ),
    # GradientBoosting / XGBoost 不支援 class_weight，改以 recall_class2 為選模指標
    ...
}
```

**③ `notebooks/03_model_us.ipynb`：三處更新**

最佳模型改以 `recall_class2` 選擇：
```python
# 修正前
best_model_name = results_df.loc[results_df['f1_macro'].idxmax(), 'model']

# 修正後
best_model_name = results_df.loc[results_df['recall_class2'].idxmax(), 'model']
```

視覺化 cell 改為動態偵測欄位（有 `recall_class2` 就顯示）：
```python
metrics_to_plot = [m for m in ['accuracy', 'f1_macro', 'roc_auc_ovr', 'recall_class2', 'recall_macro']
                   if m in results_df.columns]
```

新增業務 KPI 比較 cell，明確對比兩種選模邏輯：
```python
biz_df = results_df[['model', 'recall_class2', 'f1_macro']].sort_values('recall_class2', ascending=False)
print('[Business pick — max recall_class2] :', biz_df.iloc[0]['model'])
print('[Statistical pick — max f1_macro]   :', results_df.loc[results_df['f1_macro'].idxmax(), 'model'])
```

---

## Issue #7：LogisticRegression n_jobs FutureWarning（已修正）

**嚴重度**：補充說明

**位置**：[src/model.py](../src/model.py) → `MODELS['LogisticRegression']`

**問題描述**

sklearn >= 1.8 中 `LogisticRegression(n_jobs=-1)` 會觸發 FutureWarning：
```
'n_jobs' has no effect since 1.8 and will be removed in 1.10.
```

已在修正 Issue #1 時一併移除 `n_jobs=-1`。

---

## 修正後的新特徵集結構

```
notebooks/02_feature_selection.ipynb 執行後產出：

corr_selected (~102)  ←── Pearson + VIF(0.85) 篩選後
    ├── tree_features = corr_selected     (102 個，樹模型用)
    ├── vif_features  ⊂ corr_selected     (~55 個，VIF<10，LR 用)
    ├── lasso_selected ⊂ corr_selected    (LassoCV 非零係數)
    ├── enet_selected  ⊂ corr_selected    (ElasticNetCV 非零係數)
    ├── rf_selected    ⊂ corr_selected    (RF importance >= 平均)
    └── voted_features ⊂ corr_selected    (≥2/3 方法選到，推薦)
```

**建議模型訓練使用 `voted_features`**，它在可解釋性與預測力之間取得最佳平衡。

---

*對照依據：MDS 課程講義（Ch3、Ch5、Ch9、Ch12、附錄D）*
