# Repo Redesign Plan

> **歷史文件說明**：本文件是專案改版初期的設計草案，記錄目錄結構重組與特徵工程流程的規劃思路。
> 實際實作已完成，部分細節與本文不同（例如正式模型改為 two-stage，未建立 `model_comparison_tree.csv` 等舊版產出）。
> 若需了解目前實際架構，請參考 `README.md` 與 `docs/twostage_model.md`。

## Goal

Redesign the project structure and feature-selection pipeline so the repo can support two purposes clearly:

1. Preserve the current completed analysis as a legacy baseline.
2. Build a Taiwan-transfer-oriented pipeline that keeps transferable policy, energy, climate, housing, and socioeconomic concepts.

This plan should be followed before making broad repo changes, so the implementation does not drift.

## Key Decisions

### Keep Legacy Notebooks

The current notebooks should be preserved as legacy references instead of overwritten.

Planned structure:

```text
notebooks/
  legacy/
    01_eda_legacy.ipynb
    02_feature_selection_legacy.ipynb
    03_model_us_legacy.ipynb
    03b_model_explainability_legacy.ipynb
  01_eda.ipynb
  02_feature_selection.ipynb
  03_model_us.ipynb
  03b_model_explainability.ipynb
```

The root-level notebooks will become the redesigned transfer-oriented workflow. The `legacy/` copies preserve the previous working version and its outputs.

### Add Legacy and Transfer Output Directories

Current outputs should be moved under `legacy/`. New outputs should be written under `transfer/`.

Planned structure:

```text
data/
  processed/
    legacy/
      column_metadata.json
      us_modeling_ready.csv
    transfer/
      column_metadata.json
      us_modeling_ready_vif.csv
      us_modeling_ready_tree.csv

outputs/
  figures/
    legacy/
      01_missing_values.png
      01_tile_count_distribution.png
      02_correlation_heatmap.png
      02_vif_final.png
      03_model_comparison.png
      03_feature_importance.png
      03_confusion_matrix.png
      03b_shap_overall_bar.png
      03b_shap_summary_class_0.png
      03b_shap_summary_class_1.png
      03b_shap_summary_class_2.png
    transfer/
      ...

  results/
    legacy/
      01_eda_summary.md
      02_feature_selection_summary.md
      03_model_us_summary.md
      03b_model_explainability_summary.md
      selected_features.csv
      model_comparison.csv
      feature_importance.csv
      best_us_model.pkl
      03b_shap_importance.csv
    transfer/
      selected_features_vif.csv
      selected_features_tree.csv
      model_comparison_tree.csv
      feature_importance_tree.csv
      best_us_model_tree.pkl
      03b_shap_importance_tree.csv
      02_feature_selection_summary_transfer.md
      03_model_us_summary_tree.md
      03b_model_explainability_summary_tree.md
```

The old root-level output files should not remain as the main source of truth once migration is complete.

## Feature Classification Redesign

The current function name `get_us_only_cols()` is misleading because it mixes truly non-transferable fields with policy and energy-economic concepts.

In the redesigned pipeline, remove `get_us_only_cols()` from the main workflow and replace it with clearer groups. A small compatibility shim may remain in `src/features.py` only so notebooks under `notebooks/legacy/` can still run.

### Non-transferable Columns

These should be dropped for Taiwan-transfer modeling:

```text
race_*
voting_*
fips
state
county
```

Reason: these encode US-specific demographics, political context, or administrative geography that do not map cleanly to Taiwan.

### Policy Concept Columns

These should not be automatically dropped. They represent policy or incentive concepts that can be remapped to Taiwan.

```text
incentive_count_residential
incentive_count_nonresidential
incentive_residential_state_level
incentive_nonresidential_state_level
net_metering
feedin_tariff
cooperate_tax
property_tax
sales_tax
rebate
avg_electricity_retail_rate
```

Reason: the original US encoding may not transfer directly, but the concepts of subsidy, FIT, rebate, tax support, and electricity retail rates are meaningful for Taiwan.

### ID / Index Columns

These should be dropped from modeling inputs:

```text
lat
lon
Unnamed: 0
```

Reason: `Unnamed: 0` is an index artifact. `lat` and `lon` should not be used directly in the baseline transfer model unless a spatial modeling strategy is intentionally added.

### Target / Leakage Columns

These should be dropped from modeling inputs:

```text
tile_count
solar_system_count
total_panel_area
tile_count_residential
tile_count_nonresidential
solar_system_count_residential
solar_system_count_nonresidential
total_panel_area_residential
total_panel_area_nonresidential
solar_panel_area_divided_by_area
solar_panel_area_per_capita
number_of_solar_system_per_household
```

Reason: these are solar deployment outcomes or direct derivatives of the target.

## Planned `src/features.py` Changes

Replace the old grouping logic with:

```python
def get_non_transferable_cols():
    ...

def get_policy_concept_cols():
    ...

def get_id_cols():
    ...

def get_target_cols():
    ...

def prepare_features(
    df,
    drop_non_transferable=True,
    drop_policy_concepts=False,
    drop_targets=True,
    drop_ids=True,
):
    ...
```

Default behavior for the redesigned transfer-oriented pipeline:

```text
drop_non_transferable=True
drop_policy_concepts=False
drop_targets=True
drop_ids=True
```

This keeps transferable policy and energy-economic concepts while dropping fields that are truly non-transferable or leaky.

## Feature Set Outputs

The redesigned `02_feature_selection.ipynb` should produce two feature sets.

### VIF Feature Set

Output files:

```text
outputs/results/transfer/selected_features_vif.csv
data/processed/transfer/us_modeling_ready_vif.csv
```

Purpose:

```text
Interpretable / linear-friendly baseline
Controls multicollinearity
Useful for comparison with legacy results
```

Workflow:

```text
drop non-transferable columns
keep policy concept columns
drop target leakage
median imputation
target labeling
correlation screening
VIF screening
```

### Tree / Transfer Feature Set

Output files:

```text
outputs/results/transfer/selected_features_tree.csv
data/processed/transfer/us_modeling_ready_tree.csv
```

Purpose:

```text
XGBoost / RandomForest modeling
Taiwan transfer
Retains domain-critical and policy/economic concepts
Avoids over-pruning features that tree models can handle
```

Workflow for first redesigned version:

```text
drop non-transferable columns
keep policy concept columns
drop target leakage
median imputation
target labeling
correlation screening only
no VIF
```

No protected VIF list is needed in the first version because the tree feature set skips VIF entirely.

## Model Training Changes

`03_model_us.ipynb` should support configurable profiles and feature sets.

Example:

```python
PROFILE = "transfer"
FEATURE_SET = "tree"

DATA_PATH = f"../data/processed/{PROFILE}/us_modeling_ready_{FEATURE_SET}.csv"
RESULT_DIR = f"../outputs/results/{PROFILE}"
FIGURE_DIR = f"../outputs/figures/{PROFILE}"
```

Expected tree-transfer outputs:

```text
outputs/results/transfer/model_comparison_tree.csv
outputs/results/transfer/feature_importance_tree.csv
outputs/results/transfer/best_us_model_tree.pkl
outputs/figures/transfer/03_model_comparison_tree.png
outputs/figures/transfer/03_feature_importance_tree.png
outputs/figures/transfer/03_confusion_matrix_tree.png
```

## SHAP Explainability Changes

`03b_model_explainability.ipynb` should also support configurable profiles and feature sets.

Example:

```python
PROFILE = "transfer"
FEATURE_SET = "tree"

MODEL_PATH = f"../outputs/results/{PROFILE}/best_us_model_{FEATURE_SET}.pkl"
DATA_PATH = f"../data/processed/{PROFILE}/us_modeling_ready_{FEATURE_SET}.csv"
```

Expected outputs:

```text
outputs/results/transfer/03b_shap_importance_tree.csv
outputs/figures/transfer/03b_shap_overall_bar_tree.png
outputs/figures/transfer/03b_shap_summary_class_0_tree.png
outputs/figures/transfer/03b_shap_summary_class_1_tree.png
outputs/figures/transfer/03b_shap_summary_class_2_tree.png
```

## README Updates

Replace the current wording:

```text
移除美國特有欄位（補貼制度、族裔組成）
```

With:

```text
移除無法對應至台灣制度之美國地域性欄位，保留具一般能源經濟意義之政策、電價與氣候特徵，並於台灣遷移階段重新建立對應 mapping。
```

Also add a short explanation:

```text
本專案保留 legacy baseline 與 transfer-oriented pipeline：
- legacy：早期 US-only + VIF 篩選流程，作為 baseline 與開發紀錄。
- transfer：新版台灣遷移導向流程，區分不可轉移欄位與可重新 mapping 的政策/能源經濟概念，並提供 VIF 與 tree-based feature set。
```

## Implementation Order

1. Create `notebooks/legacy/`.
2. Copy current notebooks into `notebooks/legacy/` with `_legacy` suffix.
3. Create `data/processed/legacy`, `data/processed/transfer`, `outputs/results/legacy`, `outputs/results/transfer`, `outputs/figures/legacy`, and `outputs/figures/transfer`.
4. Move current completed outputs into `legacy/`.
5. Update `src/features.py` with the redesigned column grouping functions.
6. Update `02_feature_selection.ipynb` to output both VIF and tree feature sets under `transfer/`.
7. Run `02_feature_selection.ipynb`.
8. Verify:
   - `selected_features_vif.csv`
   - `selected_features_tree.csv`
   - `us_modeling_ready_vif.csv`
   - `us_modeling_ready_tree.csv`
9. Update `03_model_us.ipynb` to read configurable `PROFILE` and `FEATURE_SET`.
10. Run tree-transfer `03_model_us.ipynb`.
11. Update and run `03b_model_explainability.ipynb` for tree-transfer model.
12. Create updated transfer summaries.
13. Update README after the new pipeline is verified.

## Non-goals for This Refactor

Do not solve Taiwan data collection in this refactor.

Do not implement MADA yet.

Do not force a replacement for `frost_days` in this pass. It can be addressed in a later Taiwan mapping step.

Do not delete legacy outputs until the transfer pipeline is verified and reviewed.
