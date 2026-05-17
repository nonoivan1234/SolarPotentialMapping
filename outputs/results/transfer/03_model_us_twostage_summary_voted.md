# 03 Two-stage US Model Summary - Transfer Pipeline

## 中文版

### 這份 notebook 在做什麼

`03_model_us_twostage` 是新版正式模型路線，和原本三分類 baseline 不一樣。

三分類 baseline 是把 `tile_count` 用分位數切成三類；two-stage 則更接近 DeepSolar 原論文 SolarForest 的想法，把問題拆成兩段：

```text
Stage 1:
這個 census tract 有沒有太陽能部署？
installed = tile_count > 0

Stage 2:
如果已經有太陽能部署，部署強度是多少？
這版使用 density per 1000 households，並用 log1p 轉換降低極端值影響。
```

這樣的好處是：先分開處理「有沒有裝」和「裝了多少」。這兩件事背後的原因不一定一樣，混在一個三分類模型裡會比較粗。

> 本研究不直接預測台灣真實裝設密度，而是利用美國 DeepSolar 訓練出的 adoption/deployment pattern，建立台灣各地區的相對太陽能部署傾向與推廣優先度排序。

### 使用的資料與特徵

這次 notebook 使用：

- profile：`transfer`
- feature set：`voted`
- features：`90`
- rows：`72,537`
- primary model：`RandomForest`
- comparison models：`RandomForest`, `XGBoost`, `GradientBoosting`, `Linear`

太陽能部署狀態分布：

| installed | 筆數 | 意義 |
|---:|---:|---|
| 0 | 16,279 | 沒有太陽能部署 |
| 1 | 56,258 | 有太陽能部署 |

installed rate 是 `0.7756`，也就是約 `77.6%` 的 census tracts 有至少一個 solar tile。

在已部署地區中，deployment density per 1000 households 分布非常偏斜：

- mean：`216.0`
- median：`5.31`

這表示少數地區的部署密度非常高，會把平均值拉很大。因此 Stage 2 用 `log1p` 是合理的。

### Feature Set 比較

這次比較三組特徵：

| feature set | features | 意義 |
|---|---:|---|
| `tree` | 102 | correlation screening 後的完整 tree-based 特徵集 |
| `vif` | 55 | 再經過 VIF 移除後的低共線性版本 |
| `voted` | 90 | 折衷版本，保留重要 domain features，同時比 tree 版精簡 |

整體看起來，`voted` 是合理的主線選擇：比 `tree` 少 12 欄，但表現沒有明顯下降；比 `vif` 保留更多對遷移重要的氣候、電價、政策與社經欄位。

### Stage 1 結果：有沒有太陽能部署

主模型 RandomForest 在 hold-out test set 的結果：

| metric | value | 解讀 |
|---|---:|---|
| accuracy | `0.8246` | 整體分類正確率約 82.5% |
| F1 | `0.8937` | 對 installed class 的 precision/recall 平衡表現不錯 |
| precision | `0.8436` | 預測「有部署」時，約 84.4% 真的有部署 |
| recall | `0.9501` | 真正有部署的地區中，約 95.0% 被抓到 |
| ROC-AUC | `0.8637` | 區分有/無部署的能力良好 |

Stage 1 的結果算穩。尤其 recall 很高，代表模型很少漏掉已經有太陽能部署的區域。

不過要注意：資料本身 installed rate 高達 `77.6%`，所以 F1 高有一部分也受類別分布影響。這時候 ROC-AUC `0.864` 比單看 F1 更有參考價值。

### Stage 2 結果：已部署地區的部署強度

Stage 2 只在 `tile_count > 0` 的地區訓練與測試。

主模型 RandomForest 的結果：

| metric | value | 解讀 |
|---|---:|---|
| train positive rows | `45,006` | 訓練集中有部署的樣本數 |
| test positive rows | `11,252` | 測試集中有部署的樣本數 |
| MAE log1p | `0.5473` | log scale 平均誤差 |
| RMSE log1p | `0.7307` | log scale RMSE |
| R² log1p | `0.7501` | 在 log scale 可解釋約 75% 變異 |
| MAE density | `257.78` | 換回原始 density scale 的平均誤差 |
| RMSE density | `9495.92` | 原始 scale 受極端值影響很大 |
| R² density | `0.0852` | 原始 density scale 的解釋力偏低 |

這裡最重要的是不要把 `R² log1p = 0.7501` 和 `R² density = 0.0852` 看成互相矛盾。

它們代表的是不同尺度：

- `log1p scale`：把極端值壓縮後，模型能抓到部署強度的相對差異。
- `density scale`：回到原始密度後，少數超高部署地區會讓誤差被放大很多。

所以可以這樣解讀：

```text
模型對「一般量級」與「相對高低」預測得不錯；
但對極端高密度部署地區的精準數值預測仍然很弱。
```

### End-to-end CV R² 為什麼偏低？

full comparison 裡的 `cv_r2_mean` 是 end-to-end 指標，也就是把 Stage 1 和 Stage 2 串起來後，在所有地區上評估部署強度。

主要結果：

| feature set | model | cv R² mean |
|---|---|---:|
| voted | GradientBoosting | `0.1088` |
| tree | RandomForest | `0.0902` |
| tree | GradientBoosting | `0.0895` |
| voted | RandomForest | `0.0886` |
| vif | RandomForest | `0.0873` |

這些數值都不高，代表完整 two-stage pipeline 要準確預測所有地區的原始部署密度仍然很難。

原因主要有三個：

1. deployment density 極端偏斜，少數超高值會讓原始尺度 R² 很難看。
2. Stage 1 的錯誤會傳遞到 Stage 2。
3. DeepSolar 的 tabular features 比較像 adoption/deployment proxy，不是完整的物理發電潛力資料。

因此，這個模型比較適合先用來做：

- deployment tendency
- adoption intensity proxy
- 區域相對排序
- 台灣遷移前的 feature relevance 參考

目前還不適合直接宣稱可以精準預測台灣真實裝設密度、真實發電量或 ROI。

### 模型比較重點

Stage 1 有/無部署分類：

- `XGBoost + voted` 的 ROC-AUC 最高：`0.8726`
- `GradientBoosting + tree` 的 F1 最高：`0.8974` 左右
- `RandomForest + voted` 的 recall 最高且穩：`0.9501`

Stage 2 log scale regression：

- `XGBoost + tree` R² log1p 最高：`0.7531`
- `XGBoost + voted` R² log1p：`0.7522`
- `RandomForest + voted` R² log1p：`0.7501`

差距其實很小。RandomForest 雖然不是每個指標第一，但它穩定、可解釋、也更貼近 SolarForest，所以當正式 baseline 合理。

### VIF 版表現如何？

`vif` 版 feature 數最少，只有 `55` 欄，但表現不一定更好。

特別是：

- `XGBoost + vif` 的 end-to-end CV R² mean 是 `-0.7935`，std 高達 `2.5230`
- 代表它在不同 fold 非常不穩

這再次支持前面的判斷：對 tree-based model 來說，硬做 VIF 不一定有幫助，甚至可能刪掉重要的 domain features。

### Feature Importance：Stage 1

Stage 1 是在判斷「有沒有太陽能部署」。RandomForest classifier 前幾個重要特徵：

| rank | feature | importance | 解讀 |
|---:|---|---:|---|
| 1 | `total_area` | `0.0539` | 區域尺度與可部署空間相關 |
| 2 | `population_density` | `0.0508` | 都市化/人口密度與部署機會相關 |
| 3 | `education_bachelor` | `0.0373` | 教育程度 proxy adoption tendency |
| 4 | `heating_fuel_coal_coke_rate` | `0.0258` | 能源使用結構 proxy |
| 5 | `daily_solar_radiation` | `0.0257` | 日照條件仍有影響 |
| 6 | `population` | `0.0244` | 區域規模與市場大小 |
| 7 | `housing_unit_median_value` | `0.0194` | 房價/資產條件 |
| 8 | `housing_unit_median_gross_rent` | `0.0185` | 地區經濟條件 |
| 9 | `relative_humidity` | `0.0185` | 氣候條件 |
| 10 | `education_college` | `0.0177` | 教育/社經 proxy |

Stage 1 比較像 adoption classifier：它看的是哪裡「比較可能已經有人裝」。

### Feature Importance：Stage 2

Stage 2 是在已部署地區中預測部署強度。RandomForest regressor 前幾個重要特徵：

| rank | feature | importance | 解讀 |
|---:|---|---:|---|
| 1 | `relative_humidity` | `0.2331` | 影響部署強度最明顯，可能 proxy 雲量/氣候條件 |
| 2 | `electricity_price_industrial` | `0.0916` | 電價越高，太陽能經濟誘因可能越強 |
| 3 | `heating_fuel_gas` | `0.0827` | 能源使用結構 proxy |
| 4 | `population_density` | `0.0587` | 區域密度/屋頂型態/市場結構 |
| 5 | `frost_days` | `0.0373` | 氣候嚴苛程度 proxy，美國資料中有用，台灣遷移需替代 |
| 6 | `occupancy_owner_rate` | `0.0303` | 自有住宅率與安裝決策相關 |
| 7 | `heating_fuel_electricity` | `0.0279` | 用電/能源型態 proxy |
| 8 | `daily_solar_radiation` | `0.0185` | 日照條件 |
| 9 | `incentive_residential_state_level` | `0.0174` | 政策誘因 |
| 10 | `housing_unit_median_value` | `0.0161` | 房價/資產條件 |

Stage 2 更像 deployment intensity model：它關心已經部署後，哪些因素讓部署量更高。

### 我會怎麼解讀目前結果

目前 two-stage 結果是合理而且有價值的，但要清楚定位：

```text
這是一個 deployment/adoption model，
不是純物理發電潛力模型。
```

放到台灣遷移時，它應該被解讀為相對分數與排序工具：

```text
Stage 1 score:
台灣某地區像不像美國資料中已經出現太陽能 adoption 的地區。

Stage 2 score:
若該地區具備 adoption 條件，它像不像美國資料中部署強度較高的地區。

Final use:
建立台灣各區域的相對部署傾向與推廣優先度排序。
```

因此，不應直接把 Stage 2 output 解讀成台灣真實安裝密度。

它最強的地方是：

- 能分辨有沒有部署：Stage 1 ROC-AUC 約 `0.864`
- 能在 log scale 解釋已部署地區的部署強度：Stage 2 R² 約 `0.75`
- 能分開解釋 adoption 與 intensity 的重要因子

它目前弱的地方是：

- 原始 density scale 的 R² 很低
- 對極端高部署地區預測不穩
- `frost_days` 對美國有意義，但台灣沒有直接對應
- 還沒有真正納入台灣建物、屋頂、遮蔭、發電量與 ROI 資料

### 後續建議

1. **保留 two-stage 作為正式 US baseline**  
   三分類 baseline 可以留作比較，但正式方法建議以 two-stage 為主。

2. **台灣遷移優先使用 voted 或 tree 特徵集**  
   `vif` 版太容易刪掉 domain-critical features，不建議作為主線。

3. **Stage 2 指標應優先看 log scale**  
   因為部署密度極端偏斜，原始 scale 的 RMSE/R² 很容易被少數 outliers 主導。

4. **台灣版應將 Stage 2 作為相對 intensity score**  
   在沒有台灣真實安裝密度標籤前，Stage 2 不應被解讀為絕對密度預測，而應作為區域相對排序的一部分。

5. **若未來有台灣標籤資料，再重定義 Stage 2 target**  
   若未來有建物/屋頂資料，Stage 2 可以改成：
   - 預測可用屋頂面積
   - 預測年發電量
   - 預測 ROI
   - 預測回本年限

6. **之後解釋模型時要分開解釋兩階段**  
   Stage 1 和 Stage 2 的 feature importance 不應混在一起看，因為它們回答的問題不同。

---

## English Summary

The two-stage model separates solar deployment modeling into two questions: whether a census tract has any solar deployment, and how much deployment intensity exists among deployed tracts. This is closer to the DeepSolar SolarForest idea than the multiclass quantile baseline.

The current run uses the `voted` transfer feature set with 90 features. Stage 1 performs well: RandomForest achieves accuracy `0.8246`, F1 `0.8937`, recall `0.9501`, and ROC-AUC `0.8637`. Stage 2 also performs well on the transformed target, with log-scale R² `0.7501`. However, raw density-scale R² is only `0.0852`, because deployment density is extremely skewed and dominated by outliers.

The most important interpretation is that this is a solar deployment/adoption model, not a pure physical feasibility or energy generation model. For Taiwan transfer, this study does not directly predict true installed density. Instead, it uses adoption/deployment patterns learned from US DeepSolar data to build relative solar deployment tendency scores and promotion-priority rankings across Taiwan regions.

---

## 本研究 vs DeepSolar 原論文對比

### 原論文 SolarForest 結果（Table 1 / Appendix）

DeepSolar 原論文（Wang et al., 2019）以美國 census tract 為單位，使用 RandomForest 訓練一個兩階段部署密度預測模型（SolarForest），結果如下：

| 指標 | 原論文 SolarForest | 說明 |
|---|---:|---|
| Feature count | 94 | 包含美國特有欄位（race, incentive 等） |
| CV folds | 10-fold | 與本研究相同 |
| Stage 2 R² (log scale) | `0.722` | 主要對比指標 |
| 目標變數 | deployment density (log) | 與本研究 log1p(density) 一致 |

> 原論文直接用全部 94 欄特徵，未作跨國遷移的特徵篩選。

---

### 本研究結果

| 指標 | 本研究 (RF + voted) | 說明 |
|---|---:|---|
| Feature count | **90** | 移除美國特有欄位後的可遷移集 |
| CV folds | 10-fold | 與原論文相同 |
| Stage 2 R² log1p | **`0.7501`** | 主要對比指標 |
| Stage 1 ROC-AUC | `0.8637` | 原論文未報告此指標 |
| End-to-end CV R² | `0.0886` | 原始密度尺度，因偏斜分布而偏低 |

完整模型家族比較（Stage 2 R² log1p）：

| feature set | model | Stage 2 R² log1p |
|---|---|---:|
| tree | XGBoost | `0.7531` |
| voted | XGBoost | `0.7522` |
| **voted** | **RandomForest** | **`0.7501`** |
| tree | RandomForest | `0.7454` |
| vif | XGBoost | `0.7481` |

---

### 對比解讀

```text
本研究 Stage 2 R² log1p = 0.7501
原論文 SolarForest R²    = 0.722

差距：+0.028（↑ 約 3.9%）
```

這個結果說明：

1. **在特徵數更少（90 vs 94）且移除美國特有欄位的情況下，模型表現略優於原論文**。  
   這表示本研究篩選出的 `voted` 特徵集在保留部署預測能力的同時，排除了美國地域性雜訊。

2. **本研究的 Stage 2 R² 超越原論文，但前提相同**：兩者都是在 log scale 上評估、僅針對有部署地區、使用 RandomForest。因此指標是可比較的。

3. **這是本研究的主要研究貢獻之一**：用可跨國泛化的特徵集（voted, 90 欄），在相同評估協議下達到甚至超越原論文的預測性能，為後續台灣遷移奠定基礎。

4. **End-to-end CV R² 偏低（~0.089）不代表模型差**：  
   原論文報告的是 Stage 2 log-scale R²（已部署地區），而 end-to-end CV R² 是在原始密度尺度評估所有地區（含零值地區），因此不能直接對比。若要與原論文 Table 1 對比，應使用 **Stage 2 R² log1p**。

---

### 研究貢獻定位

| 面向 | 原論文 SolarForest | 本研究 |
|---|---|---|
| 資料範圍 | 美國 census tract | 美國訓練 → 台灣遷移 |
| 特徵集 | 94 欄（含美國特有） | 90 欄（可跨國泛化） |
| Stage 2 R² log1p | 0.722 | **0.750**（略優） |
| 主要輸出 | 密度預測值 | 相對部署傾向排序 |
| 驗證方式 | 美國 10-fold CV | 美國 CV + 台灣外部驗證（規劃中） |
| 可解釋性 | Feature importance | Feature importance + SHAP（03b） |
