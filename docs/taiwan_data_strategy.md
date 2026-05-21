# 台灣資料收集與 04_transfer 實作方向

> 為小組討論準備｜2026-05-17

---

## 目標回顧

台灣遷移階段的核心工作：
1. **收集台灣對應資料**：社經、氣象、政策、電力等 11 類特徵
2. **特徵對齐**：台灣資料欄位 → 美國模型欄位名稱
3. **套用模型**：用 `best_us_twostage_model_voted_tuned.pkl` 推論各地區潛力
4. **產出結果**：`taiwan_predictions_voted.csv`（各地區潛力分數）

**不做什麼**：不預測台灣真實安裝密度，只預測**相對潛力排名**。

---

## 台灣資料需求表

### 核心映射：11 類特徵 → 台灣資料來源

| 美國特徵類 | 例示美國欄位 | 台灣對應方案 | 來源 | 難度 |
|-----------|----------|----------|------|------|
| **氣候** | `cooling_degree_days`, `heating_degree_days` | 日照輻射量(年均 kWh/m²/day) | NASA POWER API / 中央氣象署 | 🟢 中等 |
| **電力** | `avg_electricity_retail_rate` | 台電住宅電價(元/度) | 台電官網電價查詢 | 🟢 容易 |
| **收入** | `median_household_income` | 家戶可支配所得 | 主計總處家庭收支調查 | 🟡 中等 |
| **人口** | `population_density`, `population` | 人口密度(人/km²) | 內政部戶政司 | 🟢 容易 |
| **住宅** | `median_home_value`, `median_household_size` | 房屋持有率, 家戶人數 | 內政部戶政司, 主計總處 | 🟡 中等 |
| **社經** | `poverty_rate`, `gini_coefficient`, `unemployment_rate` | 低收入戶比例, 失業率, 教育程度 | 社會局, 勞動部, 教育部 | 🟡 中等 |
| **政策** | `incentive_*`, `rebate`, `net_metering` | 補助金額, FIT 費率, 自用率比例 | 能源局太陽光電政策 | 🟡 中等 |

---

## 方向 A：保守方案（推薦新手組隊）⭐

### 範圍
- **資料來源**：3-4 個最容易取得的來源
- **特徵數**：~40-50 個（對應投票特徵集 90 個中的核心 50%）
- **地理尺度**：縣市級（22 縣市）或鄉鎮級（368 個）
- **時間投入**：1-2 週
- **風險**：特徵不完整，預測力可能下降 5-10%

### 具體步驟

#### 第 1 步：收集最容易的 3 類資料

**1.1 日照輻射量**（1-2 天）
```python
# 方案 A1a：NASA POWER API（推薦）
# https://power.larc.nasa.gov/api/
# 免費無需認證，按縣市中心座標查年均日照

# 方案 A1b：中央氣象署公開資料
# 下載各測站年均日照（較細膩，但手工整理工作多）

# 輸出格式：
# county  | lat    | lon     | solar_irradiance_kwh_m2_day
# 台北市  | 25.033 | 121.564 | 3.8
# 新北市  | 25.017 | 121.466 | 3.9
```

**1.2 台電電價**（1 天）
```python
# 台電官網：https://www.taipower.com.tw/
# 查詢住宅用電價格（按時間區段）

# 簡化策略：取年均邊際電價（夏季高峰加權平均）
# 輸出格式：
# county  | electricity_retail_rate_twd_per_kwh
# 台北市  | 3.65
# 新北市  | 3.65
```

**1.3 人口數與密度**（1 天）
```python
# 內政部戶政司 MyData 平台
# https://data.moi.gov.tw/

# 下載 2025 年各縣市人口統計
# 計算人口密度 = 人口 / 面積

# 輸出格式：
# county  | population | area_km2 | population_density
# 台北市  | 2580123    | 272      | 9485
```

#### 第 2 步：簡化特徵對齐

```python
# src/features.py 新增函式

def load_taiwan_data_minimal():
    """載入三個基本資料源"""
    solar_irr = pd.read_csv('data/taiwan/solar_irradiance.csv')
    elec_price = pd.read_csv('data/taiwan/electricity_price.csv')
    pop_density = pd.read_csv('data/taiwan/population_density.csv')
    
    tw_df = (solar_irr
        .merge(elec_price, on='county')
        .merge(pop_density, on='county'))
    return tw_df

def align_taiwan_features_minimal(tw_df, us_feature_names):
    """簡化版特徵對齐：只對應有的欄位"""
    mapping = {
        'solar_irradiance_kwh_m2_day': ['heating_degree_days', 'cooling_degree_days'],
        # 用日照輻射量當作氣候 proxy
        
        'electricity_retail_rate_twd_per_kwh': ['avg_electricity_retail_rate'],
        
        'population_density': ['population_density'],
        'population': ['population'],
    }
    
    aligned = pd.DataFrame(index=tw_df.index)
    for tw_col, us_cols in mapping.items():
        if tw_col in tw_df.columns:
            for us_col in us_cols:
                if us_col in us_feature_names:
                    aligned[us_col] = tw_df[tw_col].values
    
    # 缺失特徵用全國平均填補
    for col in us_feature_names:
        if col not in aligned.columns:
            aligned[col] = np.nan
    
    return aligned
```

#### 第 3 步：輕量級 04_transfer notebook

```python
# notebooks/04_transfer_taiwan.ipynb

import pickle
from src.features import load_taiwan_data_minimal, align_taiwan_features_minimal

# 載入台灣資料
tw_df = load_taiwan_data_minimal()  # 22 縣市 or 368 鄉鎮

# 載入美國訓練模型
with open('../outputs/results/transfer/best_us_twostage_model_voted_tuned.pkl', 'rb') as f:
    bundle = pickle.load(f)
    classifier = bundle['classifier']
    regressor = bundle['regressor']
    feature_names = bundle['feature_names']

# 特徵對齙
tw_features = align_taiwan_features_minimal(tw_df, feature_names)

# 填補缺失值（用全國中位數）
imputer = SimpleImputer(strategy='median')
tw_features_imputed = imputer.fit_transform(tw_features)

# Stage 1：預測「是否有太陽能」
stage1_prob = classifier.predict_proba(tw_features_imputed)[:, 1]

# Stage 2：預測密度
stage2_log_density = regressor.predict(tw_features_imputed)
stage2_density = np.expm1(stage2_log_density)

# 產出結果
results = pd.DataFrame({
    'county': tw_df['county'],
    'stage1_prob_has_solar': stage1_prob,
    'stage2_log_density': stage2_log_density,
    'stage2_density_per_1000hh': stage2_density,
    'combined_score': stage1_prob * stage2_density,  # 綜合潛力分
})

results.to_csv('../outputs/results/transfer/taiwan_predictions_minimal.csv', index=False)
```

**產出**：`taiwan_predictions_minimal.csv`（22×4 or 368×4）

---

## 方向 B：進階方案（預期重點實施）⭐⭐

### 範圍
- **資料來源**：6-7 個正式來源
- **特徵數**：~70-80 個（接近投票特徵集 90 個的 80%）
- **地理尺度**：鄉鎮級（368 個鄉鎮）
- **時間投入**：2-3 週
- **風險**：部分政策類特徵可能有時間滯後，但整體可行性高

### 新增資料源（vs. A 方案）

**B1. 家戶收入與社經指標**（3-5 天）
```
來源：主計總處家庭收支調查
https://www.dgbas.gov.tw/

內容：縣市層級的可支配所得、各所得分位數、貧窮線比例
輸出：median_household_income、gini_coefficient（自算）
```

**B2. 住宅特徵**（2-3 天）
```
來源：內政部戶政司 + 不動產資訊平台
https://www.moica.gov.tw/

內容：房屋數、獨棟住宅比例、公寓比例、年均房價
計算：median_home_value（各鄉鎮市場均價）、 median_household_size
```

**B3. 教育程度與就業**（2-3 天）
```
來源：教育部統計、勞動部統計
內容：各鄉鎮小學以上人口教育程度分布、失業率

計算欄位：
- avg_education_years（加權平均年數）
- unemployment_rate_pct
- labor_participation_rate
```

**B4. 政策補助與 FIT**（3-5 天，涉及手工搜尋）
```
來源：能源局太陽光電政策資訊、台電 FIT 價表
內容：2023-2025 年各縣市補助申請數 / 核准數、現行 FIT 費率

計算：
- incentive_count_residential（近三年平均）
- feedin_tariff_twd_per_kwh（最新 FIT 費率）
- rebate_twd_per_kwatt（如有）
```

### 進階 align_taiwan_features() 實作

```python
def align_taiwan_features_advanced(tw_df, us_feature_names):
    """進階特徵對齙，支援邏輯計算與 proxy 變數"""
    
    mapping = {
        # 氣候變數：用日照輻射量與月均溫計算
        'heating_degree_days': lambda: compute_hdd(tw_df['avg_temp_min'], base=18.3),
        'cooling_degree_days': lambda: compute_cdd(tw_df['avg_temp_max'], base=23.3),
        
        # 電力
        'avg_electricity_retail_rate': 'electricity_retail_rate_twd_per_kwh',
        
        # 收入與社經
        'median_household_income': 'median_household_income_twd',
        'gini_coefficient': 'gini_index',
        'poverty_rate': 'poverty_rate_pct',
        
        # 人口與住宅
        'population': 'population',
        'population_density': 'population_density_per_km2',
        'median_home_value': 'median_home_price_twd',
        'median_household_size': 'avg_household_size',
        
        # 教育與就業
        'education_rate': 'tertiary_education_rate_pct',
        'unemployment_rate': 'unemployment_rate_pct',
        
        # 政策
        'incentive_count_residential': 'subsidy_count_recent_3yr',
        'feedin_tariff': 'fit_rate_twd_per_kwh_latest',
        'net_metering': 'net_metering_allowed',  # 0/1 boolean
    }
    
    aligned = pd.DataFrame(index=tw_df.index)
    for us_col, tw_col_or_fn in mapping.items():
        if callable(tw_col_or_fn):
            # 計算型特徵
            aligned[us_col] = tw_col_or_fn()
        elif tw_col_or_fn in tw_df.columns:
            # 直接映射
            aligned[us_col] = tw_df[tw_col_or_fn].values
        else:
            aligned[us_col] = np.nan
    
    # 缺失欄位補 NaN（稍後由 imputer 統一處理）
    for col in us_feature_names:
        if col not in aligned.columns:
            aligned[col] = np.nan
    
    return aligned
```

---

## 方向 C：野心方案（超出課程範圍）🚀

### 範圍
- **資料來源**：全部 11 類，包含網路爬蟲
- **地理尺度**：里級（7,000+ 里）或村級
- **特徵數**：90+ 個（完整投票特徵集）
- **時間投入**：4-6 週
- **額外工作**：地理空間分析、時序補值

### 額外工作項目

**C1. 網路爬蟲**（10-15 天）
```
- 能源局補助申請系統爬蟲（縣市×月份 補助金額）
- 台電發電統計按區域別爬蟲（縣市級太陽光電併網數）
- 房仲網路房價指數（各鄉鎮）
```

**C2. 地理空間內插**（5-10 天）
```
- 氣象測站不足的鄉鎮用 Kriging 內插日照輻射量
- 用 shapefile 進行 census tract 與台灣鄉鎮的邊界對齙
- 計算美國 county 層級的台灣等效「行政區」
```

**C3. 時序補值**（3-5 天）
```
- 部分補助政策資料只有 2024-2025
- 用 forward fill 或平均法回溯至 2022
```

**成果**：與美國 voted_features（90 個）幾乎一對一的台灣特徵集

---

## 小組討論建議

### 決策表：選擇哪個方案？

| 選擇 | 適合組隊 | 優點 | 缺點 | 建議時機 |
|------|--------|------|------|---------|
| **方案 A**（保守） | 初心者、時間緊 | 快速驗證流程、風險低 | 特徵損失 5-10%、預測力有限 | **先做 A，作為 POC** |
| **方案 B**（進階） | 標準課程隊伍 | 特徵完整性 80%、可用性高 | 需多人協作、3 週工期 | **本次主線目標** |
| **方案 C**（野心） | 資料科學社團、未來研究 | 完整度 100%、發表亮點 | 超出課程時間、爬蟲維護成本 | **課後延伸** |

### 小組分工建議（方案 B 為例）

```
隊伍配置：5 人，2.5-3 週工期

週 1：資料收集
  ├─ 組員 A + B：日照 + 電價 + 人口（3 天內完成）
  ├─ 組員 C：家戶收入 + 社經（3-4 天）
  └─ 組員 D + E：住宅 + 教育 + 政策（4-5 天）

週 2：特徵對齙 + 資料驗證
  ├─ 組員 A：主導 align_taiwan_features() 進階版（2-3 天）
  ├─ 組員 B：資料品質檢查（缺失值、異常值）（2 天）
  └─ 全員：統一欄位單位、檢查範圍合理性（1 天）

週 3：模型套用 + 結果
  ├─ 組員 C：04_transfer_taiwan.ipynb 實作（2 天）
  ├─ 組員 D：結果視覺化（地圖、排序表）（1 天）
  └─ 全員：驗證 + 報告撰寫（1-2 天）
```

### 風險與解決策略

| 風險 | 徵兆 | 解決策略 |
|------|------|---------|
| **資料找不到** | 某個來源官網改版 / 下線 | 提前 2 週開始探路；準備 3 個 backup 來源 |
| **欄位不匹配** | 美國用「州級政策」，台灣「中央+地方混雜」 | 用 mapping 邏輯轉換；無法轉換時用全國平均 |
| **缺失值過多** | >30% 欄位缺失 | 改用方案 A；或依 tree model 容忍度決定 drop/impute |
| **預測效果差** | Spearman ρ < 0.5 vs 台電實際資料 | 檢查特徵分布是否與美國差異大；可補充在地化調整 |
| **時間不足** | 第 2 週末仍未完成資料蒐集 | 立即降級到方案 A；最少保留 3-4 核心特徵 |

---

## 技術細節補充

### 缺失值填補策略（04_transfer 中）

```python
# 方案 1：簡單填補（保守方案 A 用）
imputer = SimpleImputer(strategy='median')  # 全國中位數

# 方案 2：分層填補（進階方案 B 用）
# 先按都會/非都會分層，再計算該層中位數
urban_mask = tw_df['population_density'] > 1500
for col in feature_cols:
    tw_df.loc[urban_mask & tw_df[col].isna(), col] = \
        tw_df[urban_mask][col].median()
    tw_df.loc[~urban_mask & tw_df[col].isna(), col] = \
        tw_df[~urban_mask][col].median()
```

### 模型套用邏輯

```python
# 四種潛力指標，組員可選擇上報

# 1. Stage 1 概率：地區「有太陽能」的可能性
potential_stage1 = classifier.predict_proba(X_tw)[:, 1]

# 2. Stage 2 密度：若有太陽能，部署強度
potential_stage2_density = np.expm1(regressor.predict(X_tw))

# 3. 綜合分（乘積）：考慮有無 × 強度
potential_combined = potential_stage1 * potential_stage2_density

# 4. 綜合分（加權）：可調整權重
potential_weighted = 0.4 * potential_stage1 + 0.6 * potential_stage2_density

# 排名
ranking = pd.DataFrame({
    'rank': potential_combined.argsort()[::-1] + 1,
    'county_or_township': tw_df['area_name'],
    'potential_score': potential_combined,
})
```

---

## 重要提醒

1. **不預測絕對裝設密度**：台灣的 `stage2_density` 輸出只是**相對比較分數**，不是真實 kWh/戶。真實台灣裝設密度會受到：
   - 建築法規（屋頂面積限制）
   - 電網併網容量（地方限電）
   - 補助額度差異（大幅影響投資意願）
   - 在地施工成本（山區更貴）
   
   這些在美國模型中已經隱含了，但台灣特殊性無法完全轉移。

2. **驗證方式**：與台電 2023-2024 年縣市級併網數的 **Spearman rank correlation** 做對標：
   ```python
   from scipy.stats import spearmanr
   rho, p_val = spearmanr(predicted_rank, actual_tw_installation_rank)
   ```
   目標：ρ > 0.6（表示排名相關性合理）

3. **台灣資料授權**：確認使用授權：
   - 政府開放資料（多數免費可商用）
   - NASA POWER API（免費）
   - 台電、能源局資料（確認是否公開版本）

---

## 建議起手式

**第一次小組會決定事項**：
1. ☐ 選擇方案（A / B / 混合）
2. ☐ 分配第 1 個月的資料蒐集責任
3. ☐ 確認台灣資料儲存位置 & 命名規範（e.g., `data/taiwan/solar_irradiance_county_2025.csv`）
4. ☐ 設定 1-2 個里程碑（e.g., 第 2 週末前資料蒐集完 80%）
5. ☐ 誰負責 `align_taiwan_features()` 實作（建議由資料最熟的人）
6. ☐ 誰負責 `04_transfer_taiwan.ipynb` 主體程式碼

---

*祝小組順利！有問題歡迎回頭討論 ✨*
