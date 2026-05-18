# 04c Taiwan Income Data — 交接說明文件

**產出負責人**：收入資料收集階段  
**下一步負責人**：`04_transfer_taiwan.ipynb` 整合推論階段  
**資料基準年份**：民國 109 年（2020 年）  
**地理單位**：台灣 368 個鄉鎮市區（TOWNCODE）

---

## 1. 產出檔案位置

```
data/taiwan/income/
├── taiwan_income_features.csv                          ← 主要輸出，直接使用這個
├── 109年度綜稅綜合所得總額全國各縣市鄉鎮村里統計分析表.csv  ← 原始收入資料（村里層級）
├── 1.1.2低收入戶戶數及人數按鄉鎮市區別分(2015~)1150310.xlsx  ← 衛福部低收入戶原始資料
├── 109年平均每戶家庭收支 － 按區域別(縣市)分.xls           ← 縣市層級驗證用
├── 109年戶數五等分位組之所得分配比與所得差距.xls            ← 縣市 Gini 驗證用
├── 歷年可支配所得中位數 - 按區域別(縣市)分.xls             ← 歷年縣市時序（驗證用）
└── 歷年可支配所得平均數 - 按區域別(縣市)分.xls             ← 歷年縣市時序（驗證用）
```

---

## 2. 主要輸出：`taiwan_income_features.csv`

**Shape：368 行 × 8 欄**（每個鄉鎮市區一行，無缺值）

### 識別欄位

| 欄位 | 說明 | 範例 |
|---|---|---|
| `TOWNCODE` | 內政部鄉鎮市區代碼（唯一鍵） | `6300300` |
| `COUNTYNAME` | 縣市名稱 | `新竹縣` |
| `TOWNNAME` | 鄉鎮市區名稱 | `竹北市` |

### 特徵欄位（直接對應 DeepSolar 欄位名稱）

| 欄位名稱 | 單位 | 說明 | 全台均值 | 範圍 |
|---|---|---|---|---|
| `median_household_income` | 千元/年 | 鄉鎮加權中位數家戶所得 | 414.2 | 318.7 – 954.0 |
| `gini_index` | 無量綱（0–1） | 所得不平等指數（Lognormal Gini） | 0.486 | 0.352 – 0.678 |
| `poverty_family_below_poverty_level_rate` | 比例（0–1） | 低收入戶比例 | 0.025 | 0.004 – 0.100 |
| `poverty_family_below_poverty_level` | 戶 | 低收入戶絕對數量 | 398 | 1 – 4,002 |
| `tax_household_count` | 戶 | 報稅戶數（QA 用，不入模型） | 35,152 | 410 – 330,544 |

---

## 3. 如何載入資料

```python
import pandas as pd

taiwan_income = pd.read_csv(
    'data/taiwan/income/taiwan_income_features.csv',
    encoding='utf-8-sig'   # 中文縣市名稱需指定 encoding
)

print(taiwan_income.shape)      # (368, 8)
print(taiwan_income.columns.tolist())
taiwan_income.head()
```

---

## 4. 如何與其他台灣資料合併

`TOWNCODE` 是唯一鍵，與氣候、人口資料整合時以此欄位 join：

```python
taiwan_climate = pd.read_csv('data/taiwan/climate/taiwan_climate_annual.csv', encoding='utf-8-sig')
taiwan_pop     = pd.read_csv('data/taiwan/population/taiwan_population_features.csv')

taiwan_full = (
    taiwan_climate
    .merge(taiwan_income[['TOWNCODE', 'median_household_income', 'gini_index',
                           'poverty_family_below_poverty_level_rate',
                           'poverty_family_below_poverty_level']],
           on='TOWNCODE', how='left')
    .merge(taiwan_pop[['TOWNCODE', 'population_density', 'age_median',
                        'average_household_size']],
           on='TOWNCODE', how='left')
)
```

---

## 5. 如何對應 DeepSolar 模型輸入

本研究使用 `voted` 特徵集（90 個特徵），其中**收入類欄位共 4 個**，
對應 `outputs/results/transfer/selected_features_voted.csv` 排名如下：

| 欄位 | 在 voted 集的排名 | 與目標的 Pearson 相關係數 |
|---|---|---|
| `median_household_income` | 16 | +0.214 |
| `poverty_family_below_poverty_level_rate` | 28 | −0.121 |
| `gini_index` | 43 | −0.077 |
| `poverty_family_below_poverty_level` | 84 | −0.015 |

> 相關係數符號說明：  
> `median_household_income` 正相關 → 所得越高，太陽能安裝越多（符合直覺）  
> `poverty_family_below_poverty_level_rate` 負相關 → 貧窮率越高，安裝越少（符合直覺）  
> `gini_index` 負相關 → 不平等越高，安裝越少（符合直覺：財富集中但不代表廣泛普及）

---

## 6. 資料來源與計算方式

### 6a. `median_household_income`（中位數家戶所得）

**原始資料**：財政部「109年度綜稅綜合所得總額全國各縣市鄉鎮村里統計分析表」  
**計算流程**：

1. 讀取村里層級資料（共約 20,000+ 村里）
2. 從合併欄位「鄉鎮市區」（如「臺北市松山區」）解析出縣市名 + 鄉鎮名
3. 以**納稅戶數**為權重，加權平均各村里中位數，彙整至 368 個鄉鎮市區

```
加權中位數（鄉鎮） = Σ(村里中位數 × 村里納稅戶數) / Σ(村里納稅戶數)
```

> **注意**：資料為**稅前綜合所得**（非可支配所得）。DeepSolar 使用的是扣稅後可支配所得，
> 差異約 10–15%，但鄉鎮間相對排序不受影響，模型推論仍可用。

---

### 6b. `gini_index`（所得不平等係數）

**原始資料**：同上（綜稅村里資料的「變異係數」欄位）  
**計算方式**：採用計量經濟學標準的**對數常態近似公式**，從變異係數（CV）換算：

$$\text{Gini} = \text{erf}\!\left(\frac{\sigma_{\log}}{2}\right), \quad \sigma_{\log} = \sqrt{\ln\!\left(1 + \left(\frac{CV}{100}\right)^2\right)}$$

此公式在所得服從對數常態分布的假設下為精確解，與 DeepSolar 的 Gini 係數定義一致，輸出範圍 0–1。

**計算流程**：村里層級 CV → 村里 Gini → 以納稅戶數加權彙整至鄉鎮。

---

### 6c. `poverty_family_below_poverty_level_rate` / `poverty_family_below_poverty_level`

**原始資料**：衛福部「低收入戶戶數及人數按鄉鎮市區別分（2015~）」  
**使用 Sheet**：`2020`（CE 年，對應民國 109 年）  
**使用欄位**：Q4（年底）低收入戶**戶數**（欄位 56）

**計算流程**：
1. 解析 Excel 階層結構（縣市 → 鄉鎮），追蹤各鄉鎮所屬縣市
2. 修正兩個行政升格造成的名稱異動：
   - `頭份鎮`（苗栗縣）→ `頭份市`（2016 年升格）
   - `員林鎮`（彰化縣）→ `員林市`（2015 年升格）
3. 計算比例：低收入戶戶數 ÷ 鄉鎮總戶數（分母取自人口普查資料）

---

## 7. 收集程式

完整流程記錄於：`notebooks/04c_taiwan_income_features.ipynb`

若需更新資料：
| 資料 | 更新方式 |
|---|---|
| 收入資料 | 至財政部下載新年度綜稅村里資料，替換 CSV，修改 notebook 中的 `INCOME_CSV` 路徑 |
| 低收入戶 | 至衛福部下載最新版 Excel，替換檔案，修改 notebook 中的 `POVERTY_XLS` 路徑 |
| 基準年份 | 建議維持與人口資料同年（目前：109年/2020年） |

---

## 8. 驗證結果摘要

| 驗證項目 | 結果 |
|---|---|
| 覆蓋率 | 368 / 368 鄉鎮（100%） |
| 缺值 | 0（無任何缺值） |
| 所得最高鄉鎮 | 新竹縣竹北市（954.0 千元/年）✓ 新竹科學園區周邊，符合預期 |
| 所得最低鄉鎮 | 新北市三芝區（318.7 千元/年）✓ 偏遠農村，符合預期 |
| Gini 最高鄉鎮 | 新北市五股區（0.678）✓ 工業區與住宅混雜，貧富差距大 |
| Gini 最低鄉鎮 | 連江縣東引鄉（0.352）✓ 小型島嶼社區，所得結構均質 |
| 貧窮率最高鄉鎮 | 宜蘭縣南澳鄉（10.0%）✓ 原住民鄉，符合預期 |
| 貧窮率最低鄉鎮 | 新竹縣竹北市（0.43%）✓ 高所得科技區，符合預期 |
| 縣市排序方向 | 新竹市 > 新竹縣 > 臺北市 >> 農業縣 ✓ 符合主計總處公布資料 |
