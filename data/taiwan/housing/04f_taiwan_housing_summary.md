# 04f Taiwan Housing Data — 交接說明文件

**產出負責人**：住宅／家戶特徵（Housing）資料工程階段  
**下一步負責人**：`04_transfer_taiwan.ipynb` 整合推論階段  
**資料時間範圍**：家戶／房價為民國 98–114 年歷年平均；`occupancy_owner_rate` 為民國 **109** 年（表34 快照）  
**地理單位**：台灣 **368** 個鄉鎮市區（`TOWNCODE`）；`housing_unit_median_value`（縣市）為同縣市各鄉鎮相同；其餘多為**鄉鎮**層級

---

## 1. 產出檔案位置

```
data/taiwan/housing/
├── taiwan_housing_features.csv                              ← 主要輸出，直接使用這個
├── 04f_taiwan_housing_summary.md                          ← 本說明文件
├── 人口家戶_家戶結構_設有戶籍住宅之平均人口數_*.csv         ← 家戶人口（原始）
├── 住宅價格_住宅買賣_買賣契約價格平均單價_*.csv             ← 房價（原始）
└── 縣市鄉鎮住宅所有權屬資料/*.xlsx                        ← 表34 所有權屬（22 縣市）
```

---

## 2. 主要輸出：`taiwan_housing_features.csv`

**Shape：368 行 × 12 欄**（3 識別欄 + **1** 家戶 + **1** 縣市房價 + **1** 自有率 + **6** 分類型房價）

### 識別欄位

| 欄位 | 說明 | 範例 |
|---|---|---|
| `TOWNCODE` | 內政部鄉鎮市區代碼（唯一鍵） | `6300300` |
| `COUNTYNAME` | 縣市名稱 | `新竹縣` |
| `TOWNNAME` | 鄉鎮市區名稱 | `竹北市` |

### 特徵欄位

| 欄位 | 台灣來源 | 單位 | 聚合層級 | 全台均值 | 全台範圍（有值鄉鎮） |
|---|---|---|---|---|---|
| `average_household_size` | 設有戶籍住宅之平均人口數 | 人/戶 | 鄉鎮 | 3.92 | 2.52 – 8.77 |
| `housing_unit_median_value` | 買賣契約價格平均單價（**全建物類型合計**） | 萬元/坪 | **縣市**（同縣市鄉鎮相同） | 18.36 | 11.54 – 60.52 |
| `occupancy_owner_rate` | 表34：自有戶數 / 總戶數 | 比率 (0–1) | **鄉鎮**（109年） | 0.81 | 0.43 – 0.97 |
| `housing_unit_median_value_公寓` | 買賣契約價格平均單價 · 公寓 | 萬元/坪 | 鄉鎮 | 20.43 | 6.58 – 76.52 |
| `housing_unit_median_value_別墅` | 同上 · 別墅 | 萬元/坪 | 鄉鎮 | 15.66 | 9.75 – 27.52 |
| `housing_unit_median_value_套房` | 同上 · 套房 | 萬元/坪 | 鄉鎮 | 36.36 | 11.32 – 104.50 |
| `housing_unit_median_value_透天厝` | 同上 · 透天厝 | 萬元/坪 | 鄉鎮 | 18.63 | 8.17 – 73.63 |
| `housing_unit_median_value_電梯大廈` | 同上 · 電梯大廈 | 萬元/坪 | 鄉鎮 | 22.57 | 9.15 – 85.59 |
| `housing_unit_median_value_樓中樓` | 同上 · 樓中樓 | 萬元/坪 | 鄉鎮 | 20.30 | 20.17 – 20.43 |

> **兩類房價欄位**：  
> - `housing_unit_median_value`：DeepSolar 同名欄位之主要 proxy；縣市層級、不分建物類型。  
> - `housing_unit_median_value_{建物類型}`：細分建物類型，僅在該鄉鎮有交易樣本時才有值。  

---

## 3. 聚合計算方式（重點）

### 3a. 鄉鎮層級（`average_household_size`、`*_{建物類型}`）

對**每一個鄉鎮**（及每一種建物類型，若為分類型房價）：

### Step 1 — 同一年度四季平均

原始資料為季別（如 `098Q1` … `098Q4`）。對民國第 \(y\) 年：

\[
\text{年度值}_{y} = \frac{1}{4}\sum_{q=1}^{4} \text{季別值}_{y,q}
\]

實作上以 `groupby(鄉鎮, year).mean()` 對該年所有季別列取平均（若某季缺值，則為該年**有資料季別**的平均）。

### Step 2 — 跨年度平均

\[
\text{最終特徵} = \frac{1}{N}\sum_{y \in \mathcal{Y}} \text{年度值}_{y}
\]

\(\mathcal{Y}\) = 原始資料中該鄉鎮（及建物類型）實際出現的民國 **98–114** 年度；\(N\) 為有年度值的年數。

### 3b. 住宅自有率（`occupancy_owner_rate`）— 表34，109 年

**來源**：`data/taiwan/housing/縣市鄉鎮住宅所有權屬資料/` 內 22 個 XLSX（自動 glob，不 hard-code 檔名）。

**解析**（每縣市檔案）：

1. 自標題列擷取縣市（`表３４　○○縣市普通住戶之住宅所有權屬`）  
2. 自動尋找「按鄉鎮市區別分」後之資料列  
3. 自動偵測 header 之「總計」「自有」欄位（支援中英雙語）  
4. 略過縣市合計列（`level == county`），保留**鄉鎮**列  

\[
\text{occupancy\_owner\_rate} = \frac{\text{自有住宅戶數}}{\text{全部住宅戶數}}
\]

對應 DeepSolar：**owner-occupied housing rate**（屋主住宅比例；與租屋戶相對）。

| 項目 | 說明 |
|---|---|
| 時間 | 民國 109 年（單年；若未來有多期檔案可擴充為 county × year 後再平均） |
| 覆蓋 | 368 / 368 鄉鎮（22 縣市檔案皆成功解析） |
| 合理性 | 全台 \(0 \le \text{rate} \le 1\) |

Notebook 另產出 **縣市 × year** 面板（`level == county` 列）供縣市間比較；併入輸出 CSV 者為**鄉鎮**欄位。

---

### 3c. 縣市層級（`housing_unit_median_value`）— **同縣市各鄉鎮相同**

使用原始買賣表中 **`鄉鎮市區` 空白且 `建物類型` 空白** 的列（縣市總體平均單價，已混合所有建物類型）。

計算步驟與 3a 相同（四季平均 → 跨年度平均），但分組鍵為 **`COUNTYNAME` 僅縣市**：

\[
\text{housing\_unit\_median\_value}_{\text{縣市}} = \text{mean}_{y}\!\left[\text{mean}_{q}(\text{縣市季別值}_{y,q})\right]
\]

合併至 368 鄉鎮表時，以 `COUNTYNAME` 左合併：**同一縣市內所有鄉鎮得到相同數值**（與 `04e` FIT 由縣市下放鄉鎮的做法一致）。

| 項目 | 說明 |
|---|---|
| 覆蓋縣市 | 21 / 22（**連江縣**無縣市層買賣統計 → 4 鄉皆為 NaN） |
| 與分類型欄位差異 | 分類型欄位為鄉鎮×建物類型，缺值多；本欄保證有縣市資料之鄉鎮皆有值 |

### 資料列篩選

| 特徵 | 列篩選 |
|---|---|
| `average_household_size` | `鄉鎮市區` 非空白；`縣市` 為現行 22 縣市 |
| `housing_unit_median_value` | `鄉鎮市區` **空白**、`建物類型` **空白**；`縣市` ≠ 全國 |
| `housing_unit_median_value_*` | `鄉鎮市區` 非空白 + `建物類型` ∈ {公寓, 別墅, 套房, 透天厝, 電梯大廈, 樓中樓} |

### 名稱對齊（與人口表合併前）

| 類型 | 處理 |
|---|---|
| 縣市 | `台` → `臺`；排除 `臺中縣(99年以前)` 等舊制標籤 |
| 鄉鎮升格 | 頭份鎮→頭份市、員林鎮→員林市、台西鄉→臺西鄉、霧台鄉→霧臺鄉、台東市→臺東市 |
| 新竹市、嘉義市 | 原始表僅「新竹市」「嘉義市」一列 → 複製至東區／北區／香山區（新竹）或東區／西區（嘉義） |

---

## 4. 與 DeepSolar 的對應

| DeepSolar 欄位 | 美國語意 | 台灣本輸出 |
|---|---|---|
| `average_household_size` | 戶內平均人口數 | 鄉鎮層級歷年平均（本檔） |
| `housing_unit_median_value` | 自有住宅**中位數**房價 | 縣市層契約**平均**單價（全類型合計，下放鄉鎮） |
| `occupancy_owner_rate` | 自有住宅比例 | 表34 自有戶 / 總戶（109年，鄉鎮） |
| （細分） | — | 另提供 6 欄 `housing_unit_median_value_{建物類型}`（鄉鎮層） |
| `housing_unit_median_gross_rent` | 租屋中位數租金 | **未產出** |

**與 `taiwan_population_features.csv` 的 `average_household_size`**：人口表為民國 109 年單年快照；本檔為 98–114 鄉鎮歷年平均，數值與定義皆不同，合併時請加後綴或擇一使用。

**Transfer 使用建議**：

- 模型若只接**一個**房價欄位：優先使用 **`housing_unit_median_value`**（縣市層、覆蓋 364/368 鄉鎮、對應 DeepSolar 欄位名）。  
- 若需依建物型態細分：再選 `housing_unit_median_value_{建物類型}` 中與案場最接近者（缺值較多）。

---

## 5. 如何載入資料

```python
import pandas as pd

taiwan_housing = pd.read_csv(
    'data/taiwan/housing/taiwan_housing_features.csv',
    encoding='utf-8-sig'
)

print(taiwan_housing.shape)      # (368, 12)
print(taiwan_housing.columns.tolist())
taiwan_housing.head()
```

---

## 6. 如何與其他台灣資料合併

`TOWNCODE` 是唯一鍵：

```python
taiwan_pop = pd.read_csv(
    'data/taiwan/population/taiwan_population_features.csv',
    encoding='utf-8-sig'
)

housing_cols = [c for c in taiwan_housing.columns if c not in (
    'TOWNCODE', 'COUNTYNAME', 'TOWNNAME'
)]

taiwan_full = taiwan_pop.merge(
    taiwan_housing[['TOWNCODE'] + housing_cols],
    on='TOWNCODE',
    how='left',
)
```

---

## 7. 原始資料檔（自動偵測）

Notebook 以 `pathlib` 掃描 `data/taiwan/housing/`，依檔名關鍵字配對：

| 類型 | 檔名關鍵字 |
|---|---|
| 家戶人口 | `平均人口` 或 `家戶結構` + `平均` |
| 房價 | `買賣契約` 或 `買賣` + `單價` |

---

## 8. 收集程式

`notebooks/04f_taiwan_housing_features.ipynb` — 重跑後覆寫 `taiwan_housing_features.csv`。

| 檢查項 | 預期 |
|---|---|
| 列數 | 368 |
| 識別欄 | `TOWNCODE`, `COUNTYNAME`, `TOWNNAME` |
| 家戶欄 | 1（`average_household_size`） |
| 縣市房價欄 | 1（`housing_unit_median_value`） |
| 自有率欄 | 1（`occupancy_owner_rate`） |
| 分類型房價欄 | 6（`housing_unit_median_value_{建物類型}`） |
| `average_household_size` 缺值 | 0 |
| `occupancy_owner_rate` 缺值 | 0 |
| `housing_unit_median_value` 缺值 | 4（連江縣 4 鄉） |
| 至少一種分類型房價有值之鄉鎮 | 224 / 368 |
| 所有權屬檔案載入 | 22 / 22（單檔失敗不中止 pipeline） |

---

## 9. 驗證結果摘要

| 驗證項目 | 結果 |
|---|---|
| 鄉鎮覆蓋率 | 368 / 368（皆對齊人口表） |
| `average_household_size` 缺值 | 0 |
| `occupancy_owner_rate` 缺值 | 0 |
| `occupancy_owner_rate` 範圍 | 0.43 – 0.97（皆在 [0,1]） |
| `housing_unit_median_value` 缺值 | 4（連江縣） |
| `housing_unit_median_value` 同縣市唯一值 | 是（每縣市 1 個數值複製至轄下鄉鎮） |
| 任一分類型房價有值 | 224 鄉鎮 |
| 六類分類型房價皆缺值 | 144 鄉鎮（多為交易量少之鄉鎮） |
| `樓中樓` 有值鄉鎮 | 2（全台交易極少，多數為 NaN） |

**參考排序（歷年平均）**：

| 指標 | 較高 | 較低 |
|---|---|---|
| 家戶人口 | 金門縣烈嶼鄉（8.77） | — |
| 自有率（鄉鎮） | 新竹縣竹北市（0.97） | 臺中市中區（0.43） |
| 自有率（縣市合計列） | 彰化縣（0.87） | 連江縣（0.70） |
| 縣市房價 `housing_unit_median_value` | 臺北市（60.52 萬元/坪，轄下各區相同） | 嘉義縣（11.54） |
| 電梯大廈單價（鄉鎮） | 臺北市大安區（85.59） | 雲林縣虎尾鎮（9.15） |

---

## 10. 缺值說明（房價欄）

### `housing_unit_median_value`（縣市層）

僅 **連江縣** 在原始統計中無縣市層買賣列，故馬祖 4 鄉為 NaN；其餘縣市轄下鄉鎮皆有相同值。

### `housing_unit_median_value_{建物類型}`（鄉鎮層）

房價來自**實價登錄買賣統計**，僅在該鄉鎮該建物類型有足夠申報樣本時才有季別資料。因此：

- 都會核心區：`電梯大廈`、`公寓` 覆蓋較完整  
- 郊區／農業縣：常僅 `透天厝` 有值，`套房`／`別墅` 多為 NaN  
- `樓中樓`：全台僅極少鄉鎮有紀錄，**不可**視為全台可比欄位  

缺值**不代表**該鄉鎮無住宅，僅代表統計期間內無該類型之買賣申報。

---

## 11. 使用建議（Transfer 階段）

1. 家戶人口以本檔 `average_household_size` 為準（鄉鎮、多年平均）。  
2. **`occupancy_owner_rate`** 為 109 年鄉鎮快照，與美國 ACS 自有住宅率語意相近，可直接作為 transfer proxy。  
3. 單一房價 proxy 請用 **`housing_unit_median_value`**（縣市下放、覆蓋率高）；細分建物請用 `_*` 欄位。  
4. 同縣市內 `housing_unit_median_value` **完全相同**；`occupancy_owner_rate` 則**因鄉鎮而異**。  
5. 勿將台灣契約平均單價（萬元/坪）與美國 ACS 中位數房價直接比較數值尺度。  
6. 需要單一年度或單季面板時，請回到原始 CSV 重新聚合（本檔僅提供歷年平均／109 快照）。
