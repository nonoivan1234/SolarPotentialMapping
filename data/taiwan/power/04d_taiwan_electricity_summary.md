# 04d Taiwan Electricity Features

**產出負責人**：電力資料收集與 mapping 階段  
**下一步負責人**：`04_transfer_taiwan.ipynb` 整合推論階段  
**資料時間範圍**：民國 107–111 年（2018–2022）五年均值  
**地理單位**：台灣 368 個鄉鎮市區（TOWNCODE），本資料覆蓋 **368 筆**

---

## 1. 產出檔案位置

```
data/taiwan/power/
├── taiwan_electricity_features.csv          ← 主要輸出，直接使用這個
├── 107年鄉鎮市(郵遞區)別用電統計資料.csv
├── 108年鄉鎮市(郵遞區)別用電統計資料.csv
├── 109年鄉鎮市(郵遞區)別用電統計資料.csv
├── 110年鄉鎮市(郵遞區)別用電統計資料.csv
├── 111年鄉鎮市(郵遞區)別用電統計資料.csv
├── 107年上半年各類電價表及計算範例_1070401.pdf
├── 107年下半年各類電價表及計算範例_1070401.pdf
│   ...（108–112年各上下半年電價表 PDF，共12份）
```

---

## 2. 主要輸出：`taiwan_electricity_features.csv`

**Shape：368 行 × 10 欄**（每個鄉鎮市區一行，NaN TOWNCODE 列已移除）

### 識別欄位

| 欄位 | 說明 | 範例 |
|---|---|---|
| `TOWNCODE` | 內政部鄉鎮市區代碼（唯一鍵） | `63000050` |
| `COUNTYNAME` | 縣市名稱 | `臺北市` |
| `TOWNNAME` | 鄉鎮市區名稱 | `中正區` |
| `郵遞區號` | 台電資料原始鍵（3碼） | `100` |
| `行政區` | 台電資料原始鄉鎮名 | `中正區` |

### 電力特徵欄位（對應 DeepSolar 欄位）

| 欄位名稱 | 單位 | 說明 | DeepSolar 對應欄位 | r with tile_count |
|---|---|---|---|---|
| `electricity_consume_residential` | kWh/用戶/年 | 表燈非營業用，107–111年均值 per 用戶 | 同名 | −0.208 |
| `electricity_consume_commercial` | kWh/用戶/年 | 表燈營業用，107–111年均值 per 用戶 | 同名 | −0.034 |
| `electricity_consume_total` | kWh/年 | 全區合計用電量，107–111年均值 | 同名 | −0.045 |
| `electricity_price_industrial` | $/kWh | 低壓工業電費年均（全台常數）≈ 0.0875 | 同名 | +0.336 |
| `electricity_price_commercial` | $/kWh | 表燈營業電費年均（全台常數）≈ 0.1096 | 同名 | +0.294 |

---

## 3. 如何載入資料

```python
import pandas as pd

taiwan_elec = pd.read_csv(
    'data/taiwan/power/taiwan_electricity_features.csv',
    encoding='utf-8-sig'
)

print(taiwan_elec.shape)   # (368, 10)
print(taiwan_elec.columns.tolist())
taiwan_elec.head()
```

---

## 4. 如何與其他台灣資料合併

`TOWNCODE` 是跨資料集的唯一鍵：

```python
import pandas as pd

taiwan_pop  = pd.read_csv('data/taiwan/population/taiwan_population_features.csv', encoding='utf-8-sig')
taiwan_elec = pd.read_csv('data/taiwan/power/taiwan_electricity_features.csv', encoding='utf-8-sig')

# 只保留需要的欄位，以 TOWNCODE join
elec_cols = ['TOWNCODE',
             'electricity_consume_residential', 'electricity_consume_commercial',
             'electricity_consume_total', 'electricity_price_industrial',
             'electricity_price_commercial']

taiwan_full = taiwan_pop.merge(taiwan_elec[elec_cols], on='TOWNCODE', how='left')
print(taiwan_full.shape)   # (368, ...)
```

---

## 5. 如何對應 DeepSolar 模型輸入

本研究使用 `voted` 特徵集（90 個特徵），其中電力欄位共 5 個：

| 欄位 | r with tile_count | 備註 |
|---|---|---|
| `electricity_price_industrial` | **+0.336**（最高） | 全台常數，區域無差異 |
| `electricity_price_commercial` | **+0.294** | 全台常數，區域無差異 |
| `electricity_consume_residential` | −0.208 | 各鄉鎮不同，有鑑別力 |
| `electricity_consume_total` | −0.045 | 各鄉鎮不同 |
| `electricity_consume_commercial` | −0.034 | 各鄉鎮不同 |

> **注意**：`electricity_price_industrial` 和 `electricity_price_commercial` 是全台統一常數，在推論時對所有鄉鎮套用相同值，無法提供鄉鎮間的分辨能力，僅影響模型截距。  
> `avg_electricity_retail_rate` **未入選** voted 90 特徵，不需要 mapping。

---

## 6. 資料來源與計算方式

### 6.1 用電量欄位（consume）

**原始資料來源**：台電「鄉鎮市（郵遞區）別用電統計資料」，民國 107–111 年

**年份選擇邏輯：**
| 年份 | 是否使用 | 原因 |
|---|---|---|
| 107–111年 | ✅ 全部使用 | 皆有「售電度數(當年累計)」欄，取月份==12 一行即得年度值 |
| 106年 | ❌ 排除 | 資料較舊，避免影響均值 |
| 112年 | ❌ 排除 | 欄名改版（`用電種類`→`項目`）且無累計欄，需加總12個月，解析複雜 |

**各欄位計算方式：**

| 欄位 | 篩選條件 | 計算 |
|---|---|---|
| `electricity_consume_residential` | `用電種類.startswith('1表燈非營業用')` | 各年各鄉鎮 kWh / 用戶數 → 5年均值 |
| `electricity_consume_commercial` | `用電種類.startswith('2表燈營業用')` | 各年各鄉鎮 kWh / 用戶數 → 5年均值 |
| `electricity_consume_total` | `用電種類.contains('總計')` | 各年各鄉鎮年度總 kWh → 5年均值 |

CSV 年份 Schema 差異說明：

| 年份 | 用電種類欄名 | 數字格式 | 總計項目 |
|---|---|---|---|
| 107–109 | `用電種類` | 有逗號（`"60,504"`） | `23總計` |
| 110 | `用電種類` | 無逗號 | `23總計` |
| 111 | `用電種類` | 無逗號 | `26總計`（EV 新增 3 行）|
| 112 | `項目`（改名）| 無逗號 | `26總計` |

### 6.2 電價欄位（price）

**原始資料來源**：台電各類電價表 PDF（107–112年，各上下半年，共12份）

**電價查驗結果**（逐份 PDF 閱讀確認）：
- 107–111年全部標示「**107年4月1日起實施**」，費率完全相同
- **112年4月1日**才正式調漲（居住高用電段、商業新增分段）

**費率計算（加權年均）：**

| 欄位 | 費率來源 | 夏月（6–9月）| 非夏月 | 年均（4:8加權）|
|---|---|---|---|---|
| `electricity_price_industrial` | 低壓電力非時間流動電費 | 2.58 元/度 | 2.45 元/度 | **≈ 2.493 元/度** |
| `electricity_price_commercial` | 表燈營業用第2段（331–700度） | 3.55 元/度 | 2.91 元/度 | **≈ 3.123 元/度** |

換算 USD/kWh（匯率 28.5）：
- `electricity_price_industrial` = 2.493 / 28.5 ≈ **0.0875 $/kWh**
- `electricity_price_commercial` = 3.123 / 28.5 ≈ **0.1096 $/kWh**

**載入程式位置**：`src/features.py`
- `load_taiwan_electricity_features()` — 載入並計算五年均值
- `add_towncode_to_electricity()` — 對應 TOWNCODE，處理特殊展開邏輯

---

## 7. 特殊處理說明

### 7.1 郵遞區號→TOWNCODE 對應

電力資料以「郵遞區號（3碼）」為地理單位，需對應至其他資料集使用的 `TOWNCODE`。

`add_towncode_to_electricity()` 採三層匹配策略：
1. `(stripped_行政區名, 縣市名)` 精確查詢（以 `_POSTAL_COUNTY` dict 由郵遞區號推斷縣市）
2. `(完整行政區名, 縣市名)` 備選
3. 無縣市衝突時以 stripped 名唯一匹配

### 7.2 新竹市（300）/ 嘉義市（600）展開

**問題**：新竹市三個行政區（東區、北區、香山區）和嘉義市兩個行政區（東區、西區）全部共用同一郵遞前三碼，電力 CSV 只有市級整體一筆資料。

**處理方式**：`add_towncode_to_electricity()` 第二階段自動偵測「`行政區` == 縣市名」的列，複製展開為該縣市各行政區各一筆（電力數值相同，TOWNCODE/COUNTYNAME/TOWNNAME 各別填入）。

| 電力CSV原始列 | 展開結果 |
|---|---|
| 300 新竹市 | 300 東區 + 300 北區 + 300 香山區（3筆）|
| 600 嘉義市 | 600 東區 + 600 西區（2筆）|

### 7.3 NaN 欄位說明

`electricity_consume_residential` 和 `electricity_consume_commercial` 共 5 筆 NaN：

| 郵遞區號 | 原因 |
|---|---|
| 896 | 5 年內用戶數均為 `*`（統計保護，無法計算 per-user 值）|
| 其他 | 同上 |

### 7.4 Drop 列說明（共 4 筆）

| 郵遞區號 | 行政區 | 移除原因 |
|---|---|---|
| 290 | 中山區 | 不明特殊郵遞區號，用電量為 0 |
| 703 | 西區 | 台南縣市合併後廢止（2010年），無現行 TOWNCODE |
| 817 | 東沙 | 離島哨所，用電量為 0 |
| 819 | 南沙 | 離島哨所，用電量為 0 |

---

## 8. 驗證結果摘要

| 驗證項目 | 結果 |
|---|---|
| 覆蓋率 | 368 / 368 鄉鎮（100%）|
| NaN TOWNCODE | 0（全部已對應或合理移除）|
| 電價唯一值數 | 1（全台統一常數，符合台電獨占制度）|
| 合理性：台北市中正區 >> 連江縣 | ✅ 總用電量符合預期 |
| 電價費率 PDF 查驗 | ✅ 107–111年費率完全一致（107年4月1日起實施），112年才調漲 |
| 新竹市三區各有獨立列 | ✅ 東區、北區、香山區各一筆 |
| 嘉義市兩區各有獨立列 | ✅ 東區、西區各一筆 |

---

## 9. 注意事項

- **電價為全台常數**：台電為獨占事業，`electricity_price_industrial` 和 `electricity_price_commercial` 對所有 368 個鄉鎮套用相同值，模型推論時無鄉鎮間分辨能力，僅影響全台整體估算的截距。
- **展開列的電力數值相同**：新竹市三個行政區、嘉義市兩個行政區各自共用同一份電力消費數值（因來自同一筆市級整體資料），在使用 `electricity_consume_residential/commercial/total` 進行鄉鎮比較時需留意這個限制。
- **112年電價已調漲**：若日後要更新到更近年份的資料，需重新計算電價欄位（商業最高段從 6.43 漲至 6.75 元/度，住宅高用電段 701 度以上也調漲）。
- **112年 CSV 格式改版**：欄名 `用電種類`→`項目`，且無累計欄，需逐月加總，`load_taiwan_electricity_features()` 目前不支援（`_col_alias` 僅做欄名對齊，但無累計欄處理），若要納入需擴充解析邏輯。
