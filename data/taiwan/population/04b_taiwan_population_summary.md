# 04b 

**產出負責人**：人口資料收集階段  
**下一步負責人**：`04_transfer_taiwan.ipynb` 整合推論階段  
**資料時間範圍**：民國 109 年（2020 年）  
**地理單位**：台灣 368 個鄉鎮市區（TOWNCODE）

---

## 1. 產出檔案位置

```
data/processed/
└── taiwan_population_features.csv    ← 主要輸出，直接使用這個

data/taiwan/                          ← 原始資料來源（勿修改）
├── 民國97-114年各縣市鄉鎮市區土地面積及人口密度.xls
├── 民國109年常住人口之年齡結構/
│   ├── 臺北市常住人口之年齡結構.xlsx
│   ├── 新北市常住人口之年齡結構.xlsx
│   └── ...（22 縣市，共 22 個 xlsx）
└── 民國109年住戶數、常住人口數及平均每戶人口數/
    └── ...（22 縣市，共 22 個 xlsx）
```

---

## 2. 主要輸出：`taiwan_population_features.csv`

**Shape：368 行 × 16 欄**（每個鄉鎮市區一行，無缺值）

### 識別欄位

| 欄位 | 說明 | 範例 |
|---|---|---|
| `TOWNCODE` | 內政部鄉鎮市區代碼（唯一鍵） | `6300300` |
| `COUNTYNAME` | 縣市名稱 | `臺北市` |
| `TOWNNAME` | 鄉鎮市區名稱 | `大安區` |

### 人口特徵欄位（對應 DeepSolar 欄位名稱）

| 欄位名稱 | 單位 | 說明 | DeepSolar 對應欄位 | 全台均值 |
|---|---|---|---|---|
| `population` | 人 | 常住人口數 | 同名 | 64,025 |
| `population_density` | 人/km² | 常住人口密度 | 同名 | 2,748 |
| `land_area_km2` | km² | 土地面積（輔助欄位） | — | — |
| `age_median` | 歲 | 平均年齡（近似中位數） | 同名 | 43.4 |
| `age_35_44_rate` | 比率 | 35–44 歲人口比 | 同名 | — |
| `age_45_54_rate` | 比率 | 45–54 歲人口比 | 同名 | — |
| `age_55_64_rate` | 比率 | 55–64 歲人口比 | 同名 | — |
| `age_65_74_rate` | 比率 | 65–74 歲人口比（近似） | 同名 | — |
| `age_more_than_85_rate` | 比率 | 85 歲以上人口比（近似） | 同名 | — |
| `age_5_9_rate` | 比率 | 5–9 歲人口比（近似） | 同名 | — |
| `age_10_14_rate` | 比率 | 10–14 歲人口比（近似） | 同名 | — |
| `age_15_17_rate` | 比率 | 15–17 歲人口比（近似） | 同名 | — |
| `average_household_size` | 人/戶 | 平均每戶人口數 | 同名 | 2.98 |

---

## 3. 如何載入資料

```python
import pandas as pd

taiwan_pop = pd.read_csv(
    'data/processed/taiwan_population_features.csv',
    encoding='utf-8-sig'   # 中文縣市名稱需指定 encoding
)

print(taiwan_pop.shape)      # (368, 16)
print(taiwan_pop.columns.tolist())
taiwan_pop.head()
```

---

## 4. 如何與其他台灣資料合併

`TOWNCODE` 是唯一鍵，與氣候資料等其他台灣資料整合時以此欄位 join：

```python
import pandas as pd

taiwan_climate = pd.read_csv('data/taiwan/climate/taiwan_climate_annual.csv', encoding='utf-8-sig')
taiwan_pop     = pd.read_csv('data/processed/taiwan_population_features.csv', encoding='utf-8-sig')

taiwan_full = taiwan_climate.merge(taiwan_pop.drop(columns=['COUNTYNAME','TOWNNAME']),
                                   on='TOWNCODE', how='left')
print(taiwan_full.shape)   # (368, ...)
```

---

## 5. 如何對應 DeepSolar 模型輸入

本研究使用 `voted` 特徵集（90 個特徵），其中人口欄位 13 個，
依 SHAP 重要性排名：

| 欄位 | Stage 1 SHAP 排名 | Stage 2 SHAP 排名 |
|---|---|---|
| `population_density` | 2 | 3 |
| `population` | 4 | — |
| `age_median` | — | — |
| `age_35_44_rate` ～ `age_more_than_85_rate` | — | — |
| `average_household_size` | — | — |

> `population_density` 與 `population` 是高重要性特徵，請確認其合理性後再推論。

---

## 6. 資料來源與計算方式

**原始資料來源**：內政部戶政司（民國109年，2020年）

| 來源檔案 | 提供欄位 |
|---|---|
| `民國97-114年各縣市鄉鎮市區土地面積及人口密度.xls`（Sheet 109） | `population`、`population_density`、`land_area_km2` |
| `民國109年常住人口之年齡結構/*.xlsx` | 所有 `age_*` 欄位 |
| `民國109年住戶數、常住人口數及平均每戶人口數/*.xlsx` | `average_household_size` |

**年齡欄位的計算方式**：

台灣統計提供的年齡組距（10 年區間）與 DeepSolar（5 年區間）不完全一致，
部分欄位需做近似換算：

| DeepSolar 欄位 | 台灣資料來源 | 換算方式 |
|---|---|---|
| `age_35_44_rate` | 35–44 歲組 | 直接對應 |
| `age_45_54_rate` | 45–54 歲組 | 直接對應 |
| `age_55_64_rate` | 55–64 歲組 | 直接對應 |
| `age_median` | 平均年齡欄位 | 直接對應（以平均年齡近似中位數） |
| `age_5_9_rate` | 未滿 15 歲組 | `<15歲 × (5/15)`（均勻分布假設） |
| `age_10_14_rate` | 未滿 15 歲組 | `<15歲 × (5/15)`（均勻分布假設） |
| `age_15_17_rate` | 15–24 歲組 | `15-24歲 × (3/10)`（均勻分布假設） |
| `age_65_74_rate` | 65 歲以上組 | `65+歲 × 0.60`（台灣 65+ 年齡分布近似） |
| `age_more_than_85_rate` | 65 歲以上組 | `65+歲 × 0.08`（台灣 65+ 年齡分布近似） |

> 注意：以上三類近似欄位（`age_5_9`、`age_10_14`、`age_15_17`、`age_65_74`、`age_more_than_85`）的 SHAP 重要性均低，近似誤差對模型輸出影響有限。

**載入程式位置**：`src/features.py` → `align_taiwan_population_features()`

---

## 7. 驗證結果摘要

| 驗證項目 | 結果 |
|---|---|
| 收集覆蓋率 | 368 / 368 鄉鎮（100%） |
| 缺值 | 0（最終 CSV 無任何缺值） |
| 人口密度最高鄉鎮 | 新北市永和區（38,392 人/km²）✓ 符合預期 |
| 人口密度最低鄉鎮 | 高雄市桃源區（4.5 人/km²）✓ 原住民山區符合預期 |
| 平均年齡最高鄉鎮 | 臺南市龍崎區（55.5 歲）✓ 南台灣老齡農村符合預期 |
| 平均年齡最低鄉鎮 | 連江縣東引鄉（31.7 歲）✓ 軍事駐紮島嶼，青壯年為主 |
| 平均每戶人數 | 全台均值 2.98 人/戶 ✓ 符合台灣家庭結構 |

---

## 8. 注意事項

- **`household_type_family_rate`**（家庭戶比率）：DeepSolar voted 特徵集包含此欄，但 SHAP 重要性極低（未進前 15），且台灣無對應公開資料，推論時可填全台均值（約 0.80）或直接以全台常數代替。
- **XLS 特殊格式**：原始人口密度 XLS 中，原住民族山地鄉（如復興區、和平區、茂林區等共 32 個）縮排格式與一般鄉鎮不同，`src/features.py` 已處理，若換用其他年份資料需注意。
- **基隆市**：年齡結構檔案曾誤放人口密度表（表１），正確檔案為表３（含平均年齡欄位）。
