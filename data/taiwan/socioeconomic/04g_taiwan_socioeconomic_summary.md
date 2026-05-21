# 04g Taiwan Socioeconomic Data — 交接說明文件

**產出負責人**：社經特徵（Education / Employment）資料工程階段  
**下一步負責人**：`04_transfer_taiwan.ipynb` 整合推論階段  
**資料時間範圍**：教育程度民國 **109** 年（2020）快照；`employ_rate` 為民國 **102–113** 年（2013–2024）各縣市就業率之**歷年平均**  
**地理單位**：台灣 **368** 個鄉鎮市區（`TOWNCODE`）；原始統計為縣市層級，同縣市各鄉鎮複製相同數值

---

## 1. 產出檔案位置

```
data/taiwan/socioeconomic/
├── taiwan_socioeconomic_features.csv          ← 主要輸出，直接使用這個
├── 04g_taiwan_socioeconomic_summary.md        ← 本說明文件
├── county_unemployment_102_113.csv            ← 各縣市年度失業率（原始）
└── 縣市鄉鎮最高學歷資料/t011*.xlsx            ← 表11 最高學歷（22 縣市，各一檔）
```

---

## 2. 主要輸出：`taiwan_socioeconomic_features.csv`

**Shape：368 行 × 10 欄**（3 識別欄 + **7** 個特徵欄）

> 全缺之 DeepSolar 對應欄位（`education_bachelor`、`education_professional_school`、`education_doctoral` 及其 `*_rate`）**不輸出**。

### 識別欄位

| 欄位 | 說明 | 範例 |
|---|---|---|
| `TOWNCODE` | 內政部鄉鎮市區代碼（唯一鍵） | `6300300` |
| `COUNTYNAME` | 縣市名稱 | `新竹縣` |
| `TOWNNAME` | 鄉鎮市區名稱 | `竹北市` |

### 特徵欄位

| 欄位 | 台灣來源 | 單位 | 聚合 | 全台均值（縣市去重） | 全台範圍 |
|---|---|---|---|---|---|
| `education_college` | 表11「大專及以上」 | 人 | 109 年快照 | 396,950 | 4,336 – 1,782,577 |
| `education_high_school_graduate` | 表11「高級中等」 | 人 | 109 年快照 | 282,344 | 3,932 – 1,132,005 |
| `education_less_than_high_school` | 國小及以下 + 國（初）中 | 人 | 109 年快照 | 239,592 | 2,244 – 886,523 |
| `education_college_rate` | 大專及以上 / 總計 | 比率 (0–1) | 109 年快照 | 0.384 | 0.252 – 0.600 |
| `education_high_school_graduate_rate` | 高級中等 / 總計 | 比率 (0–1) | 109 年快照 | 0.317 | 0.240 – 0.374 |
| `education_less_than_high_school_rate` | 高中以下 / 總計 | 比率 (0–1) | 109 年快照 | 0.300 | 0.161 – 0.428 |
| `employ_rate` | `1 − (失業率 ÷ 100)` | 比率 (0–1) | **102–113 歷年平均** | 0.965 | 0.957 – 0.996 |

> **刻意未輸出**：`unemployment_rate`、全缺之 `education_bachelor*` / `education_doctoral*` 等、台灣特有衍生欄。

---

## 3. 聚合計算方式（重點）

### 3a. 教育程度（表 11）— 縣市層級，109 年快照 → 下放鄉鎮

**來源**：`縣市鄉鎮最高學歷資料/t011*.xlsx`（22 檔；notebook glob `t011*.xlsx`）

**解析**：自每檔擷取縣市合計列（`level == county`），對應 DeepSolar 欄位後以 `COUNTYNAME` 合併至 368 鄉鎮（做法同 `04e` FIT、`04f` 縣市房價）。

| DeepSolar 欄位 | 台灣表11 |
|---|---|
| `education_high_school_graduate` | 高級中等 |
| `education_less_than_high_school` | 國小及以下 + 國（初）中 |
| `education_college` | 大專及以上 |

比率：各教育人口 ÷ 同列「總計」（15 歲以上常住人口）。

### 3b. 就業率（`employ_rate`）— 縣市多年平均 → 下放鄉鎮

**來源**：`county_unemployment_102_113.csv`

**Step 1** — 每年轉換就業率：

\[
\text{employ\_rate}_{c,y} = 1 - \frac{\text{失業率}_{c,y}}{100}
\]

**Step 2** — 跨年度平均（民國 102–113，每縣市 \(N\) 為有資料年數）：

\[
\text{employ\_rate}_{c} = \frac{1}{N}\sum_{y} \text{employ\_rate}_{c,y}
\]

實作：`groupby(county)['employ_rate'].mean()`。

| 項目 | 說明 |
|---|---|
| 時間 | 102–113 年算術平均（非單年） |
| 缺年 | 連江縣、金門縣若缺 2021 年，平均時僅納入有資料年度 |
| 下放 | 同縣市 368 列中 `employ_rate` 完全相同 |

---

## 4. 與 DeepSolar 的對應

| DeepSolar 欄位 | 台灣本輸出 |
|---|---|
| `education_college` | 大專及以上（含專科～博士，未拆分） |
| `education_high_school_graduate` | 高級中等 |
| `education_less_than_high_school` | 國小及以下 + 國中 |
| `education_*_rate` | 占 15 歲以上人口比例 |
| `employ_rate` | 102–113 就業率歷年平均 |
| `education_bachelor` 等 | **不輸出**（原始表無欄位） |

**與其他台灣特徵表合併**：以 `TOWNCODE` 左合併即可（與 `taiwan_population_features.csv` 列數一致）。

---

## 5. 如何載入資料

```python
import pandas as pd

taiwan_socio = pd.read_csv(
    'data/taiwan/socioeconomic/taiwan_socioeconomic_features.csv',
    encoding='utf-8-sig',
)

print(taiwan_socio.shape)      # (368, 10)
print(taiwan_socio.columns.tolist())
taiwan_socio.head()
```

---

## 6. 如何與其他台灣資料合併

`TOWNCODE` 為唯一鍵：

```python
taiwan_pop = pd.read_csv(
    'data/taiwan/population/taiwan_population_features.csv',
    encoding='utf-8-sig',
)

socio_cols = [c for c in taiwan_socio.columns if c not in (
    'TOWNCODE', 'COUNTYNAME', 'TOWNNAME'
)]

taiwan_full = taiwan_pop.merge(
    taiwan_socio[['TOWNCODE'] + socio_cols],
    on='TOWNCODE',
    how='left',
)
```

---

## 7. 收集程式

`notebooks/04g_taiwan_socioeconomic_features.ipynb` — 重跑後覆寫 `taiwan_socioeconomic_features.csv`。

| 檢查項 | 預期 |
|---|---|
| 列數 | 368 |
| 識別欄 | `TOWNCODE`, `COUNTYNAME`, `TOWNNAME` |
| 特徵欄 | 7（不含全缺欄位） |
| `TOWNCODE` 與人口表一致 | 是 |
| 同縣市各鄉鎮特徵值相同 | 是 |
| 執行結尾訊息 | `Taiwan socioeconomic feature engineering completed.` |

---

## 8. 驗證結果摘要

| 驗證項目 | 結果 |
|---|---|
| 鄉鎮覆蓋 | 368 / 368 |
| 全缺欄位已省略 | `education_bachelor*`、`education_doctoral*` 等 |
| 比率欄位範圍 | 皆在 [0, 1] |
| `employ_rate` 範圍 | 0.957 – 0.996 |
| 縣市內唯一值檢查 | 每欄每縣市 1 個值 |

**參考排序（縣市層級）**：

| 指標 | 較高 | 較低 |
|---|---|---|
| `education_college_rate` | 臺北市（0.600） | 嘉義縣（0.252） |
| `employ_rate`（歷年平均） | 連江縣（0.996） | 基隆市（0.957） |

---

## 9. 使用建議（Transfer 階段）

1. 輸出格式已對齊 `taiwan_population_features.csv`，可直接 `merge(on='TOWNCODE')`。  
2. 教育為 **109 年橫截面**；就業率為 **102–113 歷年平均**，勿當成逐年面板。  
3. 模型若需要 `education_bachelor`，請改用 `education_college` / `education_college_rate` 或於特徵對齊階段剔除無對應欄。  
4. 同縣市內所有鄉鎮數值相同；區分鄉鎮差異需另建鄉鎮層表11 解析（本檔刻意僅輸出縣市下放版）。

---

## 10. 相關檔案

| 檔案 | 說明 |
|---|---|
| `notebooks/04g_taiwan_socioeconomic_features.ipynb` | 特徵工程主程式 |
| `data/taiwan/population/taiwan_population_features.csv` | 鄉鎮鍵參照表 |
