# Taiwan Transfer 執行快速參考

## 一句話決策

| 情況 | 建議 |
|------|------|
| ⏱️ **時間緊（< 2 週）** | 方案 A：日照 + 電價 + 人口（3 類，1 週） |
| ✅ **標準課程隊伍（2-3 週）** | 方案 B：+ 收入 + 住宅 + 教育 + 政策（7 類，2-3 週） |
| 🚀 **有興趣延伸（課後）** | 方案 C：爬蟲 + 空間內插（完整 11 類） |

---

## 三分鐘快速對比

```
成本 vs 產出品質

方案 A  ██░░░░░  成本低 | 品質 60-70% （POC 等級）
方案 B  ██████░░  成本中 | 品質 80-85% （交付級）⭐ 推薦
方案 C  ████████  成本高 | 品質 95%+  （研究級）
```

---

## 方案 A 的「一天快速驗證」流程

**Day 1：資料蒐集**
```
NASA POWER API (1小時)
  → 各縣市中心座標查日照輻射量
  → 輸出 CSV：county | solar_irradiance_kwh_m2_day

台電官網 (1小時)
  → 複製最新住宅電價表
  → 計算年均邊際價格
  → 輸出 CSV：county | electricity_retail_rate

內政部戶政司 MyData (1小時)
  → 下載 2025 人口統計
  → 計算密度 = 人口 / 面積
  → 輸出 CSV：county | population_density

合併 & 填空 (1小時)
  → merge 三個表
  → NaN 用全國平均
  → 輸出：tw_minimal_features.csv (22 rows × 5 cols)
```

**Day 2：模型套用 & 驗證**
```
04_transfer_taiwan.ipynb (4小時)
  → 載入 best_us_twostage_model_voted_tuned.pkl
  → 特徵對齙（只需 3 欄）
  → Stage 1 & 2 推論
  → 輸出排名表

結果檢查 (2小時)
  → top 5 高潛力縣市合理嗎？（e.g., 南部日照好地區應該排高）
  → 結論：POC 驗證成功 ✅
```

---

## 方案 B 的關鍵里程碑（分周目標）

### Week 1：快速資料蒐集
- [ ] Day 1-2：日照 + 電價 + 人口（3 人用 3 天）→ 輸出
- [ ] Day 3-4：收入 + 社經（1-2 人）→ 輸出
- [ ] Day 5-7：住宅 + 政策（2 人）→ 輸出
- **檢查點**：7 個資料源都有初版 CSV，欄位單位確認

### Week 2：特徵對齙 + 品質保證
- [ ] Day 1-2：實作 `align_taiwan_features()` 進階版
- [ ] Day 3-4：資料品質檢查（缺失值、異常值、分布）
- [ ] Day 5：統一單位、補值策略決定
- **檢查點**：`tw_features_aligned.csv` (368 rows × 80 cols) 產出

### Week 3：模型套用 + 驗證 + 報告
- [ ] Day 1-2：04_transfer 實作、測試、debug
- [ ] Day 3：結果視覺化（地圖 / 排序表 / 與實際數據對比）
- [ ] Day 4-5：MADA 優先排序（可選），最終報告
- **產出**：`taiwan_predictions.csv` + `taiwan_ranking.csv` + 圖表

---

## 決定資料來源時的優先序

```
「政府開放資料」優先於「爬蟲」優先於「人工蒐集」

Tier 1 ⭐⭐⭐（直接下載，5分鐘內）
  - NASA POWER API （日照輻射量）
  - 台電官網電價表
  - 內政部戶政司 MyData

Tier 2 ⭐⭐（1-2 小時搜尋 + 下載）
  - 主計總處統計資料庫（家戶所得、教育）
  - 教育部統計處（教育程度）
  - 勞動部統計（失業率）

Tier 3 ⭐（需爬蟲或 API）
  - 能源局補助系統（需分析下載或爬蟲）
  - 房仲網路房價（需爬蟲）

避免 ✗（時間黑洞）
  - 逐個鄉鎮打電話問
  - 手工掃描紙本統計書
```

---

## 「特徵對齙」的三層落實

### Layer 1：直接映射（最容易）
```python
tw_feature_col → us_feature_col  (1:1 對應)
例：population_density → population_density
```

### Layer 2：簡單轉換（稍難）
```python
tw_col1 + tw_col2 → us_col  (組合計算)
例：(avg_temp_max + avg_temp_min) / 2 → cooling_degree_days
```

### Layer 3：邏輯填補（最難）
```python
台灣沒有這個欄位 → 用 proxy 或全國平均
例：政策補助 (incentive_*) 在台灣沒有州級概念
  → 用「近 3 年全國平均核准補助數」或 0（表示無此政策）
```

**時間分配**：Layer 1 佔 60%，Layer 2 佔 30%，Layer 3 佔 10%

---

## 風險速查表 🚨

### 最常出現的 3 個問題

**Q1：「某個台灣資料怎麼找？」**
```
A: 試試這個順序：
  1. 政府資料開放平台 https://data.gov.tw/
  2. Google: "「欄位名」 台灣 統計 CSV"
  3. 該部會官網的「統計」或「公開資訊」區
  4. 若都沒有，用全國平均值填補
```

**Q2：「台灣資料缺值太多怎麼辦？」**
```
A: 優先順序
  1. 先檢查：這個欄位對模型重要嗎？
     → voted_features 的 top 20 才需要完整
  2. 缺 < 10%：用中位數 impute
  3. 缺 10-50%：考慮 drop 此欄位
  4. 缺 > 50%：改用 proxy 變數
```

**Q3：「預測結果跟台灣實際數據不符怎麼辦？」**
```
A: 很正常，因為：
  - 美國補助制度不同 → 投資意願差異
  - 台灣屋頂法規更嚴 → 裝設容量限制
  - 本地施工成本差異 → 投資報酬率差異
  
解決：
  - 改用 relative ranking 而非絕對值
  - 與台電排名的 Spearman ρ > 0.6 就算成功
```

---

## 最小可行產品（MVP）清單

要讓 Notebook 04 跑起來，最少需要：

```
✅ Mandatory（必須）
├─ 縣市或鄉鎮代碼列表 (area_id, area_name)
├─ 至少 3 個基本特徵：
│  ├─ 日照輻射量（proxy for climate）
│  ├─ 電價（proxy for policy）
│  └─ 人口密度（proxy for infrastructure）
└─ 模型檔案：best_us_twostage_model_voted_tuned.pkl

⭐ Nice-to-have（有更好）
├─ 8-10 個額外特徵（收入、教育、住宅等）
├─ 視覺化（地圖、排序圖表）
└─ 與台電實際數據的驗證分析

❌ Nice-to-not-have（可不做）
├─ 超完整 11 類全部特徵
├─ 里級別地理尺度
└─ 時序補值 & 空間內插
```

---

## 一旦開始就無法回頭的決策 ⚠️

| 決策 | 影響 | 建議 |
|------|------|------|
| **選了地理尺度** | 縣市 vs 鄉鎮 (22 vs 368 rows) | 先 proof-of-concept 用縣市，成功再升到鄉鎮 |
| **決定了 NaN 填補策略** | 全國平均 vs 分層平均 vs drop | 保守：全國平均；進階：分層 |
| **選擇了政策指標** | incentive / FIT / rebate 怎麼代表 | 能源局補助數 vs 補助額度 vs 核准率 → 早決定 |
| **公布了預測排名** | 縣市首長/民代的政治敏感度 | 只發內部報告，別公開 top 5 before 驗證 |

---

## 交付物檢查清單

最終要交出：

```python
outputs/results/transfer/
├── taiwan_predictions_{FEATURE_SET}.csv
│   └─ Columns: area_name, stage1_prob, stage2_density, combined_score, rank
├── taiwan_predictions_summary.md
│   └─ 資料來源說明 + 方法說明 + 結果限制
├── taiwan_vs_actual_validation.csv
│   └─ 若有台電實際數據：predicted_rank vs actual_rank + Spearman ρ
└── figures/
    ├── 04_taiwan_map_potential.png
    ├── 04_taiwan_top20_ranking.png
    └── 04_taiwan_vs_actual_scatter.png

notebooks/
├── 04_transfer_taiwan.ipynb
│   └─ 完全可運行，未來人執行也能複現
└── 05_mada.ipynb （如果有做 MADA）
```

---

## 討論會的 30 秒電梯簡報

```
「台灣太陽能潛力推論」

三行摘要：
1. 用美國 DeepSolar 訓練出的 two-stage 模型
2. 蒐集 7 類台灣對應資料（日照、電價、收入、人口…）
3. 推論各縣市/鄉鎮的相對太陽能部署潛力排名

工作量：2-3 週，分工 5 人

下一步：決定方案（快速 PoC vs 完整交付）& 分配資料蒐集責任
```

---

*上面這份文檔應該能加速小組討論。有具體技術問題再逐個深入！*
