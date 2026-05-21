import os
from glob import glob

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)


def get_prefix(col):
    return col.split('_')[0] if '_' in col else 'single'


def load_data(path='deepsolar_tract.csv', encoding='latin-1'):
    df = pd.read_csv(path, encoding=encoding)
    return df


def compute_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data['feature'] = features
    vif_data['VIF'] = [
        variance_inflation_factor(df[features].values, i)
        for i in range(len(features))
    ]
    return vif_data.sort_values('VIF', ascending=False)


def remove_high_corr_features(df, features, target_col, threshold=0.85):
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    target_corr = df[features + [target_col]].corr()[target_col].abs()

    to_drop = set()
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        for other in high_corr:
            if target_corr[col] >= target_corr[other]:
                to_drop.add(other)
            else:
                to_drop.add(col)
    return [f for f in features if f not in to_drop]


def label_potential(tile_count, q33, q66):
    def _label(n):
        if n <= q33:
            return 2
        elif n <= q66:
            return 1
        else:
            return 0
    return tile_count.apply(_label)


def get_non_transferable_cols():
    return [
        'race_asian', 'race_black_africa', 'race_indian_alaska',
        'race_islander', 'race_other', 'race_two_more', 'race_white',
        'race_white_rate', 'race_black_africa_rate', 'race_indian_alaska_rate',
        'race_asian_rate', 'race_islander_rate', 'race_other_rate', 'race_two_more_rate',
        'fips', 'state', 'county',
        'voting_2016_dem_percentage', 'voting_2016_gop_percentage', 'voting_2016_dem_win',
        'voting_2012_dem_percentage', 'voting_2012_gop_percentage', 'voting_2012_dem_win',
    ]


def get_policy_concept_cols():
    return [
        'incentive_count_residential', 'incentive_count_nonresidential',
        'incentive_residential_state_level', 'incentive_nonresidential_state_level',
        'net_metering', 'feedin_tariff', 'cooperate_tax', 'property_tax',
        'sales_tax', 'rebate', 'avg_electricity_retail_rate',
    ]


def get_us_only_cols():
    """Legacy compatibility for notebooks kept under notebooks/legacy."""
    return get_non_transferable_cols() + get_policy_concept_cols()


def get_id_cols():
    return ['lat', 'lon', 'Unnamed: 0']


def get_target_cols():
    return [
        'tile_count', 'solar_system_count', 'total_panel_area',
        'tile_count_residential', 'tile_count_nonresidential',
        'solar_system_count_residential', 'solar_system_count_nonresidential',
        'total_panel_area_residential', 'total_panel_area_nonresidential',
        'solar_panel_area_divided_by_area', 'solar_panel_area_per_capita',
        'number_of_solar_system_per_household',
    ]


def prepare_features(
    df,
    drop_non_transferable=True,
    drop_policy_concepts=False,
    drop_targets=True,
    drop_ids=True,
    drop_us_only=None,
):
    if drop_us_only is not None:
        drop_non_transferable = drop_us_only
        drop_policy_concepts = drop_us_only

    cols_to_drop = []
    if drop_non_transferable:
        cols_to_drop.extend(get_non_transferable_cols())
    if drop_policy_concepts:
        cols_to_drop.extend(get_policy_concept_cols())
    if drop_targets:
        cols_to_drop.extend(get_target_cols())
    if drop_ids:
        cols_to_drop.extend(get_id_cols())
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_to_drop)


def align_taiwan_features(tw_df, us_feature_names):
    """
    Align Taiwan data columns to match US feature names.
    This is a placeholder to be filled once Taiwan data sources are confirmed.
    """
    aligned = pd.DataFrame()
    for col in us_feature_names:
        if col in tw_df.columns:
            aligned[col] = tw_df[col]
        else:
            aligned[col] = np.nan
    return aligned


# ---------------------------------------------------------------------------
# Taiwan population data loaders
# ---------------------------------------------------------------------------

def _strip_ideographic(s):
    """Remove U+3000 ideographic spaces and strip whitespace."""
    return str(s).replace('　', '').strip() if pd.notna(s) else ''


def _lead_spaces(s):
    """Count leading U+3000 ideographic space characters."""
    count = 0
    for c in (str(s) if pd.notna(s) else ''):
        if c == '　':
            count += 1
        else:
            break
    return count


def _is_num(v):
    if pd.isna(v):
        return False
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_name(s):
    """Strip ideographic spaces and unify 台 → 臺 for consistent joining."""
    return _strip_ideographic(s).replace('台', '臺')


def _normalize_xls_name(s):
    """Normalize XLS names: remove U+3000, ASCII spaces, and unify 台 → 臺.

    In the multi-year XLS file, county names have ASCII spaces between characters
    (e.g., '新  北  市'), while township names start/end with U+3000.
    Stripping all spaces yields clean Chinese names for matching.
    """
    if not pd.notna(s):
        return ''
    return str(s).replace('　', '').replace(' ', '').replace('台', '臺')


_KNOWN_COUNTIES = {
    '臺北市', '新北市', '桃園市', '臺中市', '臺南市', '高雄市',
    '基隆市', '新竹市', '嘉義市',
    '宜蘭縣', '新竹縣', '苗栗縣', '彰化縣', '南投縣', '雲林縣',
    '嘉義縣', '屏東縣', '臺東縣', '花蓮縣', '澎湖縣', '連江縣', '金門縣',
}
_SKIP_NAMES = {'總計', '臺灣省', '福建省', ''}


def _load_pop_density_xls(xls_path, sheet='109'):
    """
    Parse the multi-year township land-area and population-density XLS file.

    Use sheet='109' for 2020 data (民國109年). This file covers ALL 368 townships
    and is the preferred source over the per-county XLSX files, which exclude some
    remote/indigenous townships.

    File structure in the township table (0-indexed columns):
      col 0: name — county names have ASCII spaces between chars (e.g., '新  北  市');
              township names start with one U+3000 (e.g., '　板橋區　').
      col 1: annual resident population
      col 2: land area (km²)
      col 3: population density (persons/km²)

    Detection strategy:
      lv=0 (no leading U+3000):
        - known county name → set county
        - skip-list name (總計, 臺灣省, 福建省) → skip
        - anything else → mis-indented township (e.g., indigenous areas marked ※);
          append under current county
      lv=1 (one leading U+3000): regular township row → append record
    """
    df = pd.read_excel(xls_path, sheet_name=sheet, header=None)
    records, county = [], None
    for _, row in df.iterrows():
        n = row.get(0, np.nan)
        lv = _lead_spaces(n)
        if not _is_num(row.get(1)):
            continue
        cleaned = _normalize_xls_name(n).lstrip('※')
        if lv == 0:
            if cleaned in _KNOWN_COUNTIES:
                county = cleaned
            elif cleaned not in _SKIP_NAMES and county:
                records.append({
                    'county_zh': county,
                    'town_zh': cleaned,
                    'population': int(float(row[1])),
                    'land_area_km2': float(row[2]) if _is_num(row.get(2)) else np.nan,
                    'population_density': float(row[3]) if _is_num(row.get(3)) else np.nan,
                })
        elif lv == 1 and county:
            records.append({
                'county_zh': county,
                'town_zh': _normalize_xls_name(n),
                'population': int(float(row[1])),
                'land_area_km2': float(row[2]) if _is_num(row.get(2)) else np.nan,
                'population_density': float(row[3]) if _is_num(row.get(3)) else np.nan,
            })
    return pd.DataFrame(records)


def _load_pop_density(pop_dir):
    """
    Parse all per-county population & density Excel files.

    File structure (0-indexed columns):
      col 1: Chinese name (leading U+3000 spaces indicate level)
      col 2: Grand total population
      col 5: Land area (km²)
      col 6: Population density (persons/km²)
    County total rows have 1 leading ideographic space; township rows have 2.
    Data rows alternate with English-name-only rows (no numeric values).
    """
    records, county = [], None
    pattern = os.path.join(pop_dir, '109年*各鄉鎮市常住人口數及人口密度.xlsx')
    for path in sorted(glob(pattern)):
        df = pd.read_excel(path, header=None)
        for _, row in df.iterrows():
            n = row.get(1, np.nan)
            lv = _lead_spaces(n)
            if not _is_num(row.get(2)):
                continue
            if lv == 1:
                county = _normalize_name(n)
            elif lv == 2 and county:
                records.append({
                    'county_zh': county,
                    'town_zh': _normalize_name(n),
                    'population': int(float(row[2])),
                    'land_area_km2': float(row[5]) if _is_num(row.get(5)) else np.nan,
                    'population_density': float(row[6]) if _is_num(row.get(6)) else np.nan,
                })
    return pd.DataFrame(records)


def _load_age(age_dir):
    """
    Parse all per-county age structure Excel files.

    Available age bands: Under 15, 15-24, 25-34, 35-44, 45-54, 55-64, 65+, Average age.
    File structure (0-indexed columns):
      col 1: Chinese name, col 2: total, col 3: <15, col 4: 15-24,
      col 5: 25-34, col 6: 35-44, col 7: blank, col 8: 45-54,
      col 9: 55-64, col 10: 65+, col 11: average age.

    Mapping to DeepSolar features:
      - Direct: age_35_44_rate, age_45_54_rate, age_55_64_rate, age_median
      - Approximate (low-importance features, uniform-distribution assumption):
          age_5_9_rate, age_10_14_rate from <15 bracket
          age_15_17_rate from 15-24 bracket
          age_65_74_rate, age_more_than_85_rate from 65+ bracket
    """
    records, county = [], None
    pattern = os.path.join(age_dir, '*常住人口之年齡結構.xlsx')
    for path in sorted(glob(pattern)):
        df = pd.read_excel(path, header=None)
        for _, row in df.iterrows():
            n = row.get(1, np.nan)
            lv = _lead_spaces(n)
            if not _is_num(row.get(2)):
                continue
            total = float(row[2])
            if total == 0:
                continue
            if lv == 1:
                county = _normalize_name(n)
            elif lv == 2 and county:
                u15   = float(row[3])  if _is_num(row.get(3))  else 0.0
                a1524 = float(row[4])  if _is_num(row.get(4))  else 0.0
                a3544 = float(row[6])  if _is_num(row.get(6))  else 0.0
                a4554 = float(row[8])  if _is_num(row.get(8))  else 0.0
                a5564 = float(row[9])  if _is_num(row.get(9))  else 0.0
                a65p  = float(row[10]) if _is_num(row.get(10)) else 0.0
                avg_a = float(row[11]) if _is_num(row.get(11)) else np.nan
                records.append({
                    'county_zh': county,
                    'town_zh': _normalize_name(n),
                    # Direct mappings
                    'age_35_44_rate':        a3544 / total,
                    'age_45_54_rate':        a4554 / total,
                    'age_55_64_rate':        a5564 / total,
                    'age_median':            avg_a,
                    # Approximations (uniform split within broader brackets)
                    'age_5_9_rate':          u15   / total * (5 / 15),
                    'age_10_14_rate':        u15   / total * (5 / 15),
                    'age_15_17_rate':        a1524 / total * (3 / 10),
                    # ~60% of Taiwan 65+ population is in the 65-74 cohort
                    'age_65_74_rate':        a65p  / total * 0.60,
                    # ~8% of Taiwan 65+ population is 85+
                    'age_more_than_85_rate': a65p  / total * 0.08,
                })
    return pd.DataFrame(records)


def _load_household(hh_dir):
    """
    Parse all per-county household Excel files.

    File structure (0-indexed columns):
      col 1: Chinese name
      col 2: Total households
      col 4: Average persons per household (total, including group quarters)
    """
    records, county = [], None
    pattern = os.path.join(hh_dir, '*.xlsx')
    for path in sorted(glob(pattern)):
        df = pd.read_excel(path, header=None)
        for _, row in df.iterrows():
            n = row.get(1, np.nan)
            lv = _lead_spaces(n)
            if not _is_num(row.get(2)):
                continue
            if lv == 1:
                county = _normalize_name(n)
            elif lv == 2 and county:
                records.append({
                    'county_zh': county,
                    'town_zh': _normalize_name(n),
                    'average_household_size': float(row[4]) if _is_num(row.get(4)) else np.nan,
                })
    return pd.DataFrame(records)


# 郵遞區號前三碼 → 縣市名稱（用於消歧同名鄉鎮）
_POSTAL_COUNTY: dict[int, str] = {}
for _p in [100, 103, 104, 105, 106, 107, 108, 110, 111, 112, 114, 115, 116]:
    _POSTAL_COUNTY[_p] = '臺北市'
for _p in range(200, 207):
    _POSTAL_COUNTY[_p] = '基隆市'
_POSTAL_COUNTY[209] = '連江縣'
for _p in list(range(207, 254)):
    _POSTAL_COUNTY.setdefault(_p, '新北市')
for _p in range(260, 273):
    _POSTAL_COUNTY[_p] = '宜蘭縣'
for _p in range(300, 303):
    _POSTAL_COUNTY[_p] = '新竹市'
for _p in range(320, 339):
    _POSTAL_COUNTY[_p] = '桃園市'
for _p in range(302, 316):
    _POSTAL_COUNTY.setdefault(_p, '新竹縣')
for _p in range(350, 370):
    _POSTAL_COUNTY[_p] = '苗栗縣'
for _p in range(400, 440):
    _POSTAL_COUNTY[_p] = '臺中市'
for _p in range(500, 531):
    _POSTAL_COUNTY[_p] = '彰化縣'
for _p in range(540, 559):
    _POSTAL_COUNTY[_p] = '南投縣'
for _p in range(600, 609):
    _POSTAL_COUNTY[_p] = '嘉義市'
for _p in range(602, 626):
    _POSTAL_COUNTY.setdefault(_p, '嘉義縣')
for _p in range(630, 656):
    _POSTAL_COUNTY[_p] = '雲林縣'
for _p in range(700, 746):
    _POSTAL_COUNTY[_p] = '臺南市'
for _p in range(800, 853):
    _POSTAL_COUNTY[_p] = '高雄市'
for _p in range(880, 886):
    _POSTAL_COUNTY[_p] = '澎湖縣'
for _p in range(890, 897):
    _POSTAL_COUNTY[_p] = '金門縣'
for _p in range(900, 948):
    _POSTAL_COUNTY[_p] = '屏東縣'
for _p in range(950, 967):
    _POSTAL_COUNTY[_p] = '臺東縣'
for _p in range(970, 984):
    _POSTAL_COUNTY[_p] = '花蓮縣'


def add_towncode_to_electricity(elec_df, pop_ref_df):
    """
    為電力 DataFrame 加入 TOWNCODE / COUNTYNAME / TOWNNAME 三欄。

    電力 CSV 的 `行政區` 欄位使用短名（省略 區/鄉/鎮/市 尾字），
    且部分同名鄉鎮橫跨多縣市。本函式先用 stripped 名稱匹配，
    再以郵遞區號前綴推斷縣市來消歧。

    Parameters
    ----------
    elec_df : pd.DataFrame
        來自 load_taiwan_electricity_features() 的輸出（含 郵遞區號、行政區）。
    pop_ref_df : pd.DataFrame
        含 TOWNCODE, COUNTYNAME, TOWNNAME 的參考表（如 taiwan_population_features.csv）。

    Returns
    -------
    pd.DataFrame
        elec_df 前端插入 TOWNCODE, COUNTYNAME, TOWNNAME；
        無法對應者（東沙、南沙等）填 NaN。
    """
    from collections import defaultdict

    def _norm(s):
        return str(s).strip().replace('台', '臺') if pd.notna(s) else ''

    def _strip(s):
        return s.rstrip('區鄉鎮市縣')

    # 建立 (stripped_name, countyname) → (TOWNCODE, COUNTYNAME, TOWNNAME)
    county_stripped: dict[tuple, tuple] = {}
    stripped_only:   dict[str, list]    = defaultdict(list)
    for _, r in pop_ref_df.iterrows():
        tn  = _norm(r['TOWNNAME'])
        cn  = _norm(r['COUNTYNAME'])
        key = (_strip(tn), cn)
        val = (r['TOWNCODE'], r['COUNTYNAME'], r['TOWNNAME'])
        county_stripped[key] = val
        stripped_only[_strip(tn)].append(val)

    towncodes, countynames, townnames = [], [], []
    for _, row in elec_df.iterrows():
        postal   = int(row['郵遞區號'])
        raw_name = _norm(row['行政區'])
        stripped = _strip(raw_name)
        county   = _norm(_POSTAL_COUNTY.get(postal, ''))

        # 1. (stripped, county) 精確查詢
        match = county_stripped.get((stripped, county))
        if match is None and county:
            # 2. 嘗試 (exact name, county) — 應對已含尾字的區名（臺北市各區）
            match = county_stripped.get((raw_name, county)) or county_stripped.get((_strip(raw_name), county))
        if match is None:
            # 3. stripped 唯一時直接使用（無縣市同名衝突）
            candidates = stripped_only.get(stripped, [])
            if len(candidates) == 1:
                match = candidates[0]

        if match:
            towncodes.append(match[0])
            countynames.append(match[1])
            townnames.append(match[2])
        else:
            towncodes.append(np.nan)
            countynames.append(np.nan)
            townnames.append(np.nan)

    result = elec_df.copy()
    result.insert(0, 'TOWNNAME',   townnames)
    result.insert(0, 'COUNTYNAME', countynames)
    result.insert(0, 'TOWNCODE',   towncodes)

    # 第二階段：對「行政區 == 縣市名」的 NaN 列展開為各行政區
    # 適用情形：嘉義市 (600)、新竹市 (300) 全市各區共用同一郵遞前三碼，
    # 電力 CSV 以市名代表全市，需複製給該市每個行政區。
    county_to_towns: dict[str, list] = defaultdict(list)
    for _, r in pop_ref_df.iterrows():
        cn = _norm(r['COUNTYNAME'])
        county_to_towns[cn].append({
            'TOWNCODE':   r['TOWNCODE'],
            'COUNTYNAME': r['COUNTYNAME'],
            'TOWNNAME':   r['TOWNNAME'],
        })

    nan_mask    = result['TOWNCODE'].isna()
    expand_rows = []
    drop_idx    = []
    for idx, row in result[nan_mask].iterrows():
        raw_name = _norm(row['行政區'])
        if raw_name in county_to_towns:
            for dist in county_to_towns[raw_name]:
                new_row = row.copy()
                new_row['TOWNCODE']   = dist['TOWNCODE']
                new_row['COUNTYNAME'] = dist['COUNTYNAME']
                new_row['TOWNNAME']   = dist['TOWNNAME']
                expand_rows.append(new_row)
            drop_idx.append(idx)

    if expand_rows:
        result = result.drop(index=drop_idx)
        result = pd.concat([result, pd.DataFrame(expand_rows)], ignore_index=True)

    return result


def load_taiwan_electricity_features(
    power_dir,
    years=(107, 108, 109, 110, 111),
    price_industrial_twd=(2.58 * 4 + 2.45 * 8) / 12,   # ≈ 2.493
    price_commercial_twd=(3.55 * 4 + 2.91 * 8) / 12,   # ≈ 3.123
    exchange_rate=28.5,
):
    """
    Map 台電鄉鎮用電統計 CSV → DeepSolar electricity_* 欄位（5年均值）。

    選用 107–111 年（民國）：
      - 皆有「售電度數(當年累計)」欄位，取月份==12 一行即得年度值
      - 109年 COVID 影響被 5 年平均稀釋
      - 排除 112 年（欄名改版、無累計欄）

    電價（全台統一常數）取 107–111 年一致適用之費率（107年4月1日起實施，112年才調漲）：
      - price_industrial_twd：低壓電力非時間流動電費 夏月2.58 / 非夏月2.45 元/度，
        以夏月4個月、非夏月8個月加權年均 ≈ 2.493 元/度
      - price_commercial_twd：表燈營業用非時間電費第2段（331–700度）夏月3.55 / 非夏月2.91，
        加權年均 ≈ 3.123 元/度（代表中小型商業用戶）

    Parameters
    ----------
    power_dir : str
        Path to data/taiwan/power/ directory.
    years : tuple[int]
        民國年份，預設 107–111。
    price_industrial_twd : float
        工業用電單價（元/度），預設為低壓非時間流動電費加權年均 ≈ 2.493。
    price_commercial_twd : float
        商業用電單價（元/度），預設為表燈營業用第2段加權年均 ≈ 3.123。
    exchange_rate : float
        TWD/USD 匯率，預設 28.5。

    Returns
    -------
    pd.DataFrame  (~368 rows)
        Columns: 郵遞區號, 行政區,
                 electricity_consume_residential, electricity_consume_commercial,
                 electricity_consume_total,
                 electricity_price_industrial, electricity_price_commercial
    """
    # 112年起欄名改版，統一對應到舊欄名
    _col_alias = {
        '項目':           '用電種類',
        '用戶數(戶)':     '用戶數',
        '售電度數(度)':   '售電度數(當年累計)',
    }

    frames = []
    for yr in years:
        fname = f'{yr}年鄉鎮市(郵遞區)別用電統計資料.csv'
        path = os.path.join(power_dir, fname)
        df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
        df = df.rename(columns=_col_alias)

        df12 = df[df['月份'] == 12].copy()
        df12['kWh'] = (
            df12['售電度數(當年累計)']
            .astype(str).str.replace(',', '', regex=False)
            .apply(pd.to_numeric, errors='coerce')
        )
        df12['users'] = (
            df12['用戶數'].astype(str)
            .str.replace(',', '', regex=False)
            .replace({'*': np.nan, '**': np.nan})
            .apply(pd.to_numeric, errors='coerce')
        )
        df12['用電種類'] = df12['用電種類'].astype(str).str.strip()
        df12['year'] = yr
        frames.append(df12)

    all_df = pd.concat(frames, ignore_index=True)

    def _agg_per_user(prefix, out_col):
        """年均 per-household 用電量（kWh）。"""
        sub = all_df[all_df['用電種類'].str.startswith(prefix)]
        annual = (
            sub.groupby(['year', '郵遞區號', '行政區'])
               .agg(kWh=('kWh', 'sum'), users=('users', 'sum'))
               .assign(per_user=lambda x: x['kWh'] / x['users'].replace(0, np.nan))
               .reset_index()
        )
        return (
            annual.groupby(['郵遞區號', '行政區'])['per_user']
                  .mean().reset_index()
                  .rename(columns={'per_user': out_col})
        )

    def _agg_total(out_col):
        """年均全區總用電量（kWh）。"""
        sub = all_df[all_df['用電種類'].str.contains('總計')]
        annual = (
            sub.groupby(['year', '郵遞區號', '行政區'])
               .agg(kWh=('kWh', 'sum'))
               .reset_index()
        )
        return (
            annual.groupby(['郵遞區號', '行政區'])['kWh']
                  .mean().reset_index()
                  .rename(columns={'kWh': out_col})
        )

    resid = _agg_per_user('1表燈非營業用', 'electricity_consume_residential')
    comm  = _agg_per_user('2表燈營業用',   'electricity_consume_commercial')
    total = _agg_total('electricity_consume_total')

    result = resid.merge(comm,  on=['郵遞區號', '行政區'], how='outer')
    result = result.merge(total, on=['郵遞區號', '行政區'], how='outer')

    # 電價：全台統一常數，換算為 USD/kWh
    result['electricity_price_industrial'] = price_industrial_twd / exchange_rate
    result['electricity_price_commercial'] = price_commercial_twd / exchange_rate

    return result


def align_taiwan_population_features(data_dir, climate_csv=None, save_path=None):
    """
    Load and align Taiwan population data to DeepSolar feature names.

    Loads three data sources (民國109年, year 2020):
      - Population count & density: from the multi-year XLS file (covers all 368 townships,
        including remote/indigenous townships missing from the per-county XLSX files)
      - Age structure (per township)
      - Average household size (per township)

    Joins all sources to the 368-township climate index via (county, town) name keys,
    so the output is ready for direct concatenation with climate features.

    Parameters
    ----------
    data_dir : str
        Path to data/taiwan/ directory.
    climate_csv : str, optional
        Path to climate/taiwan_climate_annual.csv for TOWNCODE lookup.
        Defaults to <data_dir>/climate/taiwan_climate_annual.csv.
    save_path : str, optional
        If given, saves the result CSV (UTF-8 BOM for Excel compatibility).

    Returns
    -------
    pd.DataFrame (368 rows × 17 columns)
        TOWNCODE, COUNTYNAME, TOWNNAME, population, population_density,
        land_area_km2, total_area, age_* (9 features), age_median, average_household_size.
    """
    xls_path  = os.path.join(data_dir, '民國97-114年各縣市鄉鎮市區土地面積及人口密度.xls')
    age_dir   = os.path.join(data_dir, '民國109年常住人口之年齡結構')
    hh_dir    = os.path.join(data_dir, '民國109年住戶數、常住人口數及平均每戶人口數')
    if climate_csv is None:
        climate_csv = os.path.join(data_dir, 'climate', 'taiwan_climate_annual.csv')

    df_pop = _load_pop_density_xls(xls_path, sheet='109')
    df_age = _load_age(age_dir)
    df_hh  = _load_household(hh_dir)

    df_climate = pd.read_csv(climate_csv)
    df_climate['county_zh'] = df_climate['COUNTYNAME'].str.replace('台', '臺')
    df_climate['town_zh']   = df_climate['TOWNNAME'].str.replace('台', '臺')

    base = (
        df_pop
        .merge(df_age, on=['county_zh', 'town_zh'], how='outer')
        .merge(df_hh,  on=['county_zh', 'town_zh'], how='outer')
    )

    result = (
        df_climate[['TOWNCODE', 'COUNTYNAME', 'TOWNNAME', 'county_zh', 'town_zh']]
        .merge(base, on=['county_zh', 'town_zh'], how='left')
        .drop(columns=['county_zh', 'town_zh'])
    )

    # land_area_km2 (km²) → total_area (sq. miles); 1 sq. mile = 2.58999 km²
    # US total_area includes water area; Taiwan land-only is an acceptable proxy
    result['total_area'] = result['land_area_km2'] / 2.58999

    cols = [
        'TOWNCODE', 'COUNTYNAME', 'TOWNNAME',
        'population', 'population_density', 'land_area_km2', 'total_area',
        'age_5_9_rate', 'age_10_14_rate', 'age_15_17_rate',
        'age_35_44_rate', 'age_45_54_rate', 'age_55_64_rate',
        'age_65_74_rate', 'age_more_than_85_rate', 'age_median',
        'average_household_size',
    ]
    result = result.reindex(columns=cols)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        result.to_csv(save_path, index=False, encoding='utf-8-sig')

    return result
