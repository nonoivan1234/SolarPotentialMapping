import pandas as pd
import numpy as np
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
