import numpy as np
import pandas as pd


def topsis(df, weights, benefit_criteria, alternative_col=None):
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

    Parameters
    ----------
    df : pd.DataFrame
        Decision matrix. Rows are alternatives, columns are criteria.
    weights : dict
        {criterion: weight}. Weights are normalized internally.
    benefit_criteria : list
        List of criteria where higher is better.
        Criteria not in this list are treated as cost criteria.
    alternative_col : str, optional
        Column name for alternative identifiers. If None, uses index.

    Returns
    -------
    pd.DataFrame with columns: alternative, score, rank
    """
    criteria = list(weights.keys())
    mat = df[criteria].copy()

    # Replace NaNs with column mean
    mat = mat.apply(lambda col: col.fillna(col.mean()))

    # Normalize decision matrix
    norms = np.sqrt((mat ** 2).sum(axis=0))
    mat_norm = mat / norms

    # Weighted normalized matrix
    w = np.array([weights[c] for c in criteria])
    w = w / w.sum()
    mat_weighted = mat_norm * w

    # Ideal and negative-ideal solutions
    ideal = []
    nadir = []
    for c in criteria:
        if c in benefit_criteria:
            ideal.append(mat_weighted[c].max())
            nadir.append(mat_weighted[c].min())
        else:
            ideal.append(mat_weighted[c].min())
            nadir.append(mat_weighted[c].max())
    ideal = np.array(ideal)
    nadir = np.array(nadir)

    # Distances
    d_ideal = np.sqrt(((mat_weighted - ideal) ** 2).sum(axis=1))
    d_nadir = np.sqrt(((mat_weighted - nadir) ** 2).sum(axis=1))

    # Closeness coefficient
    score = d_nadir / (d_ideal + d_nadir)

    result = pd.DataFrame({
        'alternative': alternative_col if alternative_col is None else df[alternative_col],
        'score': score,
    })
    if alternative_col is None:
        result['alternative'] = df.index
    result['rank'] = result['score'].rank(ascending=False, method='min').astype(int)
    return result.sort_values('rank')
