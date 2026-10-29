"""Response targets, deviation indices and small helpers."""

import numpy as np
import pandas as pd

C1, C2 = 0.8, 0.2      # weights on the mean and the spread of the residual
TOP_RATIO = 0.1        # fraction of records retained per target


def quantile_curve(midr, percentile=16, method='gamma'):
    """Target response at a given exceedance level.

    ``midr`` is ``(n_records, n_floors)``. ``method='gamma'`` fits a gamma
    distribution to each floor and inverts its CDF; ``method='empirical'`` takes
    the sample quantile directly.
    """
    arr = np.asarray(midr, float)
    if method == 'empirical':
        return np.nanpercentile(arr, percentile, axis=0)

    from scipy.stats import gamma
    out = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        col = col[np.isfinite(col)]
        if col.size < 2:
            out.append(np.nan)
            continue
        a, loc, scale = gamma.fit(col, floc=0)
        out.append(gamma.ppf(percentile / 100.0, a, loc=loc, scale=scale))
    return np.array(out)


def label_index(midr, target):
    """Deviation index of each record from a target curve.

    Weighted sum of the mean and the standard deviation of the per-floor
    residual magnitude. Lower values indicate a closer response.
    """
    midr = np.asarray(midr, float)
    target = np.asarray(target, float)
    return np.array([C1 * np.abs(target - row).mean() + C2 * np.abs(target - row).std()
                     for row in midr])


def label_group(midr, target, top_ratio=TOP_RATIO):
    """Mark the ``top_ratio`` fraction of records closest to ``target``."""
    idx = label_index(midr, target)
    keep = max(int(round(len(idx) * top_ratio)), 1)
    labels = np.zeros(len(idx), dtype=int)
    labels[np.argsort(idx)[keep:]] = -1
    return labels, idx


def absolute_relative_deviation(curve, ref, eps=1e-8):
    """|curve - ref| / (|ref| + eps), evaluated pointwise."""
    c = np.asarray(curve, float)
    r = np.asarray(ref, float)
    if c.shape != r.shape:
        raise ValueError('curve and reference must align')
    return np.abs(c - r) / (np.abs(r) + eps)


def dual_reference_conformity(curve, target_low, target_high):
    """Mean ARD against the two target quantiles, summed, in percent."""
    ard_low = absolute_relative_deviation(curve, target_low)
    ard_high = absolute_relative_deviation(curve, target_high)
    return (float(np.nanmean(ard_low)) + float(np.nanmean(ard_high))) * 100, \
           ard_low * 100, ard_high * 100


def read_table(path):
    """Read a CSV, trying common encodings and separators."""
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin1'):
        for sep in (None, ',', '\t', ';'):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine='python')
                if df.shape[1]:
                    return df
            except Exception:
                continue
    raise RuntimeError(f'cannot read {path}')


def find_column(df, keywords, kind=object):
    """Return the first column whose name matches any keyword."""
    for kw in keywords:
        for c in df.columns:
            if kw in str(c).lower():
                return c
    for c in df.columns:
        if df[c].dtype == kind:
            return c
    return None


def show_config(**kwargs):
    """Print a run configuration."""
    print('-' * 58)
    for k, v in kwargs.items():
        print(f'{k:<28}{v}')
    print('-' * 58)