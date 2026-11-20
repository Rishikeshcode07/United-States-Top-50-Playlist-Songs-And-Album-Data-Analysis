"""
src/eda/concentration.py  -  how concentrated is the chart? Gini, HHI, Lorenz curve and top-N share.

WHAT THIS FILE IS
    Small, tested functions that turn a list of non-negative numbers (for example the chart points of every artist) into the
    measures an analyst uses to describe concentration:

        shares(values)             every holder's share of the total (the shares add up to 1)
        hhi(values)                Herfindahl-Hirschman Index: the sum of the squared percentage shares, from 0 to 10,000
        effective_number(values)   10,000 / HHI. "The chart behaves as if this many equal artists shared it"
        gini(values)               0 = everybody holds the same amount, close to 1 = one holder has almost everything
        lorenz_curve(values)       the points of the Lorenz curve (share of holders against share of the total)
        top_n_share(values, n)     the share held by the n biggest holders
        cumulative_share(values)   the running total of the shares, biggest holder first (the Pareto curve)
        concentration_summary()    all of the above in one dictionary
        period_shares()            the share of every holder in every period (for example every month)
        period_concentration()     HHI, effective number and top-1 share of every period

WHY WE NEED IT
    The brief asks for "artist dominance over time". "Taylor Swift has 7.5 % of the chart points" is a fact about one artist.
    "Ten artists hold 41 % and the Gini is 0.68" is a fact about the whole chart, and it can be compared month by month.
    The functions are short on purpose: every formula can be checked by hand, and the tests do exactly that.

WHAT HAPPENS WHEN IT RUNS
    Nothing by itself. The EDA notebook (notebooks/04_eda_playlist_song_artist.ipynb) imports these functions.
    Zeros are allowed and count as holders that own nothing: they raise the Gini and do not change the HHI.
    Negative numbers, missing numbers and a total of zero raise a ValueError, because no share can be defined for them.

EXPECTED OUTCOME
    hhi([1, 1, 1, 1]) -> 2500.0     gini([1, 1, 1, 1]) -> 0.0     effective_number([1, 1, 1, 1]) -> 4.0
    hhi([10]) -> 10000.0            gini([0, 0, 0, 10]) -> 0.75   (the largest value a Gini can take with four holders)

MOTIVE
    One definition of "concentration" for the notebook, the Streamlit app, the research paper and the Power BI DAX measures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HHI_SCALE = 10_000.0          # shares are written in percent, so one holder with everything gives 100 x 100 = 10,000


def _as_values(values):
    """Turn a list, numpy array or pandas Series into a clean float array, or explain what is wrong with it."""
    array = np.asarray(values, dtype="float64")
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional list of numbers")
    if np.isnan(array).any():
        raise ValueError("values must not contain missing numbers")
    if (array < 0).any():
        raise ValueError("values must not be negative")
    if array.sum() <= 0:
        raise ValueError("values must add up to more than zero")
    return array


def shares(values):
    """Each holder's share of the total, as numbers between 0 and 1 that add up to 1."""
    array = _as_values(values)
    return array / array.sum()


def hhi(values, scale=HHI_SCALE):
    """Herfindahl-Hirschman Index. Four equal holders -> 2,500. One holder -> 10,000. A thousand equal holders -> 10."""
    return float((shares(values) ** 2).sum() * scale)


def effective_number(values):
    """The "numbers equivalent": how many EQUAL holders would give the same HHI. Always between 1 and the number of holders."""
    return float(1.0 / (shares(values) ** 2).sum())


def gini(values):
    """
    Gini coefficient from the sorted values:  G = 2 * sum(i * x_i) / (n * sum(x)) - (n + 1) / n   for x sorted ascending, i = 1..n.
    0 means perfect equality. With n holders the largest possible value is (n - 1) / n (one holder owns everything).
    """
    array = np.sort(_as_values(values))
    n = array.size
    ranks = np.arange(1, n + 1)
    return float(2.0 * (ranks * array).sum() / (n * array.sum()) - (n + 1.0) / n)


def lorenz_curve(values):
    """
    The Lorenz curve: x = share of holders (poorest first), y = share of the total they own.
    Both arrays start at (0, 0) and end at (1, 1). The straight line from (0, 0) to (1, 1) is perfect equality.
    """
    array = np.sort(_as_values(values))
    n = array.size
    x = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    y = np.concatenate([[0.0], np.cumsum(array) / array.sum()])
    return x, y


def top_n_share(values, n):
    """The share of the total held by the n biggest holders (n larger than the number of holders gives 1.0)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    array = np.sort(_as_values(values))[::-1]
    return float(array[:n].sum() / array.sum())


def cumulative_share(values):
    """Running total of the shares, biggest holder first. The last number is always 1.0. This is the Pareto curve."""
    array = np.sort(_as_values(values))[::-1]
    return np.cumsum(array) / array.sum()


def concentration_summary(values, top_ns=(1, 5, 10, 20)):
    """Everything above in one dictionary: holders, hhi, effective_number, gini and top_<n>_share for every n in top_ns."""
    array = _as_values(values)
    summary = {
        "holders": int(array.size),
        "holders_with_a_share": int((array > 0).sum()),
        "hhi": hhi(array),
        "effective_number": effective_number(array),
        "gini": gini(array),
    }
    for n in top_ns:
        summary[f"top_{n}_share"] = top_n_share(array, n)
    return summary


def period_shares(frame, period_col, holder_col, value_col):
    """
    One row per (period, holder) with the value earned and the holder's share of that period in PERCENT.
    Holders with nothing in a period get no row. Example: period_col="year_month", holder_col="primary_artist",
    value_col="chart_points" gives every artist's share of the chart points of every month.
    """
    table = (frame.groupby([period_col, holder_col], observed=True)[value_col].sum().rename("value").reset_index())
    table = table[table["value"] > 0].copy()
    table["share_pct"] = table["value"] / table.groupby(period_col, observed=True)["value"].transform("sum") * 100.0
    return table.reset_index(drop=True)


def period_concentration(shares_table, period_col, holder_col):
    """
    From the output of `period_shares`: one row per period with the number of holders, the HHI (sum of squared percent
    shares), the effective number of holders (10,000 / HHI) and the biggest single share in percent.
    """
    grouped = shares_table.groupby(period_col, observed=True)
    summary = pd.DataFrame({
        "holders": grouped[holder_col].size(),
        "hhi": grouped["share_pct"].apply(lambda s: float((s ** 2).sum())),
        "top1_pct": grouped["share_pct"].max(),
    })
    summary["effective_number"] = HHI_SCALE / summary["hhi"]
    return summary.reset_index()
