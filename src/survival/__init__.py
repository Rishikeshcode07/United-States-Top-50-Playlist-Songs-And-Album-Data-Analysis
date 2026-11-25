"""
src/survival  -  how long does a song stay on the chart? The tools of Phase 6B.

    lives.py       the definition of a stay, and the tables built from it (one row per song, one row per chart day)
    estimators.py  survival curve, hazard, median, restricted mean and log-rank test, with weights for the artist bootstrap
    hazard.py      the day-by-day hazard model (complementary log-log) with artist-aware standard errors, and its tests
    cox.py         a Cox model with ridge penalty, the concordance index (C-index) and the artist-grouped cross-validated risk scores
    plots.py       the chart helpers of the notebook

All of it is written with numpy, pandas and scipy only, and every function is covered by the tests in tests/test_survival_*.py.
"""
