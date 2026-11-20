"""
src/cluster  -  Phase 6C: grouping songs by the shape of their first month on the chart, and artists by their chart footprint.

WHAT THIS PACKAGE IS
    Four small modules. Everything is plain numpy and scipy, so the groups come out the same on every library version we tried.

        kmeans.py      k-means (k-means++ start, several starts), the silhouette, the adjusted Rand index, label matching
        stability.py   how firm a grouping is: re-draw whole artists, and hold whole artists out
        paths.py       the first month of every song as a row of numbers (the "chart score" of 30 days)
        artists.py     one row per artist (volume, staying power, strength, bursts) and a small PCA

WHY WE NEED IT
    A clustering always returns groups, whether the data hold groups or not. The tools here are what separates a pattern from an
    accident: every grouping is judged by how much it survives when the artists are re-drawn (songs of one artist are not
    independent) and when whole artists are held out.

MOTIVE
    Describe the chart in the past. No part of this package forecasts anything.
"""
