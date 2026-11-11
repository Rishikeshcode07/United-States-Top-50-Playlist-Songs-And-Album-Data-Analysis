# %% [md]
# Phase 6B - Survival analysis: how long does a song stay on the chart, and what goes with a short stay?

**Project:** United States Top 50 Playlist Performance and Song Popularity Trend Analysis (Atlantic Recording Corporation case)

**What this notebook does.** Phase 6A asked how much of what happens to a new song can be told from what was known early. This notebook asks the question the chart poses most directly: *for how long?* Take the {{n_songs:n}} songs that arrived on the chart after its first snapshot day (by {{n_artists:n}} artists; the {{day_one_songs}} songs that were already there on day one are left out, because we did not see them arrive). For each song, count the chart days of its first stay. Some stays are still going when the file ends ({{n_censored}} of {{n_songs}}); survival analysis is the family of methods built for exactly this: a stay that is still going is not thrown away and not treated as finished, it counts as "at least this long".

The notebook answers six questions, in this order. How many songs are still on the chart after 3, 7 and 30 chart days (Section 3)? Is the chance of leaving the same on every day (Section 3)? Which kinds of songs stay longer, and by how much (Section 4)? Which of those differences survive when the others are held fixed (Sections 5 and 6)? Which places on the chart are the dangerous ones (Section 7)? And how much of the difference between songs can be read from the arrival day (Section 8)? Section 9 repeats the main results under nine different choices, to see which of them depend on a choice.

**How the words "risk" and "hazard" are used.** The brief for this project is historical analytics, not prediction and not recommendation. The *hazard* of a chart day is the chance that it is the last day of a stay, given that the song reached it. A "risk score" in Section 8 is a measuring tool: it is what a model would have said on the arrival day, checked against what really happened, to find out how much the arrival day says. Every difference in this notebook is an association among these songs. A song that arrives with a whole album is not shown to leave sooner *because* of the album; the notebook measures how much sooner, and says what else it was held against.

**Before you run it.** Phases 1 to 5 must have run, so that `data/processed/` holds the Parquet tables (if not: `python -m src.etl.run_pipeline`). `reports/p4_facts.json` is used for one cross-check if it is there. Then restart the kernel and choose *Run All*. [[RUNTIME]] It writes 12 charts to `reports/figures/p6b_*.png` and thirteen files to `reports/`: `p6b_facts.json` (every number quoted in the text), `p6b_stays.csv` (one row per song), `p6b_survival_curves.csv`, `p6b_hazard_by_age.csv`, `p6b_further_stay.csv`, `p6b_group_comparison.csv`, `p6b_debut_alone.csv`, `p6b_hazard_ratios.csv`, `p6b_effects_by_age.csv`, `p6b_danger_zone.csv`, `p6b_ladder.csv`, `p6b_risk_thirds.csv` and `p6b_stress_tests.csv`. Always restart the kernel before running again: the container of numbers refuses to store the same name twice, which is how we know that no number is silently replaced.

**If your numbers differ a little.** [[DRIFT]]

**How to read it.** Every code cell has a short plain-English explanation above it in five parts: *what this code is*, *why we need it*, *what will happen*, *expected outcome* and *motive*. Every chart has a *What to notice* note below it, and the numbers in those notes are not typed by hand: the code stores each number it computes in `FACTS`, and the sentences are written from it.

**Ground rules used throughout.**

- *One definition of a stay.* A stay ends at the last chart day before the first absence of {{gap}} snapshots or more. Shorter absences are "short breaks" inside the stay. The clock counts chart days, the days the song is actually on the chart. Section 9 tries other rules.
- *Censored stays are kept.* A stay whose end was not seen is "at least this long", and every method here uses it that way.
- *The artist is the unit of doubt.* Songs of one artist arrive together, share a fan base and leave together. Every interval comes from drawing whole artists again {{n_boot:n}} times, and every test and standard error adds up the evidence of an artist before squaring it. The textbook version, which treats the songs as independent, is shown next to it where the difference is instructive.
- *Many tests, one family.* Where several tests ask the same kind of question (five groupings, four blocks of inputs, ten inputs), the p-values are adjusted together with the Holm method.
- *"Held fixed" means in this data.* A hazard ratio compares songs that are alike in the other inputs. It is a description of these {{n_songs:n}} songs, not a promise about what would happen if one input were changed.
- *Honest cross-validation.* The risk scores of Section 8 are made by models that never saw the artist of the song they score, exactly as in Phase 6A.

| Section | What it does | Figures |
|---|---|---|
| 1 | Setup and loading | - |
| 2 | The stays: the definition, and what it does to real songs | 1 |
| 3 | The survival curve and the hazard by age | 2-3 |
| 4 | Who stays longer: the groups | 4-5 |
| 5 | Debut place, arriving alone, and the single that does not protect | 6 |
| 6 | The hazard model: the inputs held against each other | 7-8 |
| 7 | The danger zone: the place on the chart | 9 |
| 8 | What can be read from the arrival day | 10-11 |
| 9 | Nine ways to change the rule or the songs | 12 |
| 10 | Saving the results and what Phase 6B adds up to | - |

# %% [md]
## Section 1 - Setup and loading

# %% [md]
**What this code is.** The import cell. It loads the Python libraries (pandas for tables, numpy for numbers, matplotlib for charts) and our own modules: `src/survival` (the definition of a stay, the survival and hazard estimators, the day-by-day hazard model, the Cox model and the chart helpers of this phase) together with the helpers of Phases 3 to 6A for paths, the chart look, the container of facts, the artist bootstrap and the saving of tables.

**Why we need it.** Everything below uses these names. The logic that matters (where a stay ends, how a curve and its interval are calculated, how a test allows for related songs) lives in `src/survival` and is covered by automated tests, so the notebook stays short and the formulas stay checked. The two path lines make `from src...` work whether Jupyter was started in the project folder or in the `notebooks` folder.

**What will happen.** Python imports the libraries, switches on the project look and creates `FACTS`, where every number that appears in a sentence is stored once (it refuses to store a name twice and refuses numbers that are not finite). It then sets the settings of the whole notebook: the rule of the stay (a gap of {{gap}} snapshots), the nine age windows of the hazard, and {{n_boot:n}} bootstrap draws of artists.

**Expected outcome.** Four printed lines (the versions of pandas, numpy and scipy, and the project folder) and no chart. If a line fails with `ModuleNotFoundError`, run `pip install -r requirements.txt` in the project folder and restart the kernel.

**Motive.** One setup, one look, one place for the numbers. From here to the end no number in a sentence is typed by hand.

# %%
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd
import scipy
from IPython.display import display
from matplotlib.patches import Patch

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))            # makes "from src..." work when the notebook starts inside the notebooks folder

from src.config import default_paths
from src.eda import style
from src.eda.findings import Facts
from src.eda.plots import clean_text, direct_label, make_fig, panel_title, save_fig, short, style_axes, swatch_legend
from src.inference.dependence import cluster_bootstrap_weights
from src.inference.multiplicity import holm
from src.inference.register import format_p, save_table
from src.inference.survival import kaplan_meier
from src.ml import plots as MP
from src.ml.dataset import ARTIST_FEATURES, DEBUT_FEATURES, EARLY_FEATURES, RELEASE_FEATURES, TIMING_FEATURES, build_modeling_table
from src.ml.explain import build_design
from src.ml.models import LOG_COLUMNS, SEED
from src.survival import plots as SP
from src.survival.cox import CoxScorer, concordance, concordance_pairs, cross_validated_risk, fit_cox, weighted_concordance
from src.survival.estimators import (life_table, logrank_k, median_further_stay, median_stay, restricted_mean, survival_grid, window_hazard,
                                     window_labels)
from src.survival.hazard import (age_windows, equal_effects_contrasts, fit_cloglog, selection, wald_test, window_columns, window_interactions)
from src.survival.lives import GAP, build_lives, chart_with_snapshots, drop_sizes, person_days

pd.set_option("display.max_columns", 40)             # show up to 40 columns instead of hiding some
pd.set_option("display.width", 200)                  # use a wide print area so that tables do not wrap
pd.set_option("display.float_format", "{:,.4g}".format)

style.apply_style()                                  # the project look: fonts, colours, hairline grid
FACTS = Facts()                                      # every number we quote in a sentence is stored here, once

N_BOOT = 1000                                        # artist-level bootstrap draws behind every interval
AGE_EDGES = [0, 1, 2, 3, 7, 14, 30, 60, 120, 10_000]        # the nine age windows of the hazard: day 1, day 2, day 3, days 4-7, ... day 121 and later
HORIZON = 30                                         # "still on the chart after 30 chart days" and "chart days used out of the first 30"
BAND_LABELS = ["1 to 10", "11 to 20", "21 to 30", "31 to 40", "41 to 50"]            # the five debut-place bands
FACTS["gap"] = GAP
FACTS["n_boot"] = N_BOOT
FACTS["horizon"] = HORIZON

print("pandas :", pd.__version__)
print("numpy  :", np.__version__)
print("scipy  :", scipy.__version__)
print("project:", PROJECT_ROOT)

# %% [md]
**What this code is.** The loading cell. It reads the four Parquet tables that the pipeline made: the songs, the chart rows (one per song per snapshot day), the credits (which artists are on which song) and the calendar. It stores the headline counts and compares them with Phase 4.

**Why we need it.** A model can only be as right as its data. If the pipeline had been changed since Phase 4, a quiet difference would flow into every number of this notebook. The comparison with `reports/p4_facts.json` is a seat belt: it stops the notebook, with a message that says what to run, if the chart is not the one that Phases 4 and 5 described.

**What will happen.** The code checks that the four files exist, reads them, lists the snapshot days, stores the numbers of rows, songs and snapshot days and the first and last date in `FACTS`, and prints a small table. If `p4_facts.json` is there it compares song-days, songs and snapshot days with Phase 4 and stops on any difference; if the file is not there it says that the comparison is skipped.

**Expected outcome.** A table with {{rows:n}} song-days, {{songs:n}} songs and {{snapshots:n}} snapshot days (from {{first_date}} to {{last_date}}), followed by the line "Phase 4 reproduced: the same numbers of song-days, songs and snapshot days."

**Motive.** Be sure that Phase 6B measures the same chart that Phases 4 and 5 described.

# %%
paths = default_paths()
FIG_DIR = paths.reports_dir / "figures"              # all charts of this notebook are saved here as PNG files

needed = ["dim_track", "fact_chart_daily", "bridge_track_artist", "dim_date"]
missing = [name for name in needed if not (paths.processed_dir / f"{name}.parquet").exists()]
assert not missing, f"Run the pipeline first:  python -m src.etl.run_pipeline   (missing: {missing})"

dim_track = pd.read_parquet(paths.processed_dir / "dim_track.parquet")                      # one row per song
fact_chart = pd.read_parquet(paths.processed_dir / "fact_chart_daily.parquet")              # one row per song per snapshot day
bridge = pd.read_parquet(paths.processed_dir / "bridge_track_artist.parquet")               # which artists are credited on which song
dim_date = pd.read_parquet(paths.processed_dir / "dim_date.parquet")                        # one row per calendar day

snapshots = dim_date[dim_date["has_snapshot"]]
FACTS["rows"] = len(fact_chart)
FACTS["songs"] = len(dim_track)
FACTS["snapshots"] = len(snapshots)
FACTS["first_date"], FACTS["last_date"] = f"{snapshots['date'].min():%d %B %Y}", f"{snapshots['date'].max():%d %B %Y}"

display(pd.DataFrame({"table": ["fact_chart_daily (song x day)", "dim_track (songs)", "bridge_track_artist (credits)", "snapshot days"],
                      "rows": [len(fact_chart), len(dim_track), len(bridge), len(snapshots)]}))

# ---- the numbers of Phase 4 must come out again, otherwise this notebook would measure a different chart
p4_file = paths.reports_dir / "p4_facts.json"
if p4_file.exists():
    p4 = json.loads(p4_file.read_text(encoding="utf-8"))
    for label, now, before in [("song-days", len(fact_chart), p4["rows"]), ("songs", len(dim_track), p4["songs"]),
                               ("snapshot days", len(snapshots), p4["snapshots"])]:
        assert now == before, f"{label}: {now} here, {before} in Phase 4. Re-run the pipeline and notebook 04 first."
    print("Phase 4 reproduced: the same numbers of song-days, songs and snapshot days.")
else:
    print("reports/p4_facts.json is not there (notebook 04 has not been run): the cross-check with Phase 4 is skipped.")
