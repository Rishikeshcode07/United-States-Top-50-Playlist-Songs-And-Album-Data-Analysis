# %% [md]
## Section 9 - Do the findings survive other reasonable choices?

Every result so far rests on choices that could have been made differently: how long an absence must be to end a stay (7 snapshot days), which songs are included, whether one block of songs that left together is counted as it happened. This section changes the choices, one at a time, and refits the hazard model of Section 6 each time. A finding that depends on one choice is not a finding about the chart.

Nine runs are made. Five change the **gap rule** (1, 3, 7 as used, 14 and 30 snapshot days of absence to end a stay). Four change the **songs**: the first half and the second half of the file, the stays that ended on {{busy_text}} treated as unfinished (as if that block of songs had been removed rather than having left), and the ten artists with the most songs left out. Everything else, the inputs and the way the intervals are made, stays as in Section 6.

# %% [md]
**What this code is.** The nine runs. A function takes the gap rule, the group of songs to keep and whether to treat the {{busy_text}} block as unfinished, rebuilds the stays and chart days under that rule, fits the day-by-day model with the same ten inputs and the artist-aware intervals, and also computes the Kaplan-Meier share still on the chart after 30 chart days and the median stay. The hazard ratios kept for each run are the four debut-place bands, the album drop and the single.

**Why we need it.** Section 6 gives one set of ratios for one set of choices. The question is whether the ones that matter (debut place, album drop) keep their size and direction, and whether the ones that did not (the single) stay unclear, when the choices change.

**What will happen.** The code fits nine models, one after the other. The run with the gap rule of 7 and all songs is the model of Section 6 itself, and the code checks that it reproduces its ratios. It prints a table of the nine runs and sets the facts: how many of the debut-place intervals stay clear of 1, the range of the ratios of the album drop and of the single, and how the 30-day share moves with the gap rule.

**Expected outcome.** All debut-place ratios above 1 and clear of 1 in every run, an album drop above 1 in every run and clear of 1 in most, a single that never separates from 1, and a share still on the chart after 30 days that moves with the gap rule.

**Motive.** Test the main findings against the choices that were the most open to doubt.

# %%
HALF = N_SNAPSHOTS // 2
ENTRY = STAYS["entry_snapshot"].to_numpy()
TOP_ARTISTS = STAYS.groupby("primary_artist").size().sort_values(ascending=False, kind="stable").index[:10]
ALL_SONGS = np.ones(len(STAYS), dtype=bool)
TOP_SONGS = STAYS["primary_artist"].isin(TOP_ARTISTS).to_numpy()
FACTS["top_artists_songs"] = int(TOP_SONGS.sum())
FACTS["half_snapshot"] = int(HALF)
RUNS = [("gap1", "Gap rule 1", "any absence ends the stay", 1, ALL_SONGS, False),
        ("gap3", "Gap rule 3", "3 snapshot days away end it", 3, ALL_SONGS, False),
        ("gap7", f"Gap rule {GAP}", "the rule used here", GAP, ALL_SONGS, False),
        ("gap14", "Gap rule 14", "14 snapshot days away end it", 14, ALL_SONGS, False),
        ("gap30", "Gap rule 30", "30 snapshot days away end it", 30, ALL_SONGS, False),
        ("first", "First half of the file", f"songs that arrived before snapshot day {HALF}", GAP, ENTRY < HALF, False),
        ("second", "Second half of the file", f"songs that arrived from snapshot day {HALF}", GAP, ENTRY >= HALF, False),
        ("block", f"Block of {FACTS['busy_text']}", "its stays counted as unfinished", GAP, ALL_SONGS, True),
        ("top10", "Without 10 big artists", f"the 10 artists with most songs out, {FACTS['top_artists_songs']} songs", GAP, ~TOP_SONGS, False)]
STRESS_INPUTS = {"Debut place 11 to 20": "b2", "Debut place 21 to 30": "b3", "Debut place 31 to 40": "b4", "Debut place 41 to 50": "b5", "Album drop": "drop", "Single": "single"}
MIN_EVENTS_LAST_WINDOW = 5
FACTS["min_events_last_window"] = MIN_EVENTS_LAST_WINDOW


def stress_run(gap, keep, block_unfinished):
    """Rebuild the stays under `gap`, keep the songs in `keep`, refit the Section 6 model and describe the run."""
    lives = (LIVES if gap == GAP else build_lives(CHART, STAYS["track_id"], N_SNAPSHOTS, gap=gap)).copy()
    if block_unfinished:                                                  # the stays whose last chart day is the busy exit day count as unfinished
        lives.loc[(lives["last_snapshot"] == BUSY) & lives["exit_seen"], "exit_seen"] = False
    days = person_days(CHART, lives)
    rows = pd.Index(STAYS["track_id"]).get_indexer(days["track_id"])
    use = keep[rows]
    rows, days = rows[use], days[use]
    edges = list(AGE_EDGES)
    while days.loc[days["age"] > edges[-2], "event"].sum() < MIN_EVENTS_LAST_WINDOW and len(edges) > 4:
        edges = edges[:-2] + [edges[-1]]                                  # too few exits in the last window: merge it into the one before
    base = window_columns(days["age"].to_numpy(), edges)
    design = np.column_stack([base.to_numpy(), SONG_X.to_numpy()[rows]])
    fit = fit_cloglog(days["event"].to_numpy(), design, clusters=ARTISTS[rows])
    hr_, lo_, hi_ = fit.hazard_ratios()
    names = list(base.columns) + INPUT_ORDER
    T_r, E_r = lives["stay"].to_numpy(dtype=float)[keep], lives["exit_seen"].to_numpy(dtype=bool)[keep]
    curve = survival_grid(T_r, E_r, np.arange(0, int(T_r.max()) + 1))
    out = {"songs": int(keep.sum()), "artists": int(len(np.unique(ARTISTS[keep]))), "exits": int(E_r.sum()), "censored": int((~E_r).sum()), "converged": bool(fit.converged),
           "windows": len(base.columns), "s30": float(curve[30]), "median": float(median_stay(curve)), "inputs": {}}
    for name, key in STRESS_INPUTS.items():
        i = names.index(name)
        out["inputs"][key] = (float(hr_[i]), float(lo_[i]), float(hi_[i]), float(fit.p_values()[i]))
    out["all_hr"] = {name: float(hr_[names.index(name)]) for name in INPUT_ORDER}
    return out


RESULTS = {}
for key, label, detail, gap, keep, block in RUNS:
    RESULTS[key] = stress_run(gap, keep, block)
    assert RESULTS[key]["converged"], f"the run {label} did not converge"
# the run with the gap rule of 7 and all songs is the model of Section 6: its ratios must be the same
FACTS["st_main_gap"] = float(max(abs(RESULTS["gap7"]["all_hr"][name] - float(hr[MAIN_NAMES.index(name)])) for name in INPUT_ORDER))
assert FACTS["st_main_gap"] < 1e-8, "the stress run with the usual choices does not reproduce the model of Section 6"
FACTS["st_runs"] = len(RUNS)
rows_out = []
for k, (key, label, detail, gap, keep, block) in enumerate(RUNS, start=1):
    res = RESULTS[key]
    for k_name, short in STRESS_INPUTS.items():
        value, low, high, p = res["inputs"][short]
        rows_out.append({"run": key, "run_label": label, "run_detail": detail, "songs": res["songs"], "artists": res["artists"], "exits": res["exits"], "censored": res["censored"],
                         "share_after_30_days": res["s30"], "median_stay": res["median"], "input": k_name, "hazard_ratio": value, "low": low, "high": high, "p": p})
        FACTS[f"st{k}_{short}"], FACTS[f"st{k}_{short}_low"], FACTS[f"st{k}_{short}_high"] = value, low, high
    FACTS[f"st{k}_songs"], FACTS[f"st{k}_artists"], FACTS[f"st{k}_exits"] = res["songs"], res["artists"], res["exits"]
    FACTS[f"st{k}_s30"], FACTS[f"st{k}_median"], FACTS[f"st{k}_windows"] = res["s30"], res["median"], res["windows"]
STRESS_TABLE = pd.DataFrame(rows_out)
display(pd.DataFrame({"run": [f"{label}" for _, label, *_ in RUNS], "songs": [RESULTS[k]["songs"] for k, *_ in RUNS], "exits": [RESULTS[k]["exits"] for k, *_ in RUNS],
                      "30-day share": [f"{100 * RESULTS[k]['s30']:.0f}%" for k, *_ in RUNS], "median": [f"{RESULTS[k]['median']:.0f}" for k, *_ in RUNS],
                      "debut 31-40": [f"{RESULTS[k]['inputs']['b4'][0]:.2f}  ({RESULTS[k]['inputs']['b4'][1]:.2f} to {RESULTS[k]['inputs']['b4'][2]:.2f})" for k, *_ in RUNS],
                      "album drop": [f"{RESULTS[k]['inputs']['drop'][0]:.2f}  ({RESULTS[k]['inputs']['drop'][1]:.2f} to {RESULTS[k]['inputs']['drop'][2]:.2f})" for k, *_ in RUNS],
                      "single": [f"{RESULTS[k]['inputs']['single'][0]:.2f}  ({RESULTS[k]['inputs']['single'][1]:.2f} to {RESULTS[k]['inputs']['single'][2]:.2f})" for k, *_ in RUNS]}).set_index("run"))

# ---- the findings, counted
debut_cells = [RESULTS[k]["inputs"][s] for k, *_ in RUNS for s in ("b2", "b3", "b4", "b5")]
FACTS["st_debut_total"] = len(debut_cells)
FACTS["st_debut_above_one"] = int(sum(v > 1 for v, lo, hi, p in debut_cells))
FACTS["st_debut_clear"] = int(sum(lo > 1 for v, lo, hi, p in debut_cells))
FACTS["st_debut_min"], FACTS["st_debut_max"] = float(min(v for v, *_ in debut_cells)), float(max(v for v, *_ in debut_cells))
drop_cells = [RESULTS[k]["inputs"]["drop"] for k, *_ in RUNS]
FACTS["st_drop_above_one"] = int(sum(v > 1 for v, lo, hi, p in drop_cells))
FACTS["st_drop_clear"] = int(sum(lo > 1 for v, lo, hi, p in drop_cells))
FACTS["st_drop_min"], FACTS["st_drop_max"] = float(min(v for v, *_ in drop_cells)), float(max(v for v, *_ in drop_cells))
unclear = [label for (k, label, *_), (v, lo, hi, p) in zip(RUNS, drop_cells) if lo <= 1]
FACTS["st_drop_unclear"] = ", ".join(unclear).lower() if unclear else "none"
single_cells = [RESULTS[k]["inputs"]["single"] for k, *_ in RUNS]
FACTS["st_single_clear"] = int(sum(hi < 1 or lo > 1 for v, lo, hi, p in single_cells))
FACTS["st_single_min"], FACTS["st_single_max"] = float(min(v for v, *_ in single_cells)), float(max(v for v, *_ in single_cells))
FACTS["st_single_low_min"], FACTS["st_single_high_max"] = float(min(lo for v, lo, hi, p in single_cells)), float(max(hi for v, lo, hi, p in single_cells))
gap_runs = [RESULTS[k]["s30"] for k in ("gap1", "gap3", "gap7", "gap14", "gap30")]
FACTS["st_s30_gap_min"], FACTS["st_s30_gap_max"] = float(min(gap_runs)), float(max(gap_runs))
subset_runs = [RESULTS[k]["s30"] for k in ("first", "second", "block", "top10")]
FACTS["st_s30_subset_min"], FACTS["st_s30_subset_max"] = float(min(subset_runs)), float(max(subset_runs))
FACTS["st_merged_runs"] = int(sum(RESULTS[k]["windows"] < 9 for k, *_ in RUNS))
FACTS["st_merged_names"] = ", ".join(label.lower() for k, label, *_ in RUNS if RESULTS[k]["windows"] < 9) or "none"
print(f"all {FACTS['st_debut_total']} debut-place ratios: smallest {FACTS['st_debut_min']:.2f}, largest {FACTS['st_debut_max']:.2f}; {FACTS['st_debut_clear']} of {FACTS['st_debut_total']} intervals exclude 1")
print(f"album drop: {FACTS['st_drop_min']:.2f} to {FACTS['st_drop_max']:.2f}, interval excludes 1 in {FACTS['st_drop_clear']} of {FACTS['st_runs']}; not clear in: {FACTS['st_drop_unclear']}")
print(f"single: {FACTS['st_single_min']:.2f} to {FACTS['st_single_max']:.2f}, interval clear of 1 in {FACTS['st_single_clear']} runs; runs with merged last age window: {FACTS['st_merged_runs']}")

# %% [md]
### Figure 12: the nine runs side by side

# %% [md]
**What this code is.** The drawing of Figure 12. One row for each of the nine runs, in four panels that share the rows: the hazard ratio of a debut at places 31 to 40 against a top-10 debut, the hazard ratio of the album drop (for one standard deviation more), the hazard ratio of the single against the album track, and the share of songs still on the chart after 30 chart days. A dot is filled when the 95% artist-aware interval of the ratio is clear of 1 and hollow when it includes 1.

**Why we need it.** The table is exact but hides the pattern. Side by side, the reader sees at once which ratios stay on the same side of 1 whatever is changed, and where the intervals widen.

**What will happen.** Ratios on a log scale with a line at 1, intervals as bars, the value next to each, the run used in the rest of the notebook shaded, and the picture saved as `p6b_12_stress_tests.png`.

**Expected outcome.** In the first panel nine filled dots, all well to the right of 1; in the second nine filled dots a little right of 1; in the third nine hollow dots around 1; and in the last a share that rises with the gap rule.

**Motive.** Show that the main findings do not hang on one choice, and say where the size of an effect does change.

# %%
# Figure 12 - the nine runs
Y_RUN = np.array([0, 1, 2, 3, 4, 5.8, 6.8, 7.8, 8.8])
RUN_LABELS = [f"{label}\n({detail})" for _, label, detail, *_ in RUNS]
fig, axes = make_fig(11.8, 6.6, 1, 4, left=2.95, right=0.5, bottom=0.95, wspace=0.1, width_ratios=[1.15, 1, 1, 0.85], panel_titles=True,
                     title=f"The findings hold in all {FACTS['st_runs']} runs: every debut-place ratio is above 1 and clear of it ({FACTS['st_debut_clear']} of {FACTS['st_debut_total']} intervals), "
                           f"the album drop is clear of 1 in {FACTS['st_drop_clear']} of {FACTS['st_runs']}, and the single never separates from 1",
                     subtitle="The day-by-day hazard model of Section 6 refitted for each run; ratios with 95% artist-aware intervals. Filled dot: interval clear of 1; hollow: it includes 1. "
                              "The debut-place panel shows 31 to 40 against 1 to 10 (the other bands are in the table). The share after 30 chart days has no band.",
                     note_text=f"The shaded row is the choice used in the rest of the notebook. The first-half run has {FACTS['st6_songs']} songs and the second-half run {FACTS['st7_songs']}; "
                               f"the debut-place ratio is {FACTS['st6_b4']:.1f} in the first and {FACTS['st7_b4']:.1f} in the second.")
panels = [("b4", "Debut at 31 to 40", "hazard ratio (log scale)", (0.7, 16), [1, 2, 4, 8]),
          ("drop", "Album drop, per SD", "hazard ratio (log scale)", (0.8, 2.6), [1, 1.5, 2]),
          ("single", "Single vs album track", "hazard ratio (log scale)", (0.5, 2.1), [0.5, 1, 2])]
for ax, (short, title, xlabel, xlim, ticks) in zip(axes[:3], panels):
    ax.axvline(1.0, color=style.INK_2, linewidth=1.0, zorder=1)
    ax.axhspan(2 - 0.5, 2 + 0.5, color=style.AXIS, alpha=0.16, zorder=0)
    for yi, (key, *_) in zip(Y_RUN, RUNS):
        value, low, high, p = RESULTS[key]["inputs"][short]
        ax.hlines(yi, low, high, color=style.BLUE, linewidth=4.5, alpha=0.45, zorder=2)
        ax.scatter([value], [yi], s=46, facecolor=style.BLUE if low > 1 or high < 1 else style.SURFACE, edgecolor=style.BLUE, linewidth=1.6, zorder=4)
        direct_label(ax, high * 1.07, yi, f"{value:.2f}", size=8.6, color=style.INK)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel)
    panel_title(ax, title)
ax = axes[3]
ax.axhspan(2 - 0.5, 2 + 0.5, color=style.AXIS, alpha=0.16, zorder=0)
for yi, (key, *_) in zip(Y_RUN, RUNS):
    ax.hlines(yi, 0, RESULTS[key]["s30"], color=style.BLUE, linewidth=1.4, alpha=0.55, zorder=2)
    ax.scatter([RESULTS[key]["s30"]], [yi], s=46, color=style.BLUE, edgecolor=style.SURFACE, linewidth=1.2, zorder=4)
    direct_label(ax, RESULTS[key]["s30"] + 0.012, yi, f"{100 * RESULTS[key]['s30']:.0f}%", size=8.6, color=style.INK)
ax.set_xlim(0, 0.34)
ax.xaxis.set_major_formatter(SP.percent_axis)
ax.set_xlabel("still on the chart at day 30")
panel_title(ax, "Level of the curve")
HEADING = dict(facecolor=style.SURFACE, edgecolor="none", pad=2.0)
for ax in axes:
    ax.set_ylim(9.5, -1.3)
    style_axes(ax, grid="x", left_spine=False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(Y_RUN)
    ax.set_yticklabels(RUN_LABELS if ax is axes[0] else [], fontsize=8.8)
axes[0].text(0.0, -0.85, "Changing the stay rule", transform=axes[0].get_yaxis_transform(), fontsize=9.2, fontweight="bold", color=style.INK, va="center", ha="left", bbox=HEADING, zorder=6)
axes[0].text(0.0, 5.0, "Changing the songs", transform=axes[0].get_yaxis_transform(), fontsize=9.2, fontweight="bold", color=style.INK, va="center", ha="left", bbox=HEADING, zorder=6)
save_fig(fig, "p6b_12_stress_tests", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- The debut place survives every change. All {{st_debut_total}} debut-place ratios (four bands in each of the {{st_runs}} runs) are above 1, {{st_debut_clear}} of the {{st_debut_total}} intervals are clear of 1, and the smallest ratio is {{st_debut_min:f2}}. The size moves more than the direction. For a debut at 31 to 40 it runs from {{st6_b4:f1}} in the first half of the file ({{st6_b4_low:f1}} to {{st6_b4_high:f1}}) to {{st7_b4:f1}} in the second half ({{st7_b4_low:f1}} to {{st7_b4_high:f1}}), two intervals that overlap only in their outer parts. That hints that the debut place mattered more later in the file; this section does not test it, and one comparison of two halves is a hint, not a finding.
- The album drop keeps its direction and its clearness: {{st_drop_min:f2}} to {{st_drop_max:f2}} times the hazard for one standard deviation more album drop, with an interval clear of 1 in {{st_drop_clear}} of the {{st_runs}} runs. It grows with the gap rule, from {{st1_drop:f2}} when any absence ends the stay to {{st5_drop:f2}} when 30 days away are needed; a short rule splits a returning song into two stays, and the pattern suggests that the album drop goes with leaving for good more than with leaving for a few days. That reading is a guess.
- The single never separates from 1: its ratio runs from {{st_single_min:f2}} to {{st_single_max:f2}} and its interval includes 1 in every run ({{st_single_clear}} of {{st_runs}} are clear of it). It falls on both sides of 1, which fits no effect, but the intervals reach from {{st_single_low_min:f2}} to {{st_single_high_max:f2}}, so a small effect in either direction cannot be ruled out.
- The gap rule moves the level of the curve, not the findings. The share still on the chart after 30 chart days goes from {{st_s30_gap_min:pct0}} when any absence ends the stay to {{st_s30_gap_max:pct0}} when 30 days away are needed, because a song that disappears for a few days and returns is one stay under a long rule and two under a short one. The other four runs fall between {{st_s30_subset_min:pct0}} and {{st_s30_subset_max:pct0}}. The conclusions of this notebook are about ratios and order; the exact level of the curve depends on the rule, as Section 3 said.
- Counting the {{busy_text}} block as unfinished changes little: the debut-place ratio at 31 to 40 is {{st8_b4:f2}} against {{st3_b4:f2}} in the main run, and the album drop {{st8_drop:f2}} against {{st3_drop:f2}}. Leaving out the ten artists with the most songs ({{top_artists_songs}} songs) gives {{st9_b4:f2}} and {{st9_drop:f2}}: the results do not rest on a few large artists.
- What the nine runs do not vary: the inputs, the one-ratio form of the model (Section 6 found that five of the ten inputs act differently by age), the definition of an exit, and the way intervals treat artists (not days). In {{st_merged_runs}} run ({{st_merged_names}}) the last age window held fewer than {{min_events_last_window}} exits and was merged with the window before it.
