# %% [md]
## Section 8 - How much of "who stays longer" can be read early?

Sections 4 to 7 describe how stays differ. This section asks how much of that difference could have been read at the time. Two moments are used: the **arrival day** (what is known the day a song first appears) and the **end of the third chart day** (what is known after three days of chart behaviour).

The measure is the **C-index**. Take any two songs for which it is known which one left first. A model that gives every song a risk score (a higher score means "leaves sooner") is right on that pair when the song that left first has the higher score. The C-index is the share of such pairs in which the model is right: 0.5 is a coin flip, 1.0 is perfect. It uses the unfinished stays too (a pair is used only when the order of leaving is known), and it needs no cut-off.

The scores come from a Cox model, which turns the inputs into one risk number for each song. Every score in this section is made by a model that has **never seen the song's artist**: the songs are split into 5 parts by artist, each part is scored by a model fitted on the other 4, and the whole procedure is repeated with 3 different splits. Without this, the model could score well by recognising artists.

Two limits belong here at the start. The Cox model gives each input one hazard ratio for the whole stay, and Section 6 found that five of the ten inputs act differently at different ages, so what follows is the one-ratio version; a model with age-varying ratios was not tried and might rank somewhat better. And this measures how much the early information carried about the order of exits in 2024 and 2025; it is not a promise about future songs.

# %% [md]
### The ladder on the arrival day

# %% [md]
**What this code is.** A ladder of what is known on the arrival day, with four sets of inputs added one at a time: the **release** (length, number of tracks, single or album, explicit, collaboration), the **timing** (weekday of the arrival and the days of history in the file), the **artist's record** (earlier chart days, songs, top-10 days, best place, usual popularity, on the chart the day before) and the **debut day** (place, popularity and the size of the album drop). For each step the code makes a risk score for every song with models that never saw the artist, computes the C-index for each of the 3 splits, puts a 95% band on it by resampling artists, and also computes the C-index of the same model on the songs it was fitted on.

**Why we need it.** A hazard ratio in Section 6 says an input goes with a higher hazard. It does not say how much of the difference between songs that is. The ladder says how much each set of inputs adds to the ability to put two songs in the order they left, and a step that adds nothing shows up as a gain of zero.

**What will happen.** For each of the four steps the code fits 15 Cox models (5 parts, 3 splits), each on the songs of other artists only, and scores the held-out part. It then finds the C-index among the {{n_pairs:n}} pairs of songs for which the order of leaving is known, the 95% band from the {{n_boot:n}} artist resamples, and the gain of each step over the one above. One table is printed.

**Expected outcome.** A C-index near 0.55 for the release alone, almost no gain from timing or artist history, a jump to about 0.64 when the debut day is added, and a higher value (near 0.70) when the model is scored on the songs it was fitted on, which shows how much a model can flatter itself.

**Motive.** Put a number on how much of "who stays longer" can be read on the arrival day, scored the honest way.

# %%
LADDER_STEPS = [("Release", RELEASE_FEATURES, "single, length, explicit, collaboration"), ("+ Timing", TIMING_FEATURES, "weekday, days of history"),
                ("+ Artist history", ARTIST_FEATURES, "earlier chart record"), ("+ Debut day", DEBUT_FEATURES + ["log_drop"], "place, popularity, album drop")]
N_SPLITS, N_REPEATS, RIDGE = 5, 3, 1.0
PAIRS = concordance_pairs(T, E)
FACTS["n_pairs"] = int(len(PAIRS[0]))


def run_ladder(frame, time_, event, artists, pairs, boot_weights, steps, cumulative=True):
    """
    For each step (name, columns, description): out-of-fold scores, C-index of each split, C-index draws (mean over the splits), in-sample C-index.
    With cumulative=True every step adds its columns to those of the steps before it; otherwise every step lists all of its own columns.
    """
    out, so_far = [], []
    for name, columns, _ in steps:
        so_far = so_far + list(columns) if cumulative else list(columns)
        scores = cross_validated_risk(frame, time_, event, artists, so_far, LOG_COLUMNS, ridge=RIDGE, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED)
        repeats = np.array([concordance(time_, event, scores[r]) for r in range(N_REPEATS)])
        draws = np.mean([weighted_concordance(pairs, scores[r], boot_weights) for r in range(N_REPEATS)], axis=0)
        in_sample = concordance(time_, event, CoxScorer(so_far, LOG_COLUMNS, RIDGE).fit(frame, time_, event).score(frame))
        out.append({"name": name, "inputs": len(so_far), "scores": scores, "repeats": repeats, "draws": draws, "in_sample": float(in_sample)})
    return out


ARRIVAL_RUNGS = run_ladder(STAYS, T, E, ARTISTS, PAIRS, W_BOOT, LADDER_STEPS)
OOF = {r["name"]: r["scores"] for r in ARRIVAL_RUNGS}
for k, rung in enumerate(ARRIVAL_RUNGS, start=1):
    low, high = np.nanpercentile(rung["draws"], [2.5, 97.5])
    FACTS[f"c{k}"], FACTS[f"c{k}_low"], FACTS[f"c{k}_high"] = float(rung["repeats"].mean()), float(low), float(high)
    FACTS[f"c{k}_min"], FACTS[f"c{k}_max"], FACTS[f"c{k}_in"] = float(rung["repeats"].min()), float(rung["repeats"].max()), rung["in_sample"]
    FACTS[f"c{k}_inputs"] = rung["inputs"]
    if k > 1:
        gain_draws = rung["draws"] - ARRIVAL_RUNGS[k - 2]["draws"]
        FACTS[f"gain{k}"] = float(rung["repeats"].mean() - ARRIVAL_RUNGS[k - 2]["repeats"].mean())
        FACTS[f"gain{k}_low"], FACTS[f"gain{k}_high"] = (float(v) for v in np.nanpercentile(gain_draws, [2.5, 97.5]))
FACTS["c_range_max"] = float(max(r["repeats"].max() - r["repeats"].min() for r in ARRIVAL_RUNGS))
FACTS["c4_wrong"] = 1.0 - FACTS["c4"]


def ladder_rows(view, rungs, descriptions, base_index=None):
    """Table rows of one ladder; the gain of a rung is taken over the rung given by `base_index` (default: the rung above)."""
    rows = []
    for k, (rung, description) in enumerate(zip(rungs, descriptions)):
        low, high = np.nanpercentile(rung["draws"], [2.5, 97.5])
        row = {"view": view, "step": rung["name"], "detail": description, "inputs": rung["inputs"], "c_index": float(rung["repeats"].mean()), "low": float(low), "high": float(high),
               "split_min": float(rung["repeats"].min()), "split_max": float(rung["repeats"].max()), "c_index_in_sample": rung["in_sample"], "gain": np.nan, "gain_low": np.nan, "gain_high": np.nan}
        base = k - 1 if base_index is None else base_index
        if base is not None and base >= 0 and base != k:
            gain_draws = rung["draws"] - rungs[base]["draws"]
            row["gain"] = float(rung["repeats"].mean() - rungs[base]["repeats"].mean())
            row["gain_low"], row["gain_high"] = (float(v) for v in np.nanpercentile(gain_draws, [2.5, 97.5]))
        rows.append(row)
    return rows


ARRIVAL_ROWS = ladder_rows("arrival day", ARRIVAL_RUNGS, [d for _, _, d in LADDER_STEPS])


def show_ladder(rows):
    shown = pd.DataFrame(rows)
    return pd.DataFrame({"step": shown["step"], "inputs": shown["inputs"], "C-index": shown["c_index"].map("{:.3f}".format), "95% band": [f"{a:.3f} to {b:.3f}" for a, b in zip(shown["low"], shown["high"])],
                         "range over the 3 splits": [f"{a:.3f} to {b:.3f}" for a, b in zip(shown["split_min"], shown["split_max"])], "same model on its own songs": shown["c_index_in_sample"].map("{:.3f}".format),
                         "gain": ["" if np.isnan(g) else f"{g:+.3f}  ({a:+.3f} to {b:+.3f})" for g, a, b in zip(shown["gain"], shown["gain_low"], shown["gain_high"])]}).set_index("step")


display(show_ladder(ARRIVAL_ROWS))

# %% [md]
**What to notice**

- Read the C-index as a share of pairs: of the {{n_pairs:n}} pairs of songs for which the order of leaving is known, a model that has only the release facts puts the earlier leaver first in {{c1:pct0}} of them (95% band {{c1_low:pct0}} to {{c1_high:pct0}}); 50% would be a coin flip. Timing and artist history add {{gain2:sf3}} and {{gain3:sf3}} to the C-index, and the bands of both gains ({{gain2_low:sf3}} to {{gain2_high:sf3}} and {{gain3_low:sf3}} to {{gain3_high:sf3}}) include zero: the data show no gain from either.
- The debut day is the step that matters. It adds {{gain4:sf3}} ({{gain4_low:sf3}} to {{gain4_high:sf3}}, a band that stays clear of zero) and takes the C-index to {{c4:f3}}: the model orders {{c4:pct0}} of the pairs correctly. In Section 6 the debut place and the size of the album drop were the two inputs clearly tied to the hazard; here they are the step that changes the ranking.
- The honest figure is lower than the flattering one. Scored on the songs it was fitted on, the full model gets {{c4_in:f3}} against {{c4:f3}} on artists it had not seen, and the gap widens with every block of inputs (the artist-history rung: {{c3_in:f3}} against {{c3:f3}}). This is why every score here comes from models that never saw the artist.
- The three splits tell the same story: the C-index of a rung differs by no more than {{c_range_max:f3}} between splits, much less than the size of the debut-day step.
- A C-index of {{c4:f2}} is a modest ranking, not a crystal ball: about {{c4:pct0}} of the pairs right means {{c4_wrong:pct0}} wrong. It tells how much the arrival day carries, and the answer is "some, nearly all of it from the debut day".

# %% [md]
### The landmark: what is known after three chart days

# %% [md]
**What this code is.** The same ladder after the third chart day. It uses the {{lm_songs}} songs (of {{n_songs}}) that had at least three chart days in a row at the start; for them the first three chart days exist. For these songs the clock starts at the third chart day (a song whose third day was its last has a clock of 1; shifting the clock changes no order, so no C-index). Three scores are compared: the arrival-day information as before, the arrival-day information plus the first three chart days (the place on day 2, the move over the three days, the best place, the popularity on day 3 and its change), and the first three chart days alone. The method is the same: models that never saw the artist, 3 splits, artist resampling.

**Why we need it.** By the end of the third chart day a song has shown how it behaves: climbing or sliding, holding its popularity or losing it. Sections 3 and 6 found that most of the danger sits in the first three chart days, so these days may say more about what follows than the debut day does.

**What will happen.** The code fits the models on the {{lm_songs}} songs, prints a table like the one above, and finds the gain from adding the first three chart days to the arrival-day information, and what the arrival-day information adds on top of the first three days.

**Expected outcome.** A clear rise, from about 0.6 for the arrival-day information to about 0.75 once the first three chart days are included, with the first three days alone nearly as good as everything together.

**Motive.** Show how fast the picture of a stay sharpens once the song's own chart behaviour is seen. The two ladders rank different groups of songs (the second leaves out the songs that left within two chart days), so their numbers are not compared as if one model improved on the other.

# %%
LM = (STAYS["first_run_days"] >= 3).to_numpy()
FACTS["lm_songs"] = int(LM.sum())
T_LM, E_LM, FRAME_LM, ART_LM = T[LM] - 2, E[LM], STAYS[LM].reset_index(drop=True), ARTISTS[LM]
PAIRS_LM = concordance_pairs(T_LM, E_LM)
FACTS["n_pairs_lm"] = int(len(PAIRS_LM[0]))
ARRIVAL_COLUMNS = RELEASE_FEATURES + TIMING_FEATURES + ARTIST_FEATURES + DEBUT_FEATURES + ["log_drop"]
LM_STEPS = [("Arrival-day information", ARRIVAL_COLUMNS, "all four sets above"), ("+ first three chart days", ARRIVAL_COLUMNS + EARLY_FEATURES, "place, move, popularity"),
            ("First three chart days alone", EARLY_FEATURES, "without the arrival-day facts")]
LM_RUNGS = run_ladder(FRAME_LM, T_LM, E_LM, ART_LM, PAIRS_LM, W_BOOT[:, LM], LM_STEPS, cumulative=False)      # every rung lists all of its own columns
OOF_LM = {r["name"]: r["scores"] for r in LM_RUNGS}
LM_ROWS = ladder_rows("after three chart days", LM_RUNGS, [d for _, _, d in LM_STEPS])     # the second rung's gain is over the first: what the first three chart days add
for key in ("gain", "gain_low", "gain_high"):
    LM_ROWS[2][key] = np.nan                                                      # the third rung (three days alone) is not a step up from the second
for k, rung in enumerate(LM_RUNGS, start=1):
    low, high = np.nanpercentile(rung["draws"], [2.5, 97.5])
    FACTS[f"lm{k}"], FACTS[f"lm{k}_low"], FACTS[f"lm{k}_high"] = float(rung["repeats"].mean()), float(low), float(high)
    FACTS[f"lm{k}_in"], FACTS[f"lm{k}_min"], FACTS[f"lm{k}_max"] = rung["in_sample"], float(rung["repeats"].min()), float(rung["repeats"].max())
gain_lm = LM_RUNGS[1]["draws"] - LM_RUNGS[0]["draws"]                      # what the first three chart days add to the arrival-day information
extra_lm = LM_RUNGS[1]["draws"] - LM_RUNGS[2]["draws"]                     # what the arrival-day information adds to the first three days
FACTS["lm_gain"], FACTS["lm_gain_low"], FACTS["lm_gain_high"] = float(LM_RUNGS[1]["repeats"].mean() - LM_RUNGS[0]["repeats"].mean()), *(float(v) for v in np.nanpercentile(gain_lm, [2.5, 97.5]))
FACTS["lm_extra"], FACTS["lm_extra_low"], FACTS["lm_extra_high"] = float(LM_RUNGS[1]["repeats"].mean() - LM_RUNGS[2]["repeats"].mean()), *(float(v) for v in np.nanpercentile(extra_lm, [2.5, 97.5]))
display(show_ladder(LM_ROWS))

# %% [md]
### Figure 10: how much each step adds

# %% [md]
**What this code is.** The drawing of Figure 10. The rows are the seven models of the two ladders: four on the arrival day and three after three chart days. The left panel shows the C-index of each (dot, with its 95% artist band as a line, and the same model on its own songs as a hollow circle). The right panel, on the same rows, shows what each step adds to the row above it, with its band.

**Why we need it.** A table of C-indices hides what matters, which is the size of each step next to its uncertainty. On the right the question "does this step add anything?" is answered by whether the band touches the zero line.

**What will happen.** Dots and bands on a common set of rows, with the value written beside each, a dotted line at 0.5 for the coin flip, and the picture saved as `p6b_10_ladder.png`.

**Expected outcome.** On the left, three dots that hardly move away from 0.55, a jump at the debut day, and a second ladder that climbs to about 0.75. On the right, bands that straddle zero for timing and artist history and stay clear of it for the debut day and for the first three chart days.

**Motive.** Show where the ability to rank stays comes from, and that it comes from the debut day and from what the song does in its first days.

# %%
# Figure 10 - the two ladders
ROWS_ALL = ARRIVAL_ROWS + LM_ROWS
Y_ROWS = np.array([0, 1, 2, 3, 4.9, 5.9, 6.9])
ROW_LABELS = [f"{row['step']}\n({row['detail']})" for row in ROWS_ALL]
fig, (axa, axb) = make_fig(11.8, 6.4, 1, 2, left=2.85, right=0.6, bottom=0.95, wspace=0.06, width_ratios=[1.55, 1], panel_titles=True,
                           title=f"Only the debut day clearly improves the ranking of stays on the arrival day (C-index {FACTS['c4']:.2f}); once the first three chart days are known it reaches {FACTS['lm2']:.2f}",
                           subtitle="C-index: the share of pairs of songs, with a known order of leaving, in which the song that left first had the higher risk score (0.5 = coin flip). "
                                    "Every score comes from a Cox model that never saw the artist (5 folds by artist, 3 splits). Whiskers: 95% artist band. Right: gain over the row above.",
                           note_text=f"The two ladders rank different songs: all {FACTS['n_songs']} on the arrival day, and the {FACTS['lm_songs']} with at least three chart days in a row after three chart days. "
                                     "Their rows are not compared with each other.")
axa.axvline(0.5, color=style.INK_2, linewidth=1.0, linestyle=":", zorder=1)
direct_label(axa, 0.503, 7.45, "0.5 = coin flip", size=8.4, color=style.INK_2)
axa.hlines(Y_ROWS, [r["low"] for r in ROWS_ALL], [r["high"] for r in ROWS_ALL], color=style.BLUE, linewidth=5.0, alpha=0.45, zorder=2)
axa.scatter([r["c_index"] for r in ROWS_ALL], Y_ROWS, s=44, color=style.BLUE, edgecolor=style.SURFACE, linewidth=1.3, zorder=4, label="held-out artists (95% band)")
axa.scatter([r["c_index_in_sample"] for r in ROWS_ALL], Y_ROWS, s=40, facecolor="none", edgecolor=style.ORANGE, linewidth=1.6, zorder=3, label="same model, on songs it was fitted on")
for yi, row in zip(Y_ROWS, ROWS_ALL):
    direct_label(axa, max(row["high"], row["c_index_in_sample"]) + 0.012, yi, f"{row['c_index']:.3f}", size=8.8, color=style.INK)
HEADING = dict(facecolor=style.SURFACE, edgecolor="none", pad=2.0)                      # a plain box, so that the dotted line does not run through the words
axa.text(0.462, -0.78, f"On the arrival day: all {FACTS['n_songs']} songs", fontsize=9.2, fontweight="bold", color=style.INK, va="center", ha="left", bbox=HEADING, zorder=6)
axa.text(0.462, 4.1, f"After three chart days: the {FACTS['lm_songs']} songs with three chart days in a row", fontsize=9.2, fontweight="bold", color=style.INK, va="center", ha="left", bbox=HEADING, zorder=6)
axa.set_yticks(Y_ROWS)
axa.set_yticklabels(ROW_LABELS, fontsize=8.8)
axa.set_ylim(7.7, -1.25)
axa.set_xlim(0.46, 0.87)
axa.set_xticks([0.5, 0.6, 0.7, 0.8])
axa.set_xlabel("C-index")
style_axes(axa, grid="x", left_spine=False)
axa.tick_params(axis="y", length=0)
axa.legend(loc="upper right", frameon=False, fontsize=8.4, labelcolor=style.INK_2, handletextpad=0.3, borderaxespad=0.2)
panel_title(axa, "How well the order of leaving can be read")

axb.axvline(0.0, color=style.INK_2, linewidth=1.0, zorder=1)
for yi, row in zip(Y_ROWS, ROWS_ALL):
    if np.isnan(row["gain"]):
        continue
    axb.hlines(yi, row["gain_low"], row["gain_high"], color=style.BLUE, linewidth=5.0, alpha=0.45, zorder=2)
    axb.scatter([row["gain"]], [yi], s=44, color=style.BLUE, edgecolor=style.SURFACE, linewidth=1.3, zorder=4)
    direct_label(axb, row["gain_high"] + 0.009, yi, f"{row['gain']:+.3f}", size=8.8, color=style.INK)
axb.set_yticks(Y_ROWS)
axb.set_yticklabels([])
axb.set_ylim(7.7, -1.25)
axb.set_xlim(-0.065, 0.235)
axb.set_xticks([-0.05, 0.0, 0.05, 0.10, 0.15, 0.20])
axb.set_xlabel("gain in C-index over the row above")
style_axes(axb, grid="x", left_spine=False)
axb.tick_params(axis="y", length=0)
panel_title(axb, "What each step adds")
save_fig(fig, "p6b_10_ladder", FIG_DIR)
plt.show()
FACTS["lm_excluded"] = int(FACTS["n_songs"] - FACTS["lm_songs"])
GAPS = [r["c_index_in_sample"] - r["c_index"] for r in ROWS_ALL]
FACTS["ins_gap_min"], FACTS["ins_gap_max"] = float(min(GAPS)), float(max(GAPS))

# %% [md]
**What to notice**

- On the arrival day only one step moves the dot. The first three rows stay close to 0.55 with bands that overlap, the debut day takes the C-index to {{c4:f3}}, and on the right only the band of the debut-day gain ({{gain4_low:sf3}} to {{gain4_high:sf3}}) stays clear of the zero line.
- After three chart days the ranking is much sharper. Adding the first three chart days to the arrival-day facts lifts the C-index from {{lm1:f3}} to {{lm2:f3}}, a gain of {{lm_gain:sf3}} ({{lm_gain_low:sf3}} to {{lm_gain_high:sf3}}). The first three chart days alone reach {{lm3:f3}}. What the arrival-day facts add on top of them is {{lm_extra:sf3}} ({{lm_extra_low:sf3}} to {{lm_extra_high:sf3}}), a band that includes zero: once a song's own first days are known, the data show nothing more to be gained from its release, timing, artist record or debut day.
- The hollow circles sit to the right of the dots in every row: the same model scored on the songs it was fitted on looks better by {{ins_gap_min:f2}} to {{ins_gap_max:f2}}. This is the gap that scoring by artist removes.
- Two limits. The five early inputs are used together, so the notebook does not say which of them (the place on day 2, the move over three days, the best place, the popularity on day 3 or its change) carries the ranking. And the second ladder leaves out the {{lm_excluded}} songs whose first unbroken run lasted only one or two chart days: nothing known after three chart days can rank those songs, and they are the quickest leavers.

# %% [md]
### Risk thirds: what the ranking means for stays

# %% [md]
**What this code is.** A translation of the C-index into stays. For each ladder the out-of-fold risk scores of the best model (the arrival-day model with the debut day; the model with the first three chart days) are averaged over the 3 splits and the songs are cut into three equal groups: the lowest-risk third, the middle third and the highest-risk third. For each group the code computes the survival curve with its 95% artist band, the median stay, and the share still on the chart after 7 and after 30 chart days. For the second ladder the curve is of the whole stay, among songs that reached three chart days in a row.

**Why we need it.** "C-index 0.64" is hard to picture. Three curves answer the question a person would ask: if songs are sorted by what was known when they arrived, how different do the stays of the top and the bottom third look?

**What will happen.** The code cuts the scores at their one-third and two-thirds points, runs the Kaplan-Meier estimate on each group, resamples artists {{n_boot:n}} times for the bands, and prints a table with one row for each group.

**Expected outcome.** Three clearly separated curves on both ladders: a highest-risk third that is mostly gone within a few chart days and a lowest-risk third that keeps a third or more of its songs past day 30 on the arrival day, and half of them after three chart days.

**Motive.** Show in chart days, not in a score, what the ranking is worth, and that the groups it makes really differ.

# %%
THIRD_NAMES = ["lowest risk", "middle", "highest risk"]


def third_index(risk):
    """0 for the third of the songs with the lowest risk score, 1 for the middle third, 2 for the highest."""
    return np.digitize(risk, np.quantile(risk, [1 / 3, 2 / 3]))


def summarize_thirds(time_, event, artists, boot_weights, risk, shift, view):
    """
    Cut the songs into thirds of `risk` (0 = lowest risk) and describe the stays of each third. `time_ + shift` is the stay in chart days (the second ladder
    counts its clock from the third chart day), so that medians and shares are given in chart days of the stay.
    """
    third = third_index(risk)
    ages = np.arange(0, int(time_.max()) + 1)
    out = []
    for k, name in enumerate(THIRD_NAMES):
        mask = third == k
        curve = survival_grid(time_[mask], event[mask], ages)
        usable = boot_weights[:, mask].sum(axis=1) > 0
        draws = survival_grid(time_[mask], event[mask], ages, weights=boot_weights[:, mask])[usable]
        low, high = np.percentile(draws, [2.5, 97.5], axis=0)
        medians = median_stay(draws) + shift
        row = {"view": view, "third": name, "songs": int(mask.sum()), "artists": int(len(np.unique(artists[mask]))), "exits": int(event[mask].sum()),
               "median": float(median_stay(curve) + shift), "median_low": float(np.nanpercentile(medians, 2.5)), "median_high": float(np.nanpercentile(medians, 97.5))}
        for day in (7, 30):
            row[f"s{day}"], row[f"s{day}_low"], row[f"s{day}_high"] = float(curve[day - shift]), float(low[day - shift]), float(high[day - shift])
        row["curve"], row["low"], row["high"], row["stay_axis"] = curve, low, high, ages + shift
        out.append(row)
    return out


THIRDS_ARRIVAL = summarize_thirds(T, E, ARTISTS, W_BOOT, OOF["+ Debut day"].mean(axis=0), 0, "arrival day")
THIRDS_LM = summarize_thirds(T_LM, E_LM, ART_LM, W_BOOT[:, LM], OOF_LM["+ first three chart days"].mean(axis=0), 2, "after three chart days")
for prefix, rows in (("ra", THIRDS_ARRIVAL), ("rl", THIRDS_LM)):
    for k, row in enumerate(rows, start=1):
        for key in ("songs", "artists", "exits", "median", "median_low", "median_high", "s7", "s7_low", "s7_high", "s30", "s30_low", "s30_high"):
            FACTS[f"{prefix}{k}_{key}"] = row[key]
    FACTS[f"{prefix}_low_vs_high_median"] = float(rows[0]["median"] / rows[2]["median"])
    FACTS[f"{prefix}_s30_apart_12"] = int(rows[0]["s30_low"] > rows[1]["s30_high"])               # 1 when the 95% bands of the lowest and middle thirds do not overlap
    FACTS[f"{prefix}_s30_apart_23"] = int(rows[1]["s30_low"] > rows[2]["s30_high"])
    FACTS[f"{prefix}_s30_apart_13"] = int(rows[0]["s30_low"] > rows[2]["s30_high"])
THIRDS_TABLE = pd.DataFrame([{key: value for key, value in row.items() if key not in ("curve", "low", "high", "stay_axis")} for row in THIRDS_ARRIVAL + THIRDS_LM])
display(pd.DataFrame({"view": THIRDS_TABLE["view"], "third": THIRDS_TABLE["third"], "songs": THIRDS_TABLE["songs"], "artists": THIRDS_TABLE["artists"],
                      "median stay (95% band)": [f"{m:.0f}  ({a:.0f} to {b:.0f})" for m, a, b in zip(THIRDS_TABLE["median"], THIRDS_TABLE["median_low"], THIRDS_TABLE["median_high"])],
                      "still there after 7 chart days": [f"{100 * v:.0f}%  ({100 * a:.0f} to {100 * b:.0f})" for v, a, b in zip(THIRDS_TABLE["s7"], THIRDS_TABLE["s7_low"], THIRDS_TABLE["s7_high"])],
                      "still there after 30": [f"{100 * v:.0f}%  ({100 * a:.0f} to {100 * b:.0f})" for v, a, b in zip(THIRDS_TABLE["s30"], THIRDS_TABLE["s30_low"], THIRDS_TABLE["s30_high"])]}).set_index(["view", "third"]))

# %% [md]
### Figure 11: the stays of the three risk groups

# %% [md]
**What this code is.** The drawing of Figure 11. On the left, the survival curves of the three risk thirds on the arrival day, each with its 95% artist band and a label at its end that gives the median stay. On the right, the same for the songs that reached three chart days in a row, sorted by the model that also knows the first three chart days; there the curves start at 100% on chart day 2.

**Why we need it.** It shows what the C-index values mean for the length of stays, and how much sharper the groups become after three chart days.

**What will happen.** Six curves with shaded bands, darker for the lower-risk groups, labelled directly at their ends; the picture is saved as `p6b_11_risk_groups.png`.

**Expected outcome.** On the left, three curves that fan out from the start and stay apart, with a lowest-risk curve that ends near 30% at day 30 and a highest-risk curve that is near 7%. On the right, a wider fan: about half of the lowest-risk third is still there at day 30, almost none of the highest-risk third.

**Motive.** Turn the ranking into something a reader can see: how long the stays of each group last.

# %%
# Figure 11 - the curves of the risk thirds
RISK_COLORS = MP.step_colors(3)[::-1]                                  # dark = lowest risk (longest stays), light = highest risk
fig, (axa, axb) = make_fig(11.8, 5.6, 1, 2, left=0.85, right=1.75, bottom=0.9, wspace=0.58, panel_titles=True,
                           title=f"Sorted by what is known on the arrival day, the highest-risk third of songs has a median stay of {FACTS['ra3_median']:.0f} chart days and the lowest-risk third "
                                 f"{FACTS['ra1_median']:.0f}; after three chart days the medians are {FACTS['rl3_median']:.0f} and {FACTS['rl1_median']:.0f}",
                           subtitle="Share of songs still on the chart, by chart days of the stay. Thirds are cut from the risk scores of models that never saw the artist. Shaded: 95% artist band. "
                                    "Right: only the songs with three chart days in a row, so the curves start at 100% on chart day 2.",
                           note_text=f"Curves are drawn to 60 chart days. Left: {FACTS['ra1_songs']} songs in each third; right: {FACTS['rl1_songs']} or {FACTS['rl2_songs']}.")
for ax, rows, left_title in ((axa, THIRDS_ARRIVAL, f"On the arrival day (all {FACTS['n_songs']} songs)"), (axb, THIRDS_LM, f"After three chart days ({FACTS['lm_songs']} songs)")):
    curves = [{"label": f"{row['third']}: median {row['median']:.0f}", "ages": row["stay_axis"], "survival": row["curve"], "low": row["low"], "high": row["high"], "color": color}
              for row, color in zip(rows, RISK_COLORS)]
    SP.draw_survival_curves(ax, curves, x_max=60, label_gap=0.085, band_alpha=0.14)
    ax.set_xlabel("chart days of the stay")
    style_axes(ax, grid="y")
    panel_title(ax, left_title)
axa.set_xticks([0, 7, 14, 30, 60])
axb.set_xlim(2, 60)
axb.set_xticks([3, 7, 14, 30, 60])
axa.set_ylabel("share still on the chart")
save_fig(fig, "p6b_11_risk_groups", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- The groups really differ. On the arrival day the lowest-risk third has a median stay of {{ra1_median:f0}} chart days (95% band {{ra1_median_low:f0}} to {{ra1_median_high:f0}}), the middle third {{ra2_median:f0}} and the highest-risk third {{ra3_median:f0}} ({{ra3_median_low:f0}} to {{ra3_median_high:f0}}); "risk" here always means the risk of leaving. After 30 chart days {{ra1_s30:pct0}}, {{ra2_s30:pct0}} and {{ra3_s30:pct0}} of the songs are still on the chart, and of the highest-risk third only {{ra3_s7:pct0}} are there after a week. The highest-risk third is clearly apart from the other two: its band at day 30 ({{ra3_s30_low:pct0}} to {{ra3_s30_high:pct0}}) does not come near theirs. The lowest and the middle third are closer: their bands at day 30 meet at about {{ra2_s30_high:pct0}}.
- This is what a C-index of {{c4:f2}} looks like. It sorts songs into groups with clearly different stays; it does not say what any single song will do, and inside every third the stays run from one chart day to months.
- The highest-risk third is made of few artists: {{ra3_songs}} songs from {{ra3_artists}} artists, against {{ra1_songs}} songs from {{ra1_artists}} artists in the lowest. A few artists bring many songs on one day, and the songs of one artist share their fate, which is why the band of the median of that third, {{ra3_median_low:f0}} to {{ra3_median_high:f0}}, is wide for a median this small.
- After three chart days the groups fan out further. The medians are {{rl1_median:f0}}, {{rl2_median:f0}} and {{rl3_median:f0}} chart days, and {{rl1_s30:pct0}}, {{rl2_s30:pct0}} and {{rl3_s30:pct0}} of the songs are still on the chart after 30. A song in the highest-risk third after three chart days almost never reaches day 30 ({{rl3_s30_low:pct0}} to {{rl3_s30_high:pct0}}), while about half of the lowest-risk third does.
- These are descriptions of past stays, sorted by scores that models made without having seen the artist, cut at the thirds of this one sample. They are not a rule for choosing songs, and the label of a third says how songs like it behaved in 2024 and 2025, nothing about why.
