# %% [md]
## Section 6 - The hazard model: the inputs held against each other

The splits of Section 4 compared groups that differ in many ways at once, and Section 5 showed how easily that misleads. The model of this section asks about one input at a time while the others are held fixed: among songs that are alike in everything else in the model, how does the chance that a chart day is the last one change with this input? The answer is a *hazard ratio*. A ratio of 2.0 means that on every chart day the chance of it being the last is twice as high; 0.5 means half as high; 1.0 means no difference.

Ten inputs are used, all known on the arrival day: the debut place (five bands, the top 10 as the reference); the album drop (how many songs of the same artist arrived on the same day, on a log scale); how many chart days the artist had before (log scale); the artist's usual popularity and a flag for artists who have none yet; whether the artist was on the chart the day before; whether the song is a single, explicit or a collaboration; the length of the song; and its popularity on the arrival day. The number of tracks on the release is left out on purpose, and the reason is shown in this section.

# %% [md]
**What this code is.** The day-by-day hazard model and its tests. The table `DAYS` has one row for each chart day of each stay, with a 1 on the last chart day of a finished stay. The model says: the chance that the chart day is the last one equals 1 minus exp(minus exp(*baseline of its age window* plus *the inputs times their coefficients*)). This is the complementary log-log model, the day-by-day form of the Cox model. The baseline is a separate level for each of the nine age windows of Section 3, so no assumption is made about how the hazard falls with age. The exponential of a coefficient is the hazard ratio.

**Why we need it.** To hold the inputs against each other, which the splits cannot do. The day-by-day form uses every chart day, handles the many ties of a daily chart (155 songs leave on their first chart day) without approximation, and gives the baseline hazard its own level at each age. The inputs on a log scale or in standard deviations are made comparable: a numeric input is standardised, so that its hazard ratio means "per one standard deviation".

**What will happen.** The code builds the inputs with `build_design` (the artist's usual popularity is missing for {{n_no_history}} songs, whose artists have no earlier chart days; they get the median and a flag, and the popularity on the debut day, missing for {{n_missing_entry_pop}} songs, gets the median without a flag, because a flag for so few songs would let the model fit them one by one). It spreads the inputs over the chart days, fits the model with artist-clustered standard errors (the artist is the cluster, p-values and intervals use the t distribution with {{n_artists_model}} minus 1 degrees of freedom), tests each input as a block with an F test (so that the four debut-place coefficients are tested together, and the popularity and its flag together), adjusts the ten p-values with Holm, and compares the hazard ratios with those of a Cox model fitted on the stays.

**Expected outcome.** Debut place and album drop clear, with debut places 31 to 40 at about four times the hazard of a top-10 debut and one standard deviation more album drop at about 1.5 times. A lower hazard when the artist was on the chart the day before. Single, explicit, collaboration and song length near 1. The hazard ratios of the Cox model close to those of the day-by-day model.

**Motive.** Find out which of the differences of Sections 4 and 5 are left when the inputs are held against each other, and how sure the data let us be of each.

# %%
NUMERIC = {"Album drop": ("log_drop", "raw"),                                # log of the songs by the artist arriving that day
           "Artist's earlier chart days": ("artist_prior_slot_days", "log1p"),
           "Artist's usual popularity": ("artist_prior_mean_pop", "raw"),
           "Song length": ("duration_min", "log1p"),
           "Popularity on the debut day": ("entry_pop", "raw")}
BINARY = {"Single": "is_single", "Explicit": "is_explicit", "Collaboration": "is_collab", "Artist on the chart the day before": "artist_on_chart_prev_day"}
BAND_COLUMNS = [f"Debut place {b}" for b in BAND_LABELS[1:]]
NO_HISTORY = "No usual popularity yet"

FACTS["n_no_history"] = int(STAYS["artist_prior_mean_pop"].isna().sum())
FACTS["n_missing_entry_pop"] = int(STAYS["entry_pop"].isna().sum())
SONG_DESIGN, SCALE = build_design(STAYS, NUMERIC, BINARY, flag_from=50)      # flag_from=50: a missing-value flag only when 50 or more songs are missing
SONG_DESIGN = SONG_DESIGN.rename(columns={"Artist's usual popularity (missing)": NO_HISTORY})
bands = pd.get_dummies(STAYS["debut_band"].astype(str)).reindex(columns=BAND_LABELS[1:], fill_value=0).astype(float)
bands.columns = BAND_COLUMNS
INPUT_ORDER = BAND_COLUMNS + ["Album drop", "Artist's earlier chart days", "Artist's usual popularity", NO_HISTORY, "Artist on the chart the day before",
                              "Single", "Explicit", "Collaboration", "Song length", "Popularity on the debut day"]
SONG_X = pd.concat([bands, SONG_DESIGN.drop(columns="Intercept")], axis=1)[INPUT_ORDER]
assert np.isfinite(SONG_X.to_numpy()).all()

# every chart day of every stay gets the inputs of its song and the age-window columns of the baseline
SONG_ROW = pd.Index(STAYS["track_id"]).get_indexer(DAYS["track_id"])         # for each chart day: the row of its song in STAYS
assert (SONG_ROW >= 0).all()
DAY_ARTIST = ARTISTS[SONG_ROW]                                               # the cluster of each chart day
DAY_Y = DAYS["event"].to_numpy()
BASELINE = window_columns(DAYS["age"].to_numpy(), AGE_EDGES)                 # the nine age windows: the baseline of the model
BASELINE_NAMES = list(BASELINE.columns)


def day_by_day(columns, extra=None):
    """Fit the model with the chosen song-level inputs (and optional extra song-level columns); return the fit and the column names."""
    frame = SONG_X[columns] if extra is None else pd.concat([SONG_X[columns], extra], axis=1)
    design = pd.concat([BASELINE, pd.DataFrame(frame.to_numpy()[SONG_ROW], columns=list(frame.columns))], axis=1)
    return fit_cloglog(DAY_Y, design.to_numpy(), clusters=DAY_ARTIST), list(design.columns)


MAIN, MAIN_NAMES = day_by_day(INPUT_ORDER)
assert MAIN.converged, "the main model did not converge"
hr, hr_low, hr_high = MAIN.hazard_ratios()
p_coef = MAIN.p_values()
FACTS["n_artists_model"] = int(MAIN.groups)
FACTS["model_rows"], FACTS["model_events"], FACTS["model_coefficients"] = int(MAIN.n), int(DAY_Y.sum()), int(MAIN.k)
FACTS["model_iterations"] = int(MAIN.iterations)

KEYS = {"Debut place 11 to 20": "b2", "Debut place 21 to 30": "b3", "Debut place 31 to 40": "b4", "Debut place 41 to 50": "b5", "Album drop": "drop",
        "Artist's earlier chart days": "earlier", "Artist's usual popularity": "usualpop", NO_HISTORY: "nohist", "Artist on the chart the day before": "onchart",
        "Single": "single", "Explicit": "explicit", "Collaboration": "collab", "Song length": "length", "Popularity on the debut day": "debutpop"}
rows = []
for name in INPUT_ORDER:
    i = MAIN_NAMES.index(name)
    rows.append({"input": name, "hazard_ratio": hr[i], "low": hr_low[i], "high": hr_high[i], "p": p_coef[i]})
    key = KEYS[name]
    FACTS[f"hr_{key}"], FACTS[f"hr_{key}_low"], FACTS[f"hr_{key}_high"], FACTS[f"hr_{key}_p"] = float(hr[i]), float(hr_low[i]), float(hr_high[i]), float(p_coef[i])
    FACTS[f"hr_{key}_p_text"] = format_p(p_coef[i])

# what one standard deviation means, in the units of each input
sd = dict(zip(SCALE["input"], SCALE["sd"]))                                  # one standard deviation of each numeric input, on the scale it enters the model
FACTS["sd_drop_factor"] = float(np.exp(sd["Album drop"]))                    # the inputs on a log scale: one SD is a multiplication by this factor
FACTS["sd_earlier_factor"] = float(np.exp(sd["Artist's earlier chart days"]))
FACTS["sd_length_factor"] = float(np.exp(sd["Song length"]))
FACTS["sd_usualpop_points"] = float(sd["Artist's usual popularity"])         # the inputs on their own scale: one SD is this many popularity points
FACTS["sd_debutpop_points"] = float(sd["Popularity on the debut day"])
UNIT = {name: "compared with a top-10 debut" for name in BAND_COLUMNS}
UNIT["Album drop"] = f"per 1 SD = a drop {FACTS['sd_drop_factor']:.1f} times as large"
UNIT["Artist's earlier chart days"] = f"per 1 SD = {FACTS['sd_earlier_factor']:.1f} times as many days"
UNIT["Artist's usual popularity"] = f"per 1 SD = {FACTS['sd_usualpop_points']:.0f} popularity points"
UNIT["Song length"] = f"per 1 SD = {FACTS['sd_length_factor']:.2f} times as long"
UNIT["Popularity on the debut day"] = f"per 1 SD = {FACTS['sd_debutpop_points']:.0f} popularity points"
for name in [NO_HISTORY, "Artist on the chart the day before", "Explicit", "Collaboration"]:
    UNIT[name] = "yes against no"
UNIT["Single"] = "single against album"
HR_TABLE = pd.DataFrame(rows)
HR_TABLE["unit"] = HR_TABLE["input"].map(UNIT)
display(HR_TABLE.assign(**{"hazard ratio (95% interval)": [f"{a:.2f}  ({b:.2f} to {c:.2f})" for a, b, c in zip(HR_TABLE["hazard_ratio"], HR_TABLE["low"], HR_TABLE["high"])],
                           "p": [format_p(p) for p in HR_TABLE["p"]]})[["input", "unit", "hazard ratio (95% interval)", "p"]].set_index("input"))
print(f"{MAIN.n:,} chart days, {int(DAY_Y.sum())} exits, {MAIN.k} coefficients ({len(BASELINE_NAMES)} of them the age-window baseline), {MAIN.groups} artists; {MAIN.iterations} iterations.")

# ---- each input as a block: one F test, ten tests in all, adjusted together
BLOCKS = [("Debut place", BAND_COLUMNS), ("Album drop", ["Album drop"]), ("Artist's earlier chart days", ["Artist's earlier chart days"]),
          ("Artist's usual popularity", ["Artist's usual popularity", NO_HISTORY]), ("Artist on the chart the day before", ["Artist on the chart the day before"]),
          ("Single", ["Single"]), ("Explicit", ["Explicit"]), ("Collaboration", ["Collaboration"]), ("Song length", ["Song length"]),
          ("Popularity on the debut day", ["Popularity on the debut day"])]
BLOCK_KEY = {"Debut place": "band", "Album drop": "drop", "Artist's earlier chart days": "earlier", "Artist's usual popularity": "usualpop",
             "Artist on the chart the day before": "onchart", "Single": "single", "Explicit": "explicit", "Collaboration": "collab", "Song length": "length",
             "Popularity on the debut day": "debutpop"}
block_rows = []
for block_name, columns in BLOCKS:
    test = wald_test(MAIN, selection(MAIN_NAMES, columns))
    block_rows.append({"input": block_name, "coefficients": len(columns), "F": test["stat"], "df1": test["q"], "df2": test["df"], "p": test["p"]})
BLOCK_TABLE = pd.DataFrame(block_rows)
BLOCK_TABLE["p_holm"] = holm(BLOCK_TABLE["p"].to_numpy())
for row in BLOCK_TABLE.itertuples():
    key = BLOCK_KEY[row.input]
    FACTS[f"blk_{key}_F"], FACTS[f"blk_{key}_p"], FACTS[f"blk_{key}_p_holm"] = float(row.F), float(row.p), float(row.p_holm)
    FACTS[f"blk_{key}_p_text"], FACTS[f"blk_{key}_p_holm_text"] = format_p(row.p), format_p(row.p_holm)
FACTS["n_inputs"] = len(BLOCK_TABLE)
FACTS["n_inputs_clear"] = int((BLOCK_TABLE["p_holm"] < 0.05).sum())
FACTS["inputs_clear"] = ", ".join(BLOCK_TABLE.loc[BLOCK_TABLE["p_holm"] < 0.05, "input"].str.lower())
display(BLOCK_TABLE.assign(F=lambda d: d["F"].round(2), p=[format_p(p) for p in BLOCK_TABLE["p"]], after_Holm=[format_p(p) for p in BLOCK_TABLE["p_holm"]]).drop(columns="p_holm").set_index("input"))

# ---- how much the artist-aware standard errors differ from the textbook ones
ratio = MAIN.se / np.sqrt(np.diag(MAIN.model_cov))
song_level = [MAIN_NAMES.index(name) for name in INPUT_ORDER]
FACTS["se_ratio_median"], FACTS["se_ratio_max"] = float(np.median(ratio[song_level])), float(np.max(ratio[song_level]))
FACTS["se_ratio_argmax"] = INPUT_ORDER[int(np.argmax(ratio[song_level]))]

# ---- a Cox model on the stays must give nearly the same hazard ratios
COX = fit_cox(T, E, SONG_X.to_numpy(), ridge=0.0)
cox_hr = np.exp(COX.coef)
FACTS["cox_max_gap"] = float(np.max(np.abs(cox_hr - hr[[MAIN_NAMES.index(n) for n in INPUT_ORDER]])))
FACTS["cox_median_gap"] = float(np.median(np.abs(cox_hr - hr[[MAIN_NAMES.index(n) for n in INPUT_ORDER]])))
print(f"Artist-aware standard errors are {FACTS['se_ratio_median']:.2f} times the textbook ones (median over the inputs), up to {FACTS['se_ratio_max']:.2f} times for {FACTS['se_ratio_argmax'].lower()}.")
print(f"A Cox model on the stays gives hazard ratios within {FACTS['cox_max_gap']:.3f} of the day-by-day model (median gap {FACTS['cox_median_gap']:.3f}).")

# %% [md]
**What this code is.** The check on the single, in five versions of the model. Version 1 has the single as its only input (besides the age baseline). Versions 2 and 3 add the debut place and then the album drop. Version 4 is the main model above. Version 5 adds the number of tracks on the release to the main model, which the main model leaves out.

**Why we need it.** Section 5 saw that the single's advantage went away among songs that arrive alone. This cell asks the same question of the model and answers a second one: why is the number of tracks not an input? A single has one to three tracks and an album more, so "single" and "number of tracks" say nearly the same thing about a song (their correlation is {{corr_single_tracks:f2}}). When two inputs carry almost the same information a model cannot tell which of them does the work, and the coefficients of the two become large, opposite and unstable. A model with both looks as if it had found something, and it has not.

**What will happen.** The code fits the five models with the same artist-clustered method, takes the hazard ratio of the single in each, and prints a small table. For version 5 it also prints the hazard ratio of the number of tracks.

**Expected outcome.** The hazard ratio of the single starts below 1 (about 0.75: a single leaves later than an album track), stays there when the debut place is added, moves to about 1.0 once the album drop is in, and is still about 1.0 in the main model. In version 5 it jumps to more than 2 and the track count has a hazard ratio far from 1: the two inputs fight over one piece of information.

**Motive.** Show why the main model does not carry the number of tracks, and that the single's lower hazard is explained by how the single arrives.

# %%
tracks = np.log1p(STAYS["total_tracks"].astype(float).to_numpy())
tracks = np.where(np.isnan(tracks), np.nanmedian(tracks), tracks)
tracks_z = (tracks - tracks.mean()) / tracks.std(ddof=1)                      # tracks on the release: log scale, standardised like the other inputs
FACTS["corr_single_tracks"] = float(np.corrcoef(STAYS["is_single"].to_numpy().astype(float), tracks)[0, 1])   # between the single flag and the log of the track count

LADDER_FITS = [("1. Single alone", ["Single"], None),
               ("2. + debut place", ["Single"] + BAND_COLUMNS, None),
               ("3. + album drop", ["Single"] + BAND_COLUMNS + ["Album drop"], None),
               ("4. the main model (all ten inputs)", INPUT_ORDER, None),
               ("5. main model + tracks on the release", INPUT_ORDER, pd.DataFrame({"Tracks on the release": tracks_z}))]
SINGLE_LADDER = []
for k, (label, columns, extra) in enumerate(LADDER_FITS, start=1):
    fit, names = day_by_day(columns, extra)
    a, b, c = fit.hazard_ratios()
    i = names.index("Single")
    row = {"version": label, "hazard_ratio": float(a[i]), "low": float(b[i]), "high": float(c[i]), "converged": fit.converged}
    if extra is not None:
        j = names.index("Tracks on the release")
        row.update({"tracks_hazard_ratio": float(a[j]), "tracks_low": float(b[j]), "tracks_high": float(c[j])})
        FACTS["tracks_hr"], FACTS["tracks_hr_low"], FACTS["tracks_hr_high"] = row["tracks_hazard_ratio"], row["tracks_low"], row["tracks_high"]
    SINGLE_LADDER.append(row)
    FACTS[f"ladder{k}_single_hr"], FACTS[f"ladder{k}_single_low"], FACTS[f"ladder{k}_single_high"] = row["hazard_ratio"], row["low"], row["high"]
LADDER_TABLE = pd.DataFrame(SINGLE_LADDER)
display(LADDER_TABLE.assign(**{"hazard ratio of the single (95% interval)": [f"{a:.2f}  ({b:.2f} to {c:.2f})" for a, b, c in zip(LADDER_TABLE["hazard_ratio"], LADDER_TABLE["low"], LADDER_TABLE["high"])],
                               "hazard ratio of the track count": [f"{a:.2f}  ({b:.2f} to {c:.2f})" if a == a else "" for a, b, c in
                                                                    zip(LADDER_TABLE["tracks_hazard_ratio"], LADDER_TABLE["tracks_low"], LADDER_TABLE["tracks_high"])]})
        [["version", "hazard ratio of the single (95% interval)", "hazard ratio of the track count"]].set_index("version"))

FACTS["ladder_width_ratio"] = float(np.log(FACTS["ladder5_single_high"] / FACTS["ladder5_single_low"]) / np.log(FACTS["ladder4_single_high"] / FACTS["ladder4_single_low"]))

# %% [md]
**What to notice**

- On its own the single goes with a lower hazard: {{ladder1_single_hr:f2}} ({{ladder1_single_low:f2}} to {{ladder1_single_high:f2}}), a chart day of a single is about a quarter less likely to be its last than one of an album track. Adding the debut place changes little ({{ladder2_single_hr:f2}}). Adding the album drop takes it to {{ladder3_single_hr:f2}} ({{ladder3_single_low:f2}} to {{ladder3_single_high:f2}}), and the rest of the inputs leave it at {{ladder4_single_hr:f2}} ({{ladder4_single_low:f2}} to {{ladder4_single_high:f2}}). That is the answer of Section 5 given by the model: the single's advantage is how singles arrive, mostly alone.
- Version 5 shows the trap. With the track count in the model the single's hazard ratio becomes {{ladder5_single_hr:f2}} ({{ladder5_single_low:f2}} to {{ladder5_single_high:f2}}) and the track count's {{tracks_hr:f2}} ({{tracks_hr_low:f2}} to {{tracks_hr_high:f2}}). This is not a finding that singles are twice as risky. The single flag and the track count are almost the same information (correlation {{corr_single_tracks:f2}}), the model cannot say which of them carries the effect, and the interval of the single becomes about {{ladder_width_ratio:f1}} times as wide on the log scale. This is the reason that the main model leaves the track count out.

# %% [md]
**What this code is.** Figure 7. The hazard ratios of the main model as a forest plot: one row for each of the fourteen coefficients, a dot for the hazard ratio, a bar for its 95% interval on a log scale, and a line at 1. A dot is *open* when the interval includes 1.

**Why we need it.** A table of fourteen hazard ratios is easy to misread, and the chart puts the sizes and the doubts in one view: which inputs are far from 1, which are close to it, and how wide each interval is. On a log scale a ratio of 2 and a ratio of one half are the same distance from 1 on either side, which is the honest way to draw ratios.

**What will happen.** The code draws the rows grouped by input, writes the hazard ratio with its interval to the right of each row and the Holm-adjusted p-value of the whole block of the input at the far right, and notes what one standard deviation means for each numeric input. It saves the picture as `p6b_07_hazard_ratios.png`.

**Expected outcome.** The four debut-place rows far to the right of 1 with filled dots, the album drop to the right of 1 and also filled. Everything else with an interval that crosses 1 (open dots), most of them close to it. Three of the inputs (the artist on the chart the day before, the missing-popularity flag, and the popularity on the debut day) lean below 1 with wide intervals.

**Motive.** Show, in one picture, what the model finds and what it does not.

# %%
# Figure 07 - hazard ratios of the main model
labels = list(INPUT_ORDER)
pretty = {"Album drop": "Album drop (per 1 SD)", "Artist's earlier chart days": "Artist's earlier chart days (per 1 SD)", "Artist's usual popularity": "Artist's usual popularity (per 1 SD)",
          "Song length": "Song length (per 1 SD)", "Popularity on the debut day": "Popularity on the debut day (per 1 SD)", "Single": "Single (against album)",
          "Debut place 11 to 20": "Debut place 11 to 20 (against 1 to 10)", "Debut place 21 to 30": "Debut place 21 to 30", "Debut place 31 to 40": "Debut place 31 to 40",
          "Debut place 41 to 50": "Debut place 41 to 50"}
y_forest, y_now, block_start = [], 0.0, {}
previous_block = None
block_of = {column: block for block, columns in BLOCKS for column in columns}
for name in labels:
    if block_of[name] != previous_block and previous_block is not None:
        y_now += 0.55
    y_forest.append(y_now)
    y_now += 1.0
    previous_block = block_of[name]
fig, ax = make_fig(11.8, 7.4, 1, 1, left=3.1, right=3.55, bottom=1.25,
                   title="Debut place and album drop go with a clearly higher hazard; the other inputs do not stand out once these are held fixed",
                   subtitle=f"Hazard ratios from the day-by-day model: {FACTS['model_rows']:,} chart days, {FACTS['model_events']} exits, {FACTS['n_artists_model']} artists. Dot: hazard ratio; "
                            "bar: 95% interval with artist-clustered errors; open dot: the interval includes 1.",
                   note_text=f"1 SD: album drop = a drop {FACTS['sd_drop_factor']:.1f} times as large; earlier chart days = {FACTS['sd_earlier_factor']:.1f} times as many; usual popularity = "
                             f"{FACTS['sd_usualpop_points']:.0f} points; song length = {FACTS['sd_length_factor']:.2f} times as long; debut-day popularity = {FACTS['sd_debutpop_points']:.0f} points. "
                             "Right: p-value of the whole input (all its rows together), Holm-adjusted over the ten inputs.")
hr_by_name = {row.input: row for row in HR_TABLE.itertuples()}
estimates = [hr_by_name[n].hazard_ratio for n in labels]
colors = [style.BLUE if block_of[n] in ("Debut place", "Album drop") else style.INK_2 for n in labels]
MP.draw_forest(ax, y_forest, estimates, [hr_by_name[n].low for n in labels], [hr_by_name[n].high for n in labels], colors=colors, reference=1.0, log=True,
               texts=[f"{hr_by_name[n].hazard_ratio:.2f}  ({hr_by_name[n].low:.2f} to {hr_by_name[n].high:.2f})" for n in labels], text_x=1.02)
holm_by_block = dict(zip(BLOCK_TABLE["input"], BLOCK_TABLE["p_holm"]))
seen = set()
for name, y_value in zip(labels, y_forest):
    block = block_of[name]
    if block not in seen:
        seen.add(block)
        ax.text(1.4, y_value, format_p(holm_by_block[block]), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.8,
                color=style.INK, fontweight="bold" if holm_by_block[block] < 0.05 else "normal")
ax.set_yticks(y_forest)
ax.set_yticklabels([pretty.get(n, n) if n != NO_HISTORY else "No usual popularity yet (flag)" for n in labels], fontsize=9)
ax.set_ylim(y_now - 0.4, -0.7)
ax.set_xlim(0.28, 7.5)
ax.set_xticks([0.3, 0.5, 1, 2, 4, 6])
ax.set_xticklabels(["0.3", "0.5", "1", "2", "4", "6"])
ax.minorticks_off()
ax.set_xlabel("hazard ratio (log scale): below 1, a chart day is less likely to be the last; above 1, more likely")
ax.text(1.02, -0.58, "hazard ratio (95% interval)", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.6, color=style.INK_2, fontweight="bold")
ax.text(1.4, -0.58, "whole input, Holm", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.6, color=style.INK_2, fontweight="bold")
save_fig(fig, "p6b_07_hazard_ratios", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- *Debut place.* All four hazard ratios are above 1 and their intervals are clear of it: {{hr_b2:f2}} ({{hr_b2_low:f2}} to {{hr_b2_high:f2}}) at 11 to 20, {{hr_b3:f2}} ({{hr_b3_low:f2}} to {{hr_b3_high:f2}}) at 21 to 30, {{hr_b4:f2}} ({{hr_b4_low:f2}} to {{hr_b4_high:f2}}) at 31 to 40 and {{hr_b5:f2}} ({{hr_b5_low:f2}} to {{hr_b5_high:f2}}) at 41 to 50. A chart day of a song that debuted at 31 to 40 is about {{hr_b4:f0}} times as likely to be its last as one of a song that debuted in the top 10 with the same album drop, artist history, popularity and the rest. The ratio climbs from the top 10 to places 31 to 40; the dip at 41 to 50 is smaller than the width of the intervals, so the data do not show that debuts at 41 to 50 do better than those at 31 to 40. As a block, debut place has {{blk_band_p_holm_text}} after the Holm adjustment.
- *Album drop.* One standard deviation more (a drop {{sd_drop_factor:f1}} times as large) goes with a hazard {{hr_drop:f2}} times as high on every chart day (interval {{hr_drop_low:f2}} to {{hr_drop_high:f2}}; {{blk_drop_p_holm_text}} after Holm). This is the input that absorbed the single's advantage.
- *Everything else.* {{n_inputs_clear}} of the {{n_inputs}} inputs are clear after the Holm adjustment: {{inputs_clear}}. The artist's earlier chart days ({{hr_earlier:f2}}), the artist's usual popularity ({{hr_usualpop:f2}}), single ({{hr_single:f2}}), explicit ({{hr_explicit:f2}}), collaboration ({{hr_collab:f2}}) and song length ({{hr_length:f2}}) are all close to 1 with intervals that include it. Three lean below 1: the artist on the chart the day before ({{hr_onchart:f2}}, {{hr_onchart_low:f2}} to {{hr_onchart_high:f2}}), no usual popularity yet ({{hr_nohist:f2}}, {{hr_nohist_low:f2}} to {{hr_nohist_high:f2}}) and the popularity on the debut day ({{hr_debutpop:f2}}, {{hr_debutpop_low:f2}} to {{hr_debutpop_high:f2}}). They are suggestions, not findings: the intervals touch or include 1 and none survives the adjustment for ten inputs.
- *How sure.* The artist-aware standard errors are {{se_ratio_median:f2}} times the textbook ones in the middle of the inputs, and up to {{se_ratio_max:f2}} times for {{se_ratio_argmax}}, an input that an artist's songs mostly share. A Cox model fitted on the stays gives hazard ratios within {{cox_max_gap:f3}} of these (median gap {{cox_median_gap:f3}}), so the result does not depend on the day-by-day form.
- *What "held fixed" means.* The hazard ratios compare songs that are alike in the other nine inputs. They describe these {{n_songs:n}} songs; they do not say what would happen to a song if one of its inputs were changed.

# %% [md]
**What this code is.** The check that each hazard ratio is the same at every age. The model above gives an input one hazard ratio for all chart days. That is the *proportional hazards* assumption, and it can be wrong: the album drop might matter on the first three chart days and not afterwards. The code splits the chart days into three age windows (chart days 1 to 3, 4 to 14, and 15 and later) and, one input at a time, lets that input have its own hazard ratio in each window while the others keep one.

**Why we need it.** If a hazard ratio changes with age, the single number of Figure 7 is an average of different things, and it can mislead. The check says which inputs are safe to summarise with one number, and the separate ratios say how an input works at each stage of a stay. The three windows hold {{ph_events_1}}, {{ph_events_2}} and {{ph_events_3}} exits, enough for each to carry its own estimates.

**What will happen.** For each of the ten inputs the code builds a design in which the input's columns are replaced by one column per window (`window_interactions`), fits the model, and tests that the effects are the same in all three windows (an F test with the artist-clustered covariance; the debut place has four columns and so eight contrasts). The ten p-values are adjusted with Holm. The window-specific hazard ratios, with their intervals, are kept for Figure 8.

**Expected outcome.** A few inputs whose ratios change with age, and most that do not. Where the effect changes, it should be easiest to see for the debut place and the album drop, which have the strongest effects.

**Motive.** Know where "one hazard ratio for the whole stay" is a fair summary and where it is not.

# %%
PH_EDGES = [0, 3, 14, 10_000]                                                # chart days 1 to 3, 4 to 14, 15 and later
PH_LABELS = ["chart days 1 to 3", "chart days 4 to 14", "chart day 15 and later"]
PH_WINDOW = age_windows(DAYS["age"].to_numpy(), PH_EDGES)
N_PH = len(PH_LABELS)
for k in range(N_PH):
    FACTS[f"ph_events_{k + 1}"] = int(DAY_Y[PH_WINDOW == k].sum())
FACTS["ph_n_windows"] = N_PH

FULL_DESIGN = pd.concat([BASELINE, pd.DataFrame(SONG_X.to_numpy()[SONG_ROW], columns=INPUT_ORDER)], axis=1)
PH_TESTS, AGE_EFFECTS = [], []
for block_name, columns in BLOCKS:
    split_design = window_interactions(FULL_DESIGN, columns, PH_WINDOW, N_PH)
    split_fit = fit_cloglog(DAY_Y, split_design.to_numpy(), clusters=DAY_ARTIST)
    assert split_fit.converged, f"the model with separate effects for {block_name} did not converge"
    split_names = list(split_design.columns)
    test = wald_test(split_fit, equal_effects_contrasts(split_names, columns, N_PH))
    PH_TESTS.append({"input": block_name, "F": test["stat"], "df1": test["q"], "df2": test["df"], "p": test["p"]})
    a, b, c = split_fit.hazard_ratios()
    for column in columns:
        for k in range(N_PH):
            i = split_names.index(f"{column} @ {k + 1}")
            AGE_EFFECTS.append({"input": column, "block": block_name, "window": PH_LABELS[k], "window_number": k + 1, "hazard_ratio": float(a[i]), "low": float(b[i]), "high": float(c[i])})
PH_TABLE = pd.DataFrame(PH_TESTS)
PH_TABLE["p_holm"] = holm(PH_TABLE["p"].to_numpy())
AGE_TABLE = pd.DataFrame(AGE_EFFECTS)
for row in PH_TABLE.itertuples():
    key = BLOCK_KEY[row.input]
    FACTS[f"ph_{key}_F"], FACTS[f"ph_{key}_p"], FACTS[f"ph_{key}_p_holm"] = float(row.F), float(row.p), float(row.p_holm)
    FACTS[f"ph_{key}_p_text"], FACTS[f"ph_{key}_p_holm_text"] = format_p(row.p), format_p(row.p_holm)
for row in AGE_TABLE.itertuples():
    key = KEYS[row.input]
    FACTS[f"age_{key}_{row.window_number}"], FACTS[f"age_{key}_{row.window_number}_low"], FACTS[f"age_{key}_{row.window_number}_high"] = row.hazard_ratio, row.low, row.high
FACTS["ph_n_inputs"] = len(PH_TABLE)
FACTS["ph_n_flagged"] = int((PH_TABLE["p_holm"] < 0.05).sum())
FACTS["ph_flagged"] = ", ".join(PH_TABLE.loc[PH_TABLE["p_holm"] < 0.05, "input"].str.lower()) or "none"
display(PH_TABLE.assign(F=lambda d: d["F"].round(2), p=[format_p(p) for p in PH_TABLE["p"]], after_Holm=[format_p(p) for p in PH_TABLE["p_holm"]]).drop(columns="p_holm").set_index("input"))
display(AGE_TABLE.assign(**{"hazard ratio (95% interval)": [f"{a:.2f}  ({b:.2f} to {c:.2f})" for a, b, c in zip(AGE_TABLE["hazard_ratio"], AGE_TABLE["low"], AGE_TABLE["high"])]})
        .pivot(index="input", columns="window", values="hazard ratio (95% interval)").reindex(INPUT_ORDER)[PH_LABELS])

# %% [md]
**What this code is.** Figure 8. The hazard ratios of Figure 7 split by age: for every input three dots with their 95% intervals, one for each age window (chart days 1 to 3, 4 to 14, 15 and later). A line at 1, as before. At the right, the p-value of the test that the three are equal.

**Why we need it.** If the ratios of an input differ from window to window, one number cannot describe the whole stay. The figure shows which inputs behave the same at every age and which change, and how.

**What will happen.** The code draws, for every input, three intervals one above the other, coloured by window (a blue ramp from light for the first three chart days to dark for day 15 and later), on a log scale. It writes the Holm-adjusted p-value of the equality test next to each input, in bold when it is below 0.05. It saves the picture as `p6b_08_effects_by_age.png`.

**Expected outcome.** The intervals of the flat inputs (explicit, collaboration, single, song length, debut-day popularity) lie on top of each other around 1. For debut place, album drop and the three inputs about the artist's history the dots of the three windows are visibly apart.

**Motive.** Show where the single hazard ratio of Figure 7 hides a change with age, which is the most important caveat of the model.

# %%
# Figure 08 - hazard ratios by age window
ramp = MP.step_colors(3)                                                      # light for the first chart days, dark for the long stays
rows_y, y_now, previous_block = {}, 0.0, None
for name in INPUT_ORDER:
    if previous_block is not None and block_of[name] != previous_block:
        y_now += 0.6
    rows_y[name] = y_now
    y_now += 1.0
    previous_block = block_of[name]

fig, ax = make_fig(11.8, 9.4, 1, 1, left=3.1, right=1.9, bottom=1.25,
                   title="The ratios for debut place, album drop and the artist's history change with age; the other inputs look the same at every age",
                   subtitle="Hazard ratio of each input in three age windows of a stay (the other inputs keep one ratio), 95% artist-clustered intervals. Right: test that the three ratios "
                            "of an input are equal, Holm-adjusted over the ten inputs.",
                   note_text=f"Exits in the three windows: {FACTS['ph_events_1']}, {FACTS['ph_events_2']} and {FACTS['ph_events_3']}. A hazard ratio above 1: a chart day in that window is more likely to be "
                             "the last one. Debut place is compared with a top-10 debut.")
ax.axvline(1.0, color=style.AXIS, linewidth=1.2, zorder=1)
for row in AGE_TABLE.itertuples():
    y_value = rows_y[row.input] + (row.window_number - 2) * 0.27
    color = ramp[row.window_number - 1]
    ax.hlines(y_value, row.low, row.high, color=color, linewidth=2.2, zorder=2)
    open_dot = row.low <= 1.0 <= row.high
    ax.scatter([row.hazard_ratio], [y_value], s=30, facecolor=style.SURFACE if open_dot else color, edgecolor=color if not open_dot else style.INK_2, linewidth=1.3, zorder=3)
for block, key in BLOCK_KEY.items():
    first = [c for b, columns in BLOCKS if b == block for c in columns][0]
    p_holm = FACTS[f"ph_{key}_p_holm"]
    ax.text(1.02, rows_y[first], format_p(p_holm), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.8, color=style.INK, fontweight="bold" if p_holm < 0.05 else "normal")
ax.text(1.02, -0.85, "equal at every age?", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.6, color=style.INK_2, fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(0.12, 40)
ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20])
ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
ax.minorticks_off()
ax.set_yticks([rows_y[n] for n in INPUT_ORDER])
ax.set_yticklabels([pretty.get(n, n) if n != NO_HISTORY else "No usual popularity yet (flag)" for n in INPUT_ORDER], fontsize=9)
ax.set_ylim(y_now - 0.4, -1.2)
ax.set_xlabel("hazard ratio (log scale)")
swatch_legend(ax, [(label, color) for label, color in zip(PH_LABELS, ramp)], loc="lower right")
style_axes(ax, grid="x", left_spine=False)
ax.tick_params(axis="y", length=0)
save_fig(fig, "p6b_08_effects_by_age", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- *Which inputs change with age.* {{ph_n_flagged}} of the {{ph_n_inputs}} inputs have hazard ratios that differ between the three age windows after the Holm adjustment: {{ph_flagged}}. The other five are consistent with one ratio at every age: single ({{ph_single_p_holm_text}}), explicit ({{ph_explicit_p_holm_text}}), collaboration ({{ph_collab_p_holm_text}}), song length ({{ph_length_p_holm_text}}) and the popularity on the debut day ({{ph_debutpop_p_holm_text}}).
- *Debut place is mostly a first-three-days effect.* For debuts at 31 to 40 the ratio is {{age_b4_1:f1}} (interval {{age_b4_1_low:f1}} to {{age_b4_1_high:f1}}) on chart days 1 to 3, and {{age_b4_2:f1}} and {{age_b4_3:f1}} in the later windows; for debuts at 41 to 50 it is {{age_b5_1:f1}} ({{age_b5_1_low:f1}} to {{age_b5_1_high:f1}}), then {{age_b5_2:f1}} and {{age_b5_3:f1}}. The single ratios of Figure 7 ({{hr_b4:f1}} and {{hr_b5:f1}}) average a very large early effect and a smaller later one. A song that arrives low on the chart is lost mostly in its first three chart days; after that the place of arrival says much less.
- *The album drop matters after the first three chart days, not in them.* Its ratio is {{age_drop_1:f2}} ({{age_drop_1_low:f2}} to {{age_drop_1_high:f2}}) in the first window, {{age_drop_2:f2}} ({{age_drop_2_low:f2}} to {{age_drop_2_high:f2}}) on chart days 4 to 14 and {{age_drop_3:f2}} ({{age_drop_3_low:f2}} to {{age_drop_3_high:f2}}) from day 15.
- *The artist's history goes with a protection in the first three chart days.* If the artist was on the chart the day before, the ratio in the first window is {{age_onchart_1:f2}} ({{age_onchart_1_low:f2}} to {{age_onchart_1_high:f2}}), against {{age_onchart_2:f2}} and {{age_onchart_3:f2}} later, intervals that include 1. The artist's earlier chart days show the same shape ({{age_earlier_1:f2}} in the first window, {{age_earlier_2:f2}} and {{age_earlier_3:f2}} later). Artists without any usual popularity yet (their first appearance in the file) go the other way after the third day: {{age_nohist_2:f2}} on days 4 to 14 and {{age_nohist_3:f2}} from day 15. The usual popularity itself changes sign, {{age_usualpop_1:f2}} in the first window and {{age_usualpop_3:f2}} from day 15; with 42 intervals in the figure, one or two that just exclude 1 are expected by chance, so the test of equality across ages is the sturdier evidence here, not any single interval.
- *What this means for the rest of the notebook.* The ratios of Figure 7 are averages over age, and for the five inputs above they should not be read as the effect on every chart day. The ranking tool of Section 8 uses one ratio per input as well; the C-index it gives is the one for that simple form, and a model that let ratios change with age might rank differently. That was not tried here.
