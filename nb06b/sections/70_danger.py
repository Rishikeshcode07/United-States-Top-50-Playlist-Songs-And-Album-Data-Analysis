# %% [md]
## Section 7 - The danger zone: the place on the chart

The model of Section 6 used the place on the arrival day. A song's place changes every day, though, and the place of a song *today* is probably the best single clue to whether it is gone *tomorrow*. This section measures that directly, without a model: for every chart day, take the place the song held and whether that day was the last one of its stay, and count. It is a description of the chart, not of a cause: a song that has slipped to place 47 is on its way out, and the table says how often it goes.

# %% [md]
**What this code is.** A count of exits by place. The {{n_days:n}} chart days of the stays are sorted into ten bands of five places (1 to 5, 6 to 10, ... 46 to 50). For each band the code finds the chance that a chart day in it is the last day of a stay (exits divided by chart days), the share of all exits that happen in the band, and the share of all chart days that are spent in it. The same chance is also found separately for the first three chart days of a stay and for the later ones.

**Why we need it.** Section 6 told us that where a song arrives matters, mostly in the first days. This asks the other question: wherever it arrived, at what places do songs leave? If exits are spread evenly, the place is no clue; if they are concentrated in a narrow zone at the bottom, then the zone is where a chart watcher should look.

**What will happen.** The code adds up, for each song and each place band, the chart days and the exits. Multiplying the weights of each artist draw by these sums gives the chance, the share of exits and the share of chart days for every draw at once, and the 95% band is read from them. The separate first-three-days and later chances are done the same way. One table is printed.

**Expected outcome.** A chance that rises steadily with the place band, from a fraction of a per cent in the top five to more than one chart day in five at places 46 to 50. A majority of all exits from the last ten places, although they take up a much smaller share of the chart days.

**Motive.** Show where on the chart the exits happen, in a way that needs no model and no assumption.

# %%
PLACE_EDGES = list(range(0, 51, 5))                                          # places 1-5, 6-10 ... 46-50
PLACE_LABELS = [f"{low + 1}-{low + 5}" for low in PLACE_EDGES[:-1]]
N_BINS = len(PLACE_LABELS)
DAY_BIN = np.digitize(DAYS["position"].to_numpy(), PLACE_EDGES[1:-1], right=True)         # 0 ... 9: the band of the place a song held on that chart day
EARLY = DAYS["age"].to_numpy() <= 3                                          # the first three chart days of a stay

# for each song and place band: chart days and exits (all, and split by early / later)
SONG_DAYS, SONG_EXITS = np.zeros((len(STAYS), N_BINS)), np.zeros((len(STAYS), N_BINS))
EARLY_DAYS, EARLY_EXITS = np.zeros((len(STAYS), N_BINS)), np.zeros((len(STAYS), N_BINS))
np.add.at(SONG_DAYS, (SONG_ROW, DAY_BIN), 1.0)
np.add.at(SONG_EXITS, (SONG_ROW, DAY_BIN), DAY_Y.astype(float))
np.add.at(EARLY_DAYS, (SONG_ROW[EARLY], DAY_BIN[EARLY]), 1.0)
np.add.at(EARLY_EXITS, (SONG_ROW[EARLY], DAY_BIN[EARLY]), DAY_Y[EARLY].astype(float))
LATE_DAYS, LATE_EXITS = SONG_DAYS - EARLY_DAYS, SONG_EXITS - EARLY_EXITS
assert int(SONG_DAYS.sum()) == FACTS["n_days"] and int(SONG_EXITS.sum()) == FACTS["n_exits"]


def chance_with_band(exits, days):
    """Exits divided by chart days per band, for the data and with the 95% artist band (every draw is one matrix product)."""
    estimate = exits.sum(axis=0) / days.sum(axis=0)
    draws = (W_BOOT @ exits) / np.maximum(W_BOOT @ days, 1e-9)
    return estimate, np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


chance, chance_low, chance_high = chance_with_band(SONG_EXITS, SONG_DAYS)
early, early_low, early_high = chance_with_band(EARLY_EXITS, EARLY_DAYS)
late, late_low, late_high = chance_with_band(LATE_EXITS, LATE_DAYS)
share_exits = SONG_EXITS.sum(axis=0) / SONG_EXITS.sum()
share_days = SONG_DAYS.sum(axis=0) / SONG_DAYS.sum()
exits_draws = W_BOOT @ SONG_EXITS
share_exits_low, share_exits_high = np.percentile(exits_draws / exits_draws.sum(axis=1, keepdims=True), [2.5, 97.5], axis=0)

DANGER_TABLE = pd.DataFrame({"places": PLACE_LABELS, "chart_days": SONG_DAYS.sum(axis=0).astype(int), "exits": SONG_EXITS.sum(axis=0).astype(int),
                             "chance_last_day": chance, "low": chance_low, "high": chance_high, "share_of_chart_days": share_days, "share_of_exits": share_exits,
                             "share_of_exits_low": share_exits_low, "share_of_exits_high": share_exits_high,
                             "chance_first_3_days": early, "chance_first_3_low": early_low, "chance_first_3_high": early_high,
                             "chance_later": late, "chance_later_low": late_low, "chance_later_high": late_high,
                             "exits_first_3_days": EARLY_EXITS.sum(axis=0).astype(int), "chart_days_first_3": EARLY_DAYS.sum(axis=0).astype(int)})
for k, row in enumerate(DANGER_TABLE.itertuples(), start=1):
    FACTS[f"dz{k}"], FACTS[f"dz{k}_low"], FACTS[f"dz{k}_high"] = float(row.chance_last_day), float(row.low), float(row.high)
    FACTS[f"dz{k}_exits"], FACTS[f"dz{k}_days"] = int(row.exits), int(row.chart_days)
    FACTS[f"dz{k}_share_exits"], FACTS[f"dz{k}_share_days"] = float(row.share_of_exits), float(row.share_of_chart_days)
    FACTS[f"dz{k}_early"], FACTS[f"dz{k}_late"] = float(row.chance_first_3_days), float(row.chance_later)
FACTS["dz_ratio"] = float(chance[-1] / chance[0])
FACTS["dz_bottom_exit_share"] = float(share_exits[-2:].sum())                # places 41-50
low_b, high_b = np.percentile(exits_draws[:, -2:].sum(axis=1) / exits_draws.sum(axis=1), [2.5, 97.5])
FACTS["dz_bottom_exit_share_low"], FACTS["dz_bottom_exit_share_high"] = float(low_b), float(high_b)
FACTS["dz_bottom_days_share"] = float(share_days[-2:].sum())
FACTS["dz_top_exit_share"] = float(share_exits[:2].sum())                    # places 1-10
FACTS["dz_top_days_share"] = float(share_days[:2].sum())
FACTS["dz_first3_exit_share"] = float(EARLY_EXITS.sum() / SONG_EXITS.sum())
FACTS["dz_first3_days_share"] = float(EARLY_DAYS.sum() / SONG_DAYS.sum())
display(DANGER_TABLE.assign(**{"chance that the day is the last": [f"{100 * v:.1f}%  ({100 * a:.1f} to {100 * b:.1f})" for v, a, b in zip(DANGER_TABLE["chance_last_day"], DANGER_TABLE["low"], DANGER_TABLE["high"])],
                               "share of all exits": [f"{100 * v:.1f}%" for v in DANGER_TABLE["share_of_exits"]], "share of chart days": [f"{100 * v:.1f}%" for v in DANGER_TABLE["share_of_chart_days"]],
                               "first 3 chart days": [f"{100 * v:.1f}%" for v in DANGER_TABLE["chance_first_3_days"]], "later": [f"{100 * v:.1f}%" for v in DANGER_TABLE["chance_later"]]})
        [["places", "chart_days", "exits", "chance that the day is the last", "share of all exits", "share of chart days", "first 3 chart days", "later"]].set_index("places"))
FACTS["dz_steps_up"] = int((np.diff(chance) > 0).sum())                       # of the nine steps from one band to the next, how many go up
FACTS["dz_early_higher_bands"] = int((early > late).sum())

# %% [md]
**What to notice**

- The chance rises as the place gets worse. A chart day in places 1 to 5 is the last one of its stay {{dz1:pct1}} of the time (95% band {{dz1_low:pct1}} to {{dz1_high:pct1}}); in places 21 to 25, {{dz5:pct1}}; in places 41 to 45, {{dz9:pct1}}; and in places 46 to 50, {{dz10:pct1}} ({{dz10_low:pct1}} to {{dz10_high:pct1}}), about {{dz_ratio:f0}} times the rate at the top. The estimate goes up at {{dz_steps_up}} of the 9 steps from one band to the next. Next-door bands near the top overlap in their bands, so what the data show is a rise across the chart, not a difference between two neighbouring bands.
- The rise is there both in the first three chart days of a stay ({{dz1_early:pct1}} in places 1 to 5, {{dz10_early:pct1}} in places 46 to 50) and in the later chart days ({{dz1_late:pct1}} and {{dz10_late:pct1}}). The first three days carry the higher chance in {{dz_early_higher_bands}} of the 10 bands, which is the age effect of Section 3 again; the place comes on top of it.

# %% [md]
### One busy day in the data

The counts above treat every chart day alike. But exits are not spread evenly over the calendar: songs of one album leave together, and a seasonal group can leave together. Before reading the table as a description of the chart, the next cell checks whether a single day is doing much of the work.

# %% [md]
**What this code is.** A check on days. It counts the {{n_exits}} exits by calendar day, lists the five days with the most exits (how many of them were in the top ten places and how many in places 41 to 50), shows the songs that left from the top ten places on the busiest day, and repeats the chance-by-place calculation with that one day left out.

**Why we need it.** If one day held dozens of exits and most of them sat in one place band, the pattern above would partly be that day, and a reader who trusts the bands would be misled. The bands count artists as the unit but cannot see that a whole block of songs left on the same day.

**What will happen.** The code counts exits per snapshot day and prints the five busiest. For the busiest it prints the songs that left from places 1 to 10. It then removes that snapshot day from the table of chart days (every chart day on that date, with its exits) and finds the chance in each place band again, with the same artist bands. Three small tables are printed: the busiest days, the {{busy_top10}} songs that left from the top ten on the busiest day, and the chance by place with and without that day.

**Expected outcome.** One day, {{busy_text}}, with far more exits than any other ({{busy_exits}} against {{busy2_exits}} on the second busiest), a large part of them in the top ten, and a place pattern that is much the same without it.

**Motive.** Be sure that the place pattern describes the chart and not one day, and say plainly where the figures of the top places rest on very few events.

# %%
SNAP_DATES = dim_date[dim_date["has_snapshot"]].sort_values("date")["date"].reset_index(drop=True)             # the calendar date of snapshot day number s
EXIT_ROWS = DAYS[DAYS["event"] == 1]
EXITS_BY_DAY = pd.DataFrame({"exits": EXIT_ROWS.groupby("s").size(),
                             "in_top_ten": EXIT_ROWS[EXIT_ROWS["position"] <= 10].groupby("s").size(),
                             "in_places_41_50": EXIT_ROWS[EXIT_ROWS["position"] > 40].groupby("s").size()}).fillna(0).astype(int)
EXITS_BY_DAY = EXITS_BY_DAY.sort_values(["exits"], ascending=False, kind="stable")                    # equal counts keep the calendar order
BUSY = int(EXITS_BY_DAY.index[0])                                                                      # the snapshot day with the most exits
FACTS["exit_days"] = int(len(EXITS_BY_DAY))
FACTS["busy_text"] = f"{SNAP_DATES[BUSY].day} {SNAP_DATES[BUSY]:%B %Y}"
for rank in range(1, 5):
    FACTS[f"busy{rank}_exits"] = int(EXITS_BY_DAY["exits"].iloc[rank - 1])
    FACTS[f"busy{rank}_text"] = f"{SNAP_DATES[int(EXITS_BY_DAY.index[rank - 1])].day} {SNAP_DATES[int(EXITS_BY_DAY.index[rank - 1])]:%B %Y}"
FACTS["busy_exits"], FACTS["busy_share"] = FACTS["busy1_exits"], float(EXITS_BY_DAY["exits"].iloc[0] / FACTS["n_exits"])
FACTS["busy_top10"], FACTS["busy_bottom10"] = int(EXITS_BY_DAY["in_top_ten"].iloc[0]), int(EXITS_BY_DAY["in_places_41_50"].iloc[0])
FACTS["busy5_share"] = float(EXITS_BY_DAY["exits"].iloc[:5].sum() / FACTS["n_exits"])
FACTS["dz_top_exits"] = int(SONG_EXITS[:, :2].sum())                                                   # exits in places 1-10, all days
FACTS["dz_bottom_exits"] = int(SONG_EXITS[:, -2:].sum())                                               # exits in places 41-50, all days
display(EXITS_BY_DAY.head(5).assign(date=[f"{SNAP_DATES[s]:%d %b %Y}" for s in EXITS_BY_DAY.index[:5]],
                                    share_of_all_exits=[f"{100 * v / FACTS['n_exits']:.1f}%" for v in EXITS_BY_DAY["exits"].iloc[:5]])
        [["date", "exits", "share_of_all_exits", "in_top_ten", "in_places_41_50"]].rename(columns={"in_top_ten": "in places 1-10", "in_places_41_50": "in places 41-50"}))

TOP_TEN_BUSY = (EXIT_ROWS[(EXIT_ROWS["s"] == BUSY) & (EXIT_ROWS["position"] <= 10)].merge(STAYS[["track_id", "track_label"]], on="track_id")
                .sort_values("position")[["position", "track_label", "age"]].rename(columns={"position": "place that day", "track_label": "song", "age": "chart days in the stay"}))
FACTS["busy_top10_min_days"], FACTS["busy_top10_max_days"] = int(TOP_TEN_BUSY["chart days in the stay"].min()), int(TOP_TEN_BUSY["chart days in the stay"].max())
display(TOP_TEN_BUSY.set_index("place that day"))

KEEP = (DAYS["s"] != BUSY).to_numpy()
DAYS_X, EXITS_X = np.zeros((len(STAYS), N_BINS)), np.zeros((len(STAYS), N_BINS))
np.add.at(DAYS_X, (SONG_ROW[KEEP], DAY_BIN[KEEP]), 1.0)
np.add.at(EXITS_X, (SONG_ROW[KEEP], DAY_BIN[KEEP]), DAY_Y[KEEP].astype(float))
chance_x, chance_x_low, chance_x_high = chance_with_band(EXITS_X, DAYS_X)
exits_x_draws = W_BOOT @ EXITS_X
for k in range(1, N_BINS + 1):
    FACTS[f"dzx{k}"], FACTS[f"dzx{k}_low"], FACTS[f"dzx{k}_high"] = float(chance_x[k - 1]), float(chance_x_low[k - 1]), float(chance_x_high[k - 1])
    FACTS[f"dzx{k}_exits"] = int(EXITS_X[:, k - 1].sum())
FACTS["dzx_steps_up"] = int((np.diff(chance_x) > 0).sum())
FACTS["dzx_bottom_exit_share"] = float(EXITS_X[:, -2:].sum() / EXITS_X.sum())
FACTS["dzx_top_exits"] = int(EXITS_X[:, :2].sum())
display(pd.DataFrame({"places": PLACE_LABELS, "exits, all days": SONG_EXITS.sum(axis=0).astype(int), "exits, without that day": EXITS_X.sum(axis=0).astype(int),
                      "chance, all days": [f"{100 * v:.1f}%" for v in chance], "chance, without that day": [f"{100 * v:.1f}%  ({100 * a:.1f} to {100 * b:.1f})" for v, a, b in zip(chance_x, chance_x_low, chance_x_high)]}).set_index("places"))

# %% [md]
**What to notice**

- Exits come in blocks. {{busy_exits}} of the {{n_exits}} exits ({{busy_share:pct1}}) fall on one day, {{busy_text}}: a whole block of Christmas songs, {{busy_top10}} of them in the top ten after {{busy_top10_min_days}} to {{busy_top10_max_days}} chart days (the second table lists them), ends its stay on that one day and is absent for at least the next {{gap}} snapshot days. The file does not say why. Whatever the cause, it is a block of songs leaving together, not {{busy_exits}} separate slides down the chart. The next busiest days have {{busy2_exits}} ({{busy2_text}}), {{busy3_exits}} and {{busy4_exits}} exits; the five busiest days hold {{busy5_share:pct0}} of all exits, out of {{exit_days}} days with at least one.
- Without that day the chance by place has the same shape: {{dzx1:pct1}} in places 1 to 5 rising to {{dzx10:pct1}} in places 46 to 50, with the estimate going up at {{dzx_steps_up}} of the 9 steps, and {{dzx_bottom_exit_share:pct0}} of the remaining exits are still in places 41 to 50. The day matters at the top: it holds {{busy_top10}} of the {{dz_top_exits}} exits in places 1 to 10, leaving {{dzx_top_exits}}, so the figures for the top two bands rest on very few events. In places 41 to 50 it holds only {{busy_bottom10}} of {{dz_bottom_exits}}.
- One limit is worth stating. The artist bands count artists as the unit, and they cannot see that songs of different artists left on the same day. The check above is the answer for this section; the same day is not removed from the later sections, where the bands should be read with this in mind.

# %% [md]
### Figure 9: where on the chart the stays end

# %% [md]
**What this code is.** The drawing of Figure 9. Both panels list the ten place bands from the top of the chart to the bottom. On the left, the chance that a chart day in the band is the last one of its stay (bar), its 95% artist band (line) and, as an orange dot, the same chance with {{busy_text}} left out. On the right, each band's share of all exits (blue bar) with its share of all chart days marked as a black tick, which is where the bar would end if exits were spread like the chart days.

**Why we need it.** The table holds the numbers, but the picture shows at once that exits pile up at the bottom while chart days are spread almost evenly over the places, and that taking one day out hardly moves the bars.

**What will happen.** Horizontal bars with whiskers and a value at the end of each. The picture is saved as `p6b_09_danger_zone.png` and shown.

**Expected outcome.** On the left, bars that grow from almost nothing at the top to the longest at the bottom, with the orange dots slightly inside the bars at the top and almost on them at the bottom. On the right, blue bars that grow towards the bottom while the ticks (the chart days) shrink slowly, so that the bars are far left of their ticks at the top and far right of them at the bottom.

**Motive.** Show where on the chart the exits happen, and that the pattern does not come from one day.

# %%
# Figure 09 - where on the chart the stays end
fig, (axa, axb) = make_fig(11.8, 5.9, 1, 2, left=1.0, right=0.5, bottom=0.95, wspace=0.3, width_ratios=[1.15, 1], panel_titles=True,
                           title=f"Songs leave from the bottom of the chart: places 41 to 50 hold {100 * FACTS['dz_bottom_days_share']:.0f}% of the chart days "
                                 f"and {100 * FACTS['dz_bottom_exit_share']:.0f}% of the exits",
                           subtitle=f"All {FACTS['n_songs']} stays; {FACTS['n_exits']} exits over {FACTS['n_days']:,} chart days. Place = place on the chart on that day, in bands of five. "
                                    "Left: exits divided by the chart days spent in the band. Right: each band's share of all exits (bar) and of all chart days (tick). Whiskers: 95% artist band.",
                           note_text=f"The orange dots leave out {FACTS['busy_text']}, when {FACTS['busy_exits']} stays ended together (a block of Christmas songs, {FACTS['busy_top10']} of them in the top ten). "
                                     "The figure describes the chart: a song does not leave because of its place, it sits there because it is fading.")
y = np.arange(N_BINS)
axa.barh(y, chance, height=0.62, color=style.BLUE)
axa.hlines(y, chance_low, chance_high, color=style.INK_2, linewidth=1.1, zorder=4)
axa.scatter(chance_x, y, s=26, color=style.ORANGE, edgecolor=style.SURFACE, linewidth=0.9, zorder=5)
for yi, value, low, high in zip(y, chance, chance_low, chance_high):
    direct_label(axa, high + 0.007, yi, f"{100 * value:.1f}%  ({100 * low:.1f} to {100 * high:.1f})", size=8.6, color=style.INK)
axa.set_yticks(y)
axa.set_yticklabels(PLACE_LABELS)
axa.set_ylim(N_BINS - 0.5, -0.6)
axa.set_xlim(0, 0.43)
axa.xaxis.set_major_formatter(SP.percent_axis)
axa.set_xlabel("chance that the chart day is the last one of the stay")
axa.set_ylabel("place on the chart that day")
style_axes(axa, grid="x", left_spine=False)
axa.tick_params(axis="y", length=0)
swatch_legend(axa, [("all chart days", style.BLUE), (f"without {FACTS['busy_text']}", style.ORANGE, "dot")], loc="upper right")
panel_title(axa, "The chance of leaving rises with every step down")

axb.barh(y, share_exits, height=0.5, color=style.BLUE)
axb.hlines(y, share_exits_low, share_exits_high, color=style.INK_2, linewidth=1.0, zorder=4)
(tick,) = axb.plot(share_days, y, marker="|", linestyle="none", color=style.INK, markersize=13, markeredgewidth=2.2, zorder=5)
for yi, a, high, b in zip(y, share_exits, share_exits_high, share_days):
    direct_label(axb, max(a, high, b) + 0.014, yi, f"{100 * a:.0f}%  ({100 * b:.0f}% of days)", size=8.6, color=style.INK)
direct_label(axb, 0.31, 4.1, "Places 41 to 50 together:", size=8.8, color=style.INK)
direct_label(axb, 0.31, 4.7, f"{100 * FACTS['dz_bottom_exit_share']:.0f}% of the exits ({100 * FACTS['dz_bottom_exit_share_low']:.0f} to {100 * FACTS['dz_bottom_exit_share_high']:.0f})", size=8.8, color=style.INK)
direct_label(axb, 0.31, 5.3, f"{100 * FACTS['dz_bottom_days_share']:.0f}% of the chart days", size=8.8, color=style.INK)
axb.set_yticks(y)
axb.set_yticklabels(PLACE_LABELS)
axb.set_ylim(N_BINS - 0.5, -0.6)
axb.set_xlim(0, 0.64)
axb.xaxis.set_major_formatter(SP.percent_axis)
axb.set_xlabel("share of all exits that happen in the band")
style_axes(axb, grid="x", left_spine=False)
axb.tick_params(axis="y", length=0)
axb.legend(handles=[Patch(facecolor=style.BLUE), tick], labels=["bar: share of all exits", "tick: share of all chart days"], loc="upper right", fontsize=8.6,
           handletextpad=0.5, borderaxespad=0.2)
panel_title(axb, "Exits pile up at the bottom, chart days do not")
save_fig(fig, "p6b_09_danger_zone", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- Songs leave from the bottom. The last ten places hold {{dz_bottom_days_share:pct0}} of the chart days and {{dz_bottom_exit_share:pct0}} of the exits (95% band {{dz_bottom_exit_share_low:pct0}} to {{dz_bottom_exit_share_high:pct0}}), {{dz_bottom_exits}} of the {{n_exits}}; places 46 to 50 alone hold {{dz10_share_exits:pct0}} of the exits from {{dz10_share_days:pct0}} of the chart days. Places 1 to 10, a quarter of the chart days ({{dz_top_days_share:pct0}}), hold {{dz_top_exit_share:pct0}} of the exits ({{dz_top_exits}}, of which {{busy_top10}} are the block of {{busy_text}}).
- In practical terms: a song in places 41 to 45 has a chance of about {{dz9:pct0}}, and a song in places 46 to 50 about {{dz10:pct0}}, that today is the last day of its stay. In places 21 to 25 the chance is {{dz5:pct1}}. These are rates of past stays, measured as they happened; they are not a forecast for a song of today.
- The place of the day is a symptom of a fading song, not a lever. A song does not leave because it sits at place 47; it sits at 47 because it is on its way down, and leaving is where that road ends. The table says how wide the door is at each place and nothing about what pushes a song to it. For this reason none of the models in this notebook uses the place on a later day: they use what is known when the song arrives (Section 6) or after its first three chart days (Section 8), and a later place would explain the end of a stay with the stay itself.
