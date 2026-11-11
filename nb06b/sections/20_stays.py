# %% [md]
## Section 2 - The stays: one definition, and what it does to real songs

A song does not always leave the chart for good the first time it falls off. Some miss a day in the file and come straight back; some are away for a week and return; some are still on the chart when the file stops. Before any curve can be drawn, the project needs one written rule for "how long did this song stay", and the rule has to say what happens in each of those cases. This section states the rule, shows it on four real songs, and draws what the stays of all {{n_songs:n}} new songs look like.

**The rule.** A *stay* starts on the day a song arrives on the chart (the first snapshot day it appears on, after the first day of the file). The clock counts *chart days*: the days the song is actually on the chart, 1 on the arrival day, 2 on the next chart day, and so on. The stay ends on the last chart day before the first absence of {{gap}} snapshot days or more. An absence shorter than that is a *short break*: the song counts as still being on the chart, and the days of the break are not counted. What happens after a stay ends is not part of the stay: a song that comes back after a long absence starts a *later stay*, and this notebook follows only the first one.

A stay is *finished* when we saw it end: either the song came back after an absence of at least {{gap}} snapshot days, or it has not been back for {{gap}} snapshot days up to the end of the file. Otherwise the song may still be on the chart, or may have stepped away for a day or two, and the stay is *censored*: all we know is that it lasted at least as long as we saw.

# %% [md]
**What this code is.** The cell that builds the table of stays. It takes the modeling table of Phase 6A (one row per song that arrived after the first snapshot day, with everything known on the arrival day), numbers the snapshot days, and applies the rule above to every song with `build_lives`. It adds three columns that the later sections need (how many songs of the same artist arrived on the same day, and the place on the chart on the arrival day grouped into five bands) and lists the chart days of every stay with `person_days`.

**Why we need it.** Every number from here to the end is computed from two tables made here. `STAYS` has one row per song, with its stay length and whether the end was seen. `DAYS` has one row per chart day of a stay, with a 1 on the last day of a finished stay; the day-by-day hazard model of Section 6 is fitted on it. The checks in the cell stop the notebook if the two tables disagree with each other or with the modeling table of Phase 6A.

**What will happen.** The code reads the chart rows with their snapshot numbers, builds the stays, checks that the arrival snapshots match the modeling table, joins the columns, and adds `drop_size` (the number of songs of the same primary artist that arrived on the same snapshot day, the song itself included), `log_drop` (its natural logarithm) and `debut_band` (the place on the chart on the arrival day: 1 to 10, 11 to 20, 21 to 30, 31 to 40, 41 to 50). It then stores the headline counts in `FACTS`.

**Expected outcome.** {{n_songs:n}} stays by {{n_artists:n}} artists; {{n_exits:n}} finished and {{n_censored}} censored ({{censored_share:pct1}}); {{n_days:n}} chart days in all. The {{day_one_songs}} songs that were already on the chart on the first day are not in the table.

**Motive.** Make the definition of a stay concrete, in one place, with its own checks, so that no later number depends on a rule that was applied differently in two cells.

# %%
TABLE = build_modeling_table(dim_track, fact_chart, bridge, dim_date)        # one row per song that ARRIVED after the first snapshot day
CHART = chart_with_snapshots(fact_chart, dim_date)                           # (track_id, s, position): s is the number of the snapshot day, 0 = the first
N_SNAPSHOTS = int(dim_date["has_snapshot"].sum())

LIVES = build_lives(CHART, TABLE["track_id"], N_SNAPSHOTS, gap=GAP)          # one row per song: stay length and whether its end was seen
assert (LIVES["entry_snapshot"].to_numpy() == TABLE["entry_snapshot"].to_numpy()).all(), "the stays and the modeling table disagree on arrival days"

STAYS = TABLE.merge(LIVES.drop(columns="entry_snapshot"), on="track_id", how="left", validate="one_to_one")
STAYS["drop_size"] = drop_sizes(STAYS)                                       # songs of the same artist that arrived on the same day (1 = alone)
STAYS["log_drop"] = np.log(STAYS["drop_size"])
STAYS["debut_band"] = pd.cut(STAYS["entry_rank"], [0, 10, 20, 30, 40, 50], labels=BAND_LABELS)
DAYS = person_days(CHART, LIVES)                                             # one row per chart day of a stay

assert len(STAYS) == len(TABLE) and STAYS["stay"].min() >= 1
assert int(DAYS["event"].sum()) == int(STAYS["exit_seen"].sum()), "one event per finished stay"
assert len(DAYS) == int(STAYS["stay"].sum()), "one row per chart day of every stay"
assert STAYS["debut_band"].notna().all()

T = STAYS["stay"].to_numpy(dtype=float)                                      # the clock: chart days of the first stay
E = STAYS["exit_seen"].to_numpy(dtype=bool)                                  # True = the end of the stay was seen
ARTISTS = STAYS["primary_artist"].to_numpy()                                 # the unit of doubt: songs of one artist are not independent

FACTS["n_songs"] = len(STAYS)
FACTS["n_artists"] = STAYS["primary_artist"].nunique()
FACTS["day_one_songs"] = len(dim_track) - len(STAYS)                         # already on the chart on day one: their arrival was not seen
FACTS["n_exits"] = int(E.sum())
FACTS["n_censored"] = int((~E).sum())
FACTS["censored_share"] = float((~E).mean())
FACTS["n_days"] = len(DAYS)
FACTS["n_snapshots_file"] = N_SNAPSHOTS

print(f"{len(STAYS)} stays by {FACTS['n_artists']} artists: {FACTS['n_exits']} finished, {FACTS['n_censored']} censored ({100 * FACTS['censored_share']:.1f}%).")
print(f"{FACTS['n_days']:,} chart days in all.   {FACTS['day_one_songs']} songs were already on the chart on day one and are left out.")
display(STAYS[["track_label", "primary_artist", "entry_snapshot", "entry_rank", "debut_band", "drop_size", "stay", "exit_seen", "short_breaks", "returned"]].head(6))

# %% [md]
**What this code is.** The rule on four real songs. For four songs chosen by a fixed recipe (the first song by arrival date that fits each description), the table writes the chart life as runs ("9 on, 1 away, 3 on" means 9 chart days, 1 snapshot day away, 3 more chart days) and says what the rule makes of it.

**Why we need it.** The rule has one number in it (seven snapshot days) and the exact boundary matters. A song that is away for six snapshot days is still in its stay; a song that is away for seven has finished it. A table with the real runs of real songs shows this better than a sentence does. The fourth song shows what a censored stay is.

**What will happen.** For every song the code finds the lengths of the absences inside its chart life. It then picks the first song by arrival date with (1) one absence of one snapshot day and a stay of at least 8 chart days, (2) one absence of {{gap_below}} snapshot days, (3) one absence of exactly {{gap}} snapshot days, and (4) no absence at all and a stay still open at the end of the file. For each it prints the runs, the stay length, whether the end was seen and what the rule does.

**Expected outcome.** Four rows. Songs 1 and 2 have a short break inside one stay, the second right at the edge of the rule. Song 3 has two runs, but only the first is its stay. Song 4 is censored.

**Motive.** Let the reader check the rule against real songs before trusting any number built on it.

# %%
def runs_of(track_id):
    """The chart life of one song as text: chart days on the chart, snapshot days away, chart days on the chart again ..."""
    s = CHART.loc[CHART["track_id"] == track_id, "s"].to_numpy()
    pieces, start = [], 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] - s[i - 1] > 1:
            pieces.append(f"{i - start} on")
            if i < len(s):
                pieces.append(f"{s[i] - s[i - 1] - 1} away")
            start = i
    return ", ".join(pieces)


absences = {t: np.diff(g["s"].to_numpy()) - 1 for t, g in CHART.groupby("track_id")}       # snapshot days missing between two chart days
absences = {t: a[a > 0] for t, a in absences.items()}                                      # keep only the real absences
by_arrival = STAYS.sort_values("entry_snapshot", kind="mergesort")

def first_song(fits):
    """The first song by arrival date for which `fits(row, absences_of_the_song)` is true."""
    for _, row in by_arrival.iterrows():
        if fits(row, absences[row["track_id"]]):
            return row
    raise ValueError("no song fits the description")

GAP_BELOW = GAP - 1                                                                         # the longest absence that still stays inside a stay
FACTS["gap_below"] = GAP_BELOW
examples = [
    ("one day away inside the stay", first_song(lambda r, a: len(a) == 1 and a[0] == 1 and not r["returned"] and r["exit_seen"] and r["stay"] >= 8)),
    (f"{GAP_BELOW} days away, one short of the rule", first_song(lambda r, a: len(a) == 1 and a[0] == GAP_BELOW and not r["returned"] and r["exit_seen"] and r["stay"] >= 8)),
    (f"{GAP} days away, exactly the rule", first_song(lambda r, a: len(a) == 1 and a[0] == GAP and r["returned"] and r["exit_seen"] and r["stay"] >= 8)),
    ("still on the chart at the end of the file", first_song(lambda r, a: len(a) == 0 and not r["exit_seen"] and r["stay"] >= 10)),
]
SAYS = {0: "the absence is shorter than the rule: it is a short break, the stay goes on",
        1: "still shorter than the rule: the stay goes on and the break is not counted",
        2: "the absence reaches the rule: the stay ends before it, and the return is a later stay",
        3: "nothing ended it: the stay is censored, 'at least this long'"}
rows = []
for k, (what, song) in enumerate(examples):
    rows.append({"case": what, "song": short(song["track_label"], 46), "chart life": runs_of(song["track_id"]), "stay (chart days)": int(song["stay"]),
                 "end seen": "yes" if song["exit_seen"] else "no", "what the rule does": SAYS[k]})
    FACTS[f"example{k + 1}_stay"] = int(song["stay"])
display(pd.DataFrame(rows).set_index("case"))

# %% [md]
**What to notice**

- The first two songs are each one stay, and the days away are not counted: {{example1_stay}} chart days for the song that missed one day, {{example2_stay}} for the song that was away for {{gap_below}} snapshot days. The rule does not care whether the absence was one day or six. That is deliberate: a missing day in the file, a day without a new chart or a week of pause should not cut a song's stay into pieces.
- The third song crosses the line. After {{example3_stay}} chart days it is away for exactly {{gap}} snapshot days, so its stay ends there; the chart day on which it comes back belongs to a later stay and is not counted here. One snapshot day less away and the stay would have run on. A song that came back for months would change a lot by that single day, which is why Section 9 repeats the main results with other rules (an absence of 1, 3, 14 or 30 snapshot days).
- The fourth song has been on the chart for {{example4_stay}} chart days without a break, and the file ends before it does. It is not an exit and not a stay of {{example4_stay}} days either: it is "at least {{example4_stay}}", and every method below uses it that way.


# %% [md]
**What this code is.** Figure 1. Left: how many songs had a stay of 1 chart day, 2 chart days, and so on up to 30, and one bar for all the longer stays. Right: the same {{n_songs:n}} songs sorted by what their chart life looks like.

**Why we need it.** Before any method is applied, look at the raw material. How are the stays spread? How many songs are censored, and how long are those stays? How often do songs leave and come back, which is the reason the rule above had to be written?

**What will happen.** The code counts songs by stay length, separately for finished and censored stays, and draws them as stacked bars (the censored part hatched). It sorts the songs into four exclusive kinds: stay still open at the end of the file; stay ended by an absence and the song later came back; short breaks inside the stay; one unbroken run. It saves the picture as `p6b_01_stays.png`.

**Expected outcome.** A tall bar at 1 chart day and bars that fall quickly, a flat tail, and a single bar for the stays over 30 chart days that is of the same order as the first. On the right, most songs have one unbroken run, and about one song in ten comes back after a long absence.

**Motive.** See the stays before modelling them: their shape, their censoring, and how much the rule had to decide.

# %%
# Figure 01 - what the stays look like
SHOWN = 30                                                                   # one bar per stay length up to 30 chart days, then one bar for the rest
clipped = np.minimum(STAYS["stay"].to_numpy(), SHOWN + 1)
finished_counts = np.bincount(clipped[E], minlength=SHOWN + 2)[1:]
censored_counts = np.bincount(clipped[~E], minlength=SHOWN + 2)[1:]
x_bars = np.arange(1, SHOWN + 2, dtype=float)
x_bars[-1] = SHOWN + 2.2                                                     # a small gap before the bar of the long stays

kind = np.where(~E, "open", np.where(STAYS["returned"], "returned", np.where(STAYS["short_breaks"] > 0, "breaks", "one run")))
kind_counts = {name: int((kind == name).sum()) for name in ["one run", "breaks", "returned", "open"]}
FACTS["kind_one_run"], FACTS["kind_breaks"] = kind_counts["one run"], kind_counts["breaks"]
FACTS["kind_returned"], FACTS["kind_open"] = kind_counts["returned"], kind_counts["open"]
for name in kind_counts:
    FACTS[f"kind_{name.replace(' ', '_')}_share"] = kind_counts[name] / len(STAYS)
FACTS["one_run_all"] = int(((STAYS["first_run_days"] == STAYS["stay"]) & ~STAYS["returned"] & (STAYS["short_breaks"] == 0)).sum())
FACTS["stay_median_raw"] = float(np.median(T))
FACTS["stay_mean_raw"] = float(T.mean())
FACTS["share_le2"] = float((T <= 2).mean())
FACTS["share_stay_one"] = float((T == 1).mean())
FACTS["n_gt30"], FACTS["share_gt30"] = int((T > 30).sum()), float((T > 30).mean())
FACTS["n_gt120"] = int((T > 120).sum())
FACTS["stay_max"] = int(T.max())
FACTS["n_stay_one"] = int((T == 1).sum())
FACTS["n_run_differs"] = int((STAYS["first_run_days"] != STAYS["stay"]).sum())                  # songs whose stay is not their first unbroken run
FACTS["share_run_differs"] = FACTS["n_run_differs"] / len(STAYS)
FACTS["stay_median_finished"] = float(np.median(T[E]))
FACTS["stay_median_open"] = float(np.median(T[~E]))
FACTS["open_late_arrivals"] = int((~E & (STAYS["entry_snapshot"].to_numpy() >= N_SNAPSHOTS - 30)).sum())      # arrived in the last 30 snapshot days of the file
FACTS["open_long_stays"] = int((~E & (T >= 30)).sum())
FACTS["open_both"] = int((~E & (STAYS["entry_snapshot"].to_numpy() >= N_SNAPSHOTS - 30) & (T >= 30)).sum())

fig, (axa, axb) = make_fig(11.8, 5.3, 1, 2, left=0.85, right=0.45, bottom=0.95, wspace=0.18, width_ratios=[1.45, 1], panel_titles=True,
                           title=f"Half of the stays seen are {FACTS['stay_median_raw']:.0f} chart days or shorter, yet {FACTS['n_gt30']} of the {FACTS['n_songs']} songs stayed longer than {HORIZON}",
                           subtitle=f"The first stay of each of the {FACTS['n_songs']} songs that arrived after the first snapshot day. Left: songs by chart days of the stay. "
                                    "Right: songs by what their chart life looks like.",
                           note_text=f"A stay ends at the last chart day before the first absence of {GAP} snapshot days or more. Hatched = the stay was still open at the end of the file "
                                     f"(censored). The longest stay is {FACTS['stay_max']} chart days.")
axa.bar(x_bars, finished_counts, width=0.82, color=style.BLUE, label="end of the stay seen")
axa.bar(x_bars, censored_counts, width=0.82, bottom=finished_counts, color=style.SURFACE, edgecolor=style.BLUE, hatch="////", linewidth=0.8, label="stay still open")
axa.set_xticks([1] + list(range(5, SHOWN + 1, 5)) + [x_bars[-1]])
axa.set_xticklabels(["1"] + [str(v) for v in range(5, SHOWN + 1, 5)] + [f"{SHOWN + 1}+"])
axa.set_xlabel("chart days of the first stay")
axa.set_ylabel("songs")
axa.set_ylim(0, (finished_counts + censored_counts).max() * 1.12)
axa.legend(handles=[Patch(facecolor=style.BLUE, edgecolor=style.BLUE, label="end of the stay seen"),
                    Patch(facecolor=style.SURFACE, edgecolor=style.BLUE, hatch="////", label="stay still open at the end of the file (censored)")],
           loc="upper center", bbox_to_anchor=(0.5, 0.97), frameon=False, fontsize=9, labelcolor=style.INK_2, handlelength=1.6)
style_axes(axa, grid="y")
panel_title(axa, "Many stays are short, some are very long")

labels = ["One unbroken run, then gone", f"Short breaks inside the stay (under {GAP} snapshot days)", "Left for a long absence, came back later",
          "Stay still open at the end of the file"]
counts = [kind_counts["one run"], kind_counts["breaks"], kind_counts["returned"], kind_counts["open"]]
y = np.arange(len(counts))
bars = axb.barh(y, counts, height=0.3, color=style.BLUE)
bars[3].set_facecolor(style.SURFACE); bars[3].set_edgecolor(style.BLUE); bars[3].set_hatch("////"); bars[3].set_linewidth(0.8)
for yi, label, count in zip(y, labels, counts):
    axb.text(0, yi - 0.27, label, ha="left", va="bottom", fontsize=9, color=style.INK_2)
    axb.text(count + max(counts) * 0.015, yi, f"{count}  ({100 * count / len(STAYS):.0f}%)", ha="left", va="center", fontsize=9, color=style.INK_2)
axb.set_yticks([])
axb.set_ylim(len(counts) - 0.45, -0.75)
axb.set_xlim(0, max(counts) * 1.28)
axb.set_xlabel("songs")
style_axes(axb, grid="x", left_spine=False)
panel_title(axb, "What the chart life looks like")
save_fig(fig, "p6b_01_stays", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- The stays are very uneven. Half of the stays seen are {{stay_median_raw:f0}} chart days or shorter, yet the mean is {{stay_mean_raw:f1}} and the longest is {{stay_max}}: a few very long stays pull the average far above a typical stay. {{n_stay_one}} songs ({{share_stay_one:pct0}}) were on the chart for a single day, and {{n_gt30}} ({{share_gt30:pct0}}) for more than {{horizon}} chart days, {{n_gt120}} of them for more than 120. For a measure like this the average says little; curves and medians do better, and that is what the next section draws.
- {{n_censored}} stays ({{censored_share:pct1}}) are still open at the end of the file. They are not short ones that went missing: their median is {{stay_median_open:f0}} chart days against {{stay_median_finished:f0}} for the finished stays. {{open_late_arrivals}} of them arrived in the last 30 snapshot days of the file and simply have had no time to finish, and {{open_long_stays}} have already stayed 30 chart days or more. Leaving them out, or treating them as finished on their last day, would remove or cut short exactly the longest-lasting songs.
- {{kind_one_run}} songs ({{kind_one_run_share:pct0}}) have one unbroken run that ends and does not return. {{kind_breaks}} ({{kind_breaks_share:pct0}}) miss at least one snapshot day inside their stay, and {{kind_returned}} ({{kind_returned_share:pct0}}) leave for a long absence and come back. For {{n_run_differs}} songs ({{share_run_differs:pct0}}) the stay is not the same as the first unbroken run, so a rule that treated any absence as the end (the rule of Phase 5) would have cut their stays short. The definition of this notebook is the more careful one.
