# %% [md]
## Section 3 - The survival curve and the hazard by age

Two pictures describe how long songs stay. The *survival curve* S(a) gives, for every number of chart days a, the share of songs whose stay lasts *longer* than a chart days. It starts at 100% and can only go down. The *hazard* of a chart day is the chance that it is the last day of a stay, given that the song reached it. The curve tells the story in one line; the hazard says where on the way the songs are lost, and it is the quantity that the model of Section 6 works with.

A word on how the curve is made. It is the Kaplan-Meier estimate: on every chart day a, take the songs still on the chart on that day, find what share of them left after it, and multiply the shares of all days up to a. A song whose stay is still open counts as being on the chart for as many days as we saw it and is never counted as an exit; that is how the {{n_censored}} open stays are kept without being treated as finished.

# %% [md]
**What this code is.** The survival curve of all {{n_songs:n}} stays, with two kinds of band around it, and the numbers read off it. The first band comes from the artist bootstrap: draw {{n_boot:n}} sets of artists with replacement (a whole artist, with all of that artist's songs, is drawn or not drawn), recompute the curve for each set, and take the middle 95% of the results at every age. The second band is the textbook Greenwood interval, which treats every song as an independent observation.

**Why we need it.** A single curve says nothing about how sure we can be of it. Songs of one artist arrive together and leave together, and Phase 5 measured how alike they are, so {{n_songs:n}} songs carry less information than {{n_songs:n}} unrelated songs would. The artist bootstrap measures the uncertainty that this leaves; the Greenwood band shows what would have been claimed if the relation between songs had been ignored. The cell also shows what happens if the open stays are mishandled: treated as finished on their last day, or dropped.

**What will happen.** The code draws the artist weights once ({{n_boot:n}} rows of {{n_songs:n}} weights; they are reused by every later section), computes the curve of the data and of every draw, and the Greenwood band from `kaplan_meier` of Phase 5. It checks that the two Kaplan-Meier calculations agree. It stores S at 3, 7, 14, 30, 60 and 90 chart days with both bands, the median stay, the average number of chart days used out of the first {{horizon}}, and the same figures for the two mishandled versions. It prints one table.

**Expected outcome.** About 63% of the songs are still on the chart after 3 chart days, 43% after 7, 30% after 14, 19% after 30 and 7% after 90, with a median stay of 6 chart days. The artist band is visibly wider than the Greenwood band. Treating open stays as finished or dropping them lowers the share still on the chart after 30 chart days.

**Motive.** Say how long songs stay and how sure we are, in a way that does not pretend that songs of one artist are independent.

# %%
W_BOOT = np.vstack(list(cluster_bootstrap_weights(ARTISTS, N_BOOT, SEED)))   # one row per draw: how many times each song's artist was drawn (0, 1, 2 ...)
FACTS["boot_draws"] = W_BOOT.shape[0]
assert W_BOOT.shape == (N_BOOT, len(STAYS))

AGES = np.arange(0, int(T.max()) + 1)                                        # chart days 0, 1, 2 ... up to the longest stay
S_OBS = survival_grid(T, E, AGES)                                            # S(a): the share of songs whose stay lasts MORE than a chart days
S_BOOT = survival_grid(T, E, AGES, weights=W_BOOT)                           # the same curve for every artist draw (one row each)
S_LOW, S_HIGH = np.percentile(S_BOOT, [2.5, 97.5], axis=0)                   # the 95% band of the artist bootstrap

km = kaplan_meier(T, E)                                                      # the textbook estimate, with its Greenwood interval (songs treated as independent)
at = np.searchsorted(km["time"].to_numpy(), AGES, side="right") - 1
assert np.allclose(km["survival"].to_numpy()[at], S_OBS), "the two Kaplan-Meier calculations disagree"
GW_LOW, GW_HIGH = km["ci_low"].to_numpy()[at], km["ci_high"].to_numpy()[at]

KEY_AGES = [1, 2, 3, 7, 14, 30, 60, 90]
for a in KEY_AGES:
    FACTS[f"s{a}"], FACTS[f"s{a}_low"], FACTS[f"s{a}_high"] = float(S_OBS[a]), float(S_LOW[a]), float(S_HIGH[a])
    FACTS[f"gw{a}_low"], FACTS[f"gw{a}_high"] = float(GW_LOW[a]), float(GW_HIGH[a])
FACTS["band_ratio_30"] = float((S_HIGH[30] - S_LOW[30]) / (GW_HIGH[30] - GW_LOW[30]))     # how much wider the artist band is than the textbook band at 30 days

medians = median_stay(S_BOOT)
FACTS["median_stay"] = median_stay(S_OBS)
FACTS["median_stay_low"], FACTS["median_stay_high"] = (float(v) for v in np.nanpercentile(medians, [2.5, 97.5]))
FACTS["rmst30"] = float(restricted_mean(S_OBS, HORIZON))
FACTS["rmst30_low"], FACTS["rmst30_high"] = (float(v) for v in np.percentile(restricted_mean(S_BOOT, HORIZON), [2.5, 97.5]))

# ---- what happens if the open stays are mishandled
S_AS_FINISHED = survival_grid(T, np.ones(len(T), dtype=bool), AGES)          # every stay treated as finished on the last day we saw it
S_DROPPED = survival_grid(T[E], np.ones(int(E.sum()), dtype=bool), AGES)     # the open stays removed from the data
FACTS["s30_as_finished"], FACTS["s30_dropped"] = float(S_AS_FINISHED[30]), float(S_DROPPED[30])
FACTS["median_as_finished"], FACTS["median_dropped"] = median_stay(S_AS_FINISHED), median_stay(S_DROPPED)
FACTS["rmst30_as_finished"], FACTS["rmst30_dropped"] = float(restricted_mean(S_AS_FINISHED, HORIZON)), float(restricted_mean(S_DROPPED, HORIZON))

p6a_facts = json.loads((paths.reports_dir / "p6a_facts.json").read_text(encoding="utf-8")) if (paths.reports_dir / "p6a_facts.json").exists() else {}
FACTS["p6a_chart_days_mean"] = float(p6a_facts["r2_mean"]) if "r2_mean" in p6a_facts else float("nan")

def band(low, high, digits=1):
    return f"{100 * low:.{digits}f}% to {100 * high:.{digits}f}%"

SURVIVAL_TABLE = pd.DataFrame({"after chart days": KEY_AGES, "still on the chart": [f"{100 * S_OBS[a]:.1f}%" for a in KEY_AGES],
                               "artist-bootstrap band": [band(S_LOW[a], S_HIGH[a]) for a in KEY_AGES],
                               "band if songs were independent": [band(GW_LOW[a], GW_HIGH[a]) for a in KEY_AGES]}).set_index("after chart days")
display(SURVIVAL_TABLE)

display(pd.DataFrame({"how the open stays are treated": ["kept as 'at least this long' (Kaplan-Meier)", "treated as finished on their last day", "dropped"],
                      "share still on the chart after 30 days": [f"{100 * S_OBS[30]:.1f}%", f"{100 * S_AS_FINISHED[30]:.1f}%", f"{100 * S_DROPPED[30]:.1f}%"],
                      "median stay": [median_stay(S_OBS), median_stay(S_AS_FINISHED), median_stay(S_DROPPED)],
                      "chart days used of the first 30": [f"{restricted_mean(S_OBS, HORIZON):.2f}", f"{restricted_mean(S_AS_FINISHED, HORIZON):.2f}",
                                                           f"{restricted_mean(S_DROPPED, HORIZON):.2f}"]}).set_index("how the open stays are treated"))
print(f"Median stay {FACTS['median_stay']:.0f} chart days (95% band {FACTS['median_stay_low']:.0f} to {FACTS['median_stay_high']:.0f}); "
      f"chart days used of the first {HORIZON}: {FACTS['rmst30']:.2f} ({FACTS['rmst30_low']:.2f} to {FACTS['rmst30_high']:.2f}); Phase 6A counted {FACTS['p6a_chart_days_mean']:.2f} on its 832 songs.")

# %% [md]
**What this code is.** Figure 2. Left: the survival curve for the first 90 chart days, with the artist-bootstrap band (shaded) and the Greenwood band (dashed lines). Right: for six ages, the share still on the chart with both bands side by side.

**Why we need it.** The curve is the answer to the main question of the phase, and the comparison of the two bands is the reason that the notebook uses the artist bootstrap everywhere: it shows in one picture how much narrower the doubt would look if songs were treated as independent.

**What will happen.** The code draws the curve as steps (a song leaves on a whole chart day), shades the artist band, adds the dashed edges of the Greenwood band, marks the median stay and the six ages, and draws the right panel as one row per age, the artist band as a thick bar and the Greenwood band as a thin line. It saves the picture as `p6b_02_survival.png`.

**Expected outcome.** A curve that drops steeply in the first week, flattens, and is still above 5% at 90 chart days. The shaded band is wider than the dashed one at every age, most clearly in the middle of the curve (at 14 and 30 chart days).

**Motive.** Show the central result of the phase and how far to trust it.

# %%
# Figure 02 - how many songs are still on the chart after a number of chart days
SHOW_AGES = [3, 7, 14, 30, 60, 90]
fig, (axa, axb) = make_fig(11.8, 5.7, 1, 2, left=0.85, right=0.5, bottom=0.95, wspace=0.3, width_ratios=[1.5, 1], panel_titles=True,
                           title=f"{100 * FACTS['s3']:.0f}% of new songs are still on the chart after 3 chart days, {100 * FACTS['s7']:.0f}% after 7 and {100 * FACTS['s30']:.0f}% after 30; "
                                 f"the median stay is {FACTS['median_stay']:.0f} chart days",
                           subtitle=f"Kaplan-Meier estimate for the {FACTS['n_songs']} songs that arrived after the first snapshot day; open stays are kept as 'at least this long'. "
                                    "Shaded and thick: 95% band when artists are the unit of doubt (1,000 artist draws). Dashed and thin: the band if every song were an independent observation.",
                           note_text="S(a) is the share of songs whose stay lasts longer than a chart days. Chart days, not calendar days.")
SP.draw_survival_curves(axa, [{"label": None, "ages": AGES, "survival": S_OBS, "low": S_LOW, "high": S_HIGH, "color": style.BLUE}], x_max=90)
axa.step(AGES[:91], GW_LOW[:91], where="post", color=style.ORANGE, linewidth=1.1, linestyle="--", zorder=4)
axa.step(AGES[:91], GW_HIGH[:91], where="post", color=style.ORANGE, linewidth=1.1, linestyle="--", zorder=4)
axa.plot([FACTS["median_stay"]] * 2, [0, 0.5], color=style.INK_2, linewidth=1.0, linestyle=":", zorder=2)
axa.plot([0, FACTS["median_stay"]], [0.5, 0.5], color=style.INK_2, linewidth=1.0, linestyle=":", zorder=2)
direct_label(axa, FACTS["median_stay"] + 1.2, 0.05, f"median stay: {FACTS['median_stay']:.0f} chart days", size=9, color=style.INK)
for a in SHOW_AGES:
    axa.scatter([a], [S_OBS[a]], s=34, color=style.BLUE, edgecolor=style.SURFACE, linewidth=1.2, zorder=5)
    if a == SHOW_AGES[-1]:
        direct_label(axa, a - 1.5, S_OBS[a] + 0.05, f"{100 * S_OBS[a]:.0f}%", ha="right", size=9, color=style.INK)       # the last label sits left of its dot: the axis ends there
    else:
        direct_label(axa, a + 1.5, S_OBS[a] + 0.045, f"{100 * S_OBS[a]:.0f}%", size=9, color=style.INK)
axa.set_xticks([0, 3, 7, 14, 30, 60, 90])
axa.set_xlabel("chart days on the chart")
axa.set_ylabel("share of songs still on the chart")
axa.set_ylim(0, 1.0)
style_axes(axa, grid="y")
panel_title(axa, "The curve falls fast in the first week and then flattens")

y = np.arange(len(SHOW_AGES))
for yi, a in zip(y, SHOW_AGES):
    axb.hlines(yi + 0.17, GW_LOW[a], GW_HIGH[a], color=style.ORANGE, linewidth=1.6, zorder=2)
    axb.hlines(yi - 0.1, S_LOW[a], S_HIGH[a], color=style.BLUE, linewidth=6.0, alpha=0.55, zorder=2)
    axb.scatter([S_OBS[a]], [yi - 0.1], s=30, color=style.BLUE, edgecolor=style.SURFACE, linewidth=1.2, zorder=4)
    direct_label(axb, S_HIGH[a] + 0.025, yi - 0.1, f"{100 * S_OBS[a]:.0f}%  ({100 * S_LOW[a]:.0f} to {100 * S_HIGH[a]:.0f})", size=8.6, color=style.INK_2)
axb.set_yticks(y)
axb.set_yticklabels([f"after {a} days" for a in SHOW_AGES])
axb.set_ylim(len(SHOW_AGES) - 0.4, -0.55)
axb.xaxis.set_major_formatter(SP.percent_axis)
axb.set_xlim(0, 1.05)
axb.set_xlabel("share of songs still on the chart")
style_axes(axb, grid="x", left_spine=False)
axb.tick_params(axis="y", length=0)
swatch_legend(axb, [("artists as the unit (used here)", style.BLUE, "line"), ("songs as independent", style.ORANGE, "line")], loc="lower right")
panel_title(axb, "How sure: two kinds of band")
save_fig(fig, "p6b_02_survival", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- The curve falls fastest at the start. {{s1:pct0}} of the songs are still on the chart after their first chart day, {{s3:pct0}} after three, {{s7:pct0}} after a week. The median stay is {{median_stay:f0}} chart days (95% band {{median_stay_low:f0}} to {{median_stay_high:f0}}); {{s30:pct0}} are left after {{horizon}} chart days and {{s90:pct0}} after 90. A song that gets through the first week has much better odds than a new one: {{s30_given_7:pct0}} of the songs on the chart on day 7 are still there on day 30, against {{s30:pct0}} of all new songs.
- The artist band is wider than the textbook band at every age shown. At 30 chart days the artist band runs from {{s30_low:pct1}} to {{s30_high:pct1}}, the textbook band from {{gw30_low:pct1}} to {{gw30_high:pct1}}: the artist band is {{band_ratio_30:f1}} times as wide. Songs of one artist rise and fall together, so the data say less than {{n_songs:n}} independent songs would. Every band from here on is made by drawing whole artists.
- Open stays matter, and only on the right-hand side of the curve. Treating the {{n_censored}} open stays as finished on their last day lowers the share still on the chart after {{horizon}} chart days from {{s30:pct1}} to {{s30_as_finished:pct1}}, and dropping them lowers it to {{s30_dropped:pct1}}. The median stays at {{median_stay:f0}} in all three versions, because the open stays are few and mostly long: they do not change when the half-way point is passed, they change how many songs go on beyond it.
- A new song uses {{rmst30:f2}} chart days of its first {{horizon}} on average (95% band {{rmst30_low:f2}} to {{rmst30_high:f2}}). Phase 6A counted {{p6a_chart_days_mean:f2}} on the 832 songs that could be followed for 30 snapshot days. The two are computed differently (the area under the curve of all {{n_songs:n}} songs here, a plain average on a subset there), so agreement to one decimal is a useful check on both.

# %% [md]
**What this code is.** The hazard by age and the median further stay. For nine age windows (chart day 1, day 2, day 3, days 4 to 7, 8 to 14, 15 to 30, 31 to 60, 61 to 120, 121 and later) the code counts the exits and the chart days lived in the window and divides one by the other: the chance that a chart day in the window is the last day of a stay. It does the same for each artist draw to get a band. For six ages (0, 3, 7, 14, 30, 60 chart days) it also finds the *median further stay*: of the songs that are still on the chart at that age, how many more chart days until half of them have gone.

**Why we need it.** The survival curve shows how many songs are left; the hazard shows how fast they are being lost at each stage. If the chance of leaving were the same on every day, the curve would be a smooth exponential decay and one number would describe it. If it falls with age, a song that has stayed a while is a different kind of song from a new one, and any model must allow for that. The model of Section 6 gives every age window its own baseline for exactly this reason.

**What will happen.** `window_hazard` returns exits, chart days and the ratio for each window; the same call with the weight matrix gives the ratio for all {{n_boot:n}} draws, and the 2.5th and 97.5th percentiles are the band. `median_further_stay` is applied to the curve of the data and to each draw. Two tables are printed.

**Expected outcome.** A hazard of about 17% on the first chart day, falling through about 13%, 12% and 9% in the first week to a few per cent after the second week and about half a per cent after day 120. A median further stay that grows with age: about 6 chart days for a new song, 11 after 3 days, 20 after a week and more than 50 after two months.

**Motive.** Show where in a stay the songs are lost, and that the chance of leaving is far from constant.

# %%
LABELS = window_labels(AGE_EDGES)                                            # '1', '2', '3', '4-7', '8-14', '15-30', '31-60', '61-120', '121+'
exits_w, days_w, hazard_w = window_hazard(T, E, AGE_EDGES)
_, _, hazard_boot = window_hazard(T, E, AGE_EDGES, weights=W_BOOT)           # the same ratios for every artist draw
H_LOW, H_HIGH = np.nanpercentile(hazard_boot, [2.5, 97.5], axis=0)

HAZARD_TABLE = pd.DataFrame({"window": LABELS, "from_day": np.array(AGE_EDGES[:-1]) + 1, "to_day": [float(v) if v < 10_000 else np.nan for v in AGE_EDGES[1:]],
                             "songs_reaching": [int((T > edge).sum()) for edge in AGE_EDGES[:-1]], "exits": exits_w.astype(int),
                             "chart_days": days_w.astype(int), "hazard": hazard_w, "low": H_LOW, "high": H_HIGH})
for k, label in enumerate(LABELS, start=1):
    FACTS[f"hz{k}"], FACTS[f"hz{k}_low"], FACTS[f"hz{k}_high"] = float(hazard_w[k - 1]), float(H_LOW[k - 1]), float(H_HIGH[k - 1])
    FACTS[f"hz{k}_exits"], FACTS[f"hz{k}_days"] = int(exits_w[k - 1]), int(days_w[k - 1])
FACTS["hz_first_over_last"] = float(hazard_w[0] / hazard_w[-1])
FACTS["n_windows"] = len(LABELS)
assert int(exits_w.sum()) == FACTS["n_exits"] and int(days_w.sum()) == FACTS["n_days"], "every exit and every chart day belongs to exactly one window"

display(HAZARD_TABLE.assign(hazard=lambda d: [f"{100 * v:.1f}%" for v in d["hazard"]],
                            band=lambda d: [f"{100 * a:.1f}% to {100 * b:.1f}%" for a, b in zip(d["low"], d["high"])]).drop(columns=["from_day", "to_day", "low", "high"]).set_index("window"))

FURTHER_AGES = [0, 3, 7, 14, 30, 60]
further_boot = np.array([median_further_stay(S_BOOT, a) for a in FURTHER_AGES])         # one row per age, one column per artist draw
FURTHER_TABLE = pd.DataFrame({"age": FURTHER_AGES, "still_on_chart": [S_OBS[a] for a in FURTHER_AGES],
                              "median_further_stay": [median_further_stay(S_OBS, a) for a in FURTHER_AGES],
                              "low": np.nanpercentile(further_boot, 2.5, axis=1), "high": np.nanpercentile(further_boot, 97.5, axis=1)})
for row in FURTHER_TABLE.itertuples():
    FACTS[f"further{row.age}"], FACTS[f"further{row.age}_low"], FACTS[f"further{row.age}_high"] = float(row.median_further_stay), float(row.low), float(row.high)
FACTS["s30_given_7"] = float(S_OBS[30] / S_OBS[7])
FACTS["reach60_songs"] = int((T > 60).sum())                                 # songs that get past chart day 60: the data behind the last row
FACTS["reach60_artists"] = int(STAYS.loc[T > 60, "primary_artist"].nunique())
display(FURTHER_TABLE.assign(still_on_chart=lambda d: [f"{100 * v:.1f}%" for v in d["still_on_chart"]],
                             band=lambda d: [f"{a:.0f} to {b:.0f}" for a, b in zip(d["low"], d["high"])]).drop(columns=["low", "high"]).set_index("age"))

# %% [md]
**What this code is.** Figure 3. Left: the hazard of each age window as a bar with its artist band. Right: the median further stay at six ages, with its band.

**Why we need it.** The two panels say the same thing from two sides. The left one says that the chance of leaving falls steeply with age; the right one says what that means for a song: the longer it has stayed, the longer it can be expected to go on.

**What will happen.** The code draws nine bars (one for each age window, equal width although the windows are of different length), a thin line from the lower to the upper end of each band, and the value on top. On the right, one dot for each age with its band, the age on the vertical axis. It saves the picture as `p6b_03_hazard.png`.

**Expected outcome.** Bars that fall from the first window to the last, about thirty-fold. A row of dots that moves to the right with age.

**Motive.** Show that a stay has a first, dangerous part and a long, calm one, which is what the age windows of the model are for.

# %%
# Figure 03 - the chance that a chart day is the last one, and the median further stay
fig, (axa, axb) = make_fig(11.8, 5.6, 1, 2, left=0.85, right=0.5, bottom=0.95, wspace=0.28, width_ratios=[1.45, 1], panel_titles=True,
                           title=f"The chance that a chart day is the last one falls from {100 * FACTS['hz1']:.0f}% on the first day to {100 * FACTS['hz9']:.1f}% after day 120: "
                                 f"no single average chance describes a stay",
                           subtitle=f"All {FACTS['n_songs']} stays; {FACTS['n_exits']} exits over {FACTS['n_days']:,} chart days. Left: exits divided by chart days lived in each age window. "
                                    "Right: of the songs still on the chart at an age, how many more chart days until half have gone. Whiskers: 95% artist band.",
                           note_text="The bars have equal width although the windows differ in length (1 chart day, 1, 1, 4, 7, 16, 30, 60 and everything after day 120).")
x = np.arange(len(LABELS))
axa.bar(x, hazard_w, width=0.66, color=style.BLUE)
axa.vlines(x, H_LOW, H_HIGH, color=style.INK_2, linewidth=1.1, zorder=4)
for xi, value, high in zip(x, hazard_w, H_HIGH):
    direct_label(axa, xi, high + 0.006, f"{100 * value:.1f}%", ha="center", va="bottom", size=8.8, color=style.INK)
axa.set_xticks(x)
axa.set_xticklabels(LABELS)
axa.yaxis.set_major_formatter(SP.percent_axis)
axa.set_ylim(0, 0.265)
axa.set_xlabel("chart day of the stay (age window)")
axa.set_ylabel("chance that the chart day is the last one")
style_axes(axa, grid="y")
panel_title(axa, "The chance of leaving falls with age")

y = np.arange(len(FURTHER_AGES))
for yi, row in zip(y, FURTHER_TABLE.itertuples()):
    axb.hlines(yi, row.low, row.high, color=style.BLUE, linewidth=5.0, alpha=0.5, zorder=2)
    axb.scatter([row.median_further_stay], [yi], s=40, color=style.BLUE, edgecolor=style.SURFACE, linewidth=1.3, zorder=4)
    direct_label(axb, row.high + 2.5, yi, f"{row.median_further_stay:.0f}  ({row.low:.0f} to {row.high:.0f})", size=8.8, color=style.INK_2)
axb.set_yticks(y)
axb.set_yticklabels(["on arrival"] + [f"after {a} chart days" for a in FURTHER_AGES[1:]])
axb.set_ylim(len(FURTHER_AGES) - 0.5, -0.6)
axb.set_xlim(0, 140)
axb.set_xlabel("median further stay (chart days)")
style_axes(axb, grid="x", left_spine=False)
axb.tick_params(axis="y", length=0)
panel_title(axb, "The median further stay grows with age")
save_fig(fig, "p6b_03_hazard", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- The chance of leaving improves with every stage of the stay. {{hz1:pct0}} of the songs leave after their first chart day ({{hz1_exits}} of {{n_songs}}), {{hz2:pct0}} of the rest after the second and {{hz3:pct0}} after the third; in the rest of the first week {{hz4:pct0}} per chart day, from the second week {{hz5:pct0}}, then {{hz6:pct1}}, {{hz7:pct1}} and {{hz8:pct1}}; after day 120 it is {{hz9:pct1}} per chart day. The first day is about {{hz_first_over_last:f0}} times as dangerous as the days after the 120th. The bands of the first three days overlap, so the data do not show a clear difference among them; the step down comes after the third day.
- The median further stay grows with age: {{further0:f0}} chart days on arrival, {{further3:f0}} after 3 chart days, {{further7:f0}} after 7, {{further14:f0}} after 14, {{further30:f0}} after 30 and {{further60:f0}} after 60. The bands widen quickly because fewer songs get that far: only {{reach60_songs}} songs, by {{reach60_artists}} artists, stay past chart day 60, which is why its band runs from {{further60_low:f0}} to {{further60_high:f0}}.
- Why the chance falls cannot be read from this chart. Songs that were going to leave leave early, so the ones that remain are a different mix; and the chance of leaving may also fall with age for every song. The two are mixed in these numbers. The notebook does not try to separate them. It measures the chance at each age, and from Section 4 on it compares groups of songs at the same age.

