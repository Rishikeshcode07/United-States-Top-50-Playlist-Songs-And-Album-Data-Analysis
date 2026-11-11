# %% [md]
## Section 5 - Debut place, arriving alone, and the single that does not protect

Section 4 left two puzzles. The curves by debut place do not fall in a straight line: songs that arrive at 41 to 50 stay longer than songs that arrive at 21 to 40. And singles stay longer than album tracks, but singles are also the songs that tend to arrive alone, and songs that arrive alone stay longer for any release type. This section takes the two puzzles apart with the simplest tool there is: compare songs *within* the way they arrived. It is not yet a model. It is a look at who the songs are, and it prepares the model of Section 6.

# %% [md]
**What this code is.** The look at debut place and at singles by the way of arriving. First the songs are split into those that arrive alone (the only song of their artist on that snapshot day) and those that arrive with others (two or more songs of the same artist on the day). For each debut band the code finds the share that arrive alone, the share of songs still on the chart after {{horizon}} chart days within each way of arriving, and a *mix-adjusted* share: the share that the band would have if its songs were a mix of {{share_alone:pct0}} alone and the rest with others, as in the whole data. For singles it compares singles with album tracks within the songs that arrive alone.

**Why we need it.** A group can look better than another only because it contains more of a kind of song that does well anyway. If songs that arrive alone stay longer, and the bands from 31 to 50 contain mostly songs that arrive alone, the bands will look good for that reason. The mix-adjustment answers: *how would the bands compare if every band had the same mix of arrivals?* The comparison of singles with album tracks within the songs that arrive alone asks the same question for release type.

**What will happen.** For the five bands and for each way of arriving the code calculates the share still on the chart after {{horizon}} chart days with an artist band from the {{n_boot:n}} draws (draws without any artist of a group are left out of that group's band). The mix-adjusted share is {{share_alone:pct0}} times the share of those who arrive alone plus the rest times the share of those who arrive with others, calculated in each draw. For singles it does the same for the singles and the album tracks among the songs that arrive alone, adds a difference in chart days used out of the first {{horizon}} with a paired band, and runs the log-rank test inside the songs that arrive alone with artists as the unit. A table is printed.

**Expected outcome.** The share that arrive alone climbs from about a quarter in the top 10 to over 80% at places 41 to 50. In every band the songs that arrive alone stay longer than the others. The mix-adjusted curve falls and then stays low, with a much smaller recovery at the end than the crude one. Among the songs that arrive alone, singles and album tracks stay equally long.

**Motive.** Separate "which songs these are" from "what their place or release type does", as far as a look at the data can, before the model does it properly.

# %%
ALONE = STAYS["drop_size"].to_numpy() == 1                                   # the only song of its artist on its arrival day
P_ALONE = float(ALONE.mean())
FACTS["share_alone"] = P_ALONE
AGES_30 = np.arange(HORIZON + 1)                                             # chart days 0 ... 30: the last one is S(30)


def s30_point(mask):
    """Share of the songs in `mask` still on the chart after 30 chart days."""
    return float(survival_grid(T[mask], E[mask], AGES_30)[HORIZON])


def s30_draws(mask):
    """The same share for every artist draw; a draw that holds none of the group's artists is NaN."""
    draws = survival_grid(T[mask], E[mask], AGES_30, weights=W_BOOT[:, mask])[:, HORIZON]
    return np.where(W_BOOT[:, mask].sum(axis=1) > 0, draws, np.nan)


DEBUT_ROWS = []
for k, band_name in enumerate(BAND_LABELS, start=1):
    in_band = (STAYS["debut_band"].astype(str) == band_name).to_numpy()
    alone, others = in_band & ALONE, in_band & ~ALONE
    crude, s_alone, s_others = s30_point(in_band), s30_point(alone), s30_point(others)
    adjusted = P_ALONE * s_alone + (1 - P_ALONE) * s_others                  # the share this band would have with the overall mix of arrivals
    d_crude, d_alone, d_others = s30_draws(in_band), s30_draws(alone), s30_draws(others)
    d_adjusted = P_ALONE * d_alone + (1 - P_ALONE) * d_others
    row = {"band": band_name, "songs": int(in_band.sum()), "share_alone": float(alone.sum() / in_band.sum()), "alone_songs": int(alone.sum()), "others_songs": int(others.sum()),
           "crude": crude, "crude_low": np.nanpercentile(d_crude, 2.5), "crude_high": np.nanpercentile(d_crude, 97.5),
           "alone": s_alone, "alone_low": np.nanpercentile(d_alone, 2.5), "alone_high": np.nanpercentile(d_alone, 97.5),
           "others": s_others, "others_low": np.nanpercentile(d_others, 2.5), "others_high": np.nanpercentile(d_others, 97.5),
           "adjusted": adjusted, "adjusted_low": np.nanpercentile(d_adjusted, 2.5), "adjusted_high": np.nanpercentile(d_adjusted, 97.5)}
    DEBUT_ROWS.append(row)
    for name, value in row.items():
        if name != "band":
            FACTS[f"deb{k}_{name}"] = float(value)
DEBUT_TABLE = pd.DataFrame(DEBUT_ROWS)
display(DEBUT_TABLE.assign(**{"share who arrive alone": [f"{100 * v:.0f}%" for v in DEBUT_TABLE["share_alone"]],
                              "left after 30: all songs": [f"{100 * v:.0f}%  ({100 * a:.0f} to {100 * b:.0f})" for v, a, b in zip(DEBUT_TABLE["crude"], DEBUT_TABLE["crude_low"], DEBUT_TABLE["crude_high"])],
                              "arriving alone": [f"{100 * v:.0f}%  (n = {n})" for v, n in zip(DEBUT_TABLE["alone"], DEBUT_TABLE["alone_songs"])],
                              "arriving with others": [f"{100 * v:.0f}%  (n = {n})" for v, n in zip(DEBUT_TABLE["others"], DEBUT_TABLE["others_songs"])],
                              "mix-adjusted": [f"{100 * v:.0f}%  ({100 * a:.0f} to {100 * b:.0f})" for v, a, b in zip(DEBUT_TABLE["adjusted"], DEBUT_TABLE["adjusted_low"], DEBUT_TABLE["adjusted_high"])]})
        [["band", "songs", "share who arrive alone", "left after 30: all songs", "arriving alone", "arriving with others", "mix-adjusted"]].set_index("band"))

# ---- singles and album tracks, among the songs that arrive alone
IS_SINGLE = STAYS["is_single"].to_numpy() == 1
SINGLE_ROWS = {}
for name, mask in [("single_all", IS_SINGLE), ("album_all", ~IS_SINGLE), ("single_alone", IS_SINGLE & ALONE), ("album_alone", ~IS_SINGLE & ALONE)]:
    draws = s30_draws(mask)
    SINGLE_ROWS[name] = {"songs": int(mask.sum()), "s30": s30_point(mask), "low": float(np.nanpercentile(draws, 2.5)), "high": float(np.nanpercentile(draws, 97.5))}
    FACTS[f"{name}_n"], FACTS[f"{name}_s30"] = SINGLE_ROWS[name]["songs"], SINGLE_ROWS[name]["s30"]
    FACTS[f"{name}_s30_low"], FACTS[f"{name}_s30_high"] = SINGLE_ROWS[name]["low"], SINGLE_ROWS[name]["high"]
FACTS["single_arrive_alone_share"] = float(ALONE[IS_SINGLE].mean())
FACTS["album_arrive_alone_share"] = float(ALONE[~IS_SINGLE].mean())
FACTS["singles_with_others"] = int((IS_SINGLE & ~ALONE).sum())

single_alone, album_alone = IS_SINGLE & ALONE, ~IS_SINGLE & ALONE
estimate = restricted_mean(survival_grid(T[single_alone], E[single_alone], AGES_HORIZON), HORIZON) - restricted_mean(survival_grid(T[album_alone], E[album_alone], AGES_HORIZON), HORIZON)
difference = rmst_draws(single_alone) - rmst_draws(album_alone)
FACTS["single_alone_diff"] = float(estimate)
FACTS["single_alone_diff_low"], FACTS["single_alone_diff_high"] = (float(v) for v in np.nanpercentile(difference, [2.5, 97.5]))
inside = logrank_k(T[ALONE], E[ALONE], np.where(IS_SINGLE[ALONE], "single", "album"), clusters=ARTISTS[ALONE])
FACTS["lr_single_alone_p_artists"], FACTS["lr_single_alone_p_text"] = float(inside["p_artist"]), format_p(inside["p_artist"])
FACTS["lr_single_alone_chi2_artists"] = float(inside["chi2_artist"])
display(pd.DataFrame({"songs": {k: v["songs"] for k, v in SINGLE_ROWS.items()},
                      "left after 30": {k: f"{100 * v['s30']:.0f}%  ({100 * v['low']:.0f} to {100 * v['high']:.0f})" for k, v in SINGLE_ROWS.items()}}))
print(f"Among the songs that arrive alone: single minus album track = {estimate:+.2f} chart days of the first {HORIZON} "
      f"(band {FACTS['single_alone_diff_low']:+.2f} to {FACTS['single_alone_diff_high']:+.2f}); log-rank with artists as the unit: {FACTS['lr_single_alone_p_text']}.")

# %% [md]
**What this code is.** Figure 6, in four panels. Top left: the share of songs that arrive alone in each debut band. Top right: the share still on the chart after {{horizon}} chart days in each band, separately for songs that arrive alone and songs that arrive with others. Bottom left: the crude share and the mix-adjusted share by band. Bottom right: singles against album tracks, among all songs and among the songs that arrive alone.

**Why we need it.** The four panels follow one chain of thought: who arrives alone (top left), how long each kind stays (top right), what the bands look like once the mix of arrivals is the same (bottom left), and the same question for release type (bottom right). Seeing the chain in one figure is easier than reading it from a table.

**What will happen.** The code draws stacked bars, paired dots with artist bands, two lines with shaded bands and two rows of dots, using the numbers of the cell above. Cells with fewer than 30 songs are not drawn with a band; the songs that arrive with others in drops of two to four, and singles that arrive with others ({{singles_with_others}} songs), are too few to show separately. It saves the picture as `p6b_06_alone.png`.

**Expected outcome.** In the first panel the blue part grows from left to right. In the second, the blue dots are above the orange ones in every band. In the third, the dark line has a recovery at the right and the blue line has much less of one. In the fourth, a clear gap between singles and album tracks among all songs and none among the songs that arrive alone.

**Motive.** Show how much of the crude pattern is a matter of who the songs are, before a model is asked about it.

# %%
# Figure 06 - the dip and recovery by debut place, and the single, seen through the way a song arrives
recovery_crude = FACTS["deb5_crude"] - FACTS["deb4_crude"]                    # the crude rise from places 31 to 40 to places 41 to 50
recovery_adjusted = FACTS["deb5_adjusted"] - FACTS["deb4_adjusted"]           # the same rise when every band has the same mix of arrivals
FACTS["recovery_crude"], FACTS["recovery_adjusted"] = float(recovery_crude), float(recovery_adjusted)
FACTS["recovery_share_explained"] = float(1 - recovery_adjusted / recovery_crude)

x = np.arange(len(BAND_LABELS))
fig, axes = make_fig(11.8, 8.4, 2, 2, left=0.9, right=0.5, bottom=0.9, hspace=0.55, wspace=0.3, panel_titles=True,
                     title="Most of the recovery at the bottom of the chart, and all of the single's advantage, go with arriving alone",
                     subtitle=f"{FACTS['n_songs']} new songs. 'Arrives alone' = the only song of its artist on its arrival day. Share still on the chart after {HORIZON} chart days; "
                              "bars and bands: 95% artist band.",
                     note_text="Mix-adjusted: every band is given the overall mix of arrivals (" f"{100 * P_ALONE:.0f}% alone, {100 * (1 - P_ALONE):.0f}% with others) before the shares are compared. "
                               "Cells with fewer than 30 songs are not drawn.")
axa, axb, axc, axd = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

share = DEBUT_TABLE["share_alone"].to_numpy()
axa.bar(x, share, width=0.62, color=style.BLUE)
axa.bar(x, 1 - share, width=0.62, bottom=share, color=style.ORANGE)
for xi, s in zip(x, share):
    axa.text(xi, s / 2, f"{100 * s:.0f}%", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    axa.text(xi, s + (1 - s) / 2, f"{100 * (1 - s):.0f}%", ha="center", va="center", color=style.INK, fontsize=9.5)
axa.set_xticks(x)
axa.set_xticklabels(BAND_LABELS)
axa.yaxis.set_major_formatter(SP.percent_axis)
axa.set_ylim(0, 1.17)
axa.set_ylabel("share of the songs in the band")
axa.set_xlabel("debut place")
swatch_legend(axa, [("arrives alone", style.BLUE), ("arrives with others", style.ORANGE)], loc="upper center", ncol=2)
style_axes(axa, grid="y")
panel_title(axa, "Who arrives alone")

for series, color, dx in [("alone", style.BLUE, -0.1), ("others", style.ORANGE, 0.1)]:
    axb.vlines(x + dx, DEBUT_TABLE[f"{series}_low"], DEBUT_TABLE[f"{series}_high"], color=color, linewidth=4.5, alpha=0.5, zorder=2)
    axb.plot(x + dx, DEBUT_TABLE[series], color=color, linewidth=1.2, alpha=0.8, zorder=3)
    axb.scatter(x + dx, DEBUT_TABLE[series], s=42, color=color, edgecolor=style.SURFACE, linewidth=1.2, zorder=4)
axb.set_xticks(x)
axb.set_xticklabels(BAND_LABELS)
axb.yaxis.set_major_formatter(SP.percent_axis)
axb.set_ylim(0, 0.72)
axb.set_ylabel(f"share still on the chart after {HORIZON} days")
axb.set_xlabel("debut place")
swatch_legend(axb, [("arrives alone", style.BLUE), ("arrives with others", style.ORANGE)], loc="upper right")
style_axes(axb, grid="y")
panel_title(axb, "Songs that arrive alone stay longer in every band")

for series, color, marker, label in [("crude", style.INK_2, "s", "all songs"), ("adjusted", style.BLUE, "o", "same mix of arrivals")]:
    axc.fill_between(x, DEBUT_TABLE[f"{series}_low"], DEBUT_TABLE[f"{series}_high"], color=color, alpha=0.13, linewidth=0, zorder=1)
    axc.plot(x, DEBUT_TABLE[series], color=color, linewidth=2.0, marker=marker, markersize=6.5, markeredgecolor=style.SURFACE, markeredgewidth=1.2, zorder=3)
    direct_label(axc, x[-1] + 0.15, DEBUT_TABLE[series].iloc[-1], label, size=8.8, color=style.INK)
axc.set_xticks(x)
axc.set_xticklabels(BAND_LABELS)
axc.set_xlim(-0.3, x[-1] + 1.9)
axc.yaxis.set_major_formatter(SP.percent_axis)
axc.set_ylim(0, 0.5)
axc.set_ylabel(f"share still on the chart after {HORIZON} days")
axc.set_xlabel("debut place")
style_axes(axc, grid="y")
panel_title(axc, "The recovery at the bottom is mostly the mix")

rows = [("All songs", "album_all", "single_all"), ("Songs that arrive alone", "album_alone", "single_alone")]
for yi, (label, album_key, single_key) in enumerate(rows):
    for key, color, dy in [(album_key, style.ALBUM_COLOR, -0.14), (single_key, style.SINGLE_COLOR, 0.14)]:
        r = SINGLE_ROWS[key]
        axd.hlines(yi + dy, r["low"], r["high"], color=color, linewidth=5.0, alpha=0.5, zorder=2)
        axd.scatter([r["s30"]], [yi + dy], s=44, color=color, edgecolor=style.SURFACE, linewidth=1.2, zorder=4)
        direct_label(axd, r["high"] + 0.012, yi + dy, f"{100 * r['s30']:.0f}%  (n = {r['songs']})", size=8.8, color=style.INK_2)
axd.set_yticks(range(len(rows)))
axd.set_yticklabels([r[0] for r in rows])
axd.set_ylim(len(rows) - 0.45, -0.55)
axd.xaxis.set_major_formatter(SP.percent_axis)
axd.set_xlim(0, 0.62)
axd.set_xlabel(f"share still on the chart after {HORIZON} days")
swatch_legend(axd, [("album track", style.ALBUM_COLOR), ("single", style.SINGLE_COLOR)], loc="upper right")
style_axes(axd, grid="x", left_spine=False)
axd.tick_params(axis="y", length=0)
panel_title(axd, "Singles against album tracks")
save_fig(fig, "p6b_06_alone", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- *Who arrives alone.* At the top of the chart a new song is usually part of a drop: only {{deb1_share_alone:pct0}} of the songs that debut in the top 10 arrive alone, and the share is {{deb2_share_alone:pct0}} at 11 to 20 and {{deb3_share_alone:pct0}} at 21 to 30. It then climbs to {{deb4_share_alone:pct0}} at 31 to 40 and {{deb5_share_alone:pct0}} at 41 to 50. The bottom of the chart is where lone songs arrive.
- *Alone against with others.* In every band the songs that arrive alone are still on the chart after {{horizon}} chart days more often than the songs that arrive with others: {{deb1_alone:pct0}} against {{deb1_others:pct0}} in the top 10, {{deb2_alone:pct0}} against {{deb2_others:pct0}} at 11 to 20, {{deb3_alone:pct0}} against {{deb3_others:pct0}} at 21 to 30, {{deb4_alone:pct0}} against {{deb4_others:pct0}} at 31 to 40 and {{deb5_alone:pct0}} against {{deb5_others:pct0}} at 41 to 50. Among the songs that arrive alone the dip is still there: the share falls from {{deb1_alone:pct0}} in the top 10 to {{deb3_alone:pct0}} at 21 to 30 ({{deb3_alone_songs}} songs).
- *The mix.* If every band had the overall mix of arrivals, the share still on the chart after {{horizon}} chart days would be {{deb1_adjusted:pct0}}, {{deb2_adjusted:pct0}}, {{deb3_adjusted:pct0}}, {{deb4_adjusted:pct0}} and {{deb5_adjusted:pct0}} from the top 10 to places 41 to 50. The crude rise from the fourth to the fifth band is {{recovery_crude:pct0}} points, the mix-adjusted rise is {{recovery_adjusted:pct0}} points: about {{recovery_share_explained:pct0}} of the crude recovery goes with who arrives alone. What is left of it is small, and the bands of the two bands overlap, so the data do not show a real recovery. The first step, from the top 10 to places 11 to 20, does not go away: {{deb1_adjusted:pct0}} (band {{deb1_adjusted_low:pct0}} to {{deb1_adjusted_high:pct0}}) against {{deb2_adjusted:pct0}} ({{deb2_adjusted_low:pct0}} to {{deb2_adjusted_high:pct0}}).
- *Singles.* Among all songs, {{single_all_s30:pct0}} of singles and {{album_all_s30:pct0}} of album tracks are still on the chart after {{horizon}} chart days. But {{single_arrive_alone_share:pct0}} of singles arrive alone ({{single_alone_n}} of {{single_all_n}}) against {{album_arrive_alone_share:pct0}} of album tracks ({{album_alone_n}} of {{album_all_n}}). Among the songs that arrive alone, singles and album tracks are level: {{single_alone_s30:pct0}} against {{album_alone_s30:pct0}}, a difference of {{single_alone_diff:sf1}} chart days of {{horizon}} (band {{single_alone_diff_low:sf1}} to {{single_alone_diff_high:sf1}}; log-rank with artists as the unit {{lr_single_alone_p_text}}). Only {{singles_with_others}} singles arrive with others, too few to compare.
- What this does not show. "Arrives alone" is a description of how a song arrived, and a song that arrives alone differs from one that arrives in a drop in many ways (which artist, how well known, how big the release is). The look above holds one thing fixed. Section 6 holds several fixed at once.
