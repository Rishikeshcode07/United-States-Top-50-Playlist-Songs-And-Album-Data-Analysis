# %% [md]
## Section 4 - Who stays longer: five splits of the songs

The first look at what goes with a longer or shorter stay is the simplest one: split the songs into groups by one thing that is known on the arrival day, and compare the groups' survival curves. Five splits are used: the place on the chart on the arrival day (five bands), the number of songs by the same artist that arrived on the same day (an album drop, or a single song arriving alone), the type of release (single or album), whether the song is explicit, and whether it is a collaboration. Each split compares groups that differ in other ways as well, so this section cannot say what *causes* a longer stay. It finds out which splits show a difference worth explaining; Sections 5 and 6 then ask which of the differences remain when the others are held fixed.

# %% [md]
**What this code is.** The group comparison. For each of the five splits, the code works out the survival curve of every group with its 95% artist band, the median stay, the share still on the chart after 7 and after {{horizon}} chart days, and the number of chart days used out of the first {{horizon}}. For each split it also runs the log-rank test of the question "do all groups leave the chart at the same rate on every chart day?", once in the textbook form and once in the artist-aware form.

**Why we need it.** A difference between two curves can be luck, and it is easier to be fooled here than it looks: songs of one artist travel together, so an artist with fifteen songs that all arrive on one day and all leave a week later puts fifteen songs into one group and counts as fifteen pieces of evidence. The textbook log-rank test counts them as fifteen; the artist-aware test (the score test of a Cox model with the variance of Lin and Wei) adds up the evidence of each artist before squaring it, so that the artist counts as the one piece of evidence it is. Five splits are asked of one data set, so the five artist-aware p-values are adjusted together with the Holm method.

**What will happen.** The groups are built from the columns of the stays table (the album-drop split uses 1 song, 2 to 4 songs and 5 or more songs). For each group the curve is computed for the data and for every one of the {{n_boot:n}} artist draws (the draws that do not contain any artist of the group are left out of that group's band). `logrank_k` is run for each split with and without the artists as clusters, and `holm` adjusts the five artist-aware p-values. The table of groups and the table of tests are printed.

**Expected outcome.** Fourteen groups in five splits. Debut place, the number of songs arriving together and release type differ clearly, even after the adjustment; explicit and collaboration do not. The textbook p-values are far smaller than the artist-aware ones (for debut place about 10 to the power of minus 19 against 0.0007), but the same three splits are left.

**Motive.** Find out which differences between groups of songs are visible in the stays, with the doubt measured by artists and not by songs.

# %%
DROP_LABELS = ["1 song (alone)", "2 to 4 songs", "5 or more songs"]
STAYS["drop_group"] = pd.cut(STAYS["drop_size"], [0, 1, 4, 10_000], labels=DROP_LABELS)
STAYS["release"] = np.where(STAYS["is_single"] == 1, "Single", "Album or compilation")
STAYS["explicit"] = np.where(STAYS["is_explicit"] == 1, "Explicit", "Not explicit")
STAYS["collab"] = np.where(STAYS["is_collab"] == 1, "Collaboration", "Solo")

# the five splits: test key, name, column, groups in order, and one short key for every group (used to store the facts)
SPLITS = [("band", "Debut place", "debut_band", BAND_LABELS, ["b1", "b2", "b3", "b4", "b5"]),
          ("drop", "Songs by the artist arriving the same day", "drop_group", DROP_LABELS, ["d1", "d2", "d3"]),
          ("single", "Release type", "release", ["Album or compilation", "Single"], ["album", "single"]),
          ("explicit", "Explicit content", "explicit", ["Not explicit", "Explicit"], ["clean", "explicit"]),
          ("collab", "Collaboration", "collab", ["Solo", "Collaboration"], ["solo", "collab"])]


def summarize_group(mask):
    """The curve of one group with its artist band and the headline numbers: median stay, share left after 7 and 30 chart days, chart days used of 30."""
    curve = survival_grid(T[mask], E[mask], AGES)
    usable = W_BOOT[:, mask].sum(axis=1) > 0                                 # a draw without any artist of this group says nothing about it
    draws = survival_grid(T[mask], E[mask], AGES, weights=W_BOOT[:, mask])[usable]
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    medians = median_stay(draws)
    return {"songs": int(mask.sum()), "artists": int(STAYS.loc[mask, "primary_artist"].nunique()), "exits": int(E[mask].sum()),
            "curve": curve, "low": low, "high": high, "median": median_stay(curve),
            "median_low": float(np.nanpercentile(medians, 2.5)), "median_high": float(np.nanpercentile(medians, 97.5)),
            "s7": float(curve[7]), "s30": float(curve[30]), "s30_low": float(low[30]), "s30_high": float(high[30]),
            "rmst30": float(restricted_mean(curve, HORIZON))}


GROUP_DATA, GROUP_ROWS, CURVE_ROWS = {}, [], []
for age, value, low, high in zip(AGES[:121], S_OBS[:121], S_LOW[:121], S_HIGH[:121]):
    CURVE_ROWS.append({"split": "All songs", "group": "All new songs", "age": int(age), "survival": value, "low": low, "high": high})
for test_key, split_name, column, order, short_keys in SPLITS:
    for group, short_key in zip(order, short_keys):
        info = summarize_group((STAYS[column].astype(str) == group).to_numpy())
        GROUP_DATA[(test_key, group)] = info
        for name in ["songs", "artists", "median", "median_low", "median_high", "s7", "s30", "s30_low", "s30_high", "rmst30"]:
            FACTS[f"{test_key}_{short_key}_{name}"] = info[name]
        GROUP_ROWS.append({"split": split_name, "group": group, "songs": info["songs"], "artists": info["artists"], "exits": info["exits"],
                           "median_stay": info["median"], "median_low": info["median_low"], "median_high": info["median_high"],
                           "share_left_after_7": info["s7"], "share_left_after_30": info["s30"], "after_30_low": info["s30_low"],
                           "after_30_high": info["s30_high"], "chart_days_used_of_30": info["rmst30"]})
        for age in range(121):
            CURVE_ROWS.append({"split": split_name, "group": group, "age": age, "survival": info["curve"][age], "low": info["low"][age], "high": info["high"][age]})

TEST_ROWS = []
for test_key, split_name, column, order, short_keys in SPLITS:
    result = logrank_k(T, E, STAYS[column].astype(str).to_numpy(), clusters=ARTISTS)
    TEST_ROWS.append({"key": test_key, "split": split_name, "groups": len(order), "chi2_songs": result["chi2"], "p_songs": result["p"],
                      "chi2_artists": result["chi2_artist"], "p_artists": result["p_artist"]})
TESTS = pd.DataFrame(TEST_ROWS)
TESTS["p_holm"] = holm(TESTS["p_artists"].to_numpy())                       # the five artist-aware tests are one family of questions
for row in TESTS.itertuples():
    FACTS[f"lr_{row.key}_chi2"], FACTS[f"lr_{row.key}_p"] = float(row.chi2_songs), float(row.p_songs)
    FACTS[f"lr_{row.key}_chi2_artists"], FACTS[f"lr_{row.key}_p_artists"] = float(row.chi2_artists), float(row.p_artists)
    FACTS[f"lr_{row.key}_p_holm"], FACTS[f"lr_{row.key}_p_holm_text"] = float(row.p_holm), format_p(row.p_holm)
    FACTS[f"lr_{row.key}_p_artists_text"], FACTS[f"lr_{row.key}_p_text"] = format_p(row.p_artists), format_p(row.p_songs)
FACTS["n_splits_tested"] = len(TESTS)
FACTS["n_splits_clear"] = int((TESTS["p_holm"] < 0.05).sum())

display(pd.DataFrame(GROUP_ROWS).assign(**{"median stay": lambda d: [f"{m:.0f}  ({a:.0f} to {b:.0f})" for m, a, b in zip(d["median_stay"], d["median_low"], d["median_high"])],
                                           "left after 7": lambda d: [f"{100 * v:.0f}%" for v in d["share_left_after_7"]],
                                           "left after 30 (95% band)": lambda d: [f"{100 * v:.0f}%  ({100 * a:.0f} to {100 * b:.0f})" for v, a, b in zip(d["share_left_after_30"], d["after_30_low"], d["after_30_high"])],
                                           "chart days of 30": lambda d: [f"{v:.1f}" for v in d["chart_days_used_of_30"]]})
        [["split", "group", "songs", "artists", "median stay", "left after 7", "left after 30 (95% band)", "chart days of 30"]].set_index(["split", "group"]))
display(TESTS.assign(**{"textbook chi2": lambda d: d["chi2_songs"].round(1), "textbook p": [format_p(p) for p in TESTS["p_songs"]],
                        "artist-aware chi2": lambda d: d["chi2_artists"].round(1), "artist-aware p": [format_p(p) for p in TESTS["p_artists"]],
                        "after Holm": [format_p(p) for p in TESTS["p_holm"]]})
        [["split", "groups", "textbook chi2", "textbook p", "artist-aware chi2", "artist-aware p", "after Holm"]].set_index("split"))

# %% [md]
**What this code is.** Figure 4. Five small charts of survival curves, one for each split, and a sixth that sets the two versions of the log-rank test side by side.

**Why we need it.** The table gives the numbers; the curves show where in the stay the groups part company (on the first day? after a week?) and whether they cross, which a single p-value cannot say. The sixth panel shows what the artist-aware test does to the evidence: every p-value gets larger, but not by the same factor, and the question is which splits are left.

**What will happen.** The code draws the curves up to 60 chart days, each with its 95% artist band and a direct label at its right end. In the sixth panel every split is a row with three marks on a logarithmic axis: the textbook p-value (hollow), the artist-aware p-value (filled) and the artist-aware p-value after the Holm adjustment (diamond), with the 0.05 line. It saves the picture as `p6b_04_groups.png`.

**Expected outcome.** Clear separation in the debut-place panel (the top-10 debuts well above the rest), in the same-day panel (songs that arrive alone above those that arrive with an album) and a smaller one for release type. The curves of explicit against not explicit and of collaboration against solo lie on top of each other. In the sixth panel three splits to the left of the 0.05 line after Holm, two to the right.

**Motive.** See the differences, and see how much of the textbook evidence survives when the artist is the unit.

# %%
# Figure 04 - survival curves of the groups, and the tests
PALETTE = {"band": MP.step_colors(5)[::-1], "drop": MP.step_colors(3)[::-1], "single": [style.ALBUM_COLOR, style.SINGLE_COLOR],
           "explicit": [style.CLEAN_COLOR, style.EXPLICIT_COLOR], "collab": [style.BLUE, style.GREEN]}
SHORT = {"1 song (alone)": "1 song", "2 to 4 songs": "2-4 songs", "5 or more songs": "5+ songs", "Album or compilation": "album", "Collaboration": "collab"}
PANEL_TITLES = {"band": "Debut place", "drop": "Songs by the artist arriving that day", "single": "Release type", "explicit": "Explicit content", "collab": "Collaboration"}
X_MAX = 60

fig, axes = make_fig(11.8, 8.3, 2, 3, left=0.85, right=0.45, bottom=0.85, hspace=0.5, wspace=0.62, panel_titles=True,
                     title="Debut place, the number of songs an artist brings on the same day and release type separate the survival curves; explicit content and collaboration do not",
                     subtitle=f"Share of songs still on the chart, by chart days of the stay, for {FACTS['n_songs']} new songs. Shaded: 95% artist band. "
                              "Bottom right: log-rank tests that the groups leave at the same rate, with the songs treated as independent (hollow) and with artists as the unit (filled).",
                     note_text=f"Curves are drawn to {X_MAX} chart days. Holm adjustment over the five artist-aware tests.")
slots = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
for ax, (test_key, split_name, column, order, short_keys) in zip(slots, SPLITS):
    curves = [{"label": SHORT.get(group, group), "ages": AGES, "survival": GROUP_DATA[(test_key, group)]["curve"], "low": GROUP_DATA[(test_key, group)]["low"],
               "high": GROUP_DATA[(test_key, group)]["high"], "color": color} for group, color in zip(order, PALETTE[test_key])]
    SP.draw_survival_curves(ax, curves, x_max=X_MAX, label_gap=0.075, band_alpha=0.12)
    ax.set_xticks([0, 7, 14, 30, 60])
    ax.set_xlabel("chart days")
    style_axes(ax, grid="y")
    panel_title(ax, PANEL_TITLES[test_key])
axes[0, 0].set_ylabel("share still on the chart")
axes[1, 0].set_ylabel("share still on the chart")

axt = axes[1, 2]
y = np.arange(len(TESTS))
axt.axvline(0.05, color=style.INK_2, linewidth=1.0, linestyle=":")
axt.scatter(TESTS["p_songs"], y, s=46, facecolor="none", edgecolor=style.ORANGE, linewidth=1.6, zorder=3, label="songs as independent")
axt.scatter(TESTS["p_artists"], y, s=46, color=style.BLUE, zorder=4, label="artists as the unit")
axt.scatter(TESTS["p_holm"], y, s=52, marker="D", color=style.INK, zorder=5, label="artist-aware, after Holm")
axt.set_xscale("log")
axt.set_xlim(1e-21, 3)
axt.set_xticks([1e-20, 1e-15, 1e-10, 1e-5, 1e-2, 1])
axt.set_xticklabels(["1e-20", "1e-15", "1e-10", "1e-5", "0.01", "1"], fontsize=8.5)
axt.minorticks_off()
axt.set_yticks(y)
axt.set_yticklabels(["Debut place", "Arriving together", "Release type", "Explicit", "Collaboration"], fontsize=9)
axt.set_ylim(len(TESTS) - 0.4, -0.6)
axt.set_xlabel("p-value of the log-rank test (log scale)")
style_axes(axt, grid="x", left_spine=False)
axt.tick_params(axis="y", length=0)
direct_label(axt, 0.06, 4.45, "0.05", size=8.5, color=style.INK_2)
axt.legend(loc="lower left", frameon=False, fontsize=8.2, labelcolor=style.INK_2, handletextpad=0.3, borderaxespad=0.2)
panel_title(axt, "Which splits differ?")
save_fig(fig, "p6b_04_groups", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- *Debut place.* A song that arrives in the top 10 has a median stay of {{band_b1_median:f0}} chart days (95% band {{band_b1_median_low:f0}} to {{band_b1_median_high:f0}}); {{band_b1_s7:pct0}} are still there after a week and {{band_b1_s30:pct0}} after {{horizon}} chart days. For the bands 11 to 20, 21 to 30 and 31 to 40 the median falls to {{band_b2_median:f0}}, {{band_b3_median:f0}} and {{band_b4_median:f0}}. The last band does not continue the fall: the median for debuts at 41 to 50 is {{band_b5_median:f0}} and {{band_b5_s30:pct0}} of them are still on the chart after {{horizon}} chart days, against {{band_b4_s30:pct0}} for 31 to 40 and {{band_b3_s30:pct0}} for 21 to 30. The crude pattern is a dip and a recovery, not a straight line. Section 5 looks at why.
- *Songs arriving the same day.* Songs that arrive alone have a median stay of {{drop_d1_median:f0}} chart days and {{drop_d1_s30:pct0}} are left after {{horizon}}; songs that arrive with one to three others by the same artist have a median of {{drop_d2_median:f0}} and {{drop_d2_s30:pct0}}; songs that come in a drop of five or more have a median of {{drop_d3_median:f0}} and {{drop_d3_s30:pct0}}. The drop group of five or more has {{drop_d3_songs}} songs but only {{drop_d3_artists}} artists, and its band shows how little {{drop_d3_songs}} songs by {{drop_d3_artists}} artists say.
- *Release type.* Singles stay longer than album tracks: median {{single_single_median:f0}} chart days ({{single_single_median_low:f0}} to {{single_single_median_high:f0}}) against {{single_album_median:f0}} ({{single_album_median_low:f0}} to {{single_album_median_high:f0}}), and {{single_single_s30:pct0}} against {{single_album_s30:pct0}} still on the chart after {{horizon}} chart days. Section 5 asks whether this is a property of singles or of how singles arrive.
- *Explicit content and collaboration.* The two curves lie on top of each other in both panels: median {{explicit_clean_median:f0}} against {{explicit_explicit_median:f0}} chart days for explicit content, {{collab_solo_median:f0}} against {{collab_collab_median:f0}} for collaboration, and the shares left after {{horizon}} chart days are {{explicit_clean_s30:pct0}} against {{explicit_explicit_s30:pct0}} and {{collab_solo_s30:pct0}} against {{collab_collab_s30:pct0}}.
- *The tests.* With songs treated as independent, all three differences look overwhelming: debut place {{lr_band_p_text}} (chi-square {{lr_band_chi2:f1}}), songs arriving together {{lr_drop_p_text}} ({{lr_drop_chi2:f1}}), release type {{lr_single_p_text}} ({{lr_single_chi2:f1}}). With artists as the unit the statistics shrink to {{lr_band_chi2_artists:f1}}, {{lr_drop_chi2_artists:f1}} and {{lr_single_chi2_artists:f1}}; the p-values are {{lr_band_p_artists_text}}, {{lr_drop_p_artists_text}} and {{lr_single_p_artists_text}}. After the Holm adjustment for five tests they are {{lr_band_p_holm_text}}, {{lr_drop_p_holm_text}} and {{lr_single_p_holm_text}}: {{n_splits_clear}} of the {{n_splits_tested}} splits are clearly different. Explicit content ({{lr_explicit_p_artists_text}}) and collaboration ({{lr_collab_p_artists_text}}) show no sign of a difference in either version of the test.

# %% [md]
**What this code is.** The size of the differences, in the unit that matters. For each split the code compares every group with a reference group (the top-10 debuts, the songs that arrive alone, album tracks, songs that are not explicit, solo songs) by the number of chart days a group uses out of the first {{horizon}}, and puts a 95% artist band on each difference.

**Why we need it.** A p-value says whether a difference is likely to be more than luck; it says nothing about how big it is. "Chart days used out of the first {{horizon}}" is the area under the survival curve up to {{horizon}} chart days, and a difference between two groups in this unit is a number of chart days that a reader can picture. For the splits that show no difference the band has a second use: it says how large a difference the data can rule out.

**What will happen.** For each group the code computes the average chart days used out of the first {{horizon}} on the data and on every artist draw, using the same draws for both groups of a comparison (so that the difference is measured inside each draw), and takes the 2.5th and 97.5th percentiles of the differences. The results go in a table and in Figure 5.

**Expected outcome.** The top-10 debuts use about 7 to 10 more chart days than the other bands, and the bands of the differences stay clear of zero. Songs that arrive alone use about 5 more than songs that arrive in a drop, and singles about 4 more than album tracks. For explicit content and collaboration the differences are close to zero and the bands run on both sides of it.

**Motive.** Turn "these groups differ" into "by how many chart days", and say how much a difference of zero is worth when nothing was found.

# %%
CONTRASTS = [("band", "Debut place", "11 to 20", "1 to 10"), ("band", "Debut place", "21 to 30", "1 to 10"),
             ("band", "Debut place", "31 to 40", "1 to 10"), ("band", "Debut place", "41 to 50", "1 to 10"),
             ("drop", "Songs by the artist arriving the same day", "2 to 4 songs", "1 song (alone)"),
             ("drop", "Songs by the artist arriving the same day", "5 or more songs", "1 song (alone)"),
             ("single", "Release type", "Single", "Album or compilation"),
             ("explicit", "Explicit content", "Explicit", "Not explicit"),
             ("collab", "Collaboration", "Collaboration", "Solo")]
COLUMN_OF = {key: column for key, _, column, _, _ in SPLITS}
AGES_HORIZON = np.arange(HORIZON)                                            # chart days 0 ... 29: S(0) + ... + S(29) is the chart days used of the first 30


def rmst_draws(mask):
    """Chart days used of the first 30 for every artist draw (a draw without any artist of the group is NaN)."""
    draws = restricted_mean(survival_grid(T[mask], E[mask], AGES_HORIZON, weights=W_BOOT[:, mask]), HORIZON)
    return np.where(W_BOOT[:, mask].sum(axis=1) > 0, draws, np.nan)


CONTRAST_ROWS = []
for index, (test_key, split_name, group, reference) in enumerate(CONTRASTS):
    mask = (STAYS[COLUMN_OF[test_key]].astype(str) == group).to_numpy()
    base = (STAYS[COLUMN_OF[test_key]].astype(str) == reference).to_numpy()
    estimate = GROUP_DATA[(test_key, group)]["rmst30"] - GROUP_DATA[(test_key, reference)]["rmst30"]
    difference = rmst_draws(mask) - rmst_draws(base)                         # the same artist draw for both groups: the difference is measured inside each draw
    low, high = np.nanpercentile(difference, [2.5, 97.5])
    CONTRAST_ROWS.append({"split": split_name, "group": group, "reference": reference, "songs": int(mask.sum()), "reference_songs": int(base.sum()),
                          "difference": estimate, "low": float(low), "high": float(high)})
    FACTS[f"diff{index + 1}"], FACTS[f"diff{index + 1}_low"], FACTS[f"diff{index + 1}_high"] = float(estimate), float(low), float(high)
CONTRAST_TABLE = pd.DataFrame(CONTRAST_ROWS)
FACTS["n_contrasts"] = len(CONTRAST_TABLE)
display(CONTRAST_TABLE.assign(**{"chart days used of 30, difference": [f"{d:+.1f}  ({a:+.1f} to {b:+.1f})" for d, a, b in zip(CONTRAST_TABLE["difference"], CONTRAST_TABLE["low"], CONTRAST_TABLE["high"])]})
        [["split", "group", "reference", "songs", "chart days used of 30, difference"]].set_index(["split", "group"]))

# %% [md]
**What this code is.** Figure 5. The nine differences of the table as a forest plot: a dot for the difference and a bar for its 95% artist band, grouped by split, with zero marked.

**Why we need it.** One glance shows which differences are large, which have bands that stay on one side of zero, and what the band looks like when there is nothing to find.

**What will happen.** The code draws one row for each comparison with a bold heading for each split, the reference group named in the heading, the zero line, and the value with its band at the right of each row. It saves the picture as `p6b_05_differences.png`.

**Expected outcome.** The four debut-place rows far to the left of zero, the same-day rows and the single row on either side of it as the table says, and the explicit and collaboration rows sitting on the zero line with bands a few chart days wide.

**Motive.** Show how large the differences are and how much the data can rule out.

# %%
# Figure 05 - how many more or fewer chart days of the first 30 each group uses
fig, ax = make_fig(11.8, 6.3, 1, 1, left=3.7, right=2.0, bottom=0.95,
                   title="A top-10 debut, arriving alone and being a single go with more chart days used out of the first 30; explicit content and collaboration go with no difference",
                   subtitle=f"Average chart days used out of the first {HORIZON}, each group compared with a reference group. Dot: difference in the data. Bar: 95% band from drawing whole artists "
                            f"({N_BOOT:,} draws, the same draws for both groups of a comparison).",
                   note_text="Chart days used of the first 30 is the area under the survival curve up to 30 chart days. A bar that does not cross the zero line is a difference "
                             "with the same sign in at least 97.5% of the artist draws.")
heights, y_position, row_of = [], 0.0, []
for split_key, split_name in [("band", "Debut place  (compared with 1 to 10)"), ("drop", "Artist's songs that day  (compared with 1 song)"),
                              ("single", "Release type  (compared with album or compilation)"), ("explicit", "Explicit content  (compared with not explicit)"),
                              ("collab", "Collaboration  (compared with solo)")]:
    heights.append((y_position, split_name))
    y_position += 0.9
    for index, (test_key, _, group, _) in enumerate(CONTRASTS):
        if test_key == split_key:
            row_of.append((y_position, index))
            y_position += 1.0
    y_position += 0.35
ax.axvline(0, color=style.INK_2, linewidth=1.0, zorder=1)
for y_row, index in row_of:
    row = CONTRAST_TABLE.iloc[index]
    color = style.BLUE if row["high"] < 0 or row["low"] > 0 else style.AXIS
    ax.hlines(y_row, row["low"], row["high"], color=color, linewidth=5.0, alpha=0.55, zorder=2)
    ax.scatter([row["difference"]], [y_row], s=46, color=style.BLUE if color == style.BLUE else style.INK_2, edgecolor=style.SURFACE, linewidth=1.3, zorder=4)
    ax.text(1.02, y_row, f"{row['difference']:+.1f}  ({row['low']:+.1f} to {row['high']:+.1f})", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.8, color=style.INK_2)
    ax.text(-0.02, y_row, f"{row['group']}  (n = {row['songs']})", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, color=style.INK_2)
for y_head, name in heights:
    ax.text(-0.02, y_head, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, fontweight="bold", color=style.INK)
ax.set_ylim(y_position - 0.5, -0.7)
ax.set_xlim(-14, 8)
ax.set_yticks([])
ax.set_xlabel(f"difference in chart days used out of the first {HORIZON}")
style_axes(ax, grid="x", left_spine=False)
save_fig(fig, "p6b_05_differences", FIG_DIR)
plt.show()

# %% [md]
**What to notice**

- *Debut place.* Every one of the four differences is large and clear of zero. Compared with a top-10 debut, a song that arrives at 11 to 20 uses {{diff1:sf1}} chart days of its first {{horizon}} (band {{diff1_low:sf1}} to {{diff1_high:sf1}}), one at 21 to 30 uses {{diff2:sf1}}, one at 31 to 40 {{diff3:sf1}} and one at 41 to 50 {{diff4:sf1}}. The largest gap is in the middle of the chart and the last band is closer to the top than the two before it, the dip and recovery of the curves.
- *Songs arriving the same day.* A song that arrives with one to three others by its artist uses {{diff5:sf1}} chart days compared with a song that arrives alone (band {{diff5_low:sf1}} to {{diff5_high:sf1}}), and one that arrives in a drop of five or more uses {{diff6:sf1}} ({{diff6_low:sf1}} to {{diff6_high:sf1}}). The bands are wide because these groups come from only {{drop_d2_artists}} and {{drop_d3_artists}} artists.
- *Release type.* A single uses {{diff7:sf1}} more chart days than an album track (band {{diff7_low:sf1}} to {{diff7_high:sf1}}).
- *Explicit content and collaboration.* The differences are {{diff8:sf1}} (band {{diff8_low:sf1}} to {{diff8_high:sf1}}) and {{diff9:sf1}} ({{diff9_low:sf1}} to {{diff9_high:sf1}}). "No difference" here means no difference larger than about two or three chart days of {{horizon}}; the data cannot rule out smaller ones.
- These are crude differences. The groups also differ in other ways: a song in a big drop is probably an album track, and a song that arrives alone is more likely to be a single. Whether any difference remains when the others are held fixed is the question of the next two sections.
