# %% [md]
## Section 10 - Saving the results, and what Phase 6B adds up to

# %% [md]
**What this code is.** The saving cell. It collects the result tables of the notebook, puts each in the shape other tools like (one row per song, one row per point of a curve, one row per input), and writes them to `reports/` as CSV files with UTF-8 and a byte-order mark, so that Excel on Windows shows the text correctly. It also writes `p6b_facts.json`, which holds every number quoted in the sentences of this notebook.

**Why we need it.** Later phases need these numbers without running the notebook again: the dashboard of Phase 7 and the Power BI report of Phase 8 read the CSV files, and the paper of Phase 9 quotes the facts file. A file written once by the notebook also cannot disagree with the notebook.

**What will happen.** The code builds twelve tables (the song table gets two extra columns, the risk third of each song on the arrival day and after three chart days), writes each one, reads it back to check that it has the same shape, checks that all 12 charts are in `reports/figures/`, and writes the facts file. It prints the files with their numbers of rows and columns.

**Expected outcome.** A list of thirteen files: twelve CSV files and the facts file. The song file has {{n_songs}} rows, one for each song; the curve file is the longest.

**Motive.** Keep the results outside the notebook, in files that other tools can read and that have been checked.

# %%
REPORT_DIR = paths.reports_dir


def as_frame(table):
    """Some of the tables of the earlier sections are lists of rows; the saving code works on data frames."""
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)


# ---- one row per song, with the risk thirds of Section 8
STAY_COLUMNS = ["track_id", "track_label", "primary_artist", "entry_date", "entry_snapshot", "entry_rank", "debut_band", "drop_size", "total_tracks", "is_single", "is_explicit", "is_collab",
                "duration_min", "entry_pop", "artist_prior_slot_days", "artist_prior_mean_pop", "artist_on_chart_prev_day", "stay", "exit_seen", "last_snapshot", "short_breaks",
                "first_run_days", "returned"]
SONGS_OUT = STAYS[STAY_COLUMNS].copy()
SONGS_OUT["debut_band"] = SONGS_OUT["debut_band"].astype(str)
SONGS_OUT["risk_third_arrival"] = pd.Series(THIRD_NAMES)[third_index(OOF["+ Debut day"].mean(axis=0))].to_numpy()
SONGS_OUT["risk_third_after_3_days"] = pd.Series([None] * len(SONGS_OUT), dtype=object)
SONGS_OUT.loc[LM, "risk_third_after_3_days"] = pd.Series(THIRD_NAMES)[third_index(OOF_LM["+ first three chart days"].mean(axis=0))].to_numpy()

# ---- the five groupings: the group rows with the log-rank test of the split and the RMST difference against the reference group
TEST_COLUMNS = {"p_songs": "log_rank_p_songs_as_independent", "p_artists": "log_rank_p_artist_aware", "p_holm": "log_rank_p_artist_aware_holm"}
CONTRAST_COLUMNS = {"reference": "reference_group", "difference": "chart_days_of_first_30_difference", "low": "difference_low", "high": "difference_high"}
GROUP_OUT = (as_frame(GROUP_ROWS).merge(as_frame(TESTS)[["split"] + list(TEST_COLUMNS)].rename(columns=TEST_COLUMNS), on="split", how="left")
             .merge(as_frame(CONTRAST_TABLE)[["split", "group"] + list(CONTRAST_COLUMNS)].rename(columns=CONTRAST_COLUMNS), on=["split", "group"], how="left"))
assert len(GROUP_OUT) == len(as_frame(GROUP_ROWS)), "the merges changed the number of group rows"

# ---- the hazard ratios with the test of the block each input belongs to; the effects by age with the test of equal effects
BLOCK_OF = {column: block for block, columns in BLOCKS for column in columns}
RATIOS_OUT = (as_frame(HR_TABLE).rename(columns={"p": "p_coefficient"}).assign(block=lambda d: d["input"].map(BLOCK_OF))
              .merge(as_frame(BLOCK_TABLE).rename(columns={"input": "block", "F": "block_F", "df1": "block_df1", "df2": "block_df2", "p": "block_p", "p_holm": "block_p_holm"}), on="block", how="left"))
EFFECTS_OUT = as_frame(AGE_TABLE).merge(as_frame(PH_TABLE).rename(columns={"input": "block", "F": "equal_effects_F", "df1": "equal_effects_df1", "df2": "equal_effects_df2", "p": "equal_effects_p", "p_holm": "equal_effects_p_holm"}),
                              on="block", how="left")
assert len(RATIOS_OUT) == len(as_frame(HR_TABLE)) and len(EFFECTS_OUT) == len(as_frame(AGE_TABLE)), "the merges changed the number of rows"

# ---- the danger zone with the check that leaves out the busiest exit day
DANGER_OUT = as_frame(DANGER_TABLE).assign(chance_last_day_without_busiest_day=chance_x, without_busiest_day_low=chance_x_low, without_busiest_day_high=chance_x_high)
LADDER_OUT = pd.DataFrame(ARRIVAL_ROWS + LM_ROWS)

TABLES = {"p6b_stays": SONGS_OUT, "p6b_survival_curves": as_frame(CURVE_ROWS), "p6b_hazard_by_age": as_frame(HAZARD_TABLE), "p6b_further_stay": as_frame(FURTHER_TABLE), "p6b_group_comparison": GROUP_OUT,
          "p6b_debut_alone": as_frame(DEBUT_TABLE), "p6b_hazard_ratios": RATIOS_OUT, "p6b_effects_by_age": EFFECTS_OUT, "p6b_danger_zone": DANGER_OUT, "p6b_ladder": LADDER_OUT,
          "p6b_risk_thirds": THIRDS_TABLE, "p6b_stress_tests": STRESS_TABLE}
FACTS["n_tables_saved"] = len(TABLES)
FACTS["n_figures_saved"] = len(sorted(FIG_DIR.glob("p6b_*.png")))
written = []
for name, frame in TABLES.items():
    path = REPORT_DIR / f"{name}.csv"
    save_table(frame, path)
    back = pd.read_csv(path, encoding="utf-8-sig")
    assert back.shape == frame.shape, f"{name}: the file read back has the shape {back.shape}, not {frame.shape}"
    written.append({"file": f"{name}.csv", "rows": len(frame), "columns": frame.shape[1]})
assert FACTS["n_figures_saved"] == 12, f"expected 12 charts in {FIG_DIR}, found {FACTS['n_figures_saved']}"
FACTS.save(REPORT_DIR / "p6b_facts.json")
written.append({"file": "p6b_facts.json", "rows": len(FACTS), "columns": 2})
display(pd.DataFrame(written).set_index("file"))
print(f"{FACTS['n_tables_saved']} tables and the facts file written to {REPORT_DIR}; {FACTS['n_figures_saved']} charts are in {FIG_DIR}.")

# %% [md]
### What Phase 6B adds up to

**How long songs stay.** Of the {{n_songs:n}} songs that arrived after the first snapshot day, {{s1:pct0}} are still on the chart after one chart day, {{s7:pct0}} after a week and {{s30:pct0}} after {{horizon}} chart days (95% band {{s30_low:pct0}} to {{s30_high:pct0}}); the median first stay is {{median_stay:f0}} chart days. The chance that a chart day is the last one of a stay falls steeply with the age of the stay, from {{hz1:pct0}} on the first chart day to {{hz9:pct1}} after day 120, so no single average chance describes a stay. The mean of the raw stays ({{stay_mean_raw:f1}} chart days) is far above the median because a small number of songs stay for months (the longest stay is {{stay_max}} chart days).

**What goes with a short stay.** Three groupings separate the stays after the artist-aware tests and the Holm adjustment: the debut place ({{lr_band_p_holm_text}}), the number of songs the artist brought on the same day ({{lr_drop_p_holm_text}}) and the release type ({{lr_single_p_holm_text}}); explicit content and collaboration do not. When the inputs are held against each other, two remain. A debut at places 31 to 40 goes with about {{hr_b4:f1}} times the hazard of a top-10 debut (95% interval {{hr_b4_low:f1}} to {{hr_b4_high:f1}}), and one standard deviation more album drop (a drop {{sd_drop_factor:f1}} times as large) with about {{hr_drop:f1}} times ({{hr_drop_low:f1}} to {{hr_drop_high:f1}}). The single's advantage in Section 4 is a mix effect: its ratio in the model is {{hr_single:f2}} ({{hr_single_low:f2}} to {{hr_single_high:f2}}).

**Where and how early it shows.** A chart day in places 1 to 5 is the last one of its stay {{dz1:pct1}} of the time, in places 46 to 50 {{dz10:pct0}}, and {{dz_bottom_exit_share:pct0}} of all exits happen in places 41 to 50, which hold {{dz_bottom_days_share:pct0}} of the chart days. The arrival day carries some information about the order in which songs leave (cross-validated C-index {{c4:f2}}, and the debut day is the only step that clearly adds to the release facts), and the first three chart days carry much more ({{lm2:f2}}). Sorted by the arrival-day model, the median stay of the lowest-risk third is {{ra1_median:f0}} chart days and that of the highest-risk third {{ra3_median:f0}}.

**How firm it is.** The intervals count artists, not songs, as the unit. Nine runs that change the gap rule or the songs keep every debut-place ratio above 1 and clear of it ({{st_debut_clear}} of {{st_debut_total}} intervals), keep the album drop clear of 1 in {{st_drop_clear}} of {{st_runs}} runs, and leave the single unclear in all of them. The level of the curve moves with the gap rule ({{st_s30_gap_min:pct0}} to {{st_s30_gap_max:pct0}} after {{horizon}} chart days), and the size of the debut-place ratio differs between the first and the second half of the file.

**What it does not say.**

- Every result is an association among these {{n_songs:n}} songs between {{first_date}} and {{last_date}}. Nothing here shows what would happen to a song if its debut place or its album drop were changed.
- The one-ratio model is a simplification: {{ph_n_flagged}} of the {{n_inputs}} inputs act differently at different ages (Section 6), most of all the debut place, whose effect sits in the first chart days. A model with age-varying ratios was not tried.
- The {{day_one_songs}} songs that were already on the chart on the first day are not in the analysis, and the {{n_censored}} stays that had not ended when the file ends are counted as "at least this long".
- The artist bands treat artists as independent, but songs of different artists can leave on the same day: {{busy_exits}} stays ended on {{busy_text}}. Section 7 shows that the place pattern survives without that day; the other sections keep it.
- A C-index of {{c4:f2}} measures how much the arrival day carried in this file. It is not a promise about future songs.

**Where this leads.** Phase 6C groups songs and artists by the shape of their first month on the chart and asks whether the groups are stable and whether they line up with the survival results above. Phase 7 puts these results into the dashboard and Phase 8 into the Power BI report; the files written in this section are what they read.
