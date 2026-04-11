"""Horse performance features based on past race history."""

from __future__ import annotations

import polars as pl


def add_performance_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add per-horse historical performance features (time-series safe)."""
    if df.is_empty():
        return df
    return (
        df.sort(["blood_registration_number", "race_date", "race_number", "post_position"])
        .group_by("blood_registration_number", maintain_order=True)
        .map_groups(_add_horse_performance)
    )


def _add_horse_performance(group: pl.DataFrame) -> pl.DataFrame:
    from collections.abc import Sequence
    from datetime import date

    import numpy as np

    race_dates: Sequence[date] = group["race_date"].to_list()
    finish_positions = group["finish_position"].to_list()
    last_3f_times = group["last_3f_time"].to_list()
    first_3f_times = group["first_3f_time"].to_list() if "first_3f_time" in group.columns else [None] * len(race_dates)
    race_time_secs = group["race_time_sec"].to_list() if "race_time_sec" in group.columns else [None] * len(race_dates)
    bw_changes = group["body_weight_change"].to_list() if "body_weight_change" in group.columns else [None] * len(race_dates)
    jrdb_idm_posts = group["jrdb_idm_post"].to_list() if "jrdb_idm_post" in group.columns else [None] * len(race_dates)
    ten_actuals = group["ten_index_actual"].to_list() if "ten_index_actual" in group.columns else [None] * len(race_dates)
    agari_actuals = group["agari_index_actual"].to_list() if "agari_index_actual" in group.columns else [None] * len(race_dates)
    pace_actuals = group["pace_index_actual"].to_list() if "pace_index_actual" in group.columns else [None] * len(race_dates)
    base_scores = group["base_score"].to_list() if "base_score" in group.columns else [None] * len(race_dates)
    course_positions = group["course_position"].to_list() if "course_position" in group.columns else [None] * len(race_dates)
    z_race_times = group["z_race_time"].to_list() if "z_race_time" in group.columns else [None] * len(race_dates)
    z_last_3fs = group["z_last_3f"].to_list() if "z_last_3f" in group.columns else [None] * len(race_dates)
    z_first_3fs = group["z_first_3f"].to_list() if "z_first_3f" in group.columns else [None] * len(race_dates)
    z_idm_posts = group["z_idm_post"].to_list() if "z_idm_post" in group.columns else [None] * len(race_dates)
    z_agari_actuals = group["z_agari_actual"].to_list() if "z_agari_actual" in group.columns else [None] * len(race_dates)

    prev_finish: list[list[float | None]] = [[] for _ in range(5)]
    prev_last_3f: list[list[float | None]] = [[] for _ in range(5)]
    prev_first_3f: list[list[float | None]] = [[] for _ in range(5)]
    prev_time: list[list[float | None]] = [[] for _ in range(5)]
    win_rate_10: list[float | None] = []
    top3_rate_10: list[float | None] = []
    avg_finish_10: list[float | None] = []
    avg_last_3f_5: list[float | None] = []
    avg_first_3f_5: list[float | None] = []
    ewm_finish: list[float | None] = []
    days_since: list[int | None] = []
    total_races: list[int] = []
    weight_change_trend: list[float | None] = []
    avg_rest_days: list[float | None] = []
    form_score_finish: list[float | None] = []
    win_streak_top3: list[int | None] = []
    days_since_last_win: list[int | None] = []
    # Z-value rolling features
    prev1_z_time: list[float | None] = []
    prev2_z_time: list[float | None] = []
    prev3_z_time: list[float | None] = []
    avg_z_time_5: list[float | None] = []
    best_z_time: list[float | None] = []
    std_z_time_5: list[float | None] = []
    avg_z_last3f_5: list[float | None] = []
    best_z_last3f: list[float | None] = []
    avg_z_idm_5: list[float | None] = []
    avg_z_agari_5: list[float | None] = []
    best_z_agari: list[float | None] = []
    trend_z_time: list[float | None] = []
    form_score_z: list[float | None] = []
    recent_vs_career_z: list[float | None] = []

    rolling_idm_post: list[float | None] = []
    rolling_ten_actual: list[float | None] = []
    rolling_agari_actual: list[float | None] = []
    rolling_pace_actual: list[float | None] = []
    rolling_base_score: list[float | None] = []
    best_agari_actual: list[float | None] = []
    avg_course_position: list[float | None] = []

    history: list[tuple[date, int | None, float | None, float | None, float | None]] = []
    bw_history: list[float | None] = []
    rest_history: list[int] = []
    idm_post_history: list[float] = []
    ten_actual_history: list[float] = []
    agari_actual_history: list[float] = []
    pace_actual_history: list[float] = []
    base_score_history: list[float] = []
    course_pos_history: list[float] = []
    z_time_history: list[float] = []
    z_last3f_history: list[float] = []
    z_first3f_history: list[float] = []
    z_idm_history: list[float] = []
    z_agari_history: list[float] = []

    for idx, current_date in enumerate(race_dates):
        prior = [(d, fp, l3, rt, f3) for d, fp, l3, rt, f3 in history if d < current_date]
        fps = [fp for _, fp, _, _, _ in prior if fp is not None]
        l3s = [l3 for _, _, l3, _, _ in prior if l3 is not None]
        f3s = [f3 for _, _, _, _, f3 in prior if f3 is not None]
        rts = [rt for _, _, _, rt, _ in prior if rt is not None]

        for n in range(5):
            prev_finish[n].append(fps[-(n + 1)] if len(fps) > n else None)
            prev_last_3f[n].append(l3s[-(n + 1)] if len(l3s) > n else None)
            prev_first_3f[n].append(f3s[-(n + 1)] if len(f3s) > n else None)
            prev_time[n].append(rts[-(n + 1)] if len(rts) > n else None)

        last_10 = fps[-10:]
        win_rate_10.append(sum(1 for x in last_10 if x == 1) / len(last_10) if last_10 else None)
        top3_rate_10.append(sum(1 for x in last_10 if x <= 3) / len(last_10) if last_10 else None)
        avg_finish_10.append(float(np.mean(last_10)) if last_10 else None)

        last_5_l3 = l3s[-5:]
        avg_last_3f_5.append(float(np.mean(last_5_l3)) if last_5_l3 else None)

        last_5_f3 = f3s[-5:]
        avg_first_3f_5.append(float(np.mean(last_5_f3)) if last_5_f3 else None)

        if len(fps) >= 2:
            weights = np.array([0.5 ** i for i in range(len(fps))])[::-1]
            ewm_finish.append(float(np.average(fps, weights=weights)))
        elif fps:
            ewm_finish.append(float(fps[-1]))
        else:
            ewm_finish.append(None)

        if prior:
            gap = (current_date - prior[-1][0]).days
            days_since.append(gap)
        else:
            days_since.append(None)
        total_races.append(len(prior))

        prior_bw = [bw for bw in bw_history if bw is not None]
        last_3_bw = prior_bw[-3:]
        weight_change_trend.append(float(np.mean(last_3_bw)) if last_3_bw else None)

        if len(rest_history) >= 2:
            avg_rest_days.append(float(np.mean(rest_history[-5:])))
        else:
            avg_rest_days.append(None)

        if len(fps) >= 3:
            fs = 0.5 * fps[-1] + 0.3 * fps[-2] + 0.2 * fps[-3]
            form_score_finish.append(float(fs))
        else:
            form_score_finish.append(None)

        streak = 0
        for fp_val in reversed(fps):
            if fp_val <= 3:
                streak += 1
            else:
                break
        win_streak_top3.append(streak if fps else None)

        win_dates = [d for d, fp_val, _, _, _ in prior if fp_val == 1]
        if win_dates:
            days_since_last_win.append((current_date - win_dates[-1]).days)
        else:
            days_since_last_win.append(None)

        # Z-value rolling computations
        zt = [v for v in z_time_history]
        zl = [v for v in z_last3f_history]
        zi = [v for v in z_idm_history]
        za = [v for v in z_agari_history]

        prev1_z_time.append(zt[-1] if len(zt) >= 1 else None)
        prev2_z_time.append(zt[-2] if len(zt) >= 2 else None)
        prev3_z_time.append(zt[-3] if len(zt) >= 3 else None)

        last5_zt = zt[-5:]
        avg_z_time_5.append(float(np.mean(last5_zt)) if last5_zt else None)
        best_z_time.append(float(min(zt)) if zt else None)
        std_z_time_5.append(float(np.std(last5_zt)) if len(last5_zt) >= 3 else None)

        last5_zl = zl[-5:]
        avg_z_last3f_5.append(float(np.mean(last5_zl)) if last5_zl else None)
        best_z_last3f.append(float(min(zl)) if zl else None)

        last5_zi = zi[-5:]
        avg_z_idm_5.append(float(np.mean(last5_zi)) if last5_zi else None)

        last5_za = za[-5:]
        avg_z_agari_5.append(float(np.mean(last5_za)) if last5_za else None)
        best_z_agari.append(float(min(za)) if za else None)

        if len(zt) >= 3:
            recent = zt[-3:]
            slope = (recent[-1] - recent[0]) / 2.0
            trend_z_time.append(float(slope))
        else:
            trend_z_time.append(None)

        if len(zt) >= 3:
            fs = 0.5 * zt[-1] + 0.3 * zt[-2] + 0.2 * zt[-3]
            form_score_z.append(float(fs))
        else:
            form_score_z.append(None)

        if len(zt) >= 5:
            recent_avg = np.mean(zt[-3:])
            career_avg = np.mean(zt)
            recent_vs_career_z.append(float(recent_avg - career_avg))
        else:
            recent_vs_career_z.append(None)

        last_5_idm = [v for v in idm_post_history if v is not None][-5:]
        rolling_idm_post.append(float(np.mean(last_5_idm)) if last_5_idm else None)

        last_5_ten = [v for v in ten_actual_history if v is not None][-5:]
        rolling_ten_actual.append(float(np.mean(last_5_ten)) if last_5_ten else None)

        last_5_agari = [v for v in agari_actual_history if v is not None][-5:]
        rolling_agari_actual.append(float(np.mean(last_5_agari)) if last_5_agari else None)
        all_agari = [v for v in agari_actual_history if v is not None]
        best_agari_actual.append(float(min(all_agari)) if len(all_agari) >= 3 else None)

        last_5_pace = [v for v in pace_actual_history if v is not None][-5:]
        rolling_pace_actual.append(float(np.mean(last_5_pace)) if last_5_pace else None)

        last_5_bs = [v for v in base_score_history if v is not None][-5:]
        rolling_base_score.append(float(np.mean(last_5_bs)) if last_5_bs else None)

        last_5_cp = [v for v in course_pos_history if v is not None][-5:]
        avg_course_position.append(float(np.mean(last_5_cp)) if last_5_cp else None)

        fp = finish_positions[idx]
        l3 = last_3f_times[idx]
        rt = race_time_secs[idx]
        f3 = first_3f_times[idx]
        if fp is not None and fp >= 1:
            history.append((current_date, int(fp), l3, rt, f3))

        bw_changes_val = bw_changes[idx]
        bw_history.append(bw_changes_val)

        zrt = z_race_times[idx]
        if zrt is not None and not (isinstance(zrt, float) and np.isnan(zrt)):
            z_time_history.append(float(zrt))
        zl3 = z_last_3fs[idx]
        if zl3 is not None and not (isinstance(zl3, float) and np.isnan(zl3)):
            z_last3f_history.append(float(zl3))
        zf3 = z_first_3fs[idx]
        if zf3 is not None and not (isinstance(zf3, float) and np.isnan(zf3)):
            z_first3f_history.append(float(zf3))
        zidm = z_idm_posts[idx]
        if zidm is not None and not (isinstance(zidm, float) and np.isnan(zidm)):
            z_idm_history.append(float(zidm))
        zag = z_agari_actuals[idx]
        if zag is not None and not (isinstance(zag, float) and np.isnan(zag)):
            z_agari_history.append(float(zag))

        idm_v = jrdb_idm_posts[idx]
        if idm_v is not None:
            idm_post_history.append(float(idm_v))
        ten_v = ten_actuals[idx]
        if ten_v is not None:
            ten_actual_history.append(float(ten_v))
        agari_v = agari_actuals[idx]
        if agari_v is not None:
            agari_actual_history.append(float(agari_v))
        pace_v = pace_actuals[idx]
        if pace_v is not None:
            pace_actual_history.append(float(pace_v))
        bs_v = base_scores[idx]
        if bs_v is not None:
            base_score_history.append(float(bs_v))
        cp_v = course_positions[idx]
        if cp_v is not None:
            course_pos_history.append(float(cp_v))

        if prior:
            rest_history.append((current_date - prior[-1][0]).days)

    cols = [
        *[pl.Series(f"prev{n+1}_finish_pos", prev_finish[n], dtype=pl.Float64) for n in range(5)],
        *[pl.Series(f"prev{n+1}_last_3f", prev_last_3f[n], dtype=pl.Float64) for n in range(5)],
        *[pl.Series(f"prev{n+1}_first_3f", prev_first_3f[n], dtype=pl.Float64) for n in range(5)],
        *[pl.Series(f"prev{n+1}_race_time_sec", prev_time[n], dtype=pl.Float64) for n in range(5)],
        pl.Series("win_rate_10", win_rate_10, dtype=pl.Float64),
        pl.Series("top3_rate_10", top3_rate_10, dtype=pl.Float64),
        pl.Series("avg_finish_10", avg_finish_10, dtype=pl.Float64),
        pl.Series("avg_last_3f_5", avg_last_3f_5, dtype=pl.Float64),
        pl.Series("avg_first_3f_5", avg_first_3f_5, dtype=pl.Float64),
        pl.Series("ewm_finish", ewm_finish, dtype=pl.Float64),
        pl.Series("days_since_last_race", days_since, dtype=pl.Int64),
        pl.Series("total_races", total_races, dtype=pl.Int64),
        pl.Series("weight_change_trend", weight_change_trend, dtype=pl.Float64),
        pl.Series("avg_rest_days", avg_rest_days, dtype=pl.Float64),
        pl.Series("prev1_z_time", prev1_z_time, dtype=pl.Float64),
        pl.Series("prev2_z_time", prev2_z_time, dtype=pl.Float64),
        pl.Series("prev3_z_time", prev3_z_time, dtype=pl.Float64),
        pl.Series("avg_z_time_5", avg_z_time_5, dtype=pl.Float64),
        pl.Series("best_z_time", best_z_time, dtype=pl.Float64),
        pl.Series("std_z_time_5", std_z_time_5, dtype=pl.Float64),
        pl.Series("avg_z_last3f_5", avg_z_last3f_5, dtype=pl.Float64),
        pl.Series("best_z_last3f", best_z_last3f, dtype=pl.Float64),
        pl.Series("avg_z_idm_5", avg_z_idm_5, dtype=pl.Float64),
        pl.Series("avg_z_agari_5", avg_z_agari_5, dtype=pl.Float64),
        pl.Series("best_z_agari", best_z_agari, dtype=pl.Float64),
        pl.Series("trend_z_time", trend_z_time, dtype=pl.Float64),
        pl.Series("form_score_z", form_score_z, dtype=pl.Float64),
        pl.Series("recent_vs_career_z", recent_vs_career_z, dtype=pl.Float64),
        pl.Series("form_score_finish", form_score_finish, dtype=pl.Float64),
        pl.Series("win_streak_top3", win_streak_top3, dtype=pl.Int64),
        pl.Series("days_since_last_win", days_since_last_win, dtype=pl.Int64),
        pl.Series("rolling_idm_post", rolling_idm_post, dtype=pl.Float64),
        pl.Series("rolling_ten_actual", rolling_ten_actual, dtype=pl.Float64),
        pl.Series("rolling_agari_actual", rolling_agari_actual, dtype=pl.Float64),
        pl.Series("rolling_pace_actual", rolling_pace_actual, dtype=pl.Float64),
        pl.Series("rolling_base_score", rolling_base_score, dtype=pl.Float64),
        pl.Series("best_agari_actual", best_agari_actual, dtype=pl.Float64),
        pl.Series("avg_course_position", avg_course_position, dtype=pl.Float64),
    ]
    return group.with_columns(cols)
