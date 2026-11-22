"""Formal-trial selection and signal-detection summaries."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def formal_trials(events: pd.DataFrame) -> pd.DataFrame:
    starts = events.loc[events["trial_type"].eq("started_n_back"), "onset"]
    ends = events.loc[events["trial_type"].eq("finished_n_back"), "onset"]
    if len(starts) != 1 or len(ends) != 1 or ends.iloc[0] <= starts.iloc[0]:
        raise ValueError("Expected exactly one valid formal-task start/end marker")
    trials = events[
        events["trial_type"].str.fullmatch(r"[1-4]-back", na=False)
        & events["onset"].between(float(starts.iloc[0]), float(ends.iloc[0]))
    ].copy()
    trials["level"] = trials["trial_type"].str[0].astype(int)
    trials["key_press_bool"] = trials["key_press"].astype(str).str.lower().eq("true")
    trials["matched_bool"] = trials["matched"].astype(str).str.lower().eq("true")
    if trials.empty or set(trials["level"]) != {1, 2, 3, 4}:
        raise ValueError("Expected formal trials at each level 1–4")
    return trials


def behaviour_row(g: pd.DataFrame) -> pd.Series:
    target = g["matched_bool"].to_numpy()
    response = g["key_press_bool"].to_numpy()
    hits = int(np.sum(target & response))
    misses = int(np.sum(target & ~response))
    false_alarms = int(np.sum(~target & response))
    correct_rejections = int(np.sum(~target & ~response))
    hit_raw = hits/(hits+misses) if hits+misses else np.nan
    cr_raw = correct_rejections/(false_alarms+correct_rejections) if false_alarms+correct_rejections else np.nan
    hit = (hits+0.5)/(hits+misses+1)
    fa = (false_alarms+0.5)/(false_alarms+correct_rejections+1)
    return pd.Series({
        "trials": len(g), "target_prevalence": target.mean(),
        "accuracy": g["response_accuracy"].mean(),
        "balanced_accuracy": 0.5*(hit_raw+cr_raw), "hits": hits, "misses": misses,
        "false_alarms": false_alarms, "correct_rejections": correct_rejections,
        "hit_rate_corrected": hit, "false_alarm_rate_corrected": fa,
        "d_prime": norm.ppf(hit)-norm.ppf(fa),
    })


def summarize_behaviour(trials: pd.DataFrame) -> pd.DataFrame:
    rows = [{"level": level, **behaviour_row(g).to_dict()} for level, g in trials.groupby("level", sort=True)]
    result = pd.DataFrame(rows)
    for col in ("trials", "hits", "misses", "false_alarms", "correct_rejections"):
        result[col] = result[col].astype(int)
    return result
