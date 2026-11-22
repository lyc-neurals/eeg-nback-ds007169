"""Dropout and robust channel quality metrics."""

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation


def dropout_events(events: pd.DataFrame) -> pd.DataFrame:
    dropped = events.loc[events["trial_type"].eq("dropped_samples")].copy()
    dropped["end"] = dropped["onset"] + dropped["duration"]
    return dropped


def summarize_dropouts(dropped: pd.DataFrame, trials: pd.DataFrame,
                       n_samples: int, sfreq: float) -> pd.DataFrame:
    intervals = [(str(level), float(g["onset"].min()-1.0), float(g["onset"].max()))
                 for level, g in trials.groupby("level")]
    def assign(t):
        for label, start, end in intervals:
            if start <= t <= end:
                return label
        return "outside formal blocks"
    labeled = dropped.copy()
    labeled["block"] = labeled["onset"].map(assign)
    summary = labeled.groupby("block", sort=False).agg(
        events=("trial_type", "size"), dropped_samples=("dropped_samples", "sum"),
        duration_s=("duration", "sum")).reset_index()
    summary["percent_of_full_recording"] = 100*summary["duration_s"]/(n_samples/sfreq)
    return summary


def screen_channels(raw, names: list[str], robust_z_threshold: float) -> pd.DataFrame:
    uv = raw.get_data()*1e6
    ranges = np.percentile(uv, 99, axis=1)-np.percentile(uv, 1, axis=1)
    mad = median_abs_deviation(ranges, scale="normal")
    if not np.isfinite(mad) or mad == 0:
        raise ValueError("Channel-range MAD is zero or invalid; inspect channels manually")
    z = (ranges-np.median(ranges))/mad
    return pd.DataFrame({"channel": names, "p01_p99_range_uV": ranges,
                         "robust_z": z, "flagged": z > robust_z_threshold}).sort_values(
                             "robust_z", ascending=False)
