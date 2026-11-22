"""Descriptive stimulus-locked epochs and QC aligned to retained MNE events."""

import mne
import numpy as np
import pandas as pd


def stimulus_locked_epochs(raw, trials: pd.DataFrame, dropped: pd.DataFrame,
                           cfg: dict, stimulus_lag_s: float):
    sfreq = raw.info["sfreq"]
    marker = trials["onset"].to_numpy(float)
    levels = trials["level"].to_numpy(int)
    samples = np.rint((marker-stimulus_lag_s)*sfreq).astype(int)
    events = np.c_[samples, np.zeros(len(samples), dtype=int), levels]
    epochs = mne.Epochs(raw, events, event_id={f"{i}-back": i for i in range(1, 5)},
                        tmin=cfg["tmin_s"], tmax=cfg["tmax_s"],
                        baseline=(cfg["tmin_s"], cfg["baseline_end_s"]),
                        preload=True, reject_by_annotation=False, verbose=False)
    data = epochs.get_data(copy=False)
    selection = epochs.selection
    if len(selection) == 0:
        raise ValueError("No epochs within the recording")
    p2p = np.ptp(data, axis=2).max(axis=1)*1e6
    onsets = dropped["onset"].to_numpy(float)
    counts = dropped["dropped_samples"].fillna(0).to_numpy(float)
    # Original QC window: marker − 1.2 s to marker. Kept configurable.
    marker_sel = marker[selection]
    drop_counts = np.array([counts[(onsets >= t-cfg["dropout_lookback_s"]) &
                                   (onsets < t)].sum() for t in marker_sel])
    keep = (drop_counts <= cfg["max_dropped_samples"]) & (p2p <= cfg["max_p2p_uv"])
    qc = pd.DataFrame({"level": levels[selection], "marker_onset_s": marker_sel,
                       "inferred_stimulus_onset_s": marker_sel-stimulus_lag_s,
                       "dropped_samples": drop_counts, "max_p2p_uV": p2p, "keep": keep})
    counts_by_level = qc.groupby("level").agg(candidate_epochs=("keep", "size"),
                                               retained_epochs=("keep", "sum")).reset_index()
    if set(qc.loc[qc["keep"], "level"]) != set(trials["level"]):
        raise ValueError("No ERP epochs retained for at least one level")
    return epochs, data, qc, counts_by_level
