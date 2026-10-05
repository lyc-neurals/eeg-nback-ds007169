"""Nonoverlapping windows, dropout QC and Welch relative band powers."""

import numpy as np
import pandas as pd
from scipy.signal import welch


def integrate_band(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        raise ValueError(f"Frequency grid is too coarse for {lo}–{hi} Hz")
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def spectral_windows(raw, trials: pd.DataFrame, dropped: pd.DataFrame,
                     names: list[str], cfg: dict, stimulus_lag_s: float):
    sfreq = raw.info["sfreq"]
    onsets = dropped["onset"].to_numpy(float)
    counts = dropped["dropped_samples"].fillna(0).to_numpy(float)
    window_s = float(cfg["window_s"])
    nperseg = int(round(float(cfg["welch_segment_s"])*sfreq))
    overlap = int(round(float(cfg["welch_overlap_s"])*sfreq))
    fm = [names.index(ch) for ch in cfg["frontal_midline_channels"]]
    posterior = [names.index(ch) for ch in cfg["posterior_channels"]]
    rows, psd_rows = [], []
    for level, g in trials.groupby("level", sort=True):
        block_start = float(g["onset"].min()-stimulus_lag_s)
        block_end = float(g["onset"].max())
        starts = np.arange(block_start, block_end-window_s+1e-9, window_s)
        for start in starts:
            stop = start+window_s
            n_dropped = float(counts[(onsets >= start) & (onsets < stop)].sum())
            a, b = int(round(start*sfreq)), int(round(stop*sfreq))
            if a < 0 or b > raw.n_times:
                continue  # outside the recording; never use a partial window
            data = raw.get_data(start=a, stop=b)
            if data.shape[1] < nperseg:
                continue
            p2p = float(np.ptp(data, axis=1).max()*1e6)
            keep = n_dropped <= cfg["max_dropped_samples"] and p2p <= cfg["max_p2p_uv"]
            freqs, psd = welch(data, fs=sfreq, window="hamming", nperseg=nperseg,
                               noverlap=overlap, detrend="constant", axis=-1)
            total = integrate_band(freqs, psd, *cfg["normalization_band_hz"])
            theta = integrate_band(freqs, psd, *cfg["theta_band_hz"])/total
            alpha = integrate_band(freqs, psd, *cfg["alpha_band_hz"])/total
            beta = integrate_band(freqs, psd, *cfg["beta_band_hz"])/total
            rows.append({"level": int(level), "start_s": float(start),
                         "dropped_samples": n_dropped, "max_p2p_uV": p2p, "keep": bool(keep),
                         "frontal_midline_theta_rel": float(np.mean(theta[fm])),
                         "posterior_alpha_rel": float(np.mean(alpha[posterior])),
                         "global_theta_rel": float(np.mean(theta)),
                         "global_alpha_rel": float(np.mean(alpha)),
                         "global_beta_rel": float(np.mean(beta))})
            global_db = 10*np.log10(np.maximum(np.median(psd, axis=0)*1e12,
                                                 np.finfo(float).tiny))
            psd_rows.append((int(level), float(start), bool(keep), freqs, global_db))
    windows = pd.DataFrame(rows)
    if windows.empty:
        raise ValueError("No full-length spectral windows available")
    window_qc = windows.groupby("level").agg(
        candidate_windows=("keep", "size"), retained_windows=("keep", "sum"),
        median_dropped_samples=("dropped_samples", "median"),
        median_max_p2p_uV=("max_p2p_uV", "median")).reset_index()
    kept = windows[windows["keep"]].copy()
    if set(kept["level"]) != set(trials["level"]):
        raise ValueError("No spectral windows retained for at least one level")
    spectral_summary = kept.groupby("level").agg(
        n_windows=("keep", "size"),
        frontal_theta_median=("frontal_midline_theta_rel", "median"),
        frontal_theta_q25=("frontal_midline_theta_rel", lambda x: x.quantile(.25)),
        frontal_theta_q75=("frontal_midline_theta_rel", lambda x: x.quantile(.75)),
        posterior_alpha_median=("posterior_alpha_rel", "median"),
        posterior_alpha_q25=("posterior_alpha_rel", lambda x: x.quantile(.25)),
        posterior_alpha_q75=("posterior_alpha_rel", lambda x: x.quantile(.75)),
    ).reset_index()
    return windows, window_qc, spectral_summary, psd_rows
