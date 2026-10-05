"""End-to-end single-subject analysis; no raw data or intermediate arrays are saved."""

import json
from pathlib import Path

import mne
import pandas as pd
import yaml

from .behavior import formal_trials, summarize_behaviour
from .erp import stimulus_locked_epochs
from .io import load_recording, subject_files, unit_validation
from .plotting import make_figures
from .preprocessing import filter_xdf_eeg, reference_interpolate_annotate
from .qc import dropout_events, screen_channels, summarize_dropouts
from .spectral import spectral_windows


def load_config(config_path: Path) -> dict:
    with Path(config_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or not all(k in config for k in
            ("preprocessing", "qc", "spectral", "erp", "plotting", "timing")):
        raise ValueError("Configuration must contain preprocessing, qc, spectral, erp, plotting and timing")
    return config


def run_subject(subject: str, data_root: Path, output_root: Path,
                config_path: Path, figures: bool = True) -> dict:
    """Run the unchanged scientific workflow with subject-independent file resolution."""
    cfg = load_config(config_path)
    mne.set_log_level("WARNING")
    files = subject_files(Path(data_root), subject)
    out = Path(output_root).expanduser().resolve()/subject
    out.mkdir(parents=True, exist_ok=True)
    xdf, names, indices, labels, sfreq, streams, timing = load_recording(files)
    unit_qc = unit_validation(files, xdf, indices, labels, cfg["qc"]["brainvision_probe_samples"])
    events = pd.read_csv(files.events, sep="\t")
    trials = formal_trials(events)
    behaviour = summarize_behaviour(trials)
    dropped = dropout_events(events)
    drop_summary = summarize_dropouts(dropped, trials, len(xdf), sfreq)

    unreferenced = filter_xdf_eeg(xdf, indices, names, sfreq, cfg["preprocessing"])
    channel_qc = screen_channels(unreferenced, names, cfg["qc"]["channel_robust_z_threshold"])
    bads = channel_qc.loc[channel_qc["flagged"], "channel"].tolist()
    raw = reference_interpolate_annotate(unreferenced, bads, dropped)

    lag = float(cfg["timing"]["marker_to_stimulus_lag_s"])
    windows, window_qc, spectral_summary, psd_rows = spectral_windows(
        raw, trials, dropped, names, cfg["spectral"], lag)
    epochs, epoch_data, erp_qc, erp_counts = stimulus_locked_epochs(
        raw, trials, dropped, cfg["erp"], lag)

    tables = {
        "stream_summary.csv": streams,
        "timing_qc.csv": pd.Series(timing, name="value").rename_axis("metric").reset_index(),
        "unit_validation.csv": unit_qc,
        "behaviour_by_level.csv": behaviour,
        "dropout_summary.csv": drop_summary,
        "channel_qc.csv": channel_qc,
        "spectral_windows.csv": windows,
        "spectral_window_qc.csv": window_qc,
        "spectral_summary.csv": spectral_summary,
        "erp_epoch_qc.csv": erp_qc,
        "erp_epoch_counts.csv": erp_counts,
    }
    for filename, frame in tables.items():
        frame.to_csv(out/filename, index=False)
    if figures:
        make_figures(out, behaviour, drop_summary, channel_qc, raw, windows,
                     psd_rows, epochs, epoch_data, erp_qc, cfg)

    summary = {
        "subject": subject, "formal_trials": int(len(trials)),
        "accuracy_by_level": {str(int(r.level)): round(float(r.accuracy), 3)
                              for r in behaviour.itertuples()},
        "bad_channels_interpolated": bads,
        "dropout_events": int(len(dropped)),
        "dropped_samples_total": int(dropped["dropped_samples"].sum()),
        "dropped_duration_s": round(float(dropped["duration"].sum()), 3),
        "spectral_windows_retained": {str(int(r.level)): int(r.retained_windows)
                                      for r in window_qc.itertuples()},
        "erp_epochs_retained": {str(int(r.level)): int(r.retained_epochs)
                                for r in erp_counts.itertuples()},
        "unit_correction": "XDF numeric microvolts multiplied by 1e-6 to obtain volts",
        "inference_scope": "single-subject descriptive only; fixed-order confound",
    }
    (out/"analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return {"summary": summary, "tables": tables, "output_dir": out}
