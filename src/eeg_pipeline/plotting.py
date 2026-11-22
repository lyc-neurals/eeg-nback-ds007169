"""Small descriptive figure set; every figure is labeled by its inference scope."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _save(fig, path: Path):
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(out: Path, behaviour, dropout_summary, channel_qc, raw, windows,
                 psd_rows, epochs, epoch_data, erp_qc, cfg):
    sns.set_theme(style="whitegrid", context="notebook")
    levels = sorted(behaviour["level"].astype(int).unique())
    colors = dict(zip(levels, sns.color_palette("viridis", len(levels))))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    melted = behaviour.melt(id_vars="level", value_vars=["accuracy", "balanced_accuracy"],
                            var_name="metric", value_name="value")
    melted["metric"] = melted["metric"].map({"accuracy": "Raw accuracy",
                                               "balanced_accuracy": "Balanced accuracy"})
    sns.barplot(data=melted, x="level", y="value", hue="metric",
                palette=["#4C78A8", "#72B7B2"], ax=axes[0])
    axes[0].set(ylim=(0, 1), xlabel="n-back level", ylabel="Score",
                title="Performance under target imbalance")
    axes[0].legend(frameon=False, fontsize=8)
    sns.barplot(data=behaviour, x="level", y="d_prime", color="#F58518", ax=axes[1])
    axes[1].axhline(0, ls="--", lw=1, color="0.35")
    axes[1].set(xlabel="n-back level", ylabel="d′", title="Signal-detection sensitivity")
    _save(fig, out/"01_behaviour.png")

    plot_drop = dropout_summary[dropout_summary["block"].isin([str(x) for x in levels])]
    if not plot_drop.empty:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.barplot(data=plot_drop, x="block", y="dropped_samples", color="#E45756", ax=ax)
        ax.set(xlabel="Formal n-back block", ylabel="Dropped samples",
               title="Acquisition dropouts by block")
        _save(fig, out/"02_dropouts.png")

    fig, ax = plt.subplots(figsize=(7, 5.2))
    sns.barplot(data=channel_qc, y="channel", x="p01_p99_range_uV", hue="flagged",
                palette={False: "#4C78A8", True: "#E45756"}, dodge=False, ax=ax)
    ax.set(xlabel="Filtered 1st–99th percentile range (µV)", ylabel="",
           title="Channel amplitude screen")
    ax.legend(title="Flagged", frameon=False)
    _save(fig, out/"03_channel_qc.png")

    picks = [ch for ch in cfg["plotting"]["trace_channels"] if ch in raw.ch_names]
    trace_s = float(cfg["plotting"]["trace_start_s"])
    duration = float(cfg["plotting"]["trace_duration_s"])
    if picks and raw.times[-1] >= duration:
        trace_s = min(trace_s, float(raw.times[-1]-duration))
        data = raw.get_data(picks=picks, start=int(trace_s*raw.info["sfreq"]),
                            stop=int((trace_s+duration)*raw.info["sfreq"]))*1e6
        t = np.arange(data.shape[1])/raw.info["sfreq"]
        offset = float(cfg["plotting"]["trace_offset_uv"])
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for i, y in enumerate(data):
            ax.plot(t, y+i*offset, lw=.7)
        ax.set_yticks(np.arange(len(picks))*offset, picks)
        ax.set(xlabel=f"Time from {trace_s:.0f} s (s)", ylabel=f"Channel + {offset:g} µV offset",
               title="Representative preprocessed trace")
        _save(fig, out/"04_preprocessed_trace.png")

    kept = windows.loc[windows["keep"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, col, color, title in [
        (axes[0], "frontal_midline_theta_rel", "#72B7B2", "Frontal-midline theta (4–7 Hz)"),
        (axes[1], "posterior_alpha_rel", "#F2CF5B", "Posterior alpha (8–13 Hz)")]:
        sns.boxplot(data=kept, x="level", y=col, color=color, showfliers=False, ax=ax)
        sns.stripplot(data=kept, x="level", y=col, color="0.25", size=2.2, alpha=.5, ax=ax)
        ax.set(xlabel="n-back level (fixed block order)", ylabel="Relative power", title=title)
        ax.text(.01, .98, "Descriptive only", transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color="#B22222")
    _save(fig, out/"05_roi_bandpower.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for level in levels:
        curves = np.array([r[4] for r in psd_rows if r[0] == level and r[2]])
        if not len(curves):
            continue
        freq = next(r[3] for r in psd_rows if r[0] == level and r[2])
        median = np.median(curves, axis=0)
        q25, q75 = np.quantile(curves, [.25, .75], axis=0)
        mask = (freq >= 1) & (freq <= 40)
        ax.plot(freq[mask], median[mask], color=colors[level], lw=1.7, label=f"{level}-back")
        ax.fill_between(freq[mask], q25[mask], q75[mask], color=colors[level], alpha=.13, linewidth=0)
    ax.set(xlabel="Frequency (Hz)", ylabel="Global median PSD (dB µV²/Hz)",
           title="Welch spectra of retained windows")
    ax.legend(frameon=False, ncol=2)
    _save(fig, out/"06_psd_by_level.png")

    roi = [ch for ch in cfg["erp"]["plot_channels"] if ch in raw.ch_names]
    if roi:
        fig, axes = plt.subplots(1, len(roi), figsize=(4.3*len(roi), 3.7), squeeze=False, sharex=True)
        for ax, ch in zip(axes[0], roi):
            idx = raw.ch_names.index(ch)
            for level in levels:
                sel = erp_qc["keep"].to_numpy(bool) & (erp_qc["level"].to_numpy(int) == level)
                count = int(sel.sum())
                if not count:
                    continue
                mean = epoch_data[sel, idx, :].mean(axis=0)*1e6
                sem = epoch_data[sel, idx, :].std(axis=0, ddof=1)/np.sqrt(count)*1e6 if count > 1 else np.zeros_like(mean)
                ax.plot(epochs.times, mean, color=colors[level], lw=1.4, label=f"{level}-back")
                ax.fill_between(epochs.times, mean-sem, mean+sem, color=colors[level], alpha=.12, linewidth=0)
            ax.axvline(0, color="0.25", lw=1, ls="--")
            ax.axhline(0, color="0.5", lw=.7)
            ax.set(title=ch, xlabel="Time from inferred letter onset (s)")
        axes[0][0].set_ylabel("Amplitude (µV)")
        axes[0][-1].legend(frameon=False, fontsize=8)
        fig.suptitle("Stimulus-locked ERPs (mean ± SEM; descriptive)", y=1.02)
        _save(fig, out/"07_stimulus_locked_erp.png")
