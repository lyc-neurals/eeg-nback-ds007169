"""Dataset layout, XDF metadata and optional BrainVision scale audit."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf


@dataclass(frozen=True)
class SubjectFiles:
    xdf: Path
    events: Path
    channels: Path
    brainvision_header: Path | None
    brainvision_binary: Path | None


def subject_files(data_root: Path, subject: str) -> SubjectFiles:
    """Accept the original BIDS tree or a flattened legacy sub_001 directory."""
    if not subject.startswith("sub-") or not subject[4:].isdigit():
        raise ValueError("Subject must have the form sub-001")
    num = subject[4:]
    root = Path(data_root).expanduser().resolve()
    bids_eeg = root / subject / "eeg"
    legacy = root / f"sub_{num}"
    eeg_dir = bids_eeg if bids_eeg.exists() else legacy
    xdf = root / "sourcedata" / "xdf" / f"{num}_nback.xdf"
    if not xdf.exists():
        xdf = eeg_dir / f"{num}_nback.xdf"
    stem = f"{subject}_task-nback_"
    header = eeg_dir / (stem + "eeg.vhdr")
    binary = eeg_dir / (stem + "eeg.eeg")
    files = SubjectFiles(
        xdf, eeg_dir / (stem + "events.tsv"),
        eeg_dir / (stem + "channels.tsv"),
        header if header.exists() else None, binary if binary.exists() else None,
    )
    missing = [str(p) for p in (files.xdf, files.events, files.channels) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required dataset files:\n" + "\n".join(missing))
    return files


def load_recording(files: SubjectFiles):
    streams, _ = pyxdf.load_xdf(str(files.xdf), dejitter_timestamps=True, verbose=False)
    matches = [s for s in streams if s["info"]["name"][0] == "EEGStream"]
    if len(matches) != 1:
        raise ValueError(f"Expected one EEGStream; found {len(matches)}")
    stream = matches[0]
    data = np.asarray(stream["time_series"], dtype=float)
    times = np.asarray(stream["time_stamps"], dtype=float)
    metadata = stream["info"]["desc"][0]["channels"][0]["channel"]
    labels = [c["label"][0] for c in metadata]
    units = [c.get("unit", [""])[0] for c in metadata]
    channels = pd.read_csv(files.channels, sep="\t")
    names = channels.loc[channels["type"].eq("EEG"), "name"].tolist()
    if not names or any(name not in labels for name in names):
        raise ValueError("EEG channel labels in channels.tsv do not match XDF metadata")
    indices = [labels.index(name) for name in names]
    if set(units[i].lower().replace("µ", "u").replace("μ", "u") for i in indices) != {"microvolts"}:
        raise ValueError("XDF EEG units differ from microvolts; inspect before scaling")
    if data.shape[1] != len(labels) or len(times) != len(data):
        raise ValueError("Inconsistent XDF EEG dimensions")
    dt = np.diff(times)
    if len(dt) == 0 or np.any(dt <= 0):
        raise ValueError("Missing or nonmonotonic XDF timestamps")
    sfreq = float(stream["info"]["nominal_srate"][0])
    if sfreq <= 0:
        raise ValueError("Invalid nominal EEG sampling rate")
    summary = pd.DataFrame([{
        "name": s["info"]["name"][0], "type": s["info"]["type"][0],
        "nominal_srate_Hz": float(s["info"]["nominal_srate"][0]),
        "n_channels": int(s["info"]["channel_count"][0]),
        "n_samples": len(s["time_stamps"]),
    } for s in streams])
    timing = {"samples": int(len(times)), "duration_s": float(times[-1]-times[0]),
              "median_dt_ms": float(1e3*np.median(dt)), "max_dt_ms": float(1e3*np.max(dt)),
              "effective_sfreq_Hz": float(1/np.median(dt))}
    return data, names, indices, labels, sfreq, summary, timing


def unit_validation(files: SubjectFiles, xdf: np.ndarray, indices: list[int],
                    labels: list[str], n_probe: int = 5000) -> pd.DataFrame:
    """Diagnostic comparison only; never use the BrainVision signal for analysis."""
    rows = [{"check": "XDF EEG metadata unit", "value": "microvolts"},
            {"check": "Scale used here (XDF numeric to volts)", "value": "1e-6"}]
    if files.brainvision_binary is None or files.brainvision_header is None:
        rows.append({"check": "BrainVision comparison", "value": "unavailable (optional files absent)"})
        return pd.DataFrame(rows)
    header = files.brainvision_header.read_text(encoding="utf-8-sig", errors="replace")
    # Do not silently compare columns if the exported layout differs from XDF.
    ch_lines = [line for line in header.splitlines() if line.startswith("Ch") and "=" in line]
    header_names = [line.split("=", 1)[1].split(",", 1)[0].strip() for line in ch_lines]
    if [name.casefold() for name in header_names] != [name.casefold() for name in labels]:
        rows.append({"check": "BrainVision comparison", "value": "skipped (channel order differs)"})
        return pd.DataFrame(rows)
    count = min(n_probe, len(xdf))
    binary = np.fromfile(files.brainvision_binary, dtype="<f4", count=count*len(labels))
    if len(binary) != count*len(labels):
        rows.append({"check": "BrainVision comparison", "value": "skipped (binary too short)"})
        return pd.DataFrame(rows)
    bv = binary.reshape(count, len(labels))[:, indices]
    probe = xdf[:count, indices]
    valid = np.isfinite(probe) & (np.abs(probe) > 1e-9)
    ratio = float(np.median(np.abs(bv[valid]/probe[valid]))) if valid.any() else float("nan")
    rows.extend([
        {"check": "BrainVision binary-count / XDF numeric ratio", "value": f"{ratio:.3g}"},
        {"check": "Expected ratio from V→µV and 0.1 µV resolution", "value": "1e7"},
    ])
    return pd.DataFrame(rows)
