"""XDF amplitude scaling, filtering, referencing and dropout annotations."""

import mne
import numpy as np


def filter_xdf_eeg(xdf: np.ndarray, indices: list[int], names: list[str],
                   sfreq: float, cfg: dict):
    eeg_uv = xdf[:, indices].T
    eeg_v = (eeg_uv-np.median(eeg_uv, axis=1, keepdims=True))*1e-6
    info = mne.create_info(names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg_v, info, verbose=False)
    raw.set_montage(cfg["montage"], match_case=False, on_missing="raise")
    raw.info["line_freq"] = float(cfg["line_freq_hz"])
    raw.filter(float(cfg["highpass_hz"]), float(cfg["lowpass_hz"]),
               fir_design="firwin", verbose=False)
    return raw


def reference_interpolate_annotate(raw_unreferenced, bad_channels: list[str], dropped):
    raw = raw_unreferenced.copy()
    raw.info["bads"] = bad_channels
    raw.set_eeg_reference("average", projection=False, verbose=False)
    if bad_channels:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    onset = dropped["onset"].to_numpy(float)
    duration = np.minimum(dropped["duration"].to_numpy(float),
                          np.maximum(0, raw.times[-1]-onset))
    raw.set_annotations(mne.Annotations(onset=onset, duration=duration,
                                         description=["BAD_dropped_samples"]*len(dropped)))
    return raw
