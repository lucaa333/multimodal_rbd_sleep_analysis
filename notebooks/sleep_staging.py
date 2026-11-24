"""Sleep staging with YASA and GSSC."""

import contextlib
import io
import warnings
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.signal import resample_poly

STAGES = ("W", "N1", "N2", "N3", "R")
GSSC_SAMPLES = 2560


def resample(values, sfreq, target):
    rate = Fraction(sfreq).limit_denominator(1000)
    if not np.isclose(float(rate), sfreq):
        raise ValueError(f"The sampling rate {sfreq} Hz cannot be resampled exactly.")
    ratio = Fraction(target) / rate
    values = np.asarray(values, dtype=float)
    return values if ratio == 1 else resample_poly(values, ratio.numerator, ratio.denominator)


def _raw(eeg, eog, sfreq):
    import mne

    names, types, signals = ["EEG1"], ["eeg"], [eeg]
    if eog is not None:
        names, types, signals = names + ["EOG1"], types + ["eog"], signals + [eog]
    length = min(len(signal) for signal in signals)
    data = np.vstack([signal[:length] for signal in signals]) * 1e-6
    return mne.io.RawArray(data, mne.create_info(names, sfreq, ch_types=types), verbose=False)


def yasa_probabilities(eeg, eog, sfreq, metadata=None):
    """Return the YASA probabilities (epochs x STAGES) for signals in microvolts."""
    import yasa
    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        staging = yasa.SleepStaging(_raw(eeg, eog, sfreq), eeg_name="EEG1",
                                    eog_name="EOG1" if eog is not None else None, metadata=metadata)
        proba = staging.predict().proba
    return proba[[{"W": "WAKE", "R": "REM"}.get(stage, stage) for stage in STAGES]].to_numpy(dtype=float)


@lru_cache(maxsize=1)
def _gssc_networks():
    import torch
    from importlib.resources import files

    networks = [torch.load(files("gssc.nets") / name, weights_only=False, map_location="cpu")
                for name in ("sig_net_v1.pt", "gru_net_v1.pt")]
    return [network.eval() for network in networks]


def gssc_probabilities(eeg, eog, sfreq):
    """Return the GSSC probabilities (epochs x STAGES), following gssc.infer.EEGInfer.mne_infer."""
    import mne
    import torch
    from gssc.utils import epo_arr_zscore, prepare_inst

    signal_network, context_network = _gssc_networks()
    raw = _raw(eeg, eog, sfreq)
    raw.filter(0.3, 30.0, picks=raw.ch_names, verbose=False)
    with mne.utils.use_log_level("error"), contextlib.redirect_stdout(io.StringIO()):
        epochs, _ = prepare_inst(raw, GSSC_SAMPLES, "back")
        data = epo_arr_zscore(epochs.get_data(picks=raw.ch_names) * 1e6)
    inputs = {kind: torch.tensor(data[:, index, :GSSC_SAMPLES], dtype=torch.float32).reshape(-1, 1, GSSC_SAMPLES)
              for index, kind in enumerate(raw.get_channel_types())}
    gru = context_network.gru
    hidden = torch.zeros(gru.num_layers * (1 + gru.bidirectional), 1, gru.hidden_size)
    with torch.no_grad():
        log_probabilities, _ = context_network(signal_network(inputs, rep_output="rep_only").swapaxes(-1, 1), hidden)
    probabilities = torch.exp(log_probabilities[:, 0]).numpy().astype(float)
    return probabilities / probabilities.sum(axis=1, keepdims=True)
