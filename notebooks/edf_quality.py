"""Shared EDF reading, channel selection and technical window check."""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

INT16_MIN, INT16_MAX = -32768, 32767
PLATEAU_RATIO = 10
PLATEAU_MIN_SAMPLES = 10
PLATEAU_EDGE_FRACTION = 0.01


def saturation_limits(digital, ratio=PLATEAU_RATIO, min_samples=PLATEAU_MIN_SAMPLES,
                      edge_fraction=PLATEAU_EDGE_FRACTION):
    """Find the digital values at which a channel saturates.

    EDF headers often state a display range instead of the converter range, so the
    limits come from the data. A saturated channel piles up samples at or just inside
    its observed extremes. A value near an extreme counts as a plateau when it occurs
    at least `ratio` times as often as the typical value just further inward and at
    least `min_samples` times. Without a plateau, the 16-bit limits apply.
    """
    counts = np.bincount(np.asarray(digital, dtype=np.int64).ravel() - INT16_MIN,
                         minlength=INT16_MAX - INT16_MIN + 1)
    present = np.flatnonzero(counts)
    if not len(present):
        raise ValueError("The channel contains no samples.")
    first, last = int(present[0]), int(present[-1])
    edge = max(3, round(edge_fraction * (last - first)))

    def plateau(extreme, inward):
        window = extreme + inward * np.arange(edge + 1)
        window = window[(window >= first) & (window <= last)]
        reference_bins = extreme + inward * np.arange(edge + 1, 2 * edge + 1)
        reference_bins = reference_bins[(reference_bins >= first) & (reference_bins <= last)]
        reference = float(np.median(counts[reference_bins])) if len(reference_bins) else 0.0
        threshold = max(min_samples, ratio * max(reference, 1.0))
        spikes = np.flatnonzero(counts[window] >= threshold)
        if not len(spikes):
            return None
        # Start at the plateau closest to the extreme and extend inward over adjacent plateau values.
        position = int(spikes[0])
        while position + 1 < len(window) and counts[window[position + 1]] >= threshold:
            position += 1
        return int(window[position]) + INT16_MIN

    low = plateau(first, 1)
    high = plateau(last, -1)
    return (INT16_MIN if low is None else low), (INT16_MAX if high is None else high)


class NativeEDF:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            fixed = stream.read(256)
            if len(fixed) != 256 or fixed[:8].strip() != b"0":
                raise ValueError("Only standard 16-bit EDF files are supported.")
            if b"EDF+D" in fixed[192:236]:
                raise ValueError("Discontinuous EDF requires explicit time-gap handling.")
            self.header_bytes = int(fixed[184:192])
            self.record_count = int(fixed[236:244])
            self.record_seconds = float(fixed[244:252])
            self.channel_count = int(fixed[252:256])
            header = stream.read(self.channel_count * 256)
        if len(header) != self.channel_count * 256 or self.record_seconds <= 0:
            raise ValueError("Invalid EDF signal headers or record duration.")
        if self.header_bytes < 256 + self.channel_count * 256:
            raise ValueError("Invalid EDF header length.")
        def fields(offset, width):
            return [header[self.channel_count * offset + i * width:self.channel_count * offset + (i + 1) * width].decode("latin-1").strip() for i in range(self.channel_count)]
        self.labels = fields(0, 16)
        self.units = fields(96, 8)
        self.physical_min = np.array(fields(104, 8), dtype=float)
        self.physical_max = np.array(fields(112, 8), dtype=float)
        self.digital_min = np.array(fields(120, 8), dtype=int)
        self.digital_max = np.array(fields(128, 8), dtype=int)
        self.prefilters = fields(136, 80)
        self.samples_per_record = np.array(fields(216, 8), dtype=int)
        if np.any(self.samples_per_record <= 0):
            raise ValueError("EDF contains a channel without samples.")
        self.record_width = int(self.samples_per_record.sum())
        self.offsets = np.r_[0, np.cumsum(self.samples_per_record[:-1])]
        payload = self.path.stat().st_size - self.header_bytes
        bytes_per_record = self.record_width * 2
        if self.record_count == -1 and payload % bytes_per_record == 0:
            self.record_count = payload // bytes_per_record
        if self.record_count <= 0 or payload != self.record_count * bytes_per_record:
            raise ValueError("File size does not match the number of data blocks in the EDF header.")
        self.duration = self.record_count * self.record_seconds
        self._data = None
        self._digital_limits = {}

    def open(self):
        if self._data is None:
            self._data = np.memmap(self.path, dtype="<i2", mode="r", offset=self.header_bytes, shape=(self.record_count, self.record_width))
        return self

    def close(self):
        if self._data is not None:
            self._data._mmap.close()
            self._data = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def sfreq(self, index):
        return float(self.samples_per_record[index] / self.record_seconds)

    def digital_limits(self, index):
        """Return the saturation limits found in the data and whether the samples stay inside the header range."""
        if index not in self._digital_limits:
            n = int(self.samples_per_record[index])
            offset = int(self.offsets[index])
            channel = self._data[:, offset:offset + n]
            header_low, header_high = self.digital_min[index], self.digital_max[index]
            header_ok = header_low < header_high and channel.min() >= header_low and channel.max() <= header_high
            low, high = saturation_limits(channel)
            if header_ok:
                # A header range that contains all samples is a real limit, so samples on it also count as clipped.
                low, high = max(low, int(header_low)), min(high, int(header_high))
            self._digital_limits[index] = (low, high, bool(header_ok))
        return self._digital_limits[index]

    def read_digital(self, index, start, stop):
        if self._data is None:
            raise RuntimeError("Open the EDF reader first.")
        per_record = int(self.samples_per_record[index])
        if start < 0 or stop <= start or stop > self.record_count * per_record:
            raise ValueError("Requested samples are outside the recording.")
        first, last = start // per_record, (stop - 1) // per_record + 1
        offset = int(self.offsets[index])
        digital = np.array(self._data[first:last, offset:offset + per_record], dtype=np.int32).reshape(-1)
        return digital[start - first * per_record:stop - first * per_record]

    def read(self, index, start, stop):
        digital = self.read_digital(index, start, stop)
        span = self.digital_max[index] - self.digital_min[index]
        if span == 0 or self.physical_max[index] == self.physical_min[index]:
            raise ValueError("Invalid channel calibration.")
        unit = self.units[index].lower().replace("µ", "u").replace("μ", "u")
        factor = {"uv": 1.0, "mv": 1000.0, "v": 1e6}.get(unit)
        if factor is None:
            raise ValueError(f"Unsupported voltage unit for {self.labels[index]}: {self.units[index]}")
        values = (digital.astype(float) - self.digital_min[index]) * (self.physical_max[index] - self.physical_min[index]) / span + self.physical_min[index]
        low, high, _ = self.digital_limits(index)
        clipped = (digital <= low) | (digital >= high)
        return values * factor, clipped

def channel_key(label):
    return "".join(character for character in label.upper() if character.isalnum())

def resolve_channel(edf, record_id, modality, channel_overrides, channel_candidates, epoch_seconds, eeg_bands, *, require_units=True):
    override = channel_overrides.get(record_id, {}).get(modality)
    if isinstance(override, str):
        raise ValueError("Channel overrides must be tuples of labels or EDF header indices.")
    candidates = [tuple(override)] if override is not None else channel_candidates[modality]
    candidate_errors = []
    for candidate in candidates:
        indices = []
        for label in candidate:
            if isinstance(label, int):
                matches = [label] if 0 <= label < edf.channel_count else []
            else:
                matches = [i for i, actual in enumerate(edf.labels) if channel_key(actual) == channel_key(label)]
            if len(matches) != 1:
                break
            indices.append(matches[0])
        if len(indices) != len(candidate):
            continue
        if len(indices) not in {1, 2} or len(set(indices)) != len(indices):
            raise ValueError("A montage must contain one channel or two distinct channels.")
        rates = [edf.sfreq(i) for i in indices]
        if not np.allclose(rates, rates[0]):
            candidate_errors.append("Bipolar derivation requires equal native sample rates.")
            continue
        if not np.isclose(epoch_seconds * rates[0], round(epoch_seconds * rates[0])):
            candidate_errors.append("The epoch duration must contain an integer number of native samples.")
            continue
        missing_units = [edf.labels[i] for i in indices if not edf.units[i].strip()]
        if missing_units and require_units:
            candidate_errors.append(f"Missing unit for {', '.join(missing_units)}")
            continue
        unsupported = [f"{edf.labels[i]} ({edf.units[i]})" for i in indices
                       if edf.units[i].lower().replace("µ", "u").replace("μ", "u") not in {"uv", "mv", "v"}]
        if unsupported and require_units:
            candidate_errors.append(f"Unsupported voltage unit for {', '.join(unsupported)}")
            continue
        if modality == "eeg" and rates[0] / 2 < max(high for low, high in eeg_bands.values()):
            candidate_errors.append("The EEG sampling rate cannot represent the configured bands.")
            continue
        actual_labels = [edf.labels[i] for i in indices]
        primary = tuple(channel_key(s) for s in channel_candidates[modality][0])
        alternate = tuple(channel_key(s) for s in actual_labels) != primary
        return {"indices": indices, "labels": actual_labels, "sfreq": rates[0], "alternate": alternate}
    if candidate_errors:
        raise ValueError(candidate_errors[0])
    return None


def selection_review(modality, selection):
    """Explain why a non-primary selection needs a montage review."""
    if not selection["alternate"]:
        return ""
    labels = selection["labels"]
    keys = [channel_key(label) for label in labels]
    if modality == "eog" and len(keys) == 2:
        refs = [key[-2:] for key in keys]
        if set(refs) == {"A1", "A2"}:
            return "The A1 and A2 references do not cancel when the signals are subtracted."
    if modality == "emg" and len(keys) == 2 and keys == ["CHIN", "CHIN"]:
        return "The two CHIN channels have identical labels. Their recording references need review."
    if len(labels) == 2:
        return "Check the original references before subtracting these channels."
    if modality == "eeg":
        return "The EEG location differs from the primary C4-A1 selection."
    if modality == "emg":
        return "Confirm the muscle and montage represented by this EMG channel."
    return "Confirm that this EOG channel is comparable with the primary selection."


def read_montage(edf, selection, start=0, stop=None):
    combined, clipped_combined = None, None
    for index in selection["indices"]:
        end = edf.record_count * int(edf.samples_per_record[index]) if stop is None else stop
        values, clipped = edf.read(index, start, end)
        combined = values if combined is None else combined - values
        clipped_combined = clipped if clipped_combined is None else clipped_combined | clipped
    return combined, clipped_combined


def uncalibrated(edf, index):
    """Return True when the header maps digital values one-to-one to physical values."""
    return (edf.physical_min[index] == edf.digital_min[index]
            and edf.physical_max[index] == edf.digital_max[index])


def assess_window(values, clipped, modality, flat_step_reject_fraction, clipping_reject_fraction, amplitude_flag_uv):
    reject, flags = [], []
    if not np.isfinite(values).all():
        return {"reject": [f"{modality}_nonfinite"], "flags": [], "ptp_uv": np.nan, "clip_fraction": np.nan}
    peak_to_peak = float(np.ptp(values))
    flat_fraction = float(np.mean(np.diff(values) == 0))
    clip_fraction = float(np.mean(clipped))
    if peak_to_peak == 0 or flat_fraction >= flat_step_reject_fraction:
        reject.append(f"{modality}_flat")
    if clip_fraction > 0:
        (reject if clip_fraction >= clipping_reject_fraction else flags).append(f"{modality}_clipped")
    if peak_to_peak > amplitude_flag_uv[modality]:
        flags.append(f"{modality}_high_amplitude")
    return {"reject": reject, "flags": flags, "ptp_uv": peak_to_peak, "clip_fraction": clip_fraction}

def inspect_recordings(mapping, recording_path, modalities, select_channel):
    headers = {}
    plans = {}
    inventory_rows = []
    for record in tqdm(mapping.to_dict("records"), desc="Inspect EDF headers", unit="record"):
        record_id = record["record_id"]
        row = {key: record[key] for key in ["record_id", "class_name", "original_group", "edf_path"]}
        try:
            edf = NativeEDF(recording_path(record["edf_path"]))
        except (OSError, ValueError, OverflowError) as exc:
            row.update({"status": "invalid_recording", "reason": str(exc)})
            inventory_rows.append(row)
            continue

        headers[record_id] = edf
        row["duration_seconds"] = edf.duration
        plan, problems = {}, []
        for modality in modalities:
            try:
                selection = select_channel(edf, record_id, modality)
            except ValueError as exc:
                selection = None
                problems.append(f"{modality.upper()}: {exc}")
            else:
                if selection is None:
                    problems.append(f"{modality.upper()}: No channel selected")
            plan[modality] = selection
            row[f"{modality}_channels"] = " | ".join(selection["labels"]) if selection else ""
            row[f"{modality}_sfreq_hz"] = selection["sfreq"] if selection else np.nan
            row[f"{modality}_prefilter"] = " | ".join(edf.prefilters[i] for i in selection["indices"]) if selection else ""
        row["status"] = "review_channels" if problems else "ready"
        row["reason"] = "; ".join(problems)
        row["alternate_montage"] = any(s and s["alternate"] for s in plan.values())
        if not problems:
            plans[record_id] = plan
        inventory_rows.append(row)

    return headers, plans, pd.DataFrame(inventory_rows)
