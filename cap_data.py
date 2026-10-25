"""Index of the CAP Sleep Database: one row per recording with labels, age/sex and file paths.

Usage:
    from cap_data import load_index
    df = load_index()
    rbd = df[df.label_rbd == 1]
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data" / "capslpdb"

GROUPS = {
    "rbd": "REM Sleep Behavior Disorder",
    "nfle": "Nocturnal Frontal Lobe Epilepsy",
    "plm": "Periodic Limb Movements",
    "ins": "Insomnia",
    "narco": "Narcolepsy",
    "sdb": "Sleep-Disordered Breathing",
    "brux": "Bruxism",
    "n": "Healthy Controls",
}
# Record names in gender-age.xlsx that don't match the file names.
XLSX_TYPOS = {"sbd1": "sdb1", "sbd2": "sdb2", "rdb11": "rbd11"}


def load_index(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    records = [r.removesuffix(".edf") for r in (data_dir / "RECORDS").read_text().split()]
    df = pd.DataFrame({"record": records})
    df["group"] = df.record.str.rstrip("0123456789")
    df["diagnosis"] = df.group.map(GROUPS)
    df["label_rbd"] = (df.group == "rbd").astype(int)
    df["label_3class"] = df.group.map(lambda g: "RBD" if g == "rbd" else "Healthy" if g == "n" else "Other")

    meta = pd.read_excel(data_dir / "gender-age.xlsx", header=None, names=["record", "sex", "age"])
    meta["record"] = meta.record.str.strip().str.lower().replace(XLSX_TYPOS)
    df = df.merge(meta, on="record", how="left", validate="1:1")

    df["edf"] = [data_dir / f"{r}.edf" for r in records]  # all signals
    df["hypnogram"] = [data_dir / f"{r}.txt" for r in records]  # RemLogic export: sleep stages + CAP events
    df["st"] = [data_dir / f"{r}.edf.st" for r in records]  # same annotations in WFDB format

    assert df.diagnosis.notna().all(), df[df.diagnosis.isna()].record.tolist()
    assert df.age.notna().all(), df[df.age.isna()].record.tolist()
    return df


if __name__ == "__main__":
    df = load_index()
    print(df.groupby(["label_3class", "group"]).agg(n=("record", "size"), age_mean=("age", "mean")).round(1))
    missing = [p.name for col in ("edf", "hypnogram", "st") for p in df[col] if not p.exists()]
    print(f"{len(df)} recordings, {len(missing)} files missing" + (f": {missing[:10]}..." if missing else ""))
