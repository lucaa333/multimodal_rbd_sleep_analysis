"""Download the PhysioNet CAP Sleep Database (v1.0.0, ~40 GB) into data/capslpdb/.

Re-running is safe as files that already match PhysioNet's SHA-256 are skipped.
"""
import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

if __package__:
    from .organize_data import organized_path
else:
    from organize_data import organized_path

BASE_URL = "https://physionet-open.s3.amazonaws.com/capslpdb/1.0.0/"
TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "capslpdb"
TIMEOUT = 120


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, expected, bytes_bar):
    dest = organized_path(TARGET, name) if (TARGET / "recordings.csv").is_file() else TARGET / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256(dest) == expected:
        return
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(BASE_URL + name, timeout=TIMEOUT) as r, open(part, "wb") as f:
        while chunk := r.read1(1 << 20):
            f.write(chunk)
            h.update(chunk)
            bytes_bar.update(len(chunk) / 1_000_000)
    if h.hexdigest() != expected:
        part.unlink()
        raise ValueError(f"checksum mismatch for {name}")
    part.replace(dest)


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {TARGET.resolve()}", flush=True)
    with urllib.request.urlopen(BASE_URL + "SHA256SUMS.txt", timeout=TIMEOUT) as r:
        sums = [line.split(maxsplit=1)[::-1] for line in r.read().decode().splitlines() if line.strip()]
    failed = []
    files_bar = tqdm(total=len(sums), desc="Files", unit="file", position=0, dynamic_ncols=True)
    bytes_bar = tqdm(
        unit="MB", position=1, dynamic_ncols=True, mininterval=0.5,
        bar_format="{n:.1f} MB downloaded | {rate_fmt} | Elapsed {elapsed}",
    )
    with ThreadPoolExecutor(4) as pool:
        futures = {pool.submit(fetch, name, h, bytes_bar): name for name, h in sums}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                failed.append(futures[fut])
                tqdm.write(f"FAILED {futures[fut]}: {e}")
            files_bar.update()
    files_bar.close()
    bytes_bar.close()
    if failed:
        sys.exit(f"{len(failed)} files failed, re-run to retry: {failed}")


if __name__ == "__main__":
    main()
