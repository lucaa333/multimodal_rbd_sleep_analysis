"""Download the PhysioNet CAP Sleep Database (v1.0.0, ~40 GB) into data/capslpdb/.

Re-running is safe: files that already match PhysioNet's SHA-256 are skipped.
Usage: python download_data.py [target_dir]
"""
import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# PhysioNet's official AWS mirror; physionet.org itself serves ~10 kB/s.
BASE_URL = "https://physionet-open.s3.amazonaws.com/capslpdb/1.0.0/"
TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "data" / "capslpdb")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, expected, bytes_bar):
    dest = TARGET / name
    if dest.exists() and sha256(dest) == expected:
        return
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(BASE_URL + name) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            bytes_bar.update(len(chunk))
    if sha256(part) != expected:
        part.unlink()
        raise ValueError(f"checksum mismatch for {name}")
    part.replace(dest)


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE_URL + "SHA256SUMS.txt") as r:
        sums = [line.split(maxsplit=1)[::-1] for line in r.read().decode().splitlines() if line.strip()]
    print(f"{len(sums)} files -> {TARGET}")
    failed = []
    files_bar = tqdm(total=len(sums), desc="files", unit="file", position=0)
    bytes_bar = tqdm(desc="download", unit="B", unit_scale=True, unit_divisor=1024, position=1)
    # A few parallel downloads; PhysioNet is slow per connection.
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
