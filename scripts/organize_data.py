import argparse
import csv
import re
import tempfile
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "capslpdb"
GROUP_CLASSES = {
    "n": "healthy",
    "rbd": "rbd",
    "brux": "other",
    "ins": "other",
    "narco": "other",
    "nfle": "other",
    "plm": "other",
    "sdb": "other",
}
METADATA_FILES = (
    "ANNOTATORS", "RECORDS", "gender-age.xlsx", "CAP-sequence.jpg",
    "CAP.m", "ScoringReader.m", "SHA256SUMS.txt", "LICENSE.txt",
)


def organized_path(root, filename):
    match = re.fullmatch(r"(([a-z]+)[0-9]+)\.(?:edf(?:\.st)?|txt)", filename)
    if match and match[2] in GROUP_CLASSES:
        return root / GROUP_CLASSES[match[2]] / match[1] / filename
    if filename in METADATA_FILES:
        return root / "metadata" / filename
    return root / filename


def inside(root, path):
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path is outside the dataset directory: {path}")
    return path


def organize(root=DATA_DIR):
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    partials = list(root.rglob("*.part"))
    if partials:
        raise ValueError(f"Found {len(partials)} incomplete .part downloads.")
    records_file = root / "RECORDS"
    if not records_file.is_file():
        records_file = root / "metadata" / "RECORDS"
    inside(root, records_file)
    entries = records_file.read_text(encoding="utf-8").split()
    if not entries or len(entries) != len(set(entries)):
        raise ValueError("RECORDS must contain a nonempty list without duplicate entries.")

    filenames, rows = [], []
    for entry in entries:
        match = re.fullmatch(r"([a-z]+)([0-9]+)\.edf", entry)
        if not match or match[1] not in GROUP_CLASSES:
            raise ValueError(f"Unknown recording in RECORDS: {entry}")
        record_id = entry[:-4]
        group = match[1]
        class_name = GROUP_CLASSES[group]
        row = {
            "record_id": record_id, "class_name": class_name,
            "original_group": group,
        }
        for field, suffix in (("edf_path", ".edf"), ("txt_path", ".txt"), ("st_path", ".edf.st")):
            filename = record_id + suffix
            destination = organized_path(root, filename)
            row[field] = destination.relative_to(root).as_posix()
            filenames.append(filename)
        rows.append(row)

    unexpected = [p.name for p in root.iterdir() if p.is_file()
                  and p.suffix in {".edf", ".txt", ".st"}
                  and p.name not in filenames
                  and p.name not in METADATA_FILES]
    if unexpected:
        raise ValueError(f"Recording files not listed in RECORDS: {unexpected}")
    filenames.extend(name for name in METADATA_FILES
                     if (root / name).exists() or organized_path(root, name).exists())
    moves = []
    for filename in filenames:
        source = inside(root, root / filename)
        destination = inside(root, organized_path(root, filename))
        if source.exists() and destination.exists():
            raise FileExistsError(f"Both copies exist: {source}, {destination}")
        if source.is_symlink() or destination.is_symlink():
            raise ValueError(f"Symbolic link found for: {filename}")
        if source.is_file():
            for parent in destination.parents:
                if parent == root:
                    break
                if parent.exists() and not parent.is_dir():
                    raise NotADirectoryError(parent)
            moves.append((source, destination))
        elif not destination.is_file():
            raise FileNotFoundError(f"Missing file: {filename}")
    mapping = inside(root, root / "recordings.csv")
    if mapping.exists() and not mapping.is_file():
        raise IsADirectoryError(mapping)
    print(f"{len(rows)} recordings | {len(moves)} files to move | {root}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=root,
                                         prefix="recordings_", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            source.rename(destination)
        temporary.replace(mapping)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Mapping: {mapping}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Organize CAP recordings and write recordings.csv.")
    parser.add_argument("data_dir", nargs="?", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    try:
        organize(args.data_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
