"""Results cached per recording and recomputed when their settings or code change."""

import hashlib
import inspect
import json
import pickle
from pathlib import Path


def code_version(*objects):
    source = "".join(inspect.getsource(item) for item in objects)
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def cached(path, settings, compute):
    """Return the stored value and True, or compute and store it and return False."""
    path = Path(path)
    key = json.dumps(settings, sort_keys=True, default=str)
    if path.is_file():
        with path.open("rb") as stream:
            stored = pickle.load(stream)
        if stored["key"] == key:
            return stored["value"], True
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump({"key": key, "value": value}, stream)
    temporary.replace(path)
    return value, False
