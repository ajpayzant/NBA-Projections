"""Pack / unpack the data the refresh workflow carries between runs (see cloud.py).

    python publish.py pack                  # -> dist/: bundle.tar + the app's files + manifest.json
    python publish.py unpack dist/bundle.tar

bundle.tar holds what the next refresh needs: the parsed history tables (data/lake), the current
season's raw ESPN games (so only new games are fetched) and the Results snapshots and graded
results (data/tracking). The other files are what the deployed app downloads. Raw historical
odds (data/odds_hist) are never packed: the release is public.
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import tarfile

import config as C

DIST = C.ROOT / "dist"
TRACKING = C.DATA / "tracking"
STATE = C.DATA / "live" / "state.pkl"


def bundle_paths() -> list:
    return [C.LAKE, C.RAW / str(C.CURRENT_SEASON), TRACKING]


def pack() -> None:
    DIST.mkdir(exist_ok=True)
    with tarfile.open(DIST / "bundle.tar", "w") as tar:
        for p in bundle_paths():
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(C.ROOT)))
    shutil.copy(STATE, DIST / "state.pkl")
    for name in ("results_live.parquet", "results_backtest.parquet"):
        if (TRACKING / name).exists():
            shutil.copy(TRACKING / name, DIST / name)
    with open(STATE, "rb") as fh:
        built = pickle.load(fh)["built_at"]
    (DIST / "manifest.json").write_text(json.dumps({"built_at": built}))
    for f in sorted(DIST.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


def unpack(path: str) -> None:
    with tarfile.open(path) as tar:
        tar.extractall(C.ROOT, filter="data")
    print(f"unpacked {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pack"
    pack() if cmd == "pack" else unpack(sys.argv[2])
