"""
Bootstrap / validate the NBA DuckDB warehouse.
Called by the Streamlit app on startup if the DB is missing or empty.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _ROOT / "data" / "analytics_database" / "nba_warehouse.duckdb"


def db_is_valid() -> bool:
    if not DB_PATH.exists() or DB_PATH.stat().st_size < 4096:
        return False
    try:
        import duckdb
        con = duckdb.connect(str(DB_PATH), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM clean.player_game_stats").fetchone()[0]
        con.close()
        return n > 0
    except Exception:
        return False


def ensure_db() -> None:
    if db_is_valid():
        return
    build_script = _ROOT / "scripts" / "build_warehouse.py"
    if not build_script.exists():
        raise FileNotFoundError("build_warehouse.py not found")
    result = subprocess.run(
        [sys.executable, str(build_script)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Warehouse build failed:\n{result.stderr[-2000:]}")


if __name__ == "__main__":
    ensure_db()
    print("Database OK")
