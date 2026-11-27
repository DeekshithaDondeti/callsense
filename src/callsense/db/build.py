"""Build the database from the processed dataset with one command. Free: no API calls.

Starts Postgres (docker compose), applies the schema, loads and embeds the calls, then runs the
data-quality checks. Safe to re-run.

Usage:
    uv run python -m callsense.db.build               # build or update
    uv run python -m callsense.db.build --reset       # drop the tables first and rebuild from scratch
    uv run python -m callsense.db.build --no-docker   # use an already running database
"""

import argparse
import subprocess
import sys
import time

import psycopg

from callsense.db.connect import connect, database_url
from callsense.db.load import load
from callsense.db.quality import report, run_checks
from callsense.synth.store import ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="drop the calls tables first")
    parser.add_argument("--no-docker", action="store_true", help="don't run docker compose")
    args = parser.parse_args(argv)

    if not args.no_docker:
        subprocess.run(["docker", "compose", "up", "-d", "--wait"], cwd=ROOT, check=True)
    if args.reset:
        with psycopg.connect(database_url(), autocommit=True) as conn:
            conn.execute("DROP TABLE IF EXISTS call_analysis, calls")
        print("Dropped the existing tables.")

    start = time.perf_counter()
    with connect() as conn:
        result = load(conn)
        print(f"Loaded {result.loaded} calls ({result.new} new, {result.embedded} embedded) "
              f"in {time.perf_counter() - start:.0f}s.")
        if result.waiting:
            print(f"{result.waiting} calls are not redacted and analysed yet. "
                  "Run: uv run python -m callsense.synth.process (paid, asks first)")
        print("Data-quality checks:")
        return 0 if report(run_checks(conn)) else 1


if __name__ == "__main__":
    sys.exit(main())
