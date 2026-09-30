"""Download NHL schedule + play-by-play JSON for a season into data/raw/.

Usage: python3 ingest.py [season]   (default 20252026)
Idempotent: games already on disk are skipped, so it is safe to re-run.
"""
import json
import sys
import time
from pathlib import Path

import requests
from concurrent.futures import ThreadPoolExecutor

SEASON = sys.argv[1] if len(sys.argv) > 1 else "20252026"
RAW = Path(__file__).parent / "data" / "raw" / SEASON
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://api-web.nhle.com/v1"
SESSION = requests.Session()


def get(url, retries=4):
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            print(f"  {r.status_code} on {url}")
        except requests.RequestException as e:
            print(f"  error {e} on {url}")
        time.sleep(2 ** i)
    return None


def season_game_ids():
    """Union of every team's schedule -> all completed regular-season game ids."""
    teams = get(f"{BASE}/schedule/now")  # cheap call just to confirm API is up
    standings = get(f"{BASE}/standings/now")
    abbrevs = sorted({t["teamAbbrev"]["default"] for t in standings["standings"]})
    ids = {}
    for ab in abbrevs:
        sched = get(f"{BASE}/club-schedule-season/{ab}/{SEASON}")
        for g in sched["games"]:
            if g["gameType"] == 2 and g["gameState"] in ("OFF", "FINAL"):
                ids[g["id"]] = g["gameDate"]
        time.sleep(0.2)
    return sorted(ids)


def main():
    ids = season_game_ids()
    print(f"{len(ids)} completed regular-season games")
    todo = [gid for gid in ids if not (RAW / f"{gid}.json").exists()]
    print(f"{len(todo)} to download")

    def fetch(gid):
        data = get(f"{BASE}/gamecenter/{gid}/play-by-play")
        if data:
            (RAW / f"{gid}.json").write_text(json.dumps(data))
        time.sleep(0.3)

    with ThreadPoolExecutor(max_workers=1) as pool:
        for n, _ in enumerate(pool.map(fetch, todo), 1):
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print("done:", len(list(RAW.glob('*.json'))), "files")


if __name__ == "__main__":
    main()
