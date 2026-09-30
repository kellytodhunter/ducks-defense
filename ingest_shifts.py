"""Download shift charts for one team's games (default ANA) -> data/raw_shifts/<season>/<gameId>.json."""
import json, sys, time
from pathlib import Path
import requests

SEASON, TEAM = "20252026", sys.argv[1] if len(sys.argv) > 1 else "ANA"
RAW = Path(__file__).parent / "data" / "raw" / SEASON
OUT = Path(__file__).parent / "data" / "raw_shifts" / SEASON
OUT.mkdir(parents=True, exist_ok=True)

def team_game_ids():
    ids = []
    for f in sorted(RAW.glob("*.json")):
        g = json.loads(f.read_text())
        if TEAM in (g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]):
            ids.append(g["id"])
    return ids

for gid in team_game_ids():
    p = OUT / f"{gid}.json"
    if p.exists():
        continue
    for i in range(4):
        r = requests.get(f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}", timeout=20)
        if r.status_code == 200:
            p.write_text(r.text); break
        time.sleep(2 ** i)
    time.sleep(0.4)
print("done:", len(list(OUT.glob("*.json"))), "games")
