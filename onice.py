"""On-ice attribution for ANA 5v5 defence using shift charts.

For each ANA game: rebuild a per-second on-ice matrix from shifts, mask it to 5v5 (both goalies in)
using the play-by-play situation codes, then credit every attempt against ANA to the 5 skaters on ice.

Outputs data/out/onice_events.csv (one row per attempt-against x skater) and prints a validation check.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
SEASON = "20252026"
TEAM = "ANA"
T = 3900  # regulation 3600 + one OT period
SHOTS = DATA / "shots_20252026_xg.csv"


def secs(s):
    m, x = s.split(":")
    return int(m) * 60 + int(x)


def mask_5v5(plays):
    """Boolean per-second array: True while play is 5v5 with both goalies in (situationCode 1551)."""
    m = np.zeros(T, dtype=bool)
    ev, last = [], None
    for p in sorted(plays, key=lambda p: p["sortOrder"]):
        pd_ = p["periodDescriptor"]
        if pd_["periodType"] == "SO":
            continue
        t = (pd_["number"] - 1) * 1200 + secs(p["timeInPeriod"])
        last = p.get("situationCode") or last
        ev.append((pd_["number"], t, last))
    for (p1, t1, s1), (p2, t2, _) in zip(ev, ev[1:]):
        if p1 == p2 and t2 > t1 and s1 == "1551":
            m[t1:t2] = True
    return m


def main():
    shots = pd.read_csv(SHOTS)
    shots = shots[(shots.defending_team == TEAM) & (shots.skaters_for == 5) & (shots.skaters_against == 5)
                  & (shots.empty_net == 0)]
    rows, toi_rows, pair_rows, names = [], [], [], {}
    checks = []
    for pbp_path in sorted((DATA / "raw" / SEASON).glob("*.json")):
        g = json.loads(pbp_path.read_text())
        if TEAM not in (g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]):
            continue
        gid = g["id"]
        pos = {}
        for r in g["rosterSpots"]:
            if r["teamId"] == (g["homeTeam"]["id"] if g["homeTeam"]["abbrev"] == TEAM else g["awayTeam"]["id"]):
                pos[r["playerId"]] = r["positionCode"]
                names[r["playerId"]] = f'{r["firstName"]["default"]} {r["lastName"]["default"]}'
        mask = mask_5v5(g["plays"])
        sh = json.loads((DATA / "raw_shifts" / SEASON / f"{gid}.json").read_text())["data"]
        on = {}
        for s in sh:
            if s["typeCode"] != 517 or s["teamAbbrev"] != TEAM or pos.get(s["playerId"]) in (None, "G"):
                continue
            a = (s["period"] - 1) * 1200 + secs(s["startTime"])
            b = (s["period"] - 1) * 1200 + secs(s["endTime"])
            on.setdefault(s["playerId"], np.zeros(T, dtype=bool))[a:b] = True
        # validation: at 5v5 we should reconstruct 5 skaters on ice
        occ = sum(v.astype(int) for v in on.values())
        checks.append((occ[mask] == 5).mean())
        for pid, v in on.items():
            toi_rows.append(dict(game_id=gid, player_id=pid, pos=pos[pid], toi5=int((v & mask).sum())))
        dmen = [p for p in on if pos[p] == "D"]
        for i, p in enumerate(dmen):
            for q in dmen[i + 1:]:
                pair_rows.append(dict(game_id=gid, pair=tuple(sorted((p, q))), toi5=int((on[p] & on[q] & mask).sum())))
        for _, e in shots[shots.game_id == gid].iterrows():
            t = int(e.t)
            for pid, v in on.items():
                if v[t - 1]:  # on ice for (t-1, t]
                    rows.append(dict(game_id=gid, t=t, player_id=pid, pos=pos[pid], event=e.event, blocked=e.blocked,
                                     goal=e.goal, xg=e.xg, x=e.x, y=e.y, dist=e.dist, source=e.source))
    print(f"validation: share of 5v5 seconds with exactly 5 ANA skaters reconstructed: "
          f"mean {np.mean(checks):.3f}, min {np.min(checks):.3f} over {len(checks)} games")
    out = DATA / "out"
    pd.DataFrame(rows).to_csv(out / "onice_events.csv", index=False)
    pd.DataFrame(toi_rows).to_csv(out / "onice_toi.csv", index=False)
    pr = pd.DataFrame(pair_rows)
    pr["pair"] = pr.pair.astype(str)
    pr.to_csv(out / "onice_pairs_toi.csv", index=False)
    pd.Series(names).rename("name").to_csv(out / "names.csv", index_label="player_id")
    print("wrote", len(rows), "attempt-skater rows")


if __name__ == "__main__":
    main()
