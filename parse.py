"""Turn raw play-by-play JSON into one row per shot attempt with context features.

Output: data/shots_<season>.csv, data/clock_<season>.csv
Usage: python3 parse.py [season] [rush_secs]

Conventions
-----------
* Coordinates are normalised so the SHOOTING team always attacks the +x net
  (goal at x=89, y=0). Rink is 200x85 ft, blue lines at |x|=25.
* Zones are recomputed from coordinates in the acting team's perspective rather
  than trusting the feed's zoneCode (which is inconsistent for blocked shots).
* Chance-source labels are heuristics from the event sequence -- the feed has no
  puck-tracking, so they are approximations (see README limitations).
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

SEASON = sys.argv[1] if len(sys.argv) > 1 else "20252026"
RAW = Path(__file__).parent / "data" / "raw" / SEASON
RUSH_SECS = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # window after an N/D-zone event that counts as a rush
TAG = f"_r{RUSH_SECS}" if RUSH_SECS != 10 else ""
OUT = Path(__file__).parent / "data" / f"shots_{SEASON}{TAG}.csv"
TIME_OUT = Path(__file__).parent / "data" / f"clock_{SEASON}.csv"

GOAL_X = 89.0
SHOT_TYPES = {"shot-on-goal", "missed-shot", "goal", "blocked-shot"}
MEANINGFUL = SHOT_TYPES | {"hit", "giveaway", "takeaway", "faceoff"}
DEAD = {"stoppage", "penalty", "period-start", "period-end", "game-end", "delayed-penalty"}

REBOUND_SECS = 3
FACEOFF_SECS = 12
FORECHECK_SECS = 6


def clock_to_secs(s):
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def zone_from_x(x_own):
    """x in the acting team's own attacking frame (+x = their offensive end)."""
    if x_own > 25:
        return "O"
    if x_own < -25:
        return "D"
    return "N"


def attack_dir(team_id, home_id, home_def_side):
    """+1 if the team attacks toward +x, else -1."""
    home_attacks = -1 if home_def_side == "right" else 1
    return home_attacks if team_id == home_id else -home_attacks


def parse_game(path):
    g = json.loads(path.read_text())
    home, away = g["homeTeam"], g["awayTeam"]
    home_id, away_id = home["id"], away["id"]
    abbr = {home_id: home["abbrev"], away_id: away["abbrev"]}
    roster = {r["playerId"]: r["teamId"] for r in g.get("rosterSpots", [])}

    # Pass 1: normalise every event into a flat record.
    events = []
    for p in sorted(g["plays"], key=lambda p: p["sortOrder"]):
        d = p.get("details", {})
        typ = p["typeDescKey"]
        per = p["periodDescriptor"]
        if per["periodType"] == "SO":
            continue
        t = (per["number"] - 1) * 1200 + clock_to_secs(p["timeInPeriod"]) \
            if per["periodType"] == "REG" else 3600 + clock_to_secs(p["timeInPeriod"])
        team = d.get("eventOwnerTeamId")
        if typ == "blocked-shot":
            # Owner semantics are unreliable here; the shooter's roster team is not.
            team = roster.get(d.get("shootingPlayerId"), team)
        x, y = d.get("xCoord"), d.get("yCoord")
        side = p.get("homeTeamDefendingSide")
        if x is not None and team in abbr and side:
            dr = attack_dir(team, home_id, side)
            x_own, y_own = x * dr, y * dr
        else:
            x_own = y_own = None
        events.append(dict(typ=typ, t=t, per=per["number"], team=team, x=x_own, y=y_own,
                           d=d, sit=p.get("situationCode")))

    # Game-clock seconds spent in each strength state (clock stops on whistles, so this is play time).
    time_rows, last_sit = [], None
    for i, e in enumerate(events[:-1]):
        last_sit = e["sit"] or last_sit
        nxt = events[i + 1]
        if last_sit and nxt["per"] == e["per"] and nxt["t"] > e["t"]:
            time_rows.append(dict(game_id=g["id"], home=abbr[home_id], away=abbr[away_id],
                                  sit=last_sit, secs=nxt["t"] - e["t"]))

    rows = []
    score = {home_id: 0, away_id: 0}
    for i, e in enumerate(events):
        if e["typ"] == "goal":
            d = e["d"]
            score[home_id], score[away_id] = d.get("homeScore", score[home_id]), d.get("awayScore", score[away_id])
        if e["typ"] not in SHOT_TYPES or e["x"] is None or e["team"] not in abbr:
            continue
        shooter = e["team"]
        opp = away_id if shooter == home_id else home_id
        sit = e["sit"] or "0000"
        # situationCode = away goalie, away skaters, home skaters, home goalie
        a_g, a_s, h_s, h_g = (int(c) for c in sit)
        if shooter == home_id:
            skaters_for, skaters_against, goalie_against = h_s, a_s, a_g
        else:
            skaters_for, skaters_against, goalie_against = a_s, h_s, h_g

        # Score is updated on goal events themselves, so subtract the goal to get pre-shot state.
        sf, sa = score[shooter], score[opp]
        if e["typ"] == "goal":
            sf -= 1

        # ---- chance-source classification from the preceding live-play event ----
        source, gap, prev_typ = "sustained_oz", None, None
        for j in range(i - 1, -1, -1):
            pe = events[j]
            if pe["typ"] in DEAD:
                break
            if pe["typ"] not in MEANINGFUL or pe["x"] is None or pe["per"] != e["per"]:
                continue
            gap = e["t"] - pe["t"]
            prev_typ = pe["typ"]
            same = pe["team"] == shooter
            x_s = pe["x"] if same else -pe["x"]  # prior event in SHOOTER's frame
            z = zone_from_x(x_s)
            if pe["typ"] == "shot-on-goal" and same and gap <= REBOUND_SECS:
                source = "rebound"
            elif pe["typ"] == "faceoff" and same and gap <= FACEOFF_SECS:
                source = "faceoff_play"
            elif z in ("N", "D") and gap <= RUSH_SECS:
                source = "rush"
            elif z == "O" and gap <= FORECHECK_SECS and (
                (pe["typ"] == "takeaway" and same)
                or (pe["typ"] == "giveaway" and not same)
                or (pe["typ"] == "hit" and same)
            ):
                source = "forecheck_turnover"
            break

        dx = GOAL_X - e["x"]
        dist = math.hypot(dx, e["y"])
        angle = math.degrees(math.atan2(abs(e["y"]), dx))  # 0 = straight on, >90 = behind goal line
        d = e["d"]
        rows.append(dict(
            game_id=g["id"], date=g["gameDate"], period=e["per"], t=e["t"],
            shooting_team=abbr[shooter], defending_team=abbr[opp],
            is_home=int(shooter == home_id),
            event=e["typ"], blocked=int(e["typ"] == "blocked-shot"), goal=int(e["typ"] == "goal"),
            x=e["x"], y=e["y"], dist=dist, angle=angle,
            shot_type=d.get("shotType"), shooter_id=d.get("shootingPlayerId") or d.get("scoringPlayerId"),
            goalie_id=d.get("goalieInNetId"),
            skaters_for=skaters_for, skaters_against=skaters_against,
            empty_net=int(goalie_against == 0),
            score_diff=sf - sa, source=source, prev_event=prev_typ, prev_gap=gap,
        ))
    return rows, time_rows


def main():
    files = sorted(RAW.glob("*.json"))
    rows, trows = [], []
    for f in files:
        r, t = parse_game(f)
        rows.extend(r)
        trows.extend(t)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    (pd.DataFrame(trows).groupby(["game_id", "home", "away", "sit"], as_index=False).secs.sum()
        .to_csv(TIME_OUT, index=False))
    print(f"{len(files)} games -> {len(df)} shot attempts -> {OUT}")


if __name__ == "__main__":
    main()
