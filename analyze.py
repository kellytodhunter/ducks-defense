"""Team-level defensive analysis: how does ANA's 5v5 defence differ from the league, and where?

Usage: python3 analyze.py [shots_xg_csv]
Prints findings and writes tables to data/out/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
OUT = DATA / "out"
OUT.mkdir(exist_ok=True)
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "shots_20252026_xg.csv"
TEAM = "ANA"
SOURCES = ["rush", "rebound", "faceoff_play", "forecheck_turnover", "sustained_oz"]
RNG = np.random.default_rng(7)


def load():
    shots = pd.read_csv(SRC)
    clock = pd.read_csv(DATA / "clock_20252026.csv")
    return shots, clock


def toi_5v5(clock):
    """5v5 (both goalies in) minutes per team-game."""
    c = clock[clock.sit == 1551]
    per_game = c.groupby(["game_id", "home", "away"], as_index=False).secs.sum()
    long = pd.concat([per_game.rename(columns={"home": "team"})[["game_id", "team", "secs"]],
                      per_game.rename(columns={"away": "team"})[["game_id", "team", "secs"]]])
    return long


def even_strength_against(shots):
    """Unblocked + blocked attempts against each team at 5v5, goalie in net."""
    s = shots[(shots.skaters_for == 5) & (shots.skaters_against == 5) & (shots.empty_net == 0)]
    return s


def per60_by_source(against, toi):
    """Per team: attempts/xG/goals against per 60, split by chance source."""
    hours = toi.groupby("team").secs.sum() / 3600
    g = against.groupby(["defending_team", "source"]).agg(
        ca=("event", "size"), fa=("blocked", lambda b: (b == 0).sum()), xga=("xg", "sum"), ga=("goal", "sum")
    ).reset_index()
    for col in ("ca", "fa", "xga", "ga"):
        g[f"{col}_60"] = g[col] / g.defending_team.map(hours)
    return g


def team_totals(against, toi):
    hours = toi.groupby("team").secs.sum() / 3600
    t = against.groupby("defending_team").agg(
        ca=("event", "size"), fa=("blocked", lambda b: (b == 0).sum()), xga=("xg", "sum"), ga=("goal", "sum")
    )
    for col in ("ca", "fa", "xga", "ga"):
        t[f"{col}_60"] = t[col] / hours
    t["xg_per_fa"] = t.xga / t.fa
    t["hours"] = hours
    return t


def add_zone(df):
    """Shot location bucket in the shooter's frame (goal at x=89, dots at |y|=22)."""
    z = np.select(
        [df.dist <= 8, (df.x >= 64) & (df.y.abs() <= 22), df.dist <= 40, df.dist > 40],
        ["crease", "slot", "wide_mid", "point"], default="wide_mid")
    return df.assign(zone=z)


def zone_table(against, toi, col, categories):
    """ANA vs league per-60 for each category of `col` (unblocked attempts + xG)."""
    unblocked = against[against.blocked == 0]
    hours = toi.groupby("team").secs.sum() / 3600
    g = unblocked.groupby(["defending_team", col]).agg(fa=("event", "size"), xga=("xg", "sum"), ga=("goal", "sum")).reset_index()
    for c in ("fa", "xga", "ga"):
        g[f"{c}_60"] = g[c] / g.defending_team.map(hours)
    rows = []
    for cat in categories:
        d = g[g[col] == cat].set_index("defending_team")
        rows.append(dict(category=cat, ana_fa60=d.loc[TEAM, "fa_60"], league_fa60=d.drop(TEAM).fa_60.mean(),
                         fa_rank=int(d.fa_60.rank(method="min").loc[TEAM]),
                         ana_xga60=d.loc[TEAM, "xga_60"], league_xga60=d.drop(TEAM).xga_60.mean(),
                         xga_rank=int(d.xga_60.rank(method="min").loc[TEAM])))
    return pd.DataFrame(rows)


def bootstrap_excess(against, toi, source, n=2000):
    """Bootstrap ANA games -> CI on (ANA xGA/60 for `source` - league mean of other teams)."""
    other = per60_by_source(against[against.defending_team != TEAM], toi[toi.team != TEAM])
    league_mean = other[other.source == source].xga_60.mean()
    a = against[(against.defending_team == TEAM) & (against.source == source)]
    per_game = a.groupby("game_id").xg.sum()
    t = toi[toi.team == TEAM].set_index("game_id").secs
    games = t.index.values
    xg = per_game.reindex(games, fill_value=0.0).values
    secs = t.values
    idx = RNG.integers(0, len(games), (n, len(games)))
    rate = xg[idx].sum(1) / (secs[idx].sum(1) / 3600)
    ex = rate - league_mean
    return ex.mean(), *np.percentile(ex, [2.5, 97.5])


def main():
    shots, clock = load()
    toi = toi_5v5(clock)
    against = even_strength_against(shots)
    tot = team_totals(against, toi)
    n_teams = len(tot)
    print(f"{shots.game_id.nunique()} games, {n_teams} teams. ANA 5v5 hours: {tot.loc[TEAM,'hours']:.1f}\n")

    print("=== ANA 5v5 defence vs league (lower = better; rank 1 = best) ===")
    rows = []
    for col, label in [("ca_60", "Shot attempts against/60"), ("fa_60", "Unblocked attempts against/60"),
                       ("xga_60", "xG against/60"), ("ga_60", "Goals against/60"), ("xg_per_fa", "xG per unblocked attempt")]:
        rank = tot[col].rank(method="min").loc[TEAM]
        rows.append((label, tot.loc[TEAM, col], tot[col].mean(), int(rank)))
    print(pd.DataFrame(rows, columns=["metric", "ANA", "league_avg", "rank_of_%d" % n_teams]).round(3).to_string(index=False))

    src = per60_by_source(against, toi)
    print("\n=== xGA/60 by chance source: ANA vs league ===")
    rows = []
    for s in SOURCES:
        d = src[src.source == s].set_index("defending_team")
        mean_o = d.drop(TEAM).xga_60.mean()
        m, lo, hi = bootstrap_excess(against, toi, s)
        rows.append(dict(source=s, ana_xga60=d.loc[TEAM, "xga_60"], league_avg=mean_o,
                         excess=d.loc[TEAM, "xga_60"] - mean_o, ci_lo=lo, ci_hi=hi,
                         rank=int(d.xga_60.rank(method="min").loc[TEAM]),
                         ana_fa60=d.loc[TEAM, "fa_60"], league_fa60=d.drop(TEAM).fa_60.mean()))
    tbl = pd.DataFrame(rows)
    print(tbl.round(3).to_string(index=False))
    tbl.to_csv(OUT / "source_excess.csv", index=False)
    src.to_csv(OUT / "team_source_per60.csv", index=False)
    tot.to_csv(OUT / "team_totals.csv")

    against = add_zone(against)
    print("\n=== By shot location (unblocked attempts + xG against per 60) ===")
    zt = zone_table(against, toi, "zone", ["crease", "slot", "wide_mid", "point"])
    print(zt.round(3).to_string(index=False)); zt.to_csv(OUT / "zone_table.csv", index=False)
    print("\n=== Where ANA's actual goals against came from (5v5) ===")
    ga = against[(against.defending_team == TEAM) & (against.goal == 1)].source.value_counts()
    lg = against[(against.defending_team != TEAM) & (against.goal == 1)].source.value_counts(normalize=True)
    print(pd.DataFrame({"ANA_GA": ga, "ANA_share": (ga / ga.sum()).round(3), "league_share": lg.round(3)}).to_string())


if __name__ == "__main__":
    main()
