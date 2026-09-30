"""Robustness checks on the ANA defensive findings: venue split, half-season split, rush-window sensitivity."""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
import analyze as A

DATA = Path(__file__).parent / "data"

def excess_by(col, against, toi, cats, games=None):
    if games is not None:
        against = against[against.game_id.isin(games) | (against.defending_team != A.TEAM)]
        toi = toi[toi.game_id.isin(games) | (toi.team != A.TEAM)]
    t = A.zone_table(against, toi, col, cats)
    return (t.ana_xga60 - t.league_xga60).set_axis(cats)

def main():
    shots, clock = A.load()
    toi = A.toi_5v5(clock)
    ag = A.add_zone(A.even_strength_against(shots))
    zones = ["crease", "slot", "wide_mid", "point"]
    ana_games = sorted(ag[ag.defending_team == A.TEAM].game_id.unique())
    home = set(shots[(shots.defending_team == A.TEAM) & (shots.is_home == 0)].game_id)
    away = set(ana_games) - home
    cuts = {"full season": None, "Honda Center": home, "road": away,
            "first 41": set(ana_games[:41]), "last 41": set(ana_games[41:])}
    out = pd.DataFrame({k: excess_by("zone", ag, toi, zones, g) for k, g in cuts.items()})
    print("ANA xGA/60 excess vs league by location (5v5):"); print(out.round(3).to_string())
    print("\nrush-window sensitivity (excess xGA/60, rush source):")
    for r in (5, 10, 15):
        path = DATA / f"shots_20252026{'' if r == 10 else f'_r{r}'}.csv"
        if r != 10:
            subprocess.run([sys.executable, "-W", "ignore", "parse.py", "20252026", str(r)], check=True, capture_output=True)
        if not Path(str(path).replace(".csv", "_xg.csv")).exists() or r != 10:
            subprocess.run([sys.executable, "-W", "ignore", "model.py", str(path)], check=True, capture_output=True)
        s2 = pd.read_csv(str(path).replace(".csv", "_xg.csv"))
        a2 = A.even_strength_against(s2)
        m, lo, hi = A.bootstrap_excess(a2, toi, "rush")
        share = (s2.source == "rush").mean()
        print(f"  {r:>2}s window: rush share of attempts {share:.1%}, ANA excess {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

if __name__ == "__main__":
    main()
