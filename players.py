"""Per-skater and D-pair on-ice vs off-ice 'middle of the ice' (slot+crease) chances against, ANA 5v5."""
import numpy as np, pandas as pd
from pathlib import Path

D = Path(__file__).parent / "data"; O = D / "out"
RNG = np.random.default_rng(11)
MIN_TOI = 400  # 5v5 minutes

ev = pd.read_csv(O / "onice_events.csv"); toi = pd.read_csv(O / "onice_toi.csv")
names = pd.read_csv(O / "names.csv").set_index("player_id").name
shots = pd.read_csv(D / "shots_20252026_xg.csv")
team = shots[(shots.defending_team == "ANA") & (shots.skaters_for == 5) & (shots.skaters_against == 5) & (shots.empty_net == 0)]
clock = pd.read_csv(D / "clock_20252026.csv")
gm = clock[(clock.sit == 1551) & ((clock.home == "ANA") | (clock.away == "ANA"))].groupby("game_id").secs.sum()
games = gm.index.values; gidx = {g: i for i, g in enumerate(games)}; G = len(games)

def mid(df): return (df.x >= 64) & (df.y.abs() <= 22) & (df.blocked == 0)

def per_game(df, weights):  # sum of `weights` by game -> length-G array
    a = np.zeros(G); s = df.groupby("game_id")[weights].sum()
    for g, v in s.items(): a[gidx[g]] = v
    return a

team_mid_fa = per_game(team.assign(w=mid(team).astype(float)), "w")
team_mid_xg = per_game(team.assign(w=np.where(mid(team), team.xg, 0.0)), "w")
team_secs = gm.values.astype(float)

def stat(on_w, on_secs, tw, ts, idx):
    on, on_t = on_w[idx].sum(), on_secs[idx].sum()
    off, off_t = tw[idx].sum() - on, ts[idx].sum() - on_t
    return (on / on_t * 3600) - (off / off_t * 3600) if on_t > 0 and off_t > 0 else np.nan

def boot(on_w, on_secs, tw, ts, n=1000):
    pt = stat(on_w, on_secs, tw, ts, np.arange(G))
    bs = [stat(on_w, on_secs, tw, ts, RNG.integers(0, G, G)) for _ in range(n)]
    return pt, *np.nanpercentile(bs, [2.5, 97.5])

rows = []
for pid, tp in toi.groupby("player_id"):
    mins = tp.toi5.sum() / 60
    if mins < MIN_TOI: continue
    e = ev[ev.player_id == pid]
    on_fa = per_game(e.assign(w=mid(e).astype(float)), "w") if len(e) else np.zeros(G)
    on_xg = per_game(e.assign(w=np.where(mid(e), e.xg, 0.0)), "w") if len(e) else np.zeros(G)
    on_s = np.zeros(G)
    for g, v in tp.groupby("game_id").toi5.sum().items(): on_s[gidx[g]] = v
    r_fa = boot(on_fa, on_s, team_mid_fa, team_secs); r_xg = boot(on_xg, on_s, team_mid_xg, team_secs)
    on_rate = on_fa.sum() / on_s.sum() * 3600
    rows.append(dict(player=names[pid], pos=tp.pos.iloc[0], min5v5=round(mins), on_mid_fa60=on_rate,
                     rel_mid_fa60=r_fa[0], fa_lo=r_fa[1], fa_hi=r_fa[2],
                     rel_mid_xga60=r_xg[0], xg_lo=r_xg[1], xg_hi=r_xg[2]))
res = pd.DataFrame(rows).sort_values("rel_mid_xga60")
res.to_csv(O / "player_onice.csv", index=False)
team_rate = team_mid_fa.sum() / team_secs.sum() * 3600
print(f"Team: {team_mid_fa.sum():.0f} middle-of-ice unblocked attempts against, {team_rate:.1f}/60, {team_mid_xg.sum():.1f} xG")
pd.set_option("display.width", 200)
print("\nRelative (on-ice minus off-ice) middle-of-ice xGA/60 -- negative = better\n")
print(res.round(3).to_string(index=False))
