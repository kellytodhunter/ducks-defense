"""Render report charts to charts/*.png (light surface, Ducks orange vs neutral gray)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Arc, Rectangle, Circle
from scipy.ndimage import gaussian_filter
import analyze as A

INK, SUB, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORANGE, BLUE, GRAY = "#eb6834", "#2a78d6", "#a8a7a1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": SUB,
                     "xtick.color": SUB, "ytick.color": SUB, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
OUT = A.DATA.parent / "charts"; OUT.mkdir(exist_ok=True)
shots, clock = A.load(); toi = A.toi_5v5(clock); ag = A.even_strength_against(shots)
hours = toi.groupby("team").secs.sum() / 3600
mid_mask = lambda d: (d.blocked == 0) & (d.x >= 64) & (d.y.abs() <= 22)

def title(ax, t, s):
    ax.annotate(t, (0, 1), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points", fontsize=13, fontweight="bold", color=INK)
    ax.annotate(s, (0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points", fontsize=9, color=SUB)

def excess_ci(df, n=2000):
    m, lo, hi = A.bootstrap_excess(df.assign(source="all"), toi, "all", n=n)
    return m, lo, hi

# ---- 1. rink map ------------------------------------------------------------------
u = ag[(ag.blocked == 0)]
xb, yb = np.arange(25, 101, 5), np.arange(-42.5, 43, 5)
def grid(d): return np.histogram2d(d.x, d.y, bins=[xb, yb])[0]
ana = grid(u[u.defending_team == "ANA"]) / hours["ANA"]
lg = np.mean([grid(u[u.defending_team == t]) / hours[t] for t in hours.index if t != "ANA"], axis=0)
diff = gaussian_filter(ana - lg, 0.8)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
cmap = LinearSegmentedColormap.from_list("d", [BLUE, "#f1f0ec", ORANGE])
lim = np.abs(diff).max()
im = ax.pcolormesh(xb, yb, diff.T, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), shading="flat")
ax.axvline(25, color=SUB, lw=1); ax.axvline(89, color=SUB, lw=1)
ax.add_patch(Circle((69, 22), 15, fill=False, ec=SUB, lw=.8)); ax.add_patch(Circle((69, -22), 15, fill=False, ec=SUB, lw=.8))
ax.add_patch(Rectangle((89, -3), 4, 6, fill=False, ec=SUB, lw=1))
ax.add_patch(Rectangle((64, -22), 25, 44, fill=False, ec=INK, lw=1, ls=(0, (4, 3))))
ax.text(76.5, 25, "slot / \"home plate\"", ha="center", fontsize=8, color=INK)
ax.set_xlim(25, 100); ax.set_ylim(-42.5, 42.5); ax.set_aspect("equal"); ax.axis("off")
title(ax, "Where the extra chances against come from", "Unblocked attempts against per 60 at 5v5: Ducks minus league average")
cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02); cb.set_label("attempts/60 per 5x5 ft cell", color=SUB, fontsize=8)
cb.outline.set_visible(False)
ax.text(27, -40, "Defending net at right. Orange = Ducks allow more, blue = fewer", fontsize=7.5, color=SUB)
fig.savefig(OUT / "1_rink_map.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 2. excess xGA by zone -------------------------------------------------------
agz = A.add_zone(ag); ub = agz[agz.blocked == 0]
rows = [(z, *excess_ci(ub[ub.zone == z])) for z in ["point", "wide_mid", "slot", "crease"]]
lab = {"point": "Point / perimeter\n(> 40 ft)", "wide_mid": "Wide-mid\n(outside dots)", "slot": "Slot\n(inside dots)", "crease": "Crease\n(< 8 ft)"}
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for i, (z, m, lo, hi) in enumerate(rows):
    ax.barh(i, m, height=.5, color=ORANGE if m > 0.02 else GRAY, zorder=2)
    ax.plot([lo, hi], [i, i], color=INK, lw=1.2, zorder=3); ax.plot([lo, lo], [i - .1, i + .1], color=INK, lw=1.2); ax.plot([hi, hi], [i - .1, i + .1], color=INK, lw=1.2)
    ax.text(.31, i, ("0.000" if abs(m) < .0005 else f"{m:+.3f}"), va="center", ha="right", fontsize=9, color=INK)
ax.axvline(0, color=INK, lw=1); ax.set_yticks(range(4)); ax.set_yticklabels([lab[r[0]] for r in rows], color=INK, fontsize=9)
ax.set_xlim(-.15, .32); ax.grid(axis="x", color=GRID, zorder=0); ax.set_xlabel("Ducks minus league xG against/60 (95% CI)")
title(ax, "Most of the leak is in the slot", "5v5 excess expected goals against per 60, by shot location")
fig.savefig(OUT / "2_zone_excess.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 3. middle-of-ice by chance source --------------------------------------------
mid = ag[mid_mask(ag)]
names = {"sustained_oz": "Settled offensive-zone play", "rush": "Rush (<10s after N/D-zone event)", "faceoff_play": "Off a faceoff (<12s)",
         "rebound": "Rebounds (<3s)", "forecheck_turnover": "Forecheck turnovers (<6s)"}
srows = [(s, *A.bootstrap_excess(mid, toi, s, n=10000)) for s in names]
fig, ax = plt.subplots(figsize=(6.4, 3.3))
for i, (s, m, lo, hi) in enumerate(srows):
    ax.barh(i, m, height=.5, color=ORANGE if m > 0 else BLUE, zorder=2)
    ax.plot([lo, hi], [i, i], color=INK, lw=1.2, zorder=3)
    ax.text(.305, i, f"{m:+.3f}", va="center", ha="right", fontsize=9, color=INK)
ax.axvline(0, color=INK, lw=1); ax.set_yticks(range(5)); ax.set_yticklabels([names[r[0]] for r in srows], color=INK, fontsize=9)
ax.invert_yaxis(); ax.set_xlim(-.12, .32); ax.grid(axis="x", color=GRID, zorder=0)
ax.set_xlabel("Middle-of-ice xG against per 60 vs league (95% CI)")
title(ax, "Most of the excess is in settled play", "Slot + crease xG against at 5v5 by how the chance started. Intervals are wide: read direction, not decimals")
fig.savefig(OUT / "3_source.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 4. consistency across splits --------------------------------------------------
import robust as R
ana_games = sorted(agz[agz.defending_team == "ANA"].game_id.unique())
away_g = set(shots[(shots.defending_team == "ANA") & (shots.is_home == 1)].game_id)  # ANA on road = opponent is home
home_g = set(ana_games) - away_g
cuts = [("Full season", None), ("Honda Center", home_g), ("On the road", away_g), ("First 41 games", set(ana_games[:41])), ("Last 41 games", set(ana_games[41:]))]
vals = []
for k, g in cuts:
    d, t = ub, toi
    if g is not None:
        d = d[d.game_id.isin(g) | (d.defending_team != "ANA")]; t = t[t.game_id.isin(g) | (t.team != "ANA")]
    z = A.zone_table(d, t, "zone", ["slot", "crease"]); e = (z.ana_xga60 - z.league_xga60).sum(); vals.append((k, e))
fig, ax = plt.subplots(figsize=(6.4, 2.8))
for i, (k, e) in enumerate(vals):
    ax.barh(i, e, height=.5, color=ORANGE, zorder=2); ax.text(e + .006, i, f"{e:+.3f}", va="center", fontsize=9, color=INK)
ax.axvline(0, color=INK, lw=1); ax.set_yticks(range(len(vals))); ax.set_yticklabels([v[0] for v in vals], color=INK); ax.invert_yaxis()
ax.set_xlim(0, .26); ax.grid(axis="x", color=GRID, zorder=0); ax.set_xlabel("Ducks slot + crease xG against per 60 minus league")
title(ax, "Same direction in every split", "Middle-of-ice excess, 5v5 (no intervals shown: each half is a small sample)")
fig.savefig(OUT / "4_consistency.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 5. player on/off forest -------------------------------------------------------
p = pd.read_csv(A.DATA / "out" / "player_onice.csv").sort_values("rel_mid_xga60", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.4, 5.2))
for i, r in p.iterrows():
    sig = r.xg_hi < 0 or r.xg_lo > 0
    ax.plot([r.xg_lo, r.xg_hi], [i, i], color=INK if sig else GRAY, lw=1.4, zorder=2)
    ax.scatter(r.rel_mid_xga60, i, s=32, color=BLUE if r.rel_mid_xga60 < 0 else ORANGE, zorder=3, ec="white", lw=.8)
ax.axvline(0, color=INK, lw=1); ax.set_yticks(range(len(p))); ax.set_yticklabels([f"{r.player} ({r.pos})" for _, r in p.iterrows()], fontsize=8, color=INK)
ax.set_xlim(-.7, .8); ax.grid(axis="x", color=GRID, zorder=0)
ax.set_xlabel("On-ice minus off-ice middle-of-ice xGA/60 (95% CI).  Left = better")
title(ax, "Player differences look like noise", "20 skaters, 400+ 5v5 min. One of 20 clearing zero (black) is what chance alone produces")
fig.savefig(OUT / "5_players.png", dpi=200, bbox_inches="tight"); plt.close()
print("charts:", sorted(x.name for x in OUT.glob("*.png")))
