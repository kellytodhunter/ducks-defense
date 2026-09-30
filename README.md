# Where the Ducks' 5v5 defense leaks (2025-26)

A location and chance-source analysis of Anaheim's 5v5 defense, built from public NHL data.
Full write-up: [`Ducks_5v5_Defense_Analysis.pdf`](Ducks_5v5_Defense_Analysis.pdf).

**Headline:** the Ducks' defensive problem is chance *quality*, not shot volume. Roughly three-quarters of their
excess expected goals against (about +0.18 xGA/60, ~11 goals) comes from the slot. The pattern holds home/road and
in both halves of the season, and no individual skater separates from noise, which points to a system-level cause.

## Reproduce

```bash
pip install pandas numpy scikit-learn scipy matplotlib requests reportlab pillow
python3 ingest.py               # all 1,312 games -> data/raw/   (~40 min, rate-limited API; resumable)
python3 ingest_shifts.py ANA    # Ducks shift charts
python3 parse.py                # shot attempts + chance-source labels + 5v5 clock time
python3 model.py                # xG model (out-of-fold, grouped by game)
python3 analyze.py              # team vs league tables
python3 onice.py && python3 players.py   # on-ice attribution, per-skater on/off
python3 robust.py               # venue / half-season / rush-window checks
python3 make_charts.py && python3 build_report.py
```

## Pipeline

| Step | File | What it does |
|---|---|---|
| Ingest | `ingest.py`, `ingest_shifts.py` | NHL public play-by-play and shift-chart endpoints, cached to disk |
| Parse | `parse.py` | Normalises coordinates, classifies each shot as rush / rebound / faceoff / forecheck turnover / settled OZ, builds 5v5 time from the game clock |
| Model | `model.py` | Gradient-boosted xG on unblocked non-empty-net attempts, 5-fold out-of-fold by game (AUC 0.72) |
| Analyse | `analyze.py`, `robust.py` | Ducks vs league by source and location, bootstrap CIs over games, split and sensitivity checks |
| On-ice | `onice.py`, `players.py` | Rebuilds who was on the ice from shifts (validated: 5 skaters for 99.7% of 5v5 seconds), on/off per skater |

## Things I checked that changed the analysis

- **Arena scorer bias.** Shot-type labels vary by rink (Anaheim: 38% "snap" for both teams vs 24% elsewhere). I removed shot type from the model and dropped shot-type findings. Location shows no such bias.
- **Rush definition.** The feed has no zone entries, so a 4-5 s window catches almost nothing. I use 10 s and report 5/10/15 s; the Ducks' rush excess is never distinguishable from zero, so I make no rush claim.
- **Overclaiming individuals.** Only 1 of 20 skaters has an on/off interval excluding zero (chance level); heterogeneity test p = 0.66.

## Limitations

Single season. No puck or player tracking, so this shows where and when chances arise, not why. Chance-source labels are heuristics from event sequences. On/off results are confounded by usage and linemates. Bootstrap treats games as independent.
