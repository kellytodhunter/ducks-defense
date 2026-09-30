"""Expected-goals (xG) model on unblocked shot attempts.

Out-of-fold predictions (GroupKFold by game) so every shot's xG is scored by a
model that never saw that game. Adds column `xg` to the shots file.

Usage: python3 model.py [shots_csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "shots_20252026.csv"

SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around", "bat", "poke", "cradle"]
SOURCES = ["rebound", "rush", "faceoff_play", "forecheck_turnover", "sustained_oz"]


def strength(row_for, row_against):
    if row_for == row_against:
        return "even"
    return "pp" if row_for > row_against else "sh"


def features(df):
    X = pd.DataFrame(index=df.index)
    X["dist"] = df.dist
    X["log_dist"] = np.log1p(df.dist)
    X["angle"] = df.angle
    X["abs_y"] = df.y.abs()
    X["x"] = df.x
    # shot_type deliberately excluded: arena scorers label it inconsistently (e.g. 38% "snap" in
    # Anaheim vs 24% elsewhere for BOTH teams), which would leak venue bias into xG.
    for s in SOURCES:
        X[f"src_{s}"] = (df.source == s).astype(int)
    st = [strength(a, b) for a, b in zip(df.skaters_for, df.skaters_against)]
    for s in ("pp", "sh"):
        X[f"str_{s}"] = [int(v == s) for v in st]
    X["score_diff"] = df.score_diff.clip(-3, 3)
    X["is_home"] = df.is_home
    X["period"] = df.period.clip(upper=4)
    return X


def cross_val_xg(X, y, groups, make_model, n_splits=5):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def main():
    df = pd.read_csv(SRC)
    fen = df[(df.blocked == 0) & (df.empty_net == 0) & (df.x > 0)].copy()  # unblocked, goalie in net
    X, y, groups = features(fen), fen.goal, fen.game_id
    print(f"{len(fen):,} unblocked non-empty-net attempts, goal rate {y.mean():.3%}, {groups.nunique()} games")

    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "grad_boost": lambda: HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=250, l2_regularization=1.0, random_state=0),
    }
    base = log_loss(y, np.full(len(y), y.mean()))
    results = {}
    for name, mk in models.items():
        oof = cross_val_xg(X, y, groups, mk)
        results[name] = oof
        print(f"{name:11s} AUC {roc_auc_score(y, oof):.4f}  logloss {log_loss(y, oof):.4f} (baseline {base:.4f})  "
              f"sum xG {oof.sum():.0f} vs goals {y.sum()}")

    best = max(results, key=lambda k: roc_auc_score(y, results[k]))
    fen["xg"] = results[best]
    print("using:", best)

    # calibration: predicted vs actual by decile
    fen["decile"] = pd.qcut(fen.xg, 10, duplicates="drop")
    cal = fen.groupby("decile", observed=True).agg(pred=("xg", "mean"), actual=("goal", "mean"), n=("goal", "size"))
    print(cal.round(3).to_string())

    out = df.merge(fen[["xg"]], left_index=True, right_index=True, how="left")
    out["xg"] = out.xg.fillna(0.0)  # blocked / empty-net / no-model rows carry 0 xG
    out.to_csv(SRC.with_name(SRC.stem + "_xg.csv"), index=False)
    print("wrote", SRC.with_name(SRC.stem + "_xg.csv"))


if __name__ == "__main__":
    main()
