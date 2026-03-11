"""
=============================================================
PHASE 2: NBA PARLAY MODEL — TRAIN PROP MODELS
=============================================================
Updated to work with nathanlauga/nba-games Kaggle dataset.

CSV columns used:
  GAME_DATE_EST, GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID,
  SEASON, PTS_home, FG_PCT_home, FT_PCT_home, FG3_PCT_home,
  AST_home, REB_home, PTS_away, FG_PCT_away, FT_PCT_away,
  FG3_PCT_away, AST_away, REB_away, HOME_TEAM_WINS

Trains 3 models: pts O/U, reb O/U, ast O/U
using team-level rolling stats as features.
=============================================================
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

MODEL_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── 1. LOAD & RESHAPE DATA ──────────────────────────────

def load_historical_data(path="data/nba_player_stats_historical.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["GAME_DATE_EST"])
    df = df.sort_values("date").reset_index(drop=True)

    home = pd.DataFrame({
        "game_id":  df["GAME_ID"],
        "date":     df["date"],
        "season":   df["SEASON"],
        "team_id":  df["HOME_TEAM_ID"],
        "opp_id":   df["VISITOR_TEAM_ID"],
        "is_home":  1,
        "pts":      df["PTS_home"],
        "reb":      df["REB_home"],
        "ast":      df["AST_home"],
        "fg_pct":   df["FG_PCT_home"],
        "fg3_pct":  df["FG3_PCT_home"],
        "ft_pct":   df["FT_PCT_home"],
        "opp_pts":  df["PTS_away"],
        "opp_reb":  df["REB_away"],
        "opp_ast":  df["AST_away"],
        "won":      df["HOME_TEAM_WINS"],
    })

    away = pd.DataFrame({
        "game_id":  df["GAME_ID"],
        "date":     df["date"],
        "season":   df["SEASON"],
        "team_id":  df["VISITOR_TEAM_ID"],
        "opp_id":   df["HOME_TEAM_ID"],
        "is_home":  0,
        "pts":      df["PTS_away"],
        "reb":      df["REB_away"],
        "ast":      df["AST_away"],
        "fg_pct":   df["FG_PCT_away"],
        "fg3_pct":  df["FG3_PCT_away"],
        "ft_pct":   df["FT_PCT_away"],
        "opp_pts":  df["PTS_home"],
        "opp_reb":  df["REB_home"],
        "opp_ast":  df["AST_home"],
        "won":      1 - df["HOME_TEAM_WINS"],
    })

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team_id", "date"]).reset_index(drop=True)
    long = long.dropna(subset=["pts", "reb", "ast"])

    print(f"[Data] Loaded {len(long)} team-game rows across {long['team_id'].nunique()} teams")
    print(f"[Data] Seasons: {sorted(long['season'].unique())}")
    return long


# ─── 2. FEATURE ENGINEERING ──────────────────────────────

def build_rolling_features(df, stat, windows=[5, 10, 15]):
    grp = df.groupby("team_id")[stat]
    for w in windows:
        df[f"{stat}_avg{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
        df[f"{stat}_std{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=3).std())
    return df

def build_opp_rolling_features(df, stat, windows=[5, 10]):
    opp_stat = f"opp_{stat}"
    if opp_stat not in df.columns:
        return df
    grp = df.groupby("team_id")[opp_stat]
    for w in windows:
        df[f"opp_{stat}_allowed_avg{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
    return df

def build_hit_rate(df, stat, thresholds):
    grp = df.groupby("team_id")[stat]
    for t in thresholds:
        df[f"{stat}_hit_{t}"] = grp.transform(lambda x: (x.shift(1) > t).rolling(10, min_periods=3).mean())
    return df

def prep_features(df, target_stat):
    df = build_rolling_features(df, target_stat, windows=[5, 10, 15])
    df = build_opp_rolling_features(df, target_stat, windows=[5, 10])

    if target_stat == "pts":
        df = build_hit_rate(df, "pts", [95, 105, 115])
        df = build_rolling_features(df, "fg_pct", windows=[5, 10])
        df = build_rolling_features(df, "fg3_pct", windows=[5, 10])
    elif target_stat == "reb":
        df = build_hit_rate(df, "reb", [40, 45, 50])
    elif target_stat == "ast":
        df = build_hit_rate(df, "ast", [20, 25, 30])

    df["line"] = df[f"{target_stat}_avg10"]
    df["target"] = (df[target_stat] > df["line"]).astype(int)
    df["is_home"] = df["is_home"].fillna(0)
    return df

FEATURE_COLS = {
    "pts": [
        "pts_avg5", "pts_avg10", "pts_avg15",
        "pts_std5", "pts_std10",
        "pts_hit_95", "pts_hit_105", "pts_hit_115",
        "opp_pts_allowed_avg5", "opp_pts_allowed_avg10",
        "fg_pct_avg5", "fg_pct_avg10",
        "fg3_pct_avg5", "fg3_pct_avg10",
        "is_home", "line",
    ],
    "reb": [
        "reb_avg5", "reb_avg10", "reb_avg15",
        "reb_std5", "reb_std10",
        "reb_hit_40", "reb_hit_45", "reb_hit_50",
        "opp_reb_allowed_avg5", "opp_reb_allowed_avg10",
        "is_home", "line",
    ],
    "ast": [
        "ast_avg5", "ast_avg10", "ast_avg15",
        "ast_std5", "ast_std10",
        "ast_hit_20", "ast_hit_25", "ast_hit_30",
        "opp_ast_allowed_avg5", "opp_ast_allowed_avg10",
        "is_home", "line",
    ],
}


# ─── 3. TRAIN MODEL ──────────────────────────────────────

def train_model(df, stat):
    feature_cols = [c for c in FEATURE_COLS[stat] if c in df.columns]
    df_clean = df[feature_cols + ["target"]].dropna()
    X = df_clean[feature_cols]
    y = df_clean["target"]

    print(f"\n[{stat.upper()} Model] {len(X)} samples | {y.mean():.1%} overs")

    if len(X) < 100:
        print(f"  Not enough data — skipping.")
        return None, None, feature_cols, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    base = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cv  = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

    print(f"  Accuracy : {acc:.3f}")
    print(f"  ROC-AUC  : {auc:.3f}")
    print(f"  CV Mean  : {cv.mean():.3f} +/- {cv.std():.3f}")

    metrics = {
        "stat": stat, "test_accuracy": round(acc, 4), "roc_auc": round(auc, 4),
        "cv_mean": round(cv.mean(), 4), "cv_std": round(cv.std(), 4),
        "n_samples": len(X), "trained_at": datetime.now().isoformat(),
    }
    return model, scaler, feature_cols, metrics

def save_model(model, scaler, feature_cols, metrics, stat):
    joblib.dump(model,  f"{MODEL_DIR}/model_{stat}.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_{stat}.pkl")
    with open(f"{MODEL_DIR}/features_{stat}.json", "w") as f:
        json.dump(feature_cols, f)
    with open(f"{MODEL_DIR}/metrics_{stat}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [Saved] models/model_{stat}.pkl")

def load_model(stat):
    model  = joblib.load(f"{MODEL_DIR}/model_{stat}.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler_{stat}.pkl")
    with open(f"{MODEL_DIR}/features_{stat}.json") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

def predict_prop(player_features, line, stat, model, scaler, feature_cols):
    row = {**player_features, "line": line}
    X = pd.DataFrame([row])
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols].fillna(0)
    return round(float(model.predict_proba(scaler.transform(X))[0][1]), 4)


# ─── 4. MAIN ─────────────────────────────────────────────

def run_training():
    print("=" * 60)
    print("EDGEMAKER — MODEL TRAINING")
    print("=" * 60)

    try:
        df = load_historical_data()
    except FileNotFoundError:
        print("\n❌ File not found: data/nba_player_stats_historical.csv")
        print("   Download from: kaggle.com/datasets/nathanlauga/nba-games")
        print("   Rename the games CSV to nba_player_stats_historical.csv")
        print("   Place it inside the data/ folder then run again.")
        return

    for stat in ["pts", "reb", "ast"]:
        df_feat = prep_features(df.copy(), stat)
        model, scaler, feature_cols, metrics = train_model(df_feat, stat)
        if model is not None:
            save_model(model, scaler, feature_cols, metrics, stat)

    print("\n✅ All models trained and saved to models/")

if __name__ == "__main__":
    run_training()
