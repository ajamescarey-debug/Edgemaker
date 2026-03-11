"""
=============================================================
PHASE 2: NBA PARLAY MODEL — TRAIN PROP MODELS
=============================================================
Train three models:
  1. Points O/U model
  2. Rebounds O/U model
  3. Assists O/U model

Uses Kaggle historical data + rolling features from Phase 1.
Run once to train, then load saved models daily.

pip install scikit-learn xgboost pandas numpy joblib
=============================================================
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

MODEL_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. LOAD & PREP HISTORICAL DATA ──────────────────────

def load_historical_data(path="data/nba_player_stats_historical.csv"):
    """
    Load Kaggle historical player game logs.
    Expected columns: player_id, player_name, date, pts, reb, ast,
                      min, fga, fg3a, team_id, opp_team_id, home_away
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    print(f"[Data] Loaded {len(df)} game logs for {df['player_id'].nunique()} players")
    return df

def build_rolling_features(df, stat="pts", windows=[5, 10, 15]):
    """Build rolling average + std features for a stat, per player."""
    group = df.groupby("player_id")[stat]
    for w in windows:
        df[f"{stat}_avg{w}"] = group.transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
        df[f"{stat}_std{w}"] = group.transform(lambda x: x.shift(1).rolling(w, min_periods=3).std())
    return df

def build_hit_rate_features(df, stat="pts", thresholds=[15, 20, 25]):
    """Rolling hit rate (what % of last N games did player exceed threshold)."""
    group = df.groupby("player_id")[stat]
    for thresh in thresholds:
        col = f"{stat}_hit_{thresh}"
        df[col] = group.transform(lambda x: (x.shift(1) > thresh).rolling(10, min_periods=3).mean())
    return df

def prep_training_data(df, target_stat="pts", line_col="line"):
    """
    Build features and binary target (1 = went OVER the line).
    If no line column, use rolling mean as a proxy.
    """
    # Build rolling features
    df = build_rolling_features(df, stat=target_stat)
    df = build_rolling_features(df, stat="min")

    if target_stat == "pts":
        df = build_hit_rate_features(df, stat="pts", thresholds=[15, 20, 25])
        thresholds = [15, 20, 25]
    elif target_stat == "reb":
        df = build_rolling_features(df, stat="reb")
        df = build_hit_rate_features(df, stat="reb", thresholds=[5, 8])
        thresholds = [5, 8]
    elif target_stat == "ast":
        df = build_rolling_features(df, stat="ast")
        df = build_hit_rate_features(df, stat="ast", thresholds=[5, 7])
        thresholds = [5, 7]

    # If we have real lines from odds data, use them
    if line_col in df.columns:
        df["target"] = (df[target_stat] > df[line_col]).astype(int)
        df["line"] = df[line_col]
    else:
        # Use rolling mean as proxy line
        df["line"] = df[f"{target_stat}_avg10"]
        df["target"] = (df[target_stat] > df["line"]).astype(int)

    return df

FEATURE_COLS = {
    "pts": [
        "pts_avg5", "pts_avg10", "pts_avg15",
        "pts_std5", "pts_std10",
        "pts_hit_15", "pts_hit_20", "pts_hit_25",
        "min_avg5", "min_avg10",
        "line"
    ],
    "reb": [
        "reb_avg5", "reb_avg10", "reb_avg15",
        "reb_std5", "reb_std10",
        "reb_hit_5", "reb_hit_8",
        "min_avg5", "min_avg10",
        "line"
    ],
    "ast": [
        "ast_avg5", "ast_avg10", "ast_avg15",
        "ast_std5", "ast_std10",
        "ast_hit_5", "ast_hit_7",
        "min_avg5", "min_avg10",
        "line"
    ],
}


# ─── 2. TRAIN MODEL ──────────────────────────────────────

def train_prop_model(df, stat="pts"):
    """
    Train a calibrated gradient boosting classifier for prop O/U prediction.
    Returns: model, scaler, cv_accuracy, test_auc
    """
    feature_cols = [c for c in FEATURE_COLS[stat] if c in df.columns]
    df_clean = df[feature_cols + ["target"]].dropna()

    X = df_clean[feature_cols]
    y = df_clean["target"]

    print(f"\n[{stat.upper()} Model] Training on {len(X)} samples | {y.mean():.1%} overs")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base model: Gradient Boosting (outperforms logistic for this use case)
    base = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )

    # Calibrate probabilities (critical for EV calculation to be accurate)
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Cross-val
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

    print(f"  Test Accuracy:  {acc:.3f}")
    print(f"  ROC-AUC:        {auc:.3f}")
    print(f"  CV Accuracy:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return model, scaler, feature_cols, {
        "stat": stat,
        "test_accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "n_samples": len(X),
        "trained_at": datetime.now().isoformat(),
    }

def save_model(model, scaler, feature_cols, metrics, stat):
    """Save model artifacts to disk."""
    joblib.dump(model, f"{MODEL_DIR}/model_{stat}.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_{stat}.pkl")
    with open(f"{MODEL_DIR}/features_{stat}.json", "w") as f:
        json.dump(feature_cols, f)
    with open(f"{MODEL_DIR}/metrics_{stat}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] {MODEL_DIR}/model_{stat}.pkl")


# ─── 3. PREDICT (used in Phase 3 daily) ──────────────────

def load_model(stat):
    """Load a saved model for daily inference."""
    model = joblib.load(f"{MODEL_DIR}/model_{stat}.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler_{stat}.pkl")
    with open(f"{MODEL_DIR}/features_{stat}.json") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

def predict_prop(player_features, line, stat, model, scaler, feature_cols):
    """
    Predict P(over) for a player prop.
    player_features: dict from build_player_features() in Phase 1
    line: the book's line (e.g. 22.5 points)
    Returns: probability of going over
    """
    row = {**player_features, "line": line}
    X = pd.DataFrame([row])[feature_cols]
    X_scaled = scaler.transform(X.fillna(X.median()))
    prob_over = model.predict_proba(X_scaled)[0][1]
    return round(prob_over, 4)


# ─── 4. SPREADS MODEL ─────────────────────────────────────

def train_spreads_model(spreads_df):
    """
    Simpler logistic regression for ATS (against the spread) prediction.
    spreads_df expected columns:
      home_team, away_team, spread_line, home_rest_days, away_rest_days,
      home_back_to_back, away_back_to_back, actual_result (1=home covered)
    """
    feature_cols = [
        "spread_line",
        "home_rest_days", "away_rest_days",
        "home_back_to_back", "away_back_to_back",
        "home_pts_avg10", "away_pts_avg10",
        "home_pace_avg10", "away_pace_avg10",
        "home_ats_rate_last10", "away_ats_rate_last10",
    ]
    feature_cols = [c for c in feature_cols if c in spreads_df.columns]
    df_clean = spreads_df[feature_cols + ["actual_result"]].dropna()

    X = df_clean[feature_cols]
    y = df_clean["actual_result"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = CalibratedClassifierCV(
        LogisticRegression(C=1.0, max_iter=500),
        method="sigmoid", cv=5
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n[SPREADS Model] Test Accuracy: {acc:.3f}")

    joblib.dump(model, f"{MODEL_DIR}/model_spreads.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_spreads.pkl")

    return model, scaler


# ─── 5. MAIN ──────────────────────────────────────────────

def run_training():
    print("=" * 60)
    print("NBA PROP MODEL TRAINING")
    print("=" * 60)
    print("\nNote: Requires historical CSV from Kaggle.")
    print("Recommended dataset: 'NBA Player Stats' (game logs 2010-2024)")
    print("https://www.kaggle.com/datasets/nathanlauga/nba-games\n")

    try:
        df = load_historical_data()
    except FileNotFoundError:
        print("❌ Historical data not found at data/nba_player_stats_historical.csv")
        print("   Download from Kaggle and place in the data/ folder.")
        return

    for stat in ["pts", "reb", "ast"]:
        df_prep = prep_training_data(df.copy(), target_stat=stat)
        model, scaler, feature_cols, metrics = train_prop_model(df_prep, stat=stat)
        save_model(model, scaler, feature_cols, metrics, stat)

    print("\n✅ All models trained. Run phase3_ev_parlay.py next.")

if __name__ == "__main__":
    run_training()
