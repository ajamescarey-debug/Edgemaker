"""
=============================================================
PHASE 3: NBA PARLAY MODEL — DAILY EV + PARLAY BUILDER
=============================================================
Run daily after Phase 1 (data pull).
Loads models from Phase 2, scores today's props,
finds EV+ legs, builds best parlay, saves to JSON for dashboard.
=============================================================
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from phase1_data_pipeline import (
    get_todays_games, build_player_features,
    calculate_ev, calculate_edge, kelly_criterion,
    decimal_to_implied_prob
)

def get_team_players(team_id):
    return []

def get_best_line(props_df, player_name, side="Over"):
    if props_df is None or props_df.empty:
        return None
    try:
        filtered = props_df[
            (props_df["player"].str.contains(player_name, case=False, na=False)) &
            (props_df["side"] == side)
        ]
        if filtered.empty:
            return None
        return filtered.loc[filtered["odds"].idxmax()]
    except Exception:
        return None
from phase2_train_models import load_model, predict_prop

TODAY = datetime.now().strftime("%Y-%m-%d")
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "dashboard/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── CONFIG ───────────────────────────────────────────────
MIN_EDGE = 0.05          # Minimum edge to qualify (5%)
MAX_EDGE = 0.25          # Cap suspicious edges (model error)
MIN_CONFIDENCE = 0.55    # Minimum model probability
MAX_LEGS = 3             # Maximum parlay legs
MIN_LEGS = 2             # Minimum to output a parlay


# ─── 1. LOAD TODAY'S DATA ────────────────────────────────

def load_todays_props():
    """Load props pulled in Phase 1."""
    path = f"{DATA_DIR}/props_{TODAY}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No props file for {TODAY}. Run phase1_data_pipeline.py first.")
    df = pd.read_csv(path)
    print(f"[Data] Loaded {len(df)} prop lines for {TODAY}")
    return df

def load_todays_games():
    """Load games pulled in Phase 1."""
    with open(f"{DATA_DIR}/games_{TODAY}.json") as f:
        return json.load(f)


# ─── 2. SCORE PROPS ──────────────────────────────────────

STAT_MARKET_MAP = {
    "pts": "player_points",
    "reb": "player_rebounds",
    "ast": "player_assists",
}

def score_all_props(props_df):
    """
    For each player prop, get model probability and calculate EV.
    Returns list of qualifying leg dicts.
    """
    qualifying_legs = []
    processed = set()

    for stat, market in STAT_MARKET_MAP.items():
        try:
            model, scaler, feature_cols = load_model(stat)
        except FileNotFoundError:
            print(f"[Warning] No trained model for {stat}. Run phase2 first.")
            continue

        market_props = props_df[props_df["market"] == market].copy()
        if market_props.empty:
            continue

        # Get unique players with Over lines
        players_with_lines = market_props[market_props["side"] == "Over"].groupby("player")["line"].mean().to_dict()

        for player_name, line in players_with_lines.items():
            key = f"{player_name}_{stat}_{line}"
            if key in processed:
                continue
            processed.add(key)

            if pd.isna(line):
                continue

            # Get best available odds for this player
            best_over = props_df[
                (props_df["market"] == market) &
                (props_df["player"].str.contains(player_name, case=False, na=False)) &
                (props_df["side"] == "Over")
            ]
            best_under = props_df[
                (props_df["market"] == market) &
                (props_df["player"].str.contains(player_name, case=False, na=False)) &
                (props_df["side"] == "Under")
            ]

            if best_over.empty:
                continue

            best_over_odds = best_over["odds"].max()
            best_under_odds = best_under["odds"].max() if not best_under.empty else None
            best_bookmaker = best_over.loc[best_over["odds"].idxmax(), "bookmaker"]
            game_id = best_over.iloc[0]["game_id"]
            home_team = best_over.iloc[0]["home_team"]
            away_team = best_over.iloc[0]["away_team"]

            # Build player features (rolling stats from BallDontLie)
            # NOTE: In production, you'd cache these. Here we use a simplified approach.
            # For full pipeline, player_id lookup would be needed.
            # Using available rolling data from props_df as proxy
            player_features = _estimate_player_features(props_df, player_name, stat, line)
            if player_features is None:
                continue

            try:
                model_prob_over = predict_prop(player_features, line, stat, model, scaler, feature_cols)
            except Exception as e:
                continue

            # Over analysis
            edge_over = calculate_edge(model_prob_over, best_over_odds)
            ev_over = calculate_ev(model_prob_over, best_over_odds)

            # Under analysis
            model_prob_under = 1 - model_prob_over
            if best_under_odds:
                edge_under = calculate_edge(model_prob_under, best_under_odds)
                ev_under = calculate_ev(model_prob_under, best_under_odds)
            else:
                edge_under = -1
                ev_under = -1

            # Determine best side
            if edge_over > edge_under and edge_over >= MIN_EDGE and model_prob_over >= MIN_CONFIDENCE:
                side = "Over"
                edge = edge_over
                ev = ev_over
                odds = best_over_odds
                model_prob = model_prob_over
            elif edge_under >= MIN_EDGE and model_prob_under >= MIN_CONFIDENCE:
                side = "Under"
                edge = edge_under
                ev = ev_under
                odds = best_under_odds
                model_prob = model_prob_under
            else:
                continue  # No edge found

            if edge > MAX_EDGE:
                continue  # Too good = likely model error

            leg = {
                "player": player_name,
                "stat": stat.upper(),
                "side": side,
                "line": line,
                "odds": round(odds, 2),
                "model_prob": round(model_prob, 4),
                "implied_prob": round(decimal_to_implied_prob(odds), 4),
                "edge": round(edge, 4),
                "ev": round(ev, 4),
                "kelly_stake": kelly_criterion(model_prob, odds),
                "bookmaker": best_bookmaker,
                "game_id": game_id,
                "matchup": f"{away_team} @ {home_team}",
                "market": market,
                "team": _guess_team(player_name, home_team, away_team),
            }
            qualifying_legs.append(leg)

    qualifying_legs.sort(key=lambda x: x["edge"], reverse=True)
    print(f"\n[EV] Found {len(qualifying_legs)} qualifying legs (edge ≥ {MIN_EDGE:.0%}, confidence ≥ {MIN_CONFIDENCE:.0%})")
    return qualifying_legs

def _estimate_player_features(props_df, player_name, stat, line):
    """
    Simplified feature estimation from props data when we don't have
    full BallDontLie rolling stats cached. Uses line as a proxy.
    In production, replace with actual rolling stats from Phase 1.
    """
    # Placeholder — returns basic features using line as avg proxy
    # Real implementation: call build_player_features(player_id) from Phase 1
    return {
        f"{stat}_avg5": line,
        f"{stat}_avg10": line,
        f"{stat}_avg15": line,
        f"{stat}_std5": line * 0.2,
        f"{stat}_std10": line * 0.2,
        f"{stat}_hit_{int(line)}": 0.5,
        "min_avg5": 30,
        "min_avg10": 30,
    }

def _guess_team(player_name, home_team, away_team):
    """Placeholder — in production use player-team lookup."""
    return "Unknown"


# ─── 3. CORRELATION CHECK ────────────────────────────────

def is_correlated(leg1, leg2):
    """
    Legs are considered correlated if:
    - Same game AND same team (e.g. two players on same team going Over)
    - Same game total from different angles
    - Same player different stats (always correlated)
    """
    if leg1["player"] == leg2["player"]:
        return True  # Same player

    same_game = leg1["game_id"] == leg2["game_id"]
    same_team = leg1["team"] == leg2["team"] and leg1["team"] != "Unknown"

    if same_game and same_team:
        return True

    # Both team-total correlated markets in same game
    if same_game and leg1["stat"] in ["PTS"] and leg2["stat"] in ["PTS"]:
        if leg1["side"] == leg2["side"] == "Over":
            return True  # Both overs on same game = correlated

    return False


# ─── 4. BUILD PARLAY ─────────────────────────────────────

def build_best_parlay(qualifying_legs):
    """Select best uncorrelated legs and build parlay."""
    if len(qualifying_legs) < MIN_LEGS:
        return None

    selected = []
    for leg in qualifying_legs:
        if len(selected) >= MAX_LEGS:
            break
        if all(not is_correlated(leg, s) for s in selected):
            selected.append(leg)

    if len(selected) < MIN_LEGS:
        return None

    combined_prob = 1.0
    combined_odds = 1.0
    for leg in selected:
        combined_prob *= leg["model_prob"]
        combined_odds *= leg["odds"]

    combined_odds = round(combined_odds, 2)
    combined_prob = round(combined_prob, 4)
    combined_ev = round(calculate_ev(combined_prob, combined_odds), 4)
    kelly = round(kelly_criterion(combined_prob, combined_odds), 4)

    return {
        "legs": selected,
        "num_legs": len(selected),
        "combined_odds": combined_odds,
        "combined_prob": combined_prob,
        "combined_ev": combined_ev,
        "kelly_stake_pct": kelly,
        "profitable": combined_ev > 0,
        "date": TODAY,
        "generated_at": datetime.now().isoformat(),
    }


# ─── 5. LOAD HISTORY + SAVE OUTPUT ───────────────────────

def load_history():
    path = f"{OUTPUT_DIR}/history.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def save_output(qualifying_legs, parlay, history):
    output = {
        "date": TODAY,
        "generated_at": datetime.now().isoformat(),
        "qualifying_legs": qualifying_legs,
        "recommended_parlay": parlay,
        "history": history,
        "summary": {
            "total_qualifying_legs": len(qualifying_legs),
            "parlay_found": parlay is not None,
            "parlay_odds": parlay["combined_odds"] if parlay else None,
            "parlay_ev": parlay["combined_ev"] if parlay else None,
        }
    }
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Saved] {OUTPUT_DIR}/results.json")
    return output


# ─── 6. MAIN ─────────────────────────────────────────────

def run_ev_parlay():
    print("=" * 60)
    print(f"NBA EV + PARLAY BUILDER — {TODAY}")
    print("=" * 60)

    props_df = load_todays_props()
    qualifying_legs = score_all_props(props_df)

    if not qualifying_legs:
        print("\n⚠️  No qualifying legs today. No parlay generated.")
        parlay = None
    else:
        print(f"\nTop 5 qualifying legs:")
        for i, leg in enumerate(qualifying_legs[:5], 1):
            print(f"  {i}. {leg['player']} {leg['stat']} {leg['side']} {leg['line']} "
                  f"@ {leg['odds']} | Edge: {leg['edge']:.1%} | EV: {leg['ev']:+.3f}")

        parlay = build_best_parlay(qualifying_legs)

    if parlay:
        print(f"\n🎯 RECOMMENDED PARLAY ({parlay['num_legs']} legs)")
        print(f"   Combined Odds: {parlay['combined_odds']}x")
        print(f"   Model Prob:    {parlay['combined_prob']:.1%}")
        print(f"   Expected Value: {parlay['combined_ev']:+.4f}")
        print(f"   Kelly Stake:   {parlay['kelly_stake_pct']:.1%} of bankroll")
        for leg in parlay["legs"]:
            print(f"   ✓ {leg['player']} {leg['stat']} {leg['side']} {leg['line']} @ {leg['odds']}")
    else:
        print("\n⚠️  No valid parlay found today.")

    history = load_history()
    output = save_output(qualifying_legs, parlay, history)

    print("\n✅ Phase 3 complete. Dashboard will auto-update from results.json")
    return output


if __name__ == "__main__":
    run_ev_parlay()
