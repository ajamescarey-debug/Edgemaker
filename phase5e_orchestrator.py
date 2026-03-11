"""
=============================================================
PHASE 5E: MASTER ORCHESTRATOR — ALL SIGNALS COMBINED
=============================================================
Runs all Phase 5 enrichments in sequence and produces a
final ranked list of legs with a composite confidence score.

Signal Stack (applied in order):
  1. Base model probability       (Phase 2 ML model)
  2. Defensive matchup factor     (Phase 5a)
  3. Rest/B2B adjustment          (Phase 5b)
  4. Line movement signal         (Phase 5c)
  5. Injury adjustment            (Phase 5d)
  → Final composite probability + confidence score

Daily workflow:
  python phase1_data_pipeline.py
  python phase5c_line_movement.py   ← Run 3x (morning/midday/evening)
  python phase5e_orchestrator.py    ← Run once pre-game (replaces phase3)
=============================================================
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

from phase1_data_pipeline import (
    get_todays_games, build_player_features,
    calculate_ev, calculate_edge, kelly_criterion,
    decimal_to_implied_prob
)
from phase2_train_models import load_model, predict_prop
from phase3_ev_parlay import load_todays_props, build_best_parlay, is_correlated
from phase5a_defensive_ratings import enrich_legs_with_matchup, load_defensive_ratings
from phase5b_rest_flags import build_rest_table, enrich_legs_with_rest, save_rest_table
from phase5c_line_movement import load_all_snapshots, enrich_legs_with_line_movement
from phase5d_injury_report import get_injury_report, enrich_legs_with_injuries, print_injury_summary

TODAY = datetime.now().strftime("%Y-%m-%d")
DATA_DIR = "data"
OUTPUT_DIR = "dashboard/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── MINIMUM THRESHOLDS (post-enrichment) ────────────────
MIN_FINAL_PROB = 0.57     # After all adjustments, must still be 57%+
MIN_FINAL_EDGE = 0.04     # After all adjustments, must still have 4%+ edge
CONFIDENCE_WEIGHTS = {
    "model_base":    0.40,  # Base model weight
    "matchup":       0.20,  # Defensive matchup
    "rest":          0.15,  # Rest/fatigue
    "line_movement": 0.15,  # Sharp money signal
    "injury":        0.10,  # Injury context
}


# ─── LOOKUP TABLES (in production, build from BallDontLie) ───

def build_player_lookups(todays_games, props_df):
    """
    Build player → team and player → position lookups.
    In production: pull from BallDontLie players endpoint.
    Returns dicts for use in enrichment functions.
    """
    # Placeholder lookups — wire these up from BallDontLie API
    # format: { "Nikola Jokic": {"team_id": 7, "team_name": "Denver Nuggets",
    #                             "position": "C", "opp_team_id": 14} }
    player_team_lookup = {}
    player_position_lookup = {}

    # Extract what we can from props data
    if not props_df.empty:
        for _, row in props_df.iterrows():
            player = row.get("player", "")
            if player and player not in player_team_lookup:
                player_team_lookup[player] = row.get("home_team", "")

    return player_team_lookup, player_position_lookup


# ─── CONFIDENCE SCORE CALCULATOR ─────────────────────────

def calculate_confidence_score(leg):
    """
    Composite confidence score 0-100 based on all signal layers.
    Higher = more confident the bet has genuine edge.
    """
    score = 0

    # 1. Base model probability (max 40 pts)
    base_prob = leg.get("model_prob", 0.5)
    score += min((base_prob - 0.5) * 200, 40)  # 60% prob = 20pts, 70% = 40pts

    # 2. Matchup factor (max 20 pts)
    matchup_factor = leg.get("matchup_factor", 0)
    score += min(max(matchup_factor * 5, 0), 20)

    # 3. Rest signal (max 15 pts)
    rest_adj = leg.get("rest_adjustment", 0)
    if rest_adj > 0:
        score += min(rest_adj * 200, 15)
    elif rest_adj < -0.02:
        score -= 10  # Penalise B2B

    # 4. Line movement (max 15 pts)
    lm_signal = leg.get("line_movement_signal", "STABLE")
    sharp_signals = leg.get("sharp_signals", [])
    if "STEAM" in lm_signal and any("CONFIRMS" in s for s in sharp_signals):
        score += 15
    elif "SHARP" in lm_signal and any("CONFIRMS" in s for s in sharp_signals):
        score += 10
    elif any("FADES" in s for s in sharp_signals):
        score -= 12

    # 5. Injury context (max 10 pts)
    teammate_boost = leg.get("teammate_boost", 0)
    player_status = leg.get("player_injury_status", "ACTIVE")
    if teammate_boost > 0.04:
        score += 10
    elif teammate_boost > 0.01:
        score += 5
    if player_status == "QUESTIONABLE":
        score -= 8
    elif player_status == "PROBABLE":
        score -= 2

    return round(min(max(score, 0), 100), 1)


# ─── FINAL LEG SUMMARY ───────────────────────────────────

def summarise_leg(leg):
    """Build a clean summary dict for dashboard output."""
    final_prob = leg.get("model_prob_final", leg["model_prob"])
    odds = leg["odds"]
    edge = round(final_prob - decimal_to_implied_prob(odds), 4)
    ev = round(calculate_ev(final_prob, odds), 4)
    kelly = kelly_criterion(final_prob, odds)

    return {
        # Core
        "player": leg["player"],
        "stat": leg["stat"],
        "side": leg["side"],
        "line": leg["line"],
        "odds": odds,
        "bookmaker": leg.get("bookmaker", ""),
        "matchup": leg.get("matchup", ""),
        # Probabilities
        "model_prob_base": leg["model_prob"],
        "model_prob_final": final_prob,
        "implied_prob": round(decimal_to_implied_prob(odds), 4),
        "edge": edge,
        "ev": ev,
        "kelly_stake": kelly,
        # Signal layers
        "matchup_factor": leg.get("matchup_factor", 0),
        "matchup_label": leg.get("matchup_label", "Neutral"),
        "rest_days": leg.get("rest_days", "?"),
        "opp_rest_days": leg.get("opp_rest_days", "?"),
        "rest_flags": leg.get("rest_flags", []),
        "rest_adjustment": leg.get("rest_adjustment", 0),
        "line_movement_signal": leg.get("line_movement_signal", "STABLE"),
        "sharp_signals": leg.get("sharp_signals", []),
        "line_movement_adjustment": leg.get("line_movement_adjustment", 0),
        "player_injury_status": leg.get("player_injury_status", "ACTIVE"),
        "injury_risk": leg.get("injury_risk", False),
        "teammate_boost": leg.get("teammate_boost", 0),
        "impactful_absences": leg.get("impactful_absences", []),
        # Composite
        "confidence_score": calculate_confidence_score(leg),
        "game_id": leg.get("game_id", ""),
        "team": leg.get("team", ""),
        "defending_team_id": leg.get("defending_team_id"),
    }


# ─── MAIN ORCHESTRATOR ───────────────────────────────────

def run_full_pipeline():
    print("=" * 65)
    print(f"EDGEMAKER PHASE 5 — FULL SIGNAL PIPELINE — {TODAY}")
    print("=" * 65)

    # ── Load base data ──
    try:
        props_df = load_todays_props()
    except FileNotFoundError:
        print("❌ No props data. Run phase1_data_pipeline.py first.")
        return

    todays_games = []
    games_path = f"{DATA_DIR}/games_{TODAY}.json"
    if os.path.exists(games_path):
        with open(games_path) as f:
            todays_games = json.load(f)

    # ── Score base legs (Phase 3 logic) ──
    from phase3_ev_parlay import score_all_props
    qualifying_legs = score_all_props(props_df)
    print(f"\n[Base] {len(qualifying_legs)} legs passed base model filters")

    if not qualifying_legs:
        print("No qualifying legs. Exiting.")
        _save_empty_output()
        return

    # Build lookup tables
    player_team_lookup, player_position_lookup = build_player_lookups(todays_games, props_df)

    # ── Phase 5a: Defensive matchup ──
    print("\n[5a] Applying defensive matchup factors...")
    qualifying_legs = enrich_legs_with_matchup(
        qualifying_legs, player_position_lookup, season=2024
    )

    # ── Phase 5b: Rest flags ──
    print("[5b] Calculating rest days and B2B flags...")
    rest_table = build_rest_table(todays_games)
    save_rest_table(rest_table)
    qualifying_legs = enrich_legs_with_rest(qualifying_legs, rest_table, player_team_lookup)

    # ── Phase 5c: Line movement ──
    print("[5c] Analysing line movement...")
    snapshots = load_all_snapshots("player_points")
    qualifying_legs = enrich_legs_with_line_movement(qualifying_legs, snapshots)

    # ── Phase 5d: Injuries ──
    print("[5d] Checking injury report...")
    injuries = get_injury_report()
    qualifying_legs = enrich_legs_with_injuries(qualifying_legs, player_team_lookup, injuries)

    # ── Apply final thresholds ──
    final_legs = []
    for leg in qualifying_legs:
        final_prob = leg.get("model_prob_final", leg["model_prob"])
        final_edge = final_prob - decimal_to_implied_prob(leg["odds"])
        if final_prob >= MIN_FINAL_PROB and final_edge >= MIN_FINAL_EDGE:
            final_legs.append(leg)

    print(f"\n[Final] {len(final_legs)} legs passed all signal filters")

    # ── Summarise legs ──
    summarised_legs = [summarise_leg(leg) for leg in final_legs]
    summarised_legs.sort(key=lambda x: x["confidence_score"], reverse=True)

    # ── Build parlay ──
    parlay = build_best_parlay(final_legs)

    # ── Print summary ──
    if summarised_legs:
        print(f"\n{'─'*65}")
        print(f"TOP QUALIFYING LEGS (sorted by confidence):")
        print(f"{'─'*65}")
        for leg in summarised_legs[:8]:
            signals = " | ".join(filter(None, [
                leg["matchup_label"],
                ",".join(leg["rest_flags"]) if leg["rest_flags"] else None,
                leg["line_movement_signal"] if leg["line_movement_signal"] != "STABLE" else None,
                f"⚠️ {leg['player_injury_status']}" if leg["injury_risk"] else None,
            ]))
            print(f"  [{leg['confidence_score']:4.0f}] {leg['player']:22} {leg['stat']} {leg['side']} {leg['line']:5} "
                  f"@ {leg['odds']} | Edge {leg['edge']:+.1%} | {signals}")

    if parlay:
        print(f"\n{'─'*65}")
        print(f"🎯 RECOMMENDED PARLAY — {parlay['num_legs']} LEGS @ {parlay['combined_odds']}x")
        print(f"   Win Prob: {parlay['combined_prob']:.1%} | EV: {parlay['combined_ev']:+.3f} | Kelly: {parlay['kelly_stake_pct']:.1%}")
        print(f"{'─'*65}")
        for leg in parlay["legs"]:
            s = summarise_leg(leg)
            print(f"   ✓ {s['player']:22} {s['stat']} {s['side']} {s['line']} @ {s['odds']} | Conf: {s['confidence_score']:.0f}/100")

    # ── Injury summary ──
    game_teams = [g.get("home_team", {}).get("full_name", "") for g in todays_games] + \
                 [g.get("visitor_team", {}).get("full_name", "") for g in todays_games]
    print_injury_summary(injuries, game_teams)

    # ── Save output ──
    history = _load_history()
    output = {
        "date": TODAY,
        "generated_at": datetime.now().isoformat(),
        "pipeline_version": "5.0",
        "qualifying_legs": summarised_legs,
        "recommended_parlay": parlay,
        "history": history,
        "injury_report": [i for i in injuries if i["status"] != "ACTIVE"],
        "signal_summary": {
            "base_model_legs": len(qualifying_legs),
            "final_qualifying_legs": len(final_legs),
            "sharp_confirmed_legs": sum(1 for l in summarised_legs if any("CONFIRMS" in s for s in l["sharp_signals"])),
            "b2b_flagged_legs": sum(1 for l in summarised_legs if "B2B" in l["rest_flags"]),
            "injury_risk_legs": sum(1 for l in summarised_legs if l["injury_risk"]),
        },
        "summary": {
            "total_qualifying_legs": len(summarised_legs),
            "parlay_found": parlay is not None,
            "parlay_odds": parlay["combined_odds"] if parlay else None,
            "parlay_ev": parlay["combined_ev"] if parlay else None,
        },
        "model_metrics": _load_model_metrics(),
    }

    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Output saved to {OUTPUT_DIR}/results.json")
    print("   Push to GitHub → Netlify auto-deploys dashboard")
    return output


# ─── HELPERS ─────────────────────────────────────────────

def _load_history():
    path = f"{OUTPUT_DIR}/history.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def _load_model_metrics():
    metrics = {}
    for stat in ["pts", "reb", "ast"]:
        path = f"models/metrics_{stat}.json"
        if os.path.exists(path):
            with open(path) as f:
                metrics[stat] = json.load(f)
    return metrics

def _save_empty_output():
    output = {
        "date": TODAY,
        "generated_at": datetime.now().isoformat(),
        "qualifying_legs": [],
        "recommended_parlay": None,
        "history": _load_history(),
        "summary": {"total_qualifying_legs": 0, "parlay_found": False},
    }
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    run_full_pipeline()
