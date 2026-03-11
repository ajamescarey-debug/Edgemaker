"""
=============================================================
PHASE 5C: LINE MOVEMENT & SHARP MONEY TRACKING
=============================================================
Line movement is one of the best free signals in sports betting.
When sharp bettors (professionals) hammer a side, books move
the line to protect themselves. Following that movement —
especially when it goes AGAINST public consensus — is a
historically profitable strategy.

This module:
  1. Tracks opening vs current line for props and spreads
  2. Calculates line movement direction and magnitude
  3. Compares our book (AU) lines vs Pinnacle (sharpest book)
  4. Flags "sharp" vs "public" movement patterns
  5. Generates a line movement signal to boost/reduce edge

Key Concepts:
  - "Steam" move: sharp, sudden large line movement (strong signal)
  - "Reverse line movement": line moves AGAINST majority bets (sharp signal)
  - Pinnacle: world's sharpest sportsbook, their line = market truth
=============================================================
"""

import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

DATA_DIR = "data"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "YOUR_KEY_HERE")
TODAY = datetime.now().strftime("%Y-%m-%d")

# ─── SIGNAL THRESHOLDS ───────────────────────────────────
STEAM_MOVE_THRESHOLD = 1.5   # Half-point move in < 2 hours = steam
SHARP_EDGE_BOOST = 0.04      # +4% prob boost when sharp move confirms our bet
FADE_PENALTY = 0.05          # -5% prob if sharp money opposes our bet


# ─── 1. SNAPSHOT OPENING LINES ───────────────────────────

def snapshot_current_lines(market="player_points"):
    """
    Pull and save the current lines as a timestamped snapshot.
    Run this multiple times per day to track movement.
    """
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "au,us",      # AU for Betr/Sportsbet + US for Pinnacle
        "markets": market,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"[LineMove] Error: {resp.status_code}")
        return []

    data = resp.json()
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "market": market,
        "games": data,
    }

    # Save timestamped snapshot
    ts = datetime.now().strftime("%H%M")
    path = f"{DATA_DIR}/snapshot_{market}_{TODAY}_{ts}.json"
    with open(path, "w") as f:
        json.dump(snapshot, f)
    print(f"[LineMove] Snapshot saved: {path}")
    return data

def load_all_snapshots(market="player_points"):
    """Load all snapshots for today for movement comparison."""
    snapshots = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if f"snapshot_{market}_{TODAY}" in fname:
            with open(f"{DATA_DIR}/{fname}") as f:
                snapshots.append(json.load(f))
    return snapshots


# ─── 2. EXTRACT PINNACLE LINES ───────────────────────────

PINNACLE_NAMES = ["Pinnacle", "pinnacle"]

def extract_pinnacle_line(game_data, player_name, market_key="player_points"):
    """
    Extract Pinnacle's line for a player. Pinnacle = sharpest market.
    Their line is effectively the true probability before juice.
    """
    for book in game_data.get("bookmakers", []):
        if book["title"] not in PINNACLE_NAMES:
            continue
        for market in book.get("markets", []):
            if market["key"] != market_key:
                continue
            for outcome in market.get("outcomes", []):
                if player_name.lower() in outcome.get("description", "").lower():
                    return {
                        "book": "Pinnacle",
                        "line": outcome.get("point"),
                        "odds": outcome.get("price"),
                        "side": outcome.get("name"),
                    }
    return None

def pinnacle_implied_prob(pinnacle_over_odds, pinnacle_under_odds):
    """
    Calculate true no-vig probability from Pinnacle's lines.
    Pinnacle's margin is ~2-3%, so this is very close to true probability.
    """
    if not pinnacle_over_odds or not pinnacle_under_odds:
        return None, None

    raw_over = 1 / pinnacle_over_odds
    raw_under = 1 / pinnacle_under_odds
    total = raw_over + raw_under

    # Remove the vig
    true_over = raw_over / total
    true_under = raw_under / total
    return round(true_over, 4), round(true_under, 4)


# ─── 3. LINE MOVEMENT ANALYSIS ───────────────────────────

def parse_lines_from_snapshot(snapshot, player_name, market_key):
    """Extract all book lines for a player from a snapshot."""
    lines = {}
    for game in snapshot.get("games", []):
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != market_key:
                    continue
                for outcome in market.get("outcomes", []):
                    if player_name.lower() in outcome.get("description", "").lower():
                        key = f"{book['title']}_{outcome['name']}"
                        lines[key] = {
                            "book": book["title"],
                            "side": outcome["name"],
                            "line": outcome.get("point"),
                            "odds": outcome.get("price"),
                            "timestamp": snapshot.get("timestamp"),
                        }
    return lines

def calculate_line_movement(snapshots, player_name, market_key="player_points"):
    """
    Compare earliest vs latest snapshot to find line movement.
    Returns movement summary for a player.
    """
    if len(snapshots) < 2:
        return {"movement": 0, "signal": "INSUFFICIENT_DATA", "direction": None}

    earliest = parse_lines_from_snapshot(snapshots[0], player_name, market_key)
    latest = parse_lines_from_snapshot(snapshots[-1], player_name, market_key)

    movements = []
    for key in latest:
        if key in earliest:
            open_odds = earliest[key]["odds"]
            curr_odds = latest[key]["odds"]
            open_line = earliest[key].get("line")
            curr_line = latest[key].get("line")

            if open_line and curr_line:
                line_move = curr_line - open_line
            else:
                line_move = 0

            odds_move = curr_odds - open_odds if open_odds and curr_odds else 0

            movements.append({
                "book": latest[key]["book"],
                "side": latest[key]["side"],
                "open_line": open_line,
                "current_line": curr_line,
                "line_move": round(line_move, 2),
                "open_odds": open_odds,
                "current_odds": curr_odds,
                "odds_move": round(odds_move, 3),
            })

    if not movements:
        return {"movement": 0, "signal": "NO_DATA", "direction": None}

    # Average line movement across books
    avg_move = np.mean([abs(m["line_move"]) for m in movements])
    over_moves = [m for m in movements if m["side"] == "Over"]
    avg_over_line_move = np.mean([m["line_move"] for m in over_moves]) if over_moves else 0

    # Classify signal
    signal = classify_movement_signal(avg_move, avg_over_line_move)

    return {
        "player": player_name,
        "movements": movements,
        "avg_line_move": round(float(avg_over_line_move), 3),
        "movement_magnitude": round(float(avg_move), 3),
        "signal": signal,
        "direction": "over" if avg_over_line_move < 0 else "under",  # line UP = harder to go over
        "sharp_money_on_over": avg_over_line_move < -0.3,  # line dropped = sharp money on Over
        "sharp_money_on_under": avg_over_line_move > 0.3,  # line rose = sharp money on Under
    }

def classify_movement_signal(magnitude, direction_move):
    """Classify the line movement as a sharp signal."""
    if magnitude >= STEAM_MOVE_THRESHOLD:
        return "STEAM"         # Strong sharp movement
    elif magnitude >= 0.5:
        return "SHARP"         # Meaningful movement
    elif magnitude >= 0.1:
        return "MODERATE"      # Some action
    else:
        return "STABLE"        # Line unchanged, no information


# ─── 4. BOOK COMPARISON (AU vs SHARP) ────────────────────

def compare_au_vs_pinnacle(au_odds, pinnacle_true_prob, side="over"):
    """
    Compare what you're getting in AU books vs what Pinnacle
    says the true probability is.

    A positive "book edge" means AU books are offering better odds
    than the sharp market price (value opportunity).
    """
    if not au_odds or not pinnacle_true_prob:
        return {"book_edge": 0, "signal": "NO_PINNACLE_DATA"}

    au_implied = 1 / au_odds
    book_edge = pinnacle_true_prob - au_implied  # Positive = you're getting value

    return {
        "au_implied_prob": round(au_implied, 4),
        "pinnacle_true_prob": pinnacle_true_prob,
        "book_edge": round(book_edge, 4),
        "is_value": book_edge > 0.02,  # 2%+ edge vs Pinnacle = genuine value
        "signal": "VALUE" if book_edge > 0.02 else "FAIR" if book_edge > -0.02 else "OVERPRICED",
    }


# ─── 5. COMBINED LINE MOVEMENT SIGNAL ────────────────────

def get_line_movement_signal(leg, snapshots, pinnacle_game_data=None):
    """
    Master function: calculate the full line movement signal for a leg.
    Returns probability adjustment and metadata.
    """
    player = leg["player"]
    stat = leg["stat"].lower()
    side = leg["side"]
    market_map = {"pts": "player_points", "reb": "player_rebounds", "ast": "player_assists"}
    market_key = market_map.get(stat, "player_points")

    result = {
        "player": player,
        "line_movement": None,
        "pinnacle_comparison": None,
        "prob_adjustment": 0.0,
        "signals": [],
    }

    # Line movement analysis
    movement = calculate_line_movement(snapshots, player, market_key)
    result["line_movement"] = movement

    # Check if movement confirms or contradicts our bet
    if movement["signal"] in ("STEAM", "SHARP"):
        our_side = side.lower()
        sharp_on_our_side = (
            (our_side == "over" and movement["sharp_money_on_over"]) or
            (our_side == "under" and movement["sharp_money_on_under"])
        )
        if sharp_on_our_side:
            result["prob_adjustment"] += SHARP_EDGE_BOOST
            result["signals"].append(f"SHARP_CONFIRMS_{our_side.upper()}")
        else:
            result["prob_adjustment"] -= FADE_PENALTY
            result["signals"].append(f"SHARP_FADES_{our_side.upper()}")

    # Pinnacle comparison
    if pinnacle_game_data:
        pinn_over = extract_pinnacle_line(pinnacle_game_data, player, market_key)
        # Would need both over and under for proper de-vig calc
        # Placeholder for when Pinnacle data is available
        result["pinnacle_comparison"] = {"available": pinn_over is not None}

    result["prob_adjustment"] = round(result["prob_adjustment"], 4)
    return result


# ─── 6. ENRICH LEGS WITH LINE MOVEMENT ───────────────────

def enrich_legs_with_line_movement(qualifying_legs, snapshots):
    """Add line movement signals to qualifying legs."""
    enriched = []
    for leg in qualifying_legs:
        signal = get_line_movement_signal(leg, snapshots)
        adj = signal["prob_adjustment"]

        leg["line_movement_signal"] = signal["line_movement"].get("signal", "N/A")
        leg["line_movement_direction"] = signal["line_movement"].get("direction", "N/A")
        leg["line_movement_adjustment"] = adj
        leg["sharp_signals"] = signal["signals"]

        # Apply to final probability
        base = leg.get("model_prob_final", leg.get("model_prob_adjusted", leg["model_prob"]))
        leg["model_prob_final"] = round(min(max(base + adj, 0.01), 0.99), 4)

        enriched.append(leg)

    return enriched


# ─── 7. DAILY SNAPSHOT SCHEDULER ─────────────────────────

def run_snapshot_schedule():
    """
    Call this script 3x per day to build movement history:
      - Morning (10am): opening line snapshot
      - Afternoon (2pm): midday snapshot
      - Evening (5pm): pre-game snapshot
    """
    for market in ["player_points", "player_rebounds", "player_assists", "spreads"]:
        snapshot_current_lines(market)
    print(f"\n✅ Snapshot complete at {datetime.now().strftime('%H:%M')}")


if __name__ == "__main__":
    run_snapshot_schedule()
