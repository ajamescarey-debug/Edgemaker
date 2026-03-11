"""
=============================================================
PHASE 1: NBA PARLAY MODEL — DATA PIPELINE
=============================================================
Run this daily before games to pull fresh data.
Requires: pip install requests pandas numpy python-dotenv

API Keys needed:
  - The Odds API: https://the-odds-api.com (free tier = 500 req/month)
  - BallDontLie: https://www.balldontlie.io (free, no key needed)
=============================================================
"""

import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────
ODDS_API_KEY       = os.getenv("ODDS_API_KEY", "")
BALLDONTLIE_KEY    = os.getenv("BALLDONTLIE_API_KEY", "")
BDL_HEADERS        = {"Authorization": BALLDONTLIE_KEY}
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
TODAY = datetime.now().strftime("%Y-%m-%d")

# ─── 1. BALLDONTLIE — PLAYER STATS ────────────────────────

def get_todays_games():
    url = "https://api.balldontlie.io/v1/games"
    params = {"start_date": TODAY, "end_date": TODAY, "per_page": 30}
    resp = requests.get(url, params=params, headers=BDL_HEADERS)
    games = resp.json().get("data", [])
    print(f"[BallDontLie] Found {len(games)} games today ({TODAY})")
    return games

def get_player_season_stats(player_id, season=2024):
    url = "https://api.balldontlie.io/v1/season_averages"
    params = {"season": season, "player_ids[]": player_id}
    resp = requests.get(url, params=params, headers=BDL_HEADERS)
    data = resp.json().get("data", [])
    return data[0] if data else None

def get_recent_player_games(player_id, n=15):
    url = "https://api.balldontlie.io/v1/stats"
    params = {"player_ids[]": player_id, "per_page": n, "seasons[]": 2024}
    resp = requests.get(url, params=params, headers=BDL_HEADERS)
    data = resp.json().get("data", [])
    return data

def get_team_players(team_id):
    """Get active players for a team."""
    url = "https://www.balldontlie.io/api/v1/players"
    params = {"team_ids[]": team_id, "per_page": 25}
    resp = requests.get(url, params=params)
    return resp.json().get("data", [])

def build_player_features(player_id, player_name):
    """
    Build rolling features for a player used in the prop model.
    Returns a feature dict.
    """
    recent = get_recent_player_games(player_id, n=15)
    if len(recent) < 5:
        return None

    df = pd.DataFrame([{
        "pts": g["pts"] or 0,
        "reb": g["reb"] or 0,
        "ast": g["ast"] or 0,
        "min": float(g["min"].split(":")[0]) if g.get("min") and ":" in str(g["min"]) else 0,
        "fga": g["fga"] or 0,
        "fg3a": g["fg3a"] or 0,
    } for g in recent])

    features = {
        "player_id": player_id,
        "player_name": player_name,
        # Rolling averages
        "pts_avg5": df["pts"].head(5).mean(),
        "pts_avg10": df["pts"].head(10).mean(),
        "pts_avg15": df["pts"].mean(),
        "reb_avg5": df["reb"].head(5).mean(),
        "reb_avg10": df["reb"].head(10).mean(),
        "reb_avg15": df["reb"].mean(),
        "ast_avg5": df["ast"].head(5).mean(),
        "ast_avg10": df["ast"].head(10).mean(),
        "ast_avg15": df["ast"].mean(),
        # Consistency (std dev — lower = more predictable)
        "pts_std5": df["pts"].head(5).std(),
        "reb_std5": df["reb"].head(5).std(),
        "ast_std5": df["ast"].head(5).std(),
        # Usage proxy
        "avg_minutes": df["min"].mean(),
        "avg_fga": df["fga"].mean(),
        # Hit rates (useful for threshold modelling)
        "pts_over_15_rate": (df["pts"] > 15).mean(),
        "pts_over_20_rate": (df["pts"] > 20).mean(),
        "pts_over_25_rate": (df["pts"] > 25).mean(),
        "reb_over_5_rate": (df["reb"] > 5).mean(),
        "reb_over_8_rate": (df["reb"] > 8).mean(),
        "ast_over_5_rate": (df["ast"] > 5).mean(),
        "ast_over_7_rate": (df["ast"] > 7).mean(),
    }
    return features


# ─── 2. THE ODDS API — PLAYER PROPS + SPREADS ─────────────

def get_nba_odds_spreads():
    """Pull today's NBA game spreads."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "au",          # change to 'us' if needed
        "markets": "spreads,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"[OddsAPI] Error: {resp.status_code} — {resp.text}")
        return []
    games = resp.json()
    print(f"[OddsAPI] Pulled spreads for {len(games)} NBA games")
    return games

def get_nba_player_props(event_id, market="player_points"):
    """
    Pull player props for a specific game.
    Markets: player_points, player_rebounds, player_assists
    """
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "au",
        "markets": market,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"[OddsAPI] Props error: {resp.status_code}")
        return None
    return resp.json()

def extract_props_from_event(event_data, market_key):
    """
    Parse prop data from Odds API response into a clean dataframe.
    Returns rows: player_name, line, over_odds, under_odds, bookmaker
    """
    rows = []
    for bookmaker in event_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market["key"] != market_key:
                continue
            for outcome in market.get("outcomes", []):
                rows.append({
                    "bookmaker": bookmaker["title"],
                    "player": outcome.get("description", outcome.get("name", "")),
                    "side": outcome["name"],  # Over / Under
                    "line": outcome.get("point", None),
                    "odds": outcome["price"],
                    "market": market_key,
                    "game_id": event_data["id"],
                    "home_team": event_data["home_team"],
                    "away_team": event_data["away_team"],
                    "commence_time": event_data["commence_time"],
                })
    return rows

def get_best_line(props_df, player_name, side="Over"):
    """Find the best odds available for a player prop across books."""
    filtered = props_df[
        (props_df["player"].str.contains(player_name, case=False)) &
        (props_df["side"] == side)
    ]
    if filtered.empty:
        return None
    best = filtered.loc[filtered["odds"].idxmax()]
    return best


# ─── 3. EV CALCULATOR ────────────────────────────────────

def decimal_to_implied_prob(decimal_odds):
    """Convert decimal odds to implied probability (with juice removed)."""
    return 1 / decimal_odds

def calculate_ev(model_prob, decimal_odds):
    """
    Expected Value = (model_prob * profit) - ((1 - model_prob) * stake)
    Normalised to 1 unit stake.
    """
    profit = decimal_odds - 1
    ev = (model_prob * profit) - ((1 - model_prob) * 1)
    return round(ev, 4)

def calculate_edge(model_prob, decimal_odds):
    """Edge = model probability minus book implied probability."""
    implied = decimal_to_implied_prob(decimal_odds)
    return round(model_prob - implied, 4)

def kelly_criterion(model_prob, decimal_odds, fraction=0.25):
    """
    Fractional Kelly stake sizing.
    fraction=0.25 = Quarter Kelly (conservative, recommended).
    Returns recommended stake as % of bankroll.
    """
    b = decimal_odds - 1
    q = 1 - model_prob
    kelly = (model_prob * b - q) / b
    return round(max(kelly * fraction, 0), 4)  # Never negative


# ─── 4. PARLAY BUILDER ───────────────────────────────────

def check_correlation(leg1, leg2):
    """
    Basic correlation check — don't combine legs from same game
    that are positively correlated (e.g. both player points from same team).
    Returns True if legs are safe to combine.
    """
    # Same game check
    if leg1.get("game_id") == leg2.get("game_id"):
        # Same team? Likely correlated
        if leg1.get("team") == leg2.get("team"):
            return False
        # Both overs on same game total = correlated
        if leg1.get("market") == "totals" and leg2.get("market") == "totals":
            return False
    return True

def build_parlay(qualifying_legs, min_legs=2, max_legs=3):
    """
    Build the highest EV parlay from qualifying legs.
    - Only combines uncorrelated legs
    - Returns combined odds, combined probability, and EV
    """
    if len(qualifying_legs) < min_legs:
        print(f"[Parlay] Not enough qualifying legs ({len(qualifying_legs)} found, need {min_legs})")
        return None

    # Sort by edge descending
    legs = sorted(qualifying_legs, key=lambda x: x["edge"], reverse=True)
    
    selected = [legs[0]]
    for leg in legs[1:]:
        if len(selected) >= max_legs:
            break
        # Check correlation with all already selected legs
        if all(check_correlation(leg, s) for s in selected):
            selected.append(leg)

    if len(selected) < min_legs:
        print(f"[Parlay] Could not find {min_legs} uncorrelated legs")
        return None

    # Combined probability = product of individual probs
    combined_prob = 1
    combined_odds = 1
    for leg in selected:
        combined_prob *= leg["model_prob"]
        combined_odds *= leg["odds"]

    combined_ev = calculate_ev(combined_prob, combined_odds)

    parlay = {
        "legs": selected,
        "num_legs": len(selected),
        "combined_odds": round(combined_odds, 2),
        "combined_prob": round(combined_prob, 4),
        "combined_ev": combined_ev,
        "kelly_stake": kelly_criterion(combined_prob, combined_odds),
        "generated_at": datetime.now().isoformat(),
    }
    return parlay


# ─── 5. MAIN PIPELINE ─────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print(f"NBA PARLAY PIPELINE — {TODAY}")
    print("=" * 60)

    # Step 1: Pull today's games
    games = get_todays_games()
    if not games:
        print("No games today. Exiting.")
        return

    # Step 2: Pull odds
    odds_games = get_nba_odds_spreads()

    # Step 3: Pull player props for each game
    all_props = []
    for game in odds_games[:5]:  # Limit to 5 games to save API calls on free tier
        for market in ["player_points", "player_rebounds", "player_assists"]:
            props = get_nba_player_props(game["id"], market=market)
            if props:
                rows = extract_props_from_event(props, market)
                all_props.extend(rows)

    props_df = pd.DataFrame(all_props)
    print(f"\n[Props] Pulled {len(props_df)} prop lines across all games")

    # Save raw data
    props_df.to_csv(f"{DATA_DIR}/props_{TODAY}.csv", index=False)
    print(f"[Saved] {DATA_DIR}/props_{TODAY}.csv")

    # Save games
    with open(f"{DATA_DIR}/games_{TODAY}.json", "w") as f:
        json.dump(games, f, indent=2)
    print(f"[Saved] {DATA_DIR}/games_{TODAY}.json")

    # Save odds
    with open(f"{DATA_DIR}/odds_{TODAY}.json", "w") as f:
        json.dump(odds_games, f, indent=2)
    print(f"[Saved] {DATA_DIR}/odds_{TODAY}.json")

    print("\n✅ Phase 1 complete. Run phase2_train_models.py next.")
    return props_df, games, odds_games


if __name__ == "__main__":
    run_pipeline()
