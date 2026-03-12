"""
=============================================================
PHASE 1: NBA PARLAY MODEL — DATA PIPELINE
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

ODDS_API_KEY    = os.getenv("ODDS_API_KEY", "")
BALLDONTLIE_KEY = os.getenv("BALLDONTLIE_API_KEY", "")
BDL_HEADERS     = {"Authorization": BALLDONTLIE_KEY}
DATA_DIR        = "data"
os.makedirs(DATA_DIR, exist_ok=True)
TODAY = datetime.now().strftime("%Y-%m-%d")


def get_todays_games():
    url    = "https://api.balldontlie.io/v1/games"
    params = {"start_date": TODAY, "end_date": TODAY, "per_page": 30}
    try:
        resp = requests.get(url, params=params, headers=BDL_HEADERS, timeout=10)
        print(f"[BallDontLie] Status: {resp.status_code}")
        print(f"[BallDontLie] Response: {resp.text[:200]}")
        if resp.status_code == 401:
            print("[BallDontLie] Auth error — check BALLDONTLIE_API_KEY secret")
            return []
        if resp.status_code != 200:
            print(f"[BallDontLie] Error {resp.status_code}")
            return []
        games = resp.json().get("data", [])
        print(f"[BallDontLie] Found {len(games)} games today ({TODAY})")
        return games
    except Exception as e:
        print(f"[BallDontLie] Exception: {e}")
        return []


def get_player_season_stats(player_id, season=2024):
    url    = "https://api.balldontlie.io/v1/season_averages"
    params = {"season": season, "player_ids[]": player_id}
    try:
        resp = requests.get(url, params=params, headers=BDL_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        return data[0] if data else None
    except Exception:
        return None


def get_recent_player_games(player_id, n=15):
    url    = "https://api.balldontlie.io/v1/stats"
    params = {"player_ids[]": player_id, "per_page": n, "seasons[]": 2024}
    try:
        resp = requests.get(url, params=params, headers=BDL_HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        return resp.json().get("data", [])
    except Exception:
        return []


def get_team_players(team_id):
    url    = "https://api.balldontlie.io/v1/players"
    params = {"team_ids[]": team_id, "per_page": 25}
    try:
        resp = requests.get(url, params=params, headers=BDL_HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        return resp.json().get("data", [])
    except Exception:
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


def build_player_features(player_id, player_name):
    recent = get_recent_player_games(player_id, n=15)
    if len(recent) < 5:
        return None
    df = pd.DataFrame([{
        "pts":  g.get("pts") or 0,
        "reb":  g.get("reb") or 0,
        "ast":  g.get("ast") or 0,
        "min":  float(str(g.get("min", "0")).split(":")[0]) if g.get("min") else 0,
        "fga":  g.get("fga") or 0,
        "fg3a": g.get("fg3a") or 0,
    } for g in recent])
    return {
        "player_id":   player_id,
        "player_name": player_name,
        "pts_avg5":    df["pts"].head(5).mean(),
        "pts_avg10":   df["pts"].head(10).mean(),
        "pts_avg15":   df["pts"].mean(),
        "pts_std5":    df["pts"].head(5).std(),
        "pts_std10":   df["pts"].head(10).std(),
        "reb_avg5":    df["reb"].head(5).mean(),
        "reb_avg10":   df["reb"].head(10).mean(),
        "reb_avg15":   df["reb"].mean(),
        "reb_std5":    df["reb"].head(5).std(),
        "reb_std10":   df["reb"].head(10).std(),
        "ast_avg5":    df["ast"].head(5).mean(),
        "ast_avg10":   df["ast"].head(10).mean(),
        "ast_avg15":   df["ast"].mean(),
        "ast_std5":    df["ast"].head(5).std(),
        "ast_std10":   df["ast"].head(10).std(),
        "avg_minutes": df["min"].mean(),
        "pts_hit_15":  (df["pts"] > 15).mean(),
        "pts_hit_20":  (df["pts"] > 20).mean(),
        "pts_hit_25":  (df["pts"] > 25).mean(),
        "reb_hit_5":   (df["reb"] > 5).mean(),
        "reb_hit_8":   (df["reb"] > 8).mean(),
        "ast_hit_5":   (df["ast"] > 5).mean(),
        "ast_hit_7":   (df["ast"] > 7).mean(),
        "is_home":     0,
        "line":        df["pts"].mean(),
    }


def get_nba_odds_spreads():
    url    = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "spreads,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"[OddsAPI] Error: {resp.status_code} — {resp.text[:200]}")
            return []
        games = resp.json()
        print(f"[OddsAPI] Pulled spreads for {len(games)} NBA games")
        return games
    except Exception as e:
        print(f"[OddsAPI] Exception: {e}")
        return []


def get_nba_player_props(event_id, market="player_points"):
    url    = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    market,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def extract_props_from_event(event_data, market_key):
    rows = []
    for bookmaker in event_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market["key"] != market_key:
                continue
            for outcome in market.get("outcomes", []):
                rows.append({
                    "bookmaker":     bookmaker["title"],
                    "player":        outcome.get("description", outcome.get("name", "")),
                    "side":          outcome["name"],
                    "line":          outcome.get("point"),
                    "odds":          outcome["price"],
                    "market":        market_key,
                    "game_id":       event_data["id"],
                    "home_team":     event_data["home_team"],
                    "away_team":     event_data["away_team"],
                    "commence_time": event_data["commence_time"],
                })
    return rows


def decimal_to_implied_prob(decimal_odds):
    return round(1 / decimal_odds, 4)

def calculate_ev(model_prob, decimal_odds):
    profit = decimal_odds - 1
    return round((model_prob * profit) - ((1 - model_prob) * 1), 4)

def calculate_edge(model_prob, decimal_odds):
    return round(model_prob - (1 / decimal_odds), 4)

def kelly_criterion(model_prob, decimal_odds, fraction=0.25):
    b = decimal_odds - 1
    q = 1 - model_prob
    kelly = (model_prob * b - q) / b
    return round(max(kelly * fraction, 0), 4)


def check_correlation(leg1, leg2):
    if leg1.get("game_id") == leg2.get("game_id"):
        if leg1.get("team") == leg2.get("team"):
            return False
        if leg1.get("market") == "totals" and leg2.get("market") == "totals":
            return False
    return True

def build_parlay(qualifying_legs, min_legs=2, max_legs=3):
    if len(qualifying_legs) < min_legs:
        print(f"[Parlay] Not enough qualifying legs ({len(qualifying_legs)} found, need {min_legs})")
        return None
    legs     = sorted(qualifying_legs, key=lambda x: x["edge"], reverse=True)
    selected = [legs[0]]
    for leg in legs[1:]:
        if len(selected) >= max_legs:
            break
        if all(check_correlation(leg, s) for s in selected):
            selected.append(leg)
    if len(selected) < min_legs:
        return None
    combined_prob = 1
    combined_odds = 1
    for leg in selected:
        combined_prob *= leg["model_prob"]
        combined_odds *= leg["odds"]
    return {
        "legs":            selected,
        "num_legs":        len(selected),
        "combined_odds":   round(combined_odds, 2),
        "combined_prob":   round(combined_prob, 4),
        "combined_ev":     calculate_ev(combined_prob, combined_odds),
        "kelly_stake":     kelly_criterion(combined_prob, combined_odds),
        "kelly_stake_pct": kelly_criterion(combined_prob, combined_odds),
        "generated_at":    datetime.now().isoformat(),
    }


def run_pipeline():
    print("=" * 60)
    print(f"NBA PARLAY PIPELINE — {TODAY}")
    print("=" * 60)

    games = get_todays_games()
    if not games:
        print("[Pipeline] No games found — writing empty state")
        with open(f"{DATA_DIR}/games_{TODAY}.json", "w") as f:
            json.dump([], f)
        return None, [], []

    odds_games = get_nba_odds_spreads()

    all_props = []
    for game in odds_games[:5]:
        for market in ["player_points", "player_rebounds", "player_assists"]:
            props = get_nba_player_props(game["id"], market=market)
            if props:
                rows = extract_props_from_event(props, market)
                all_props.extend(rows)

    props_df = pd.DataFrame(all_props) if all_props else pd.DataFrame()
    print(f"\n[Props] Pulled {len(props_df)} prop lines")

    if not props_df.empty:
        props_df.to_csv(f"{DATA_DIR}/props_{TODAY}.csv", index=False)

    with open(f"{DATA_DIR}/games_{TODAY}.json", "w") as f:
        json.dump(games, f, indent=2)

    with open(f"{DATA_DIR}/odds_{TODAY}.json", "w") as f:
        json.dump(odds_games, f, indent=2)

    print("\n✅ Phase 1 complete.")
    return props_df, games, odds_games


if __name__ == "__main__":
    run_pipeline()
