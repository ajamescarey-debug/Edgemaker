"""
=============================================================
PHASE 5A: OPPONENT DEFENSIVE RATINGS PER POSITION
=============================================================
Builds a per-position defensive rating for each NBA team.
"How many pts/reb/ast does this team allow to PGs, SGs, SFs, PFs, Cs?"

This is one of the strongest features for prop modelling —
a player facing a weak defensive team at their position
gets a major probability boost.

Source: BallDontLie API (free)
=============================================================
"""

import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ─── POSITION MAPPING ────────────────────────────────────
# BallDontLie returns positions like "G", "F", "C", "G-F", "F-C"
# We normalise these to primary position buckets

POSITION_MAP = {
    "G": "Guard",
    "G-F": "Guard",
    "F-G": "Guard",
    "F": "Forward",
    "F-C": "Forward",
    "C-F": "Center",
    "C": "Center",
    "": "Unknown",
}

def normalize_position(pos):
    if not pos:
        return "Unknown"
    return POSITION_MAP.get(pos.strip(), "Unknown")


# ─── 1. PULL TEAM GAME LOGS ──────────────────────────────

def get_all_teams():
    """Get all NBA teams."""
    resp = requests.get("https://www.balldontlie.io/api/v1/teams?per_page=30")
    return resp.json().get("data", [])

def get_games_for_season(team_id=None, season=2024, per_page=100):
    """Pull all games for a season, optionally filtered by team."""
    url = "https://www.balldontlie.io/api/v1/games"
    params = {"seasons[]": season, "per_page": per_page}
    if team_id:
        params["team_ids[]"] = team_id

    all_games = []
    page = 1
    while True:
        params["page"] = page
        resp = requests.get(url, params=params)
        data = resp.json()
        games = data.get("data", [])
        if not games:
            break
        all_games.extend(games)
        meta = data.get("meta", {})
        if page >= meta.get("total_pages", 1):
            break
        page += 1

    return all_games

def get_player_game_stats(game_ids, per_page=100):
    """Pull all player stats for a list of game IDs."""
    all_stats = []
    for i in range(0, len(game_ids), 10):  # batch requests
        batch = game_ids[i:i+10]
        url = "https://www.balldontlie.io/api/v1/stats"
        params = {
            "per_page": per_page,
            "game_ids[]": batch,
        }
        resp = requests.get(url, params=params)
        stats = resp.json().get("data", [])
        all_stats.extend(stats)
    return all_stats


# ─── 2. BUILD DEFENSIVE RATINGS ──────────────────────────

def build_opponent_defensive_ratings(season=2024):
    """
    For each team, calculate how many pts/reb/ast they allow
    to opposing players at each position (Guard, Forward, Center).

    Returns: dict keyed by team_id → {position → {pts_allowed, reb_allowed, ast_allowed}}
    """
    print(f"[DefRatings] Building opponent defensive ratings for {season} season...")

    teams = get_all_teams()
    team_lookup = {t["id"]: t["full_name"] for t in teams}

    # Pull all season games
    games = get_games_for_season(season=season)
    game_ids = [g["id"] for g in games]
    print(f"[DefRatings] Found {len(games)} games, pulling player stats...")

    # For large datasets, use saved CSV if exists
    stats_cache = f"{DATA_DIR}/player_game_stats_{season}.json"
    if os.path.exists(stats_cache):
        with open(stats_cache) as f:
            all_stats = json.load(f)
        print(f"[DefRatings] Loaded {len(all_stats)} stats from cache")
    else:
        all_stats = get_player_game_stats(game_ids)
        with open(stats_cache, "w") as f:
            json.dump(all_stats, f)
        print(f"[DefRatings] Pulled {len(all_stats)} player-game stats")

    # Build game → teams lookup
    game_teams = {}
    for g in games:
        game_teams[g["id"]] = {
            "home_team_id": g["home_team"]["id"],
            "away_team_id": g["visitor_team"]["id"],
        }

    # Build position → opponent allowed stats
    # For each stat line: which team was defending this player?
    rows = []
    for s in all_stats:
        if not s.get("game") or not s.get("player") or not s.get("team"):
            continue

        game_id = s["game"]["id"]
        player_team_id = s["team"]["id"]
        position = normalize_position(s["player"].get("position", ""))

        if game_id not in game_teams:
            continue

        # Defending team = the OTHER team in the game
        gt = game_teams[game_id]
        if player_team_id == gt["home_team_id"]:
            defending_team_id = gt["away_team_id"]
        else:
            defending_team_id = gt["home_team_id"]

        rows.append({
            "game_id": game_id,
            "player_id": s["player"]["id"],
            "player_name": f"{s['player']['first_name']} {s['player']['last_name']}",
            "position": position,
            "defending_team_id": defending_team_id,
            "defending_team": team_lookup.get(defending_team_id, "Unknown"),
            "pts": s.get("pts") or 0,
            "reb": s.get("reb") or 0,
            "ast": s.get("ast") or 0,
            "min": float(s["min"].split(":")[0]) if s.get("min") and ":" in str(s["min"]) else 0,
        })

    df = pd.DataFrame(rows)
    # Filter out DNPs (0 minutes)
    df = df[df["min"] > 5]

    # Aggregate: per team, per position — avg stats allowed
    agg = df.groupby(["defending_team_id", "defending_team", "position"]).agg(
        pts_allowed=("pts", "mean"),
        reb_allowed=("reb", "mean"),
        ast_allowed=("ast", "mean"),
        games_sample=("game_id", "count"),
    ).reset_index()

    # Compute league average per position (for relative rating)
    league_avg = df.groupby("position").agg(
        league_pts=("pts", "mean"),
        league_reb=("reb", "mean"),
        league_ast=("ast", "mean"),
    ).reset_index()

    agg = agg.merge(league_avg, on="position", how="left")

    # Defensive rating: positive = team allows MORE than average (easier matchup)
    agg["pts_def_rating"] = agg["pts_allowed"] - agg["league_pts"]
    agg["reb_def_rating"] = agg["reb_allowed"] - agg["league_reb"]
    agg["ast_def_rating"] = agg["ast_allowed"] - agg["league_ast"]

    # Save
    output = agg.to_dict(orient="records")
    out_path = f"{DATA_DIR}/defensive_ratings_{season}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[DefRatings] Saved to {out_path}")
    return agg


# ─── 3. LOOKUP FUNCTION (used in Phase 3) ────────────────

def load_defensive_ratings(season=2024):
    path = f"{DATA_DIR}/defensive_ratings_{season}.json"
    if not os.path.exists(path):
        print(f"[DefRatings] No ratings file found. Run build_opponent_defensive_ratings() first.")
        return pd.DataFrame()
    with open(path) as f:
        return pd.DataFrame(json.load(f))

def get_matchup_factor(defending_team_id, player_position, stat, season=2024):
    """
    Returns a matchup factor for a player vs their opponent.
    Positive = favourable matchup (team allows more than average)
    Negative = tough matchup

    stat: 'pts', 'reb', or 'ast'
    """
    df = load_defensive_ratings(season)
    if df.empty:
        return 0.0

    position = normalize_position(player_position)
    match = df[
        (df["defending_team_id"] == defending_team_id) &
        (df["position"] == position)
    ]

    if match.empty:
        return 0.0

    rating_col = f"{stat}_def_rating"
    if rating_col not in match.columns:
        return 0.0

    return round(float(match.iloc[0][rating_col]), 3)

def get_top_soft_defenses(stat="pts", position="Guard", n=5, season=2024):
    """Return the N softest defenses against a position for a stat."""
    df = load_defensive_ratings(season)
    if df.empty:
        return []

    filtered = df[df["position"] == position].copy()
    filtered = filtered.sort_values(f"{stat}_def_rating", ascending=False)
    return filtered[["defending_team", "defending_team_id", f"{stat}_allowed", f"{stat}_def_rating"]].head(n).to_dict("records")


# ─── 4. ENRICH QUALIFYING LEGS (plug into Phase 3) ───────

def enrich_legs_with_matchup(qualifying_legs, player_position_lookup, season=2024):
    """
    Add matchup_factor to each qualifying leg.
    player_position_lookup: dict of player_name → BallDontLie position string
    """
    enriched = []
    for leg in qualifying_legs:
        player = leg["player"]
        position = player_position_lookup.get(player, "")
        stat = leg["stat"].lower()

        # defending_team_id needs to come from game data
        # For now, add placeholder — wire up from phase1 game data
        defending_team_id = leg.get("defending_team_id")

        if defending_team_id and position:
            factor = get_matchup_factor(defending_team_id, position, stat, season)
        else:
            factor = 0.0

        leg["matchup_factor"] = factor
        leg["matchup_label"] = "✅ Soft" if factor > 1.5 else "⚠️ Tough" if factor < -1.5 else "Neutral"

        # Adjust model prob slightly based on matchup
        # Max ±5% adjustment to avoid over-weighting this signal
        adjustment = np.clip(factor / 20, -0.05, 0.05)
        leg["model_prob_adjusted"] = round(min(max(leg["model_prob"] + adjustment, 0.01), 0.99), 4)

        enriched.append(leg)

    return enriched


if __name__ == "__main__":
    build_opponent_defensive_ratings(season=2024)
    print("\nTop 5 softest defenses vs Guards for Points:")
    for team in get_top_soft_defenses(stat="pts", position="Guard"):
        print(f"  {team['defending_team']}: +{team['pts_def_rating']:.1f} pts/game above average")
