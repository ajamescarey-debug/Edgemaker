"""
=============================================================
PHASE 5B: BACK-TO-BACK & REST DAY FLAGS
=============================================================
Rest days are one of the most consistently predictive
features in NBA prop modelling. Players on B2Bs average
~2-4% lower on counting stats. Teams on 0 rest vs 2+ rest
days show significant ATS performance differences.

This module:
  1. Calculates rest days for every team on today's slate
  2. Flags back-to-backs (0 rest days)
  3. Adds a rest-adjusted probability modifier
  4. Tracks road trip length (5+ games = fatigue signal)
=============================================================
"""

import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

DATA_DIR = "data"
TODAY = datetime.now().strftime("%Y-%m-%d")

# ─── EMPIRICAL REST ADJUSTMENTS ──────────────────────────
# Based on historical NBA data (approx. averages)
# Applied as probability multipliers to model output

REST_ADJUSTMENTS = {
    # (player_rest_days, opp_rest_days) → stat multiplier
    # Negative rest diff = player more fatigued than opponent
    "pts": {
        "b2b_player":     -0.034,   # -3.4% pts on B2B
        "b2b_opponent":   +0.021,   # +2.1% pts vs B2B opponent
        "well_rested":    +0.018,   # +1.8% on 3+ days rest
        "fatigue_road":   -0.028,   # -2.8% on 5th+ road game
    },
    "reb": {
        "b2b_player":     -0.041,
        "b2b_opponent":   +0.019,
        "well_rested":    +0.012,
        "fatigue_road":   -0.022,
    },
    "ast": {
        "b2b_player":     -0.029,
        "b2b_opponent":   +0.015,
        "well_rested":    +0.011,
        "fatigue_road":   -0.018,
    },
}


# ─── 1. CALCULATE REST DAYS ──────────────────────────────

def get_recent_schedule(team_id, days_back=14):
    """
    Pull recent games for a team to calculate rest days.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    url = "https://www.balldontlie.io/api/v1/games"
    params = {
        "team_ids[]": team_id,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "per_page": 25,
    }
    resp = requests.get(url, params=params)
    games = resp.json().get("data", [])
    return sorted(games, key=lambda g: g["date"], reverse=True)

def calculate_rest_days(team_id, reference_date=None):
    """
    Calculate rest days for a team before their next game.
    Returns dict with rest info.
    """
    if reference_date is None:
        reference_date = datetime.now()

    recent_games = get_recent_schedule(team_id, days_back=14)

    if not recent_games:
        return {
            "team_id": team_id,
            "rest_days": 2,  # assume 2 if no data
            "is_b2b": False,
            "last_game_date": None,
            "last_game_location": None,
        }

    # Most recent completed game
    last_game = recent_games[0]
    last_date = datetime.strptime(last_game["date"][:10], "%Y-%m-%d")
    rest_days = (reference_date.date() - last_date.date()).days - 1
    rest_days = max(rest_days, 0)

    # Was last game home or away? (proxy for travel)
    from_team = last_game.get("home_team", {})
    last_location = "home" if from_team.get("id") == team_id else "away"

    return {
        "team_id": team_id,
        "rest_days": rest_days,
        "is_b2b": rest_days == 0,
        "last_game_date": last_date.strftime("%Y-%m-%d"),
        "last_game_location": last_location,
    }

def calculate_road_trip_length(team_id, days_back=21):
    """
    Count consecutive away games (road trip fatigue).
    5+ consecutive road games = significant fatigue signal.
    """
    recent_games = get_recent_schedule(team_id, days_back=days_back)
    road_streak = 0
    for game in recent_games:
        is_home = game.get("home_team", {}).get("id") == team_id
        if not is_home:
            road_streak += 1
        else:
            break  # streak broken by home game
    return road_streak


# ─── 2. BUILD TODAY'S REST TABLE ─────────────────────────

def build_rest_table(todays_games):
    """
    For today's slate, build a rest/fatigue table for every team.
    todays_games: list from phase1 get_todays_games()
    Returns: dict of team_id → rest_info
    """
    rest_table = {}
    processed_teams = set()

    for game in todays_games:
        for team_key in ["home_team", "visitor_team"]:
            team = game.get(team_key, {})
            team_id = team.get("id")
            if not team_id or team_id in processed_teams:
                continue
            processed_teams.add(team_id)

            rest_info = calculate_rest_days(team_id)
            road_trip = calculate_road_trip_length(team_id)
            rest_info["road_trip_length"] = road_trip
            rest_info["fatigue_road"] = road_trip >= 5
            rest_info["team_name"] = team.get("full_name", "Unknown")
            rest_table[team_id] = rest_info

    return rest_table


# ─── 3. REST ADJUSTMENT FUNCTION ─────────────────────────

def get_rest_adjustment(player_team_id, opponent_team_id, stat, rest_table):
    """
    Calculate combined rest adjustment for a prop bet.
    Returns a probability delta (e.g. -0.034 = subtract 3.4% from model prob)
    """
    adjustments = REST_ADJUSTMENTS.get(stat.lower(), {})
    total_adj = 0.0
    flags = []

    player_rest = rest_table.get(player_team_id, {})
    opp_rest = rest_table.get(opponent_team_id, {})

    # Player on B2B
    if player_rest.get("is_b2b"):
        total_adj += adjustments.get("b2b_player", 0)
        flags.append("B2B")

    # Opponent on B2B (easier game for player)
    if opp_rest.get("is_b2b"):
        total_adj += adjustments.get("b2b_opponent", 0)
        flags.append("OPP_B2B")

    # Player well rested (3+ days)
    if player_rest.get("rest_days", 0) >= 3:
        total_adj += adjustments.get("well_rested", 0)
        flags.append("WELL_RESTED")

    # Road trip fatigue
    if player_rest.get("fatigue_road"):
        total_adj += adjustments.get("fatigue_road", 0)
        flags.append("ROAD_FATIGUE")

    return round(total_adj, 4), flags


# ─── 4. ENRICH LEGS WITH REST DATA ───────────────────────

def enrich_legs_with_rest(qualifying_legs, rest_table, player_team_lookup):
    """
    Add rest adjustments to qualifying legs.
    player_team_lookup: dict of player_name → team_id
    """
    enriched = []
    for leg in qualifying_legs:
        player = leg["player"]
        stat = leg["stat"].lower()
        player_team_id = player_team_lookup.get(player)
        opponent_team_id = leg.get("defending_team_id")

        if player_team_id and opponent_team_id:
            adj, flags = get_rest_adjustment(player_team_id, opponent_team_id, stat, rest_table)
        else:
            adj, flags = 0.0, []

        leg["rest_adjustment"] = adj
        leg["rest_flags"] = flags
        leg["rest_days"] = rest_table.get(player_team_id, {}).get("rest_days", "?")
        leg["opp_rest_days"] = rest_table.get(opponent_team_id, {}).get("rest_days", "?")

        # Apply rest adjustment to probability
        base_prob = leg.get("model_prob_adjusted", leg["model_prob"])
        leg["model_prob_final"] = round(min(max(base_prob + adj, 0.01), 0.99), 4)

        enriched.append(leg)

    return enriched


# ─── 5. SAVE REST TABLE ──────────────────────────────────

def save_rest_table(rest_table):
    out = list(rest_table.values())
    with open(f"{DATA_DIR}/rest_table_{TODAY}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Rest] Saved rest table to data/rest_table_{TODAY}.json")


if __name__ == "__main__":
    # Test with sample team IDs
    # Denver Nuggets = 7, LA Lakers = 14
    for team_id, name in [(7, "Denver Nuggets"), (14, "LA Lakers")]:
        info = calculate_rest_days(team_id)
        road = calculate_road_trip_length(team_id)
        print(f"\n{name}:")
        print(f"  Rest days: {info['rest_days']} | B2B: {info['is_b2b']}")
        print(f"  Road trip: {road} consecutive away games")
