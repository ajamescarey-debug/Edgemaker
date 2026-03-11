"""
=============================================================
PHASE 5D: INJURY REPORT & LINEUP AVAILABILITY
=============================================================
Injuries are the single biggest edge destroyer in prop betting.
A star player being OUT boosts teammates' usage significantly.
A player listed as QUESTIONABLE often plays reduced minutes.

This module:
  1. Scrapes NBA injury reports from ESPN (no key needed)
  2. Maps injury status to usage/minutes adjustments
  3. Detects teammates who benefit from absences
  4. Updates prop probabilities accordingly
  5. Flags high-risk legs (player or key teammate questionable)

Injury Status Effects (empirical):
  - OUT: teammate usage +8-15% (major boost for same-team props)
  - QUESTIONABLE: player minutes -15-25% if plays (reduce prob)
  - PROBABLE: minimal effect (~2-3% minutes reduction)
=============================================================
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime

DATA_DIR = "data"
TODAY = datetime.now().strftime("%Y-%m-%d")

# ─── INJURY STATUS DEFINITIONS ───────────────────────────

INJURY_ADJUSTMENTS = {
    "OUT": {
        "player_minutes_mult": 0.0,      # Don't bet this player
        "teammate_usage_boost": 0.12,    # +12% usage for primary teammates
        "teammate_prob_boost": 0.06,     # +6% prob boost on teammate Overs
    },
    "DOUBTFUL": {
        "player_minutes_mult": 0.5,
        "teammate_usage_boost": 0.06,
        "teammate_prob_boost": 0.03,
    },
    "QUESTIONABLE": {
        "player_minutes_mult": 0.80,
        "teammate_usage_boost": 0.02,
        "teammate_prob_boost": 0.01,
    },
    "PROBABLE": {
        "player_minutes_mult": 0.95,
        "teammate_usage_boost": 0.005,
        "teammate_prob_boost": 0.005,
    },
    "ACTIVE": {
        "player_minutes_mult": 1.0,
        "teammate_usage_boost": 0.0,
        "teammate_prob_boost": 0.0,
    },
}

# Star players whose absence significantly impacts teammates
# (top usage players — update each season)
HIGH_USAGE_PLAYERS = [
    "Luka Doncic", "Nikola Jokic", "Joel Embiid", "Giannis Antetokounmpo",
    "LeBron James", "Stephen Curry", "Kevin Durant", "Jayson Tatum",
    "Anthony Davis", "Damian Lillard", "Shai Gilgeous-Alexander",
    "Trae Young", "Donovan Mitchell", "Devin Booker", "Paul George",
    "Kawhi Leonard", "Jimmy Butler", "Zion Williamson", "Anthony Edwards",
    "Ja Morant", "Tyrese Haliburton", "De'Aaron Fox",
]


# ─── 1. ESPN INJURY SCRAPER ───────────────────────────────

def scrape_espn_injuries():
    """
    Pull NBA injury reports from ESPN's public endpoint.
    No API key required.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NBAModel/1.0)",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"[Injuries] ESPN API returned {resp.status_code}")
            return []
        data = resp.json()
        return data.get("injuries", [])
    except Exception as e:
        print(f"[Injuries] Failed to scrape ESPN: {e}")
        return []

def parse_espn_injuries(raw_injuries):
    """
    Parse ESPN injury data into clean format.
    Returns list of injury records.
    """
    parsed = []
    for team_entry in raw_injuries:
        team_name = team_entry.get("team", {}).get("displayName", "Unknown")
        team_abbr = team_entry.get("team", {}).get("abbreviation", "")
        team_id = team_entry.get("team", {}).get("id", None)

        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            status_raw = inj.get("status", "ACTIVE").upper()

            # Normalise status
            if "OUT" in status_raw:
                status = "OUT"
            elif "DOUBTFUL" in status_raw:
                status = "DOUBTFUL"
            elif "QUESTIONABLE" in status_raw:
                status = "QUESTIONABLE"
            elif "PROBABLE" in status_raw:
                status = "PROBABLE"
            else:
                status = "ACTIVE"

            parsed.append({
                "player_name": athlete.get("displayName", "Unknown"),
                "player_id_espn": athlete.get("id"),
                "team_name": team_name,
                "team_abbr": team_abbr,
                "team_id_espn": team_id,
                "status": status,
                "injury_type": inj.get("type", {}).get("description", ""),
                "comment": inj.get("longComment", ""),
                "is_high_usage": athlete.get("displayName") in HIGH_USAGE_PLAYERS,
                "scraped_at": datetime.now().isoformat(),
            })

    return parsed

def get_injury_report():
    """Main function: get parsed injury report."""
    raw = scrape_espn_injuries()
    if not raw:
        # Try backup: load yesterday's report if today's fails
        backup_path = f"{DATA_DIR}/injuries_{TODAY}.json"
        if os.path.exists(backup_path):
            with open(backup_path) as f:
                print("[Injuries] Using cached injury report")
                return json.load(f)
        return []

    injuries = parse_espn_injuries(raw)

    # Save
    with open(f"{DATA_DIR}/injuries_{TODAY}.json", "w") as f:
        json.dump(injuries, f, indent=2)

    active = [i for i in injuries if i["status"] != "ACTIVE"]
    print(f"[Injuries] Found {len(active)} players with injury designations")
    for inj in active:
        print(f"  {inj['player_name']} ({inj['team_name']}): {inj['status']} — {inj['injury_type']}")

    return injuries


# ─── 2. TEAM INJURY IMPACT ───────────────────────────────

def get_team_injury_status(team_name, injuries):
    """Get all injury-listed players for a team."""
    return [i for i in injuries if team_name.lower() in i["team_name"].lower()
            and i["status"] != "ACTIVE"]

def get_player_status(player_name, injuries):
    """Get a specific player's injury status."""
    for inj in injuries:
        if player_name.lower() in inj["player_name"].lower():
            return inj["status"]
    return "ACTIVE"

def get_teammate_boost(player_name, team_name, injuries):
    """
    If high-usage teammates are out, calculate boost for this player.
    """
    team_injuries = get_team_injury_status(team_name, injuries)
    total_boost = 0.0
    impactful_absences = []

    for inj in team_injuries:
        absent_player = inj["player_name"]
        if absent_player.lower() == player_name.lower():
            continue  # Not counting yourself
        if not inj["is_high_usage"]:
            continue  # Only star absences matter significantly

        adj = INJURY_ADJUSTMENTS.get(inj["status"], {})
        boost = adj.get("teammate_prob_boost", 0)
        total_boost += boost
        impactful_absences.append({
            "player": absent_player,
            "status": inj["status"],
            "prob_boost": boost,
        })

    return round(total_boost, 4), impactful_absences


# ─── 3. ENRICH LEGS WITH INJURY DATA ─────────────────────

def enrich_legs_with_injuries(qualifying_legs, player_team_lookup, injuries):
    """
    Adjust leg probabilities based on injury report.
    - Flags player's own status
    - Adds teammate absence boosts
    - Removes or strongly penalises legs where player is QUESTIONABLE/DOUBTFUL
    """
    enriched = []
    removed = []

    for leg in qualifying_legs:
        player = leg["player"]
        team_name = player_team_lookup.get(player, "")

        # Check player's own status
        player_status = get_player_status(player, injuries)
        adj = INJURY_ADJUSTMENTS.get(player_status, INJURY_ADJUSTMENTS["ACTIVE"])

        # If player is OUT — remove the leg entirely
        if player_status == "OUT":
            leg["removed_reason"] = "PLAYER_OUT"
            removed.append(leg)
            continue

        # Minutes multiplier effect on probability
        minutes_mult = adj["player_minutes_mult"]
        player_adj = (minutes_mult - 1.0) * 0.1  # ~10% prob sensitivity to minutes

        # Teammate boosts
        teammate_boost, absences = get_teammate_boost(player, team_name, injuries)

        # Total injury adjustment
        total_adj = player_adj + teammate_boost

        leg["player_injury_status"] = player_status
        leg["player_injury_adjustment"] = round(player_adj, 4)
        leg["teammate_boost"] = teammate_boost
        leg["impactful_absences"] = absences
        leg["injury_adjustment_total"] = round(total_adj, 4)
        leg["injury_risk"] = player_status in ("QUESTIONABLE", "DOUBTFUL")

        # Apply adjustment
        base = leg.get("model_prob_final", leg.get("model_prob_adjusted", leg["model_prob"]))
        leg["model_prob_final"] = round(min(max(base + total_adj, 0.01), 0.99), 4)

        enriched.append(leg)

    if removed:
        print(f"\n[Injuries] Removed {len(removed)} legs due to player unavailability:")
        for leg in removed:
            print(f"  ✗ {leg['player']} {leg['stat']} {leg['side']} — {leg['removed_reason']}")

    return enriched


# ─── 4. INJURY REPORT SUMMARY ────────────────────────────

def print_injury_summary(injuries, todays_game_teams):
    """Print injury summary for today's teams only."""
    print(f"\n{'='*50}")
    print(f"INJURY REPORT — {TODAY}")
    print(f"{'='*50}")

    relevant = [i for i in injuries
                if any(t.lower() in i["team_name"].lower() for t in todays_game_teams)
                and i["status"] != "ACTIVE"]

    if not relevant:
        print("No significant injuries for today's games.")
        return

    by_team = {}
    for inj in relevant:
        team = inj["team_name"]
        if team not in by_team:
            by_team[team] = []
        by_team[team].append(inj)

    for team, players in by_team.items():
        print(f"\n{team}:")
        for p in players:
            star = " ⭐" if p["is_high_usage"] else ""
            print(f"  {p['player_name']}{star}: {p['status']} ({p['injury_type']})")


if __name__ == "__main__":
    injuries = get_injury_report()
    print(f"\nTotal injury report entries: {len(injuries)}")
    flagged = [i for i in injuries if i["status"] in ("OUT", "DOUBTFUL", "QUESTIONABLE")]
    print(f"Significant designations: {len(flagged)}")
