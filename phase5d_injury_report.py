"""
=============================================================
PHASE 5D: INJURY REPORT & LINEUP AVAILABILITY
=============================================================
Scrapes NBA injury data from multiple ESPN endpoints.
No API key required.
=============================================================
"""

import requests
import json
import os
from datetime import datetime

DATA_DIR = "data"
TODAY = datetime.now().strftime("%Y-%m-%d")
os.makedirs(DATA_DIR, exist_ok=True)

INJURY_ADJUSTMENTS = {
    "OUT":          {"player_minutes_mult": 0.0,  "teammate_prob_boost": 0.06},
    "DOUBTFUL":     {"player_minutes_mult": 0.5,  "teammate_prob_boost": 0.03},
    "QUESTIONABLE": {"player_minutes_mult": 0.80, "teammate_prob_boost": 0.01},
    "PROBABLE":     {"player_minutes_mult": 0.95, "teammate_prob_boost": 0.005},
    "ACTIVE":       {"player_minutes_mult": 1.0,  "teammate_prob_boost": 0.0},
}

HIGH_USAGE_PLAYERS = [
    "Luka Doncic", "Nikola Jokic", "Joel Embiid", "Giannis Antetokounmpo",
    "LeBron James", "Stephen Curry", "Kevin Durant", "Jayson Tatum",
    "Anthony Davis", "Damian Lillard", "Shai Gilgeous-Alexander",
    "Trae Young", "Donovan Mitchell", "Devin Booker", "Paul George",
    "Kawhi Leonard", "Jimmy Butler", "Zion Williamson", "Anthony Edwards",
    "Ja Morant", "Tyrese Haliburton", "De'Aaron Fox", "Victor Wembanyama",
    "Paolo Banchero", "Cade Cunningham", "Evan Mobley",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NBAModel/1.0)", "Accept": "application/json"}


def scrape_espn_injuries_v1():
    """Primary: ESPN injuries endpoint."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        # Handle both possible response shapes
        return data.get("injuries", data.get("items", []))
    except Exception as e:
        print(f"[Injuries] ESPN v1 failed: {e}")
        return []


def scrape_espn_injuries_v2():
    """Fallback: ESPN teams endpoint — loop all 30 teams."""
    injuries = []
    # ESPN team IDs 1-30 cover all NBA teams
    for team_id in range(1, 31):
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/injuries"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            items = data.get("injuries", data.get("items", []))
            if items:
                # Wrap in team structure for parser
                team_info = data.get("team", {"displayName": f"Team {team_id}", "abbreviation": "", "id": team_id})
                injuries.append({"team": team_info, "injuries": items})
        except Exception:
            continue
    return injuries


def scrape_espn_injuries_v3():
    """Second fallback: ESPN scoreboard — pull injury tags from today's games."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    injuries = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        for event in data.get("events", []):
            for competition in event.get("competitions", []):
                for competitor in competition.get("competitors", []):
                    team_name = competitor.get("team", {}).get("displayName", "")
                    for player in competitor.get("roster", []):
                        status = player.get("status", {})
                        status_type = status.get("type", {}).get("name", "").upper()
                        if status_type and status_type != "ACTIVE":
                            injuries.append({
                                "player_name": player.get("athlete", {}).get("displayName", ""),
                                "team_name": team_name,
                                "team_abbr": competitor.get("team", {}).get("abbreviation", ""),
                                "status": normalise_status(status_type),
                                "injury_type": status.get("description", ""),
                                "comment": "",
                                "is_high_usage": player.get("athlete", {}).get("displayName", "") in HIGH_USAGE_PLAYERS,
                                "scraped_at": datetime.now().isoformat(),
                            })
    except Exception as e:
        print(f"[Injuries] ESPN v3 failed: {e}")
    return injuries


def normalise_status(raw):
    raw = raw.upper()
    if "OUT" in raw:          return "OUT"
    if "DOUBTFUL" in raw:     return "DOUBTFUL"
    if "QUESTIONABLE" in raw: return "QUESTIONABLE"
    if "PROBABLE" in raw:     return "PROBABLE"
    return "ACTIVE"


def parse_espn_injuries(raw_injuries):
    parsed = []
    for team_entry in raw_injuries:
        team_name = team_entry.get("team", {}).get("displayName", "Unknown")
        team_abbr = team_entry.get("team", {}).get("abbreviation", "")
        team_id   = team_entry.get("team", {}).get("id", None)
        for inj in team_entry.get("injuries", []):
            athlete   = inj.get("athlete", {})
            status    = normalise_status(inj.get("status", "ACTIVE"))
            player_name = athlete.get("displayName", "Unknown")
            parsed.append({
                "player_name":    player_name,
                "team_name":      team_name,
                "team_abbr":      team_abbr,
                "status":         status,
                "injury_type":    inj.get("type", {}).get("description", inj.get("longComment", "")),
                "comment":        inj.get("longComment", ""),
                "is_high_usage":  player_name in HIGH_USAGE_PLAYERS,
                "scraped_at":     datetime.now().isoformat(),
            })
    return parsed


def get_injury_report():
    """Main entry point — tries 3 sources, returns best result."""
    print("[Injuries] Trying ESPN primary endpoint...")
    raw = scrape_espn_injuries_v1()

    if raw:
        injuries = parse_espn_injuries(raw)
        print(f"[Injuries] Primary source: {len(injuries)} entries")
    else:
        print("[Injuries] Primary failed — trying per-team endpoint...")
        raw2 = scrape_espn_injuries_v2()
        if raw2:
            injuries = parse_espn_injuries(raw2)
            print(f"[Injuries] Per-team source: {len(injuries)} entries")
        else:
            print("[Injuries] Per-team failed — trying scoreboard...")
            injuries = scrape_espn_injuries_v3()
            print(f"[Injuries] Scoreboard source: {len(injuries)} entries")

    # Filter to non-active only for reporting
    flagged = [i for i in injuries if i["status"] != "ACTIVE"]

    if not flagged:
        print("[Injuries] No injury designations found today")
    else:
        print(f"[Injuries] {len(flagged)} players with designations:")
        for inj in flagged:
            star = " ⭐" if inj["is_high_usage"] else ""
            print(f"  {inj['player_name']}{star} ({inj['team_name']}): {inj['status']} — {inj['injury_type']}")

    # Save full report
    with open(f"{DATA_DIR}/injuries_{TODAY}.json", "w") as f:
        json.dump(injuries, f, indent=2)

    return injuries


def get_player_status(player_name, injuries):
    for inj in injuries:
        if player_name.lower() in inj["player_name"].lower():
            return inj["status"]
    return "ACTIVE"


def get_teammate_boost(player_name, team_name, injuries):
    total_boost = 0.0
    impactful_absences = []
    for inj in injuries:
        if team_name.lower() not in inj["team_name"].lower():
            continue
        if inj["player_name"].lower() == player_name.lower():
            continue
        if not inj["is_high_usage"]:
            continue
        boost = INJURY_ADJUSTMENTS.get(inj["status"], {}).get("teammate_prob_boost", 0)
        total_boost += boost
        impactful_absences.append({
            "player": inj["player_name"],
            "status": inj["status"],
            "prob_boost": boost,
        })
    return round(total_boost, 4), impactful_absences


def enrich_legs_with_injuries(qualifying_legs, player_team_lookup, injuries):
    enriched = []
    removed  = []
    for leg in qualifying_legs:
        player      = leg["player"]
        team_name   = player_team_lookup.get(player, "")
        player_status = get_player_status(player, injuries)
        adj = INJURY_ADJUSTMENTS.get(player_status, INJURY_ADJUSTMENTS["ACTIVE"])

        if player_status == "OUT":
            leg["removed_reason"] = "PLAYER_OUT"
            removed.append(leg)
            continue

        minutes_mult = adj["player_minutes_mult"]
        player_adj   = (minutes_mult - 1.0) * 0.1
        teammate_boost, absences = get_teammate_boost(player, team_name, injuries)
        total_adj = player_adj + teammate_boost

        leg["player_injury_status"]      = player_status
        leg["player_injury_adjustment"]  = round(player_adj, 4)
        leg["teammate_boost"]            = teammate_boost
        leg["impactful_absences"]        = absences
        leg["injury_adjustment_total"]   = round(total_adj, 4)
        leg["injury_risk"]               = player_status in ("QUESTIONABLE", "DOUBTFUL")

        base = leg.get("model_prob_final", leg.get("model_prob_adjusted", leg["model_prob"]))
        leg["model_prob_final"] = round(min(max(base + total_adj, 0.01), 0.99), 4)
        enriched.append(leg)

    if removed:
        print(f"[Injuries] Removed {len(removed)} legs — player unavailable:")
        for leg in removed:
            print(f"  ✗ {leg['player']} {leg['stat']} {leg['side']}")

    return enriched


def print_injury_summary(injuries, todays_game_teams):
    print(f"\n{'='*50}\nINJURY REPORT — {TODAY}\n{'='*50}")
    relevant = [i for i in injuries
                if any(t.lower() in i["team_name"].lower() for t in todays_game_teams)
                and i["status"] != "ACTIVE"]
    if not relevant:
        print("No significant injuries for today's games.")
        return
    by_team = {}
    for inj in relevant:
        by_team.setdefault(inj["team_name"], []).append(inj)
    for team, players in by_team.items():
        print(f"\n{team}:")
        for p in players:
            star = " ⭐" if p["is_high_usage"] else ""
            print(f"  {p['player_name']}{star}: {p['status']} — {p['injury_type']}")


if __name__ == "__main__":
    injuries = get_injury_report()
    flagged = [i for i in injuries if i["status"] in ("OUT", "DOUBTFUL", "QUESTIONABLE")]
    print(f"\nSignificant designations: {len(flagged)}")
