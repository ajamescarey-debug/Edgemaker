"""
=============================================================
PHASE 6: CLAUDE AI ENRICHMENT LAYER
=============================================================
Uses the Anthropic API to add human-readable intelligence
to every qualifying leg and the overall parlay.

Generates:
  1. Plain English bet reasoning per leg
  2. Injury impact analysis per leg
  3. Line movement commentary per leg
  4. Daily parlay summary brief (the one thing you read)

Cost estimate: ~$0.05–0.20 per day depending on legs found.
Uses claude-sonnet for speed + cost balance.
=============================================================
"""

import anthropic
import json
import os
from datetime import datetime

OUTPUT_DIR = "dashboard/data"
TODAY = datetime.now().strftime("%Y-%m-%d")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ─── 1. BET REASONING ────────────────────────────────────

def generate_bet_reasoning(leg):
    """
    Generate plain English reasoning for why this leg qualifies.
    Concise — 2-3 sentences max. Written like a sharp bettor would.
    """
    signals = []
    if leg.get("matchup_label", "").startswith("✅"):
        signals.append(f"soft defensive matchup (allows +{leg.get('matchup_factor', 0):.1f} {leg['stat'].lower()} above league avg to this position)")
    if "WELL_RESTED" in leg.get("rest_flags", []):
        signals.append(f"{leg['rest_days']} days rest")
    if "B2B" in leg.get("rest_flags", []):
        signals.append("playing on zero rest (B2B)")
    if "OPP_B2B" in leg.get("rest_flags", []):
        signals.append("opponent on B2B")
    if leg.get("sharp_signals") and any("CONFIRMS" in s for s in leg["sharp_signals"]):
        signals.append(f"sharp money confirmed on the {leg['side'].lower()}")
    if leg.get("teammate_boost", 0) > 0.03:
        absences = [a["player"] for a in leg.get("impactful_absences", [])]
        signals.append(f"usage boost from {', '.join(absences)} being out")

    signal_text = "; ".join(signals) if signals else "base model edge"

    prompt = f"""You are a sharp sports betting analyst. Write a 2-3 sentence plain English explanation for why this NBA prop bet qualifies today. Be specific and direct. No fluff.

Bet: {leg['player']} {leg['stat']} {leg['side']} {leg['line']} @ {leg['odds']} ({leg.get('bookmaker','')})
Matchup: {leg.get('matchup','')}
Model probability: {leg.get('model_prob_final', leg.get('model_prob', 0)):.1%} vs book implied {leg.get('implied_prob', 0):.1%}
Edge: {leg.get('edge', 0):.1%}
Confidence score: {leg.get('confidence_score', 0)}/100
Key signals: {signal_text}
Player injury status: {leg.get('player_injury_status', 'ACTIVE')}

Write the reasoning in 2-3 sentences. Start with the strongest signal. Do not start with "This bet" or "The model"."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


# ─── 2. INJURY IMPACT ANALYSIS ───────────────────────────

def generate_injury_analysis(leg, all_injuries):
    """
    Analyse injury context for this specific leg.
    Covers: player's own status + key teammate/opponent absences.
    """
    player_status = leg.get("player_injury_status", "ACTIVE")
    impactful = leg.get("impactful_absences", [])
    teammate_boost = leg.get("teammate_boost", 0)

    if player_status == "ACTIVE" and not impactful:
        return "No injury concerns. Player is active and no high-usage teammates are sidelined."

    context_parts = []
    if player_status != "ACTIVE":
        context_parts.append(f"Player listed as {player_status}")
    if impactful:
        names = [f"{a['player']} ({a['status']})" for a in impactful]
        context_parts.append(f"Teammate absences: {', '.join(names)}")
    if teammate_boost > 0:
        context_parts.append(f"Estimated usage boost: +{teammate_boost:.1%} probability")

    prompt = f"""You are an NBA injury analyst. Write 2 sentences on the injury context for this prop bet. Be specific about usage/minutes impact.

Bet: {leg['player']} {leg['stat']} {leg['side']} {leg['line']}
Injury context: {'; '.join(context_parts)}
Teammate boost applied: {teammate_boost:.1%}

2 sentences only. Be direct about whether this injury situation helps or hurts the bet."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


# ─── 3. LINE MOVEMENT COMMENTARY ─────────────────────────

def generate_line_movement_commentary(leg):
    """
    Explain what the line movement means for this bet in plain English.
    """
    lm_signal = leg.get("line_movement_signal", "STABLE")
    lm_direction = leg.get("line_movement_direction", "")
    lm_adjustment = leg.get("line_movement_adjustment", 0)
    sharp_signals = leg.get("sharp_signals", [])

    if lm_signal == "STABLE":
        return "Line has been stable since open — no sharp money detected. Book is comfortable with this number."

    prompt = f"""You are a betting market analyst. Write 1-2 sentences explaining what this line movement means for the bet.

Bet: {leg['player']} {leg['stat']} {leg['side']} {leg['line']}
Line movement signal: {lm_signal}
Movement direction: {lm_direction}
Sharp signals: {', '.join(sharp_signals) if sharp_signals else 'none'}
Probability adjustment applied: {lm_adjustment:+.1%}

1-2 sentences. Explain what sharp/public money is doing and whether it supports or contradicts the bet."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


# ─── 4. DAILY PARLAY SUMMARY BRIEF ───────────────────────

def generate_daily_brief(parlay, qualifying_legs, injuries, signal_summary):
    """
    The daily brief — one thing Ash reads every morning.
    Punchy, confident, written like a sharp analyst's morning note.
    """
    if not parlay:
        prompt = f"""You are a sharp NBA betting analyst. Write a 3-4 sentence morning brief explaining that no qualifying parlay was found today and why the model is sitting out.

Signal summary: {json.dumps(signal_summary, indent=2)}
Qualifying legs found: {len(qualifying_legs)}
Today's date: {TODAY}

Be direct. Explain briefly what the model looked at and why nothing met the threshold. End with one sentence about what to watch for tomorrow."""

    else:
        legs_summary = "\n".join([
            f"- {l['player']} {l['stat']} {l['side']} {l['line']} @ {l['odds']} | Edge: {l.get('edge',0):.1%} | Conf: {l.get('confidence_score',0)}/100"
            for l in parlay["legs"]
        ])

        injury_flags = [i for i in injuries if i.get("status") in ("OUT", "DOUBTFUL", "QUESTIONABLE")]
        injury_text = ", ".join([f"{i['player_name']} ({i['status']})" for i in injury_flags[:4]]) if injury_flags else "None significant"

        prompt = f"""You are a sharp NBA betting analyst writing a morning parlay brief. Write 4-5 punchy sentences that cover: what the parlay is, why it qualifies, the key signals driving it, and any risks to watch.

Today's date: {TODAY}
Recommended parlay ({parlay['num_legs']} legs @ {parlay['combined_odds']}x):
{legs_summary}

Combined model probability: {parlay['combined_prob']:.1%}
Expected value: {parlay['combined_ev']:+.3f}
Kelly stake: {parlay['kelly_stake_pct']:.1%} of bankroll
Total qualifying legs found: {len(qualifying_legs)}
Key injuries: {injury_text}
Sharp confirmed legs: {signal_summary.get('sharp_confirmed_legs', 0)}
B2B flagged legs: {signal_summary.get('b2b_flagged_legs', 0)}

Write like a confident analyst, not a robot. Call out the strongest signals. Flag any concerns honestly. Do not use bullet points — flowing sentences only."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


# ─── 5. MAIN ENRICHMENT RUNNER ───────────────────────────

def run_enrichment():
    print("=" * 60)
    print(f"PHASE 6: CLAUDE AI ENRICHMENT — {TODAY}")
    print("=" * 60)

    results_path = f"{OUTPUT_DIR}/results.json"
    if not os.path.exists(results_path):
        print(f"❌ No results.json found at {results_path}. Run phase5e first.")
        return

    with open(results_path) as f:
        data = json.load(f)

    legs = data.get("qualifying_legs", [])
    parlay = data.get("recommended_parlay")
    injuries = data.get("injury_report", [])
    signal_summary = data.get("signal_summary", {})

    print(f"\n[Claude] Enriching {len(legs)} qualifying legs...")

    # Enrich each leg
    for i, leg in enumerate(legs):
        player = leg["player"]
        stat = leg["stat"]
        print(f"  [{i+1}/{len(legs)}] {player} {stat}...")

        try:
            leg["ai_reasoning"] = generate_bet_reasoning(leg)
        except Exception as e:
            leg["ai_reasoning"] = f"Reasoning unavailable: {e}"

        try:
            leg["ai_injury_analysis"] = generate_injury_analysis(leg, injuries)
        except Exception as e:
            leg["ai_injury_analysis"] = f"Injury analysis unavailable: {e}"

        try:
            leg["ai_line_movement"] = generate_line_movement_commentary(leg)
        except Exception as e:
            leg["ai_line_movement"] = f"Line movement commentary unavailable: {e}"

    # Enrich parlay legs too
    if parlay:
        for leg in parlay.get("legs", []):
            enriched = next((l for l in legs if l["player"] == leg["player"] and l["stat"] == leg["stat"]), None)
            if enriched:
                leg["ai_reasoning"] = enriched.get("ai_reasoning", "")
                leg["ai_injury_analysis"] = enriched.get("ai_injury_analysis", "")
                leg["ai_line_movement"] = enriched.get("ai_line_movement", "")

    # Generate daily brief
    print("\n[Claude] Generating daily parlay brief...")
    try:
        daily_brief = generate_daily_brief(parlay, legs, injuries, signal_summary)
        data["daily_brief"] = daily_brief
        print(f"\n{'─'*60}")
        print("DAILY BRIEF:")
        print(f"{'─'*60}")
        print(daily_brief)
        print(f"{'─'*60}\n")
    except Exception as e:
        data["daily_brief"] = f"Brief unavailable: {e}"

    # Update results
    data["qualifying_legs"] = legs
    data["recommended_parlay"] = parlay
    data["date"] = TODAY
    data["generated_at"] = TODAY
    data["enriched_at"] = datetime.now().isoformat()
    data["enrichment_version"] = "6.0"

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Enrichment complete. results.json updated.")
    print(f"   Dashboard will auto-deploy via Netlify on next git push.")


if __name__ == "__main__":
    run_enrichment()
