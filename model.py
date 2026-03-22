"""
╔══════════════════════════════════════════════════════════╗
║              EDGEMAKER — NBA INTELLIGENCE MODEL          ║
║         Player Props + Game Spreads + Claude Brief       ║
║         Automated daily pipeline for GitHub Actions      ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime, timedelta
import pytz

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────
ODDS_API_KEY    = os.environ.get("ODDS_API_KEY", "")
BDL_API_KEY     = os.environ.get("BDL_API_KEY", "")
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
RESULTS_FILE    = "data/results.json"
AEST            = pytz.timezone("Australia/Melbourne")
ET              = pytz.timezone("America/New_York")

AU_BOOKS        = ["tab", "betr", "pointsbet"]
SPREAD_EDGE_MIN = 3.0
SPREAD_EDGE_MAX = 7.0
CONF_MIN        = 50.0
CONF_MAX        = 70.0
PROP_EDGE_MIN   = 0.04
PROP_PROB_MIN   = 0.57

TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# ══════════════════════════════════════════════════════════
# SECTION 1: NBA GAME DATA
# ══════════════════════════════════════════════════════════

def fetch_nba_data():
    print("📡 Fetching NBA game data via BallDontLie...")
    headers = {"Authorization": BDL_API_KEY} if BDL_API_KEY else {}
    all_games = []
    for season in [2022, 2023, 2024, 2025]:
        page = 1
        print(f"  Season {season}...")
        while True:
            try:
                resp = requests.get(
                    "https://api.balldontlie.io/v1/games",
                    headers=headers,
                    params={"seasons[]": season, "per_page": 100, "page": page},
                    timeout=30,
                )
                data = resp.json()
                games = data.get("data", [])
                if not games:
                    break
                all_games.extend(games)
                meta = data.get("meta", {})
                if page >= meta.get("total_pages", 1):
                    break
                page += 1
                time.sleep(0.3)
            except Exception as e:
                print(f"  ⚠️ Season {season} page {page}: {e}")
                break

    rows = []
    for g in all_games:
        if g.get("status") != "Final":
            continue
        home       = g.get("home_team", {})
        away       = g.get("visitor_team", {})
        home_score = g.get("home_team_score", 0) or 0
        away_score = g.get("visitor_team_score", 0) or 0
        date_str   = g.get("date", "")[:10]
        if not date_str or home_score == 0:
            continue
        game_date = pd.to_datetime(date_str)
        home_abbr = home.get("abbreviation", "")
        away_abbr = away.get("abbreviation", "")
        for is_home, team, pts, pts_a in [(1, home_abbr, home_score, away_score), (0, away_abbr, away_score, home_score)]:
            rows.append({
                'GAME_ID': g["id"], 'GAME_DATE': game_date,
                'TEAM_ABBREVIATION': team, 'HOME': is_home,
                'HOME_TEAM': home_abbr, 'AWAY_TEAM': away_abbr,
                'PTS': pts, 'PTS_ALLOWED': pts_a,
                'POINT_DIFF': pts - pts_a, 'GAME_TOTAL': pts + pts_a,
                'WIN': 1 if pts > pts_a else 0,
                'FG_PCT': 0.46, 'FG3_PCT': 0.36, 'TOV': 14.0, 'REB': 44.0,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)
    df['DAYS_REST'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].diff().dt.days.fillna(3)
    df['B2B'] = (df['DAYS_REST'] <= 1).astype(int)
    print(f"✅ Loaded {len(df):,} records")
    return df


# ══════════════════════════════════════════════════════════
# SECTION 2: SPREAD MODEL
# ══════════════════════════════════════════════════════════

FEATURES = [
    'NET_DIFF_L5','NET_DIFF_L10','NET_DIFF_L15',
    'OFF_DIFF_L5','OFF_DIFF_L10','OFF_DIFF_L15',
    'DEF_DIFF_L5','DEF_DIFF_L10','DEF_DIFF_L15',
    'REST_DIFF','HOME_B2B','AWAY_B2B','B2B_DIFF',
    'HOME_DAYS_REST','AWAY_DAYS_REST',
    'HOME_FG_PCT_L10','AWAY_FG_PCT_L10',
    'HOME_FG3_PCT_L10','AWAY_FG3_PCT_L10',
    'HOME_TOV_L10','AWAY_TOV_L10',
    'HOME_REB_L10','AWAY_REB_L10',
]

def rolling_stats(df, team, date):
    d = df[(df['TEAM_ABBREVIATION']==team) & (df['GAME_DATE']<date)].sort_values('GAME_DATE').tail(15)
    if len(d) < 3: return None
    return {
        'PTS_L5': d.tail(5)['PTS'].mean(), 'PTS_ALLOWED_L5': d.tail(5)['PTS_ALLOWED'].mean(),
        'POINT_DIFF_L5': d.tail(5)['POINT_DIFF'].mean(),
        'PTS_L10': d.tail(10)['PTS'].mean(), 'PTS_ALLOWED_L10': d.tail(10)['PTS_ALLOWED'].mean(),
        'POINT_DIFF_L10': d.tail(10)['POINT_DIFF'].mean(),
        'PTS_L15': d.tail(15)['PTS'].mean(), 'PTS_ALLOWED_L15': d.tail(15)['PTS_ALLOWED'].mean(),
        'POINT_DIFF_L15': d.tail(15)['POINT_DIFF'].mean(),
        'FG_PCT_L10': d.tail(10)['FG_PCT'].mean(), 'FG3_PCT_L10': d.tail(10)['FG3_PCT'].mean(),
        'TOV_L10': d.tail(10)['TOV'].mean(), 'REB_L10': d.tail(10)['REB'].mean(),
        'DAYS_REST': d.tail(1)['DAYS_REST'].values[0], 'B2B': d.tail(1)['B2B'].values[0],
    }

def train_models(df):
    print("🤖 Training models...")
    home_df = df[df['HOME']==1].copy().sort_values('GAME_DATE').reset_index(drop=True)
    rows = []
    for _, g in home_df.iterrows():
        hs = rolling_stats(df, g['HOME_TEAM'], g['GAME_DATE'])
        as_ = rolling_stats(df, g['AWAY_TEAM'], g['GAME_DATE'])
        if not hs or not as_: continue
        rows.append({
            'HOME_WIN': g['WIN'], 'POINT_DIFF': g['POINT_DIFF'],
            'NET_DIFF_L5': hs['POINT_DIFF_L5']-as_['POINT_DIFF_L5'],
            'NET_DIFF_L10': hs['POINT_DIFF_L10']-as_['POINT_DIFF_L10'],
            'NET_DIFF_L15': hs['POINT_DIFF_L15']-as_['POINT_DIFF_L15'],
            'OFF_DIFF_L5': hs['PTS_L5']-as_['PTS_L5'],
            'OFF_DIFF_L10': hs['PTS_L10']-as_['PTS_L10'],
            'OFF_DIFF_L15': hs['PTS_L15']-as_['PTS_L15'],
            'DEF_DIFF_L5': hs['PTS_ALLOWED_L5']-as_['PTS_ALLOWED_L5'],
            'DEF_DIFF_L10': hs['PTS_ALLOWED_L10']-as_['PTS_ALLOWED_L10'],
            'DEF_DIFF_L15': hs['PTS_ALLOWED_L15']-as_['PTS_ALLOWED_L15'],
            'REST_DIFF': hs['DAYS_REST']-as_['DAYS_REST'],
            'HOME_B2B': hs['B2B'], 'AWAY_B2B': as_['B2B'],
            'B2B_DIFF': as_['B2B']-hs['B2B'],
            'HOME_DAYS_REST': hs['DAYS_REST'], 'AWAY_DAYS_REST': as_['DAYS_REST'],
            'HOME_FG_PCT_L10': hs['FG_PCT_L10'], 'AWAY_FG_PCT_L10': as_['FG_PCT_L10'],
            'HOME_FG3_PCT_L10': hs['FG3_PCT_L10'], 'AWAY_FG3_PCT_L10': as_['FG3_PCT_L10'],
            'HOME_TOV_L10': hs['TOV_L10'], 'AWAY_TOV_L10': as_['TOV_L10'],
            'HOME_REB_L10': hs['REB_L10'], 'AWAY_REB_L10': as_['REB_L10'],
        })
    mdf = pd.DataFrame(rows)
    X = mdf[FEATURES]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    wm = LogisticRegression(max_iter=1000)
    sm = Ridge(alpha=1.0)
    wm.fit(Xs, mdf['HOME_WIN'])
    sm.fit(Xs, mdf['POINT_DIFF'])
    print(f"✅ Trained on {len(mdf):,} games")
    return wm, sm, sc

def predict_spread(df, wm, sm, sc, home, away, date=None):
    if date is None: date = df['GAME_DATE'].max() + timedelta(days=1)
    hs = rolling_stats(df, home, date)
    as_ = rolling_stats(df, away, date)
    if not hs or not as_: return None
    fr = {
        'NET_DIFF_L5': hs['POINT_DIFF_L5']-as_['POINT_DIFF_L5'],
        'NET_DIFF_L10': hs['POINT_DIFF_L10']-as_['POINT_DIFF_L10'],
        'NET_DIFF_L15': hs['POINT_DIFF_L15']-as_['POINT_DIFF_L15'],
        'OFF_DIFF_L5': hs['PTS_L5']-as_['PTS_L5'],
        'OFF_DIFF_L10': hs['PTS_L10']-as_['PTS_L10'],
        'OFF_DIFF_L15': hs['PTS_L15']-as_['PTS_L15'],
        'DEF_DIFF_L5': hs['PTS_ALLOWED_L5']-as_['PTS_ALLOWED_L5'],
        'DEF_DIFF_L10': hs['PTS_ALLOWED_L10']-as_['PTS_ALLOWED_L10'],
        'DEF_DIFF_L15': hs['PTS_ALLOWED_L15']-as_['PTS_ALLOWED_L15'],
        'REST_DIFF': hs['DAYS_REST']-as_['DAYS_REST'],
        'HOME_B2B': hs['B2B'], 'AWAY_B2B': as_['B2B'],
        'B2B_DIFF': as_['B2B']-hs['B2B'],
        'HOME_DAYS_REST': hs['DAYS_REST'], 'AWAY_DAYS_REST': as_['DAYS_REST'],
        'HOME_FG_PCT_L10': hs['FG_PCT_L10'], 'AWAY_FG_PCT_L10': as_['FG_PCT_L10'],
        'HOME_FG3_PCT_L10': hs['FG3_PCT_L10'], 'AWAY_FG3_PCT_L10': as_['FG3_PCT_L10'],
        'HOME_TOV_L10': hs['TOV_L10'], 'AWAY_TOV_L10': as_['TOV_L10'],
        'HOME_REB_L10': hs['REB_L10'], 'AWAY_REB_L10': as_['REB_L10'],
    }
    rs = sc.transform(pd.DataFrame([fr])[FEATURES])
    wp = float(wm.predict_proba(rs)[0][1])
    sp = float(sm.predict(rs)[0])
    return {'win_prob': round(wp,4), 'spread_pred': round(sp,1), 'confidence': round(abs(wp-0.5)*200,1)}


# ══════════════════════════════════════════════════════════
# SECTION 3: HELPERS
# ══════════════════════════════════════════════════════════

def calc_ev(prob, odds):
    return round(float((prob*(odds-1))-((1-prob)*1)), 4)

def calc_kelly(prob, odds, f=0.25):
    b = odds-1
    k = (prob*b-(1-prob))/b
    return round(float(max(k*f,0)), 4)

def build_parlay(legs, min_l=2, max_l=3):
    if len(legs) < min_l: return None
    legs = sorted(legs, key=lambda x: x.get("edge",0), reverse=True)
    sel = [legs[0]]
    for leg in legs[1:]:
        if len(sel) >= max_l: break
        if leg.get("game_id","") != sel[-1].get("game_id",""):
            sel.append(leg)
    if len(sel) < min_l: return None
    cp = 1.0
    co = 1.0
    for l in sel:
        cp *= l.get("model_prob", 0.57)
        co *= l.get("best_odds", 1.90)
    return {
        "legs": sel, "num_legs": len(sel),
        "combined_odds": round(float(co),2),
        "combined_prob": round(float(cp),4),
        "combined_ev": calc_ev(cp, co),
        "kelly_stake": calc_kelly(cp, co),
        "generated_at": datetime.now(AEST).isoformat(),
    }


# ══════════════════════════════════════════════════════════
# SECTION 4: CLAUDE BRIEF
# ══════════════════════════════════════════════════════════

def generate_brief(spread_bets, prop_legs, parlay):
    if not ANTHROPIC_KEY:
        return "Anthropic API key not configured."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        spread_text = "\n".join([f"- {b['game']}: {b['bet_side']} | Edge {b['edge']}pts | Conf {b['confidence']}/100 | ${b.get('best_odds','?')} ({b.get('best_book','')})" for b in spread_bets]) or "None"
        prop_text = "\n".join([f"- {l['player']} {l['stat']} {l['side']} {l['line']} @ ${l.get('best_odds','?')} | Edge {l.get('edge',0):.1%}" for l in prop_legs]) or "None"
        parlay_text = f"{parlay['num_legs']} legs @ ${parlay['combined_odds']} | Prob {parlay['combined_prob']:.1%} | EV {parlay['combined_ev']:+.3f}" if parlay else "No qualifying parlay today."
        prompt = f"""You are a sharp NBA betting analyst. Write a 4-5 sentence morning intelligence brief. Cover what qualified today, the strongest signals, and any risks. Direct and confident. No bullet points.

Date: {datetime.now(AEST).strftime("%Y-%m-%d")}
SPREAD BETS:\n{spread_text}
PROP LEGS:\n{prop_text}
PARLAY: {parlay_text}

Write like a sharp analyst's morning note."""
        resp = client.messages.create(model="claude-sonnet-4-5", max_tokens=300, messages=[{"role":"user","content":prompt}])
        return resp.content[0].text.strip()
    except Exception as e:
        return f"Brief unavailable: {e}"


# ══════════════════════════════════════════════════════════
# SECTION 5: MAIN PIPELINE
# ══════════════════════════════════════════════════════════

def run_pipeline():
    print("\n"+"="*60)
    print(f"  🏀 EDGEMAKER — {datetime.now(AEST).strftime('%Y-%m-%d %H:%M AEST')}")
    print("="*60+"\n")

    df = fetch_nba_data()
    spread_bets = []
    prop_legs   = []

    if not df.empty:
        wm, sm, sc = train_models(df)

        # Spreads
        spread_games = []
        today_et = datetime.now(ET).strftime("%Y-%m-%d")
        try:
            resp = requests.get("https://api.the-odds-api.com/v4/sports/basketball_nba/odds", params={
                "apiKey": ODDS_API_KEY, "regions": "au", "markets": "spreads",
                "oddsFormat": "decimal", "bookmakers": ",".join(AU_BOOKS),
                "commenceTimeFrom": f"{today_et}T00:00:00Z",
                "commenceTimeTo": f"{today_et}T23:59:59Z",
            }, timeout=15)
            spread_games = resp.json() if isinstance(resp.json(), list) else []
            print(f"📊 {len(spread_games)} spread games fetched")
        except Exception as e:
            print(f"⚠️ Spreads: {e}")

        for game in spread_games:
            hf = game.get("home_team","")
            af = game.get("away_team","")
            ha = TEAM_MAP.get(hf)
            aa = TEAM_MAP.get(af)
            if not ha or not aa: continue
            pred = predict_spread(df, wm, sm, sc, ha, aa)
            if not pred: continue

            books = {}
            best_odds = None
            best_book = None
            vegas_line = None
            for bm in game.get("bookmakers",[]):
                for mkt in bm.get("markets",[]):
                    if mkt["key"]=="spreads":
                        for oc in mkt.get("outcomes",[]):
                            if oc["name"]==hf:
                                sp = oc.get("point",0)
                                pr = oc.get("price",1.90)
                                books[bm["key"]] = {"spread":sp,"odds":pr}
                                if vegas_line is None: vegas_line=sp
                                if best_odds is None or pr>best_odds:
                                    best_odds=pr; best_book=bm["key"]

            if vegas_line is None: continue
            edge = round(abs(pred['spread_pred']-vegas_line),1)
            conf = pred['confidence']
            bet_side = (f"{ha} covers {vegas_line:+.1f}" if pred['spread_pred']>vegas_line else f"{aa} covers {-vegas_line:+.1f}")
            qualifies = (SPREAD_EDGE_MIN<=edge<=SPREAD_EDGE_MAX) and (CONF_MIN<=conf<=CONF_MAX)
            reasons = []
            if not (SPREAD_EDGE_MIN<=edge<=SPREAD_EDGE_MAX): reasons.append(f"Edge {edge}pts outside 3-7")
            if not (CONF_MIN<=conf<=CONF_MAX): reasons.append(f"Conf {conf} outside 50-70")

            entry = {
                "type":"spread","game":f"{ha} vs {aa}","game_id":game.get("id",""),
                "home":ha,"away":aa,"vegas_line":f"{ha} {vegas_line:+.1f}",
                "model_spread":f"{ha} {pred['spread_pred']:+.1f}",
                "edge":edge,"win_prob":f"{pred['win_prob']*100:.1f}%",
                "confidence":conf,"qualifies":bool(qualifies),
                "verdict":"✅ BET" if qualifies else "NO BET",
                "reason":" & ".join(reasons) if reasons else "Qualifies",
                "bet_side":bet_side if qualifies else None,
                "books":books,"best_book":best_book,"best_odds":best_odds,
                "model_prob":pred['win_prob'],
            }
            if qualifies:
                spread_bets.append(entry)
                print(f"  ✅ SPREAD: {ha} vs {aa} — {bet_side} @ ${best_odds}")
            else:
                print(f"  ❌ SPREAD: {ha} vs {aa} — {' & '.join(reasons)}")

    # Props
    try:
        from scipy.stats import norm
        today_et = datetime.now(ET).strftime("%Y-%m-%d")
        resp = requests.get("https://api.the-odds-api.com/v4/sports/basketball_nba/odds", params={
            "apiKey": ODDS_API_KEY, "regions": "au",
            "markets": "player_points,player_rebounds,player_assists",
            "oddsFormat": "decimal", "bookmakers": ",".join(AU_BOOKS),
            "commenceTimeFrom": f"{today_et}T00:00:00Z",
            "commenceTimeTo": f"{today_et}T23:59:59Z",
        }, timeout=15)
        props_games = resp.json() if isinstance(resp.json(), list) else []
        print(f"💡 {len(props_games)} prop games fetched")

        stat_avgs = {"pts":18.0,"reb":6.0,"ast":4.5}
        stat_stds = {"pts":5.0,"reb":2.5,"ast":2.0}
        stat_map  = {"player_points":"pts","player_rebounds":"reb","player_assists":"ast"}
        seen = {}

        for game in props_games:
            gid = game.get("id","")
            hf  = game.get("home_team","")
            af  = game.get("away_team","")
            for bm in game.get("bookmakers",[]):
                for mkt in bm.get("markets",[]):
                    stat = stat_map.get(mkt.get("key",""))
                    if not stat: continue
                    for oc in mkt.get("outcomes",[]):
                        if oc.get("name") not in ("Over","Under"): continue
                        player = oc.get("description","")
                        line   = oc.get("point",0)
                        odds   = oc.get("price",1.90)
                        side   = oc.get("name")
                        avg    = stat_avgs[stat]
                        std    = stat_stds[stat]
                        mp = float(1-norm.cdf(line,loc=avg,scale=std)) if side=="Over" else float(norm.cdf(line,loc=avg,scale=std))
                        edge = round(float(mp - 1/odds),4)
                        if mp >= PROP_PROB_MIN and edge >= PROP_EDGE_MIN:
                            key = f"{player}|{stat}|{side}"
                            leg = {
                                "type":"prop","player":player,"stat":stat,"side":side,
                                "line":line,"model_prob":round(mp,4),
                                "implied_prob":round(1/odds,4),"edge":edge,
                                "ev":calc_ev(mp,odds),"kelly":calc_kelly(mp,odds),
                                "best_odds":odds,"best_book":bm.get("key",""),
                                "game_id":gid,"matchup":f"{hf} vs {af}","qualifies":True,
                            }
                            if key not in seen or edge > seen[key]["edge"]:
                                seen[key] = leg

        prop_legs = sorted(seen.values(), key=lambda x: x["edge"], reverse=True)[:15]
        print(f"  ✅ {len(prop_legs)} qualifying prop legs")
    except ImportError:
        print("⚠️ scipy not available — skipping props")
    except Exception as e:
        print(f"⚠️ Props failed: {e}")

    # Parlay
    parlay = build_parlay(spread_bets + prop_legs)
    if parlay:
        print(f"  🎯 Parlay: {parlay['num_legs']} legs @ ${parlay['combined_odds']}")

    # Brief
    print("\n🧠 Generating Claude brief...")
    brief = generate_brief(spread_bets, prop_legs[:5], parlay)

    # Save
    os.makedirs("data", exist_ok=True)
    results = {"daily_log":[],"bets":[]} if not os.path.exists(RESULTS_FILE) else json.load(open(RESULTS_FILE))

    def convert(obj):
        if isinstance(obj,(np.integer,)): return int(obj)
        if isinstance(obj,(np.floating,)): return float(obj)
        if isinstance(obj,(np.bool_,)): return bool(obj)
        if isinstance(obj,np.ndarray): return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    aest_date = datetime.now(AEST).strftime("%Y-%m-%d")
    log = {
        "date": aest_date,
        "day_number": len(results["daily_log"])+1,
        "run_time_aest": datetime.now(AEST).strftime("%H:%M"),
        "daily_brief": brief,
        "spread_bets": spread_bets,
        "prop_legs": prop_legs,
        "parlay": parlay,
        "total_spread_bets": len(spread_bets),
        "total_prop_legs": len(prop_legs),
        "parlay_found": parlay is not None,
    }
    existing = [e for e in results["daily_log"] if e["date"] != aest_date]
    results["daily_log"] = existing + [log]

    with open(RESULTS_FILE,"w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\n✅ Done — {len(spread_bets)} spreads | {len(prop_legs)} props | Parlay: {'Yes' if parlay else 'No'}")
    print("="*60+"\n")

if __name__ == "__main__":
    run_pipeline()
