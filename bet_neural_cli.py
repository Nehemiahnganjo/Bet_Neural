#!/usr/bin/env python3
"""
bet_neural_cli.py — Bet Neural Command-Line Interface v3
=========================================================
Commands: predict, analytics, gameweek, scrape, train, odds, portfolio, leagues, benchmark, status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).parent


def _load_dotenv() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path) as fh:
        for line in fh:
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _parse_match(match_str: str):
    for sep in (" vs ", " v ", " - ", "-"):
        if sep in match_str:
            home, away = match_str.split(sep, 1)
            return home.strip(), away.strip()
    raise ValueError(f"Cannot parse match '{match_str}'. Use 'Team1 vs Team2' format.")


def _parse_odds(odds_str: str) -> Optional[Dict[str, float]]:
    if not odds_str:
        return None
    try:
        parts = [float(x.strip()) for x in odds_str.split(",")]
        if len(parts) == 3:
            return {"home_win": parts[0], "draw": parts[1], "away_win": parts[2]}
    except Exception:
        pass
    return None


def _league_key(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _display_prediction(result: Dict, verbose: bool = True) -> None:
    print()
    for w in result.get("warnings", []):
        print(w)
    if result.get("warnings"):
        print()

    print("🎯 BET NEURAL PREDICTION")
    print("=" * 72)
    print(f"🏟️  {result['match']}")
    if result.get("match_input") and result["match_input"] != result["match"]:
        print(f"   (entered as: {result['match_input']})")
    print(f"🏆 League:  {result['league'].replace('_', ' ').title()}")
    print(f"🔧 Engine:  {result.get('prediction_engine', 'elo')}")
    print(f"⏰ Time:    {result['prediction_time'][:19]}")
    print()

    elo = result["elo_ratings"]
    print("📊 ELO RATINGS:")
    h_name, a_name = result["match"].split(" vs ")
    print(f"   🏠 {h_name:<25} {elo['home']:.0f}")
    print(f"   ✈️  {a_name:<25} {elo['away']:.0f}")
    
    # Display form analysis if available
    form_analysis = result.get("form_analysis", {})
    if form_analysis:
        print()
        print("📈 FORM ANALYSIS:")
        if 'home_summary' in form_analysis:
            print(f"   🏠 {h_name:<25} {form_analysis['home_summary']}")
        if 'away_summary' in form_analysis:
            print(f"   ✈️  {a_name:<25} {form_analysis['away_summary']}")
        if 'home_factor' in form_analysis and 'away_factor' in form_analysis:
            home_adj = "📈" if form_analysis['home_factor'] > 1.0 else "📉" if form_analysis['home_factor'] < 0.95 else "➡️"
            away_adj = "📈" if form_analysis['away_factor'] > 1.0 else "📉" if form_analysis['away_factor'] < 0.95 else "➡️"
            print(f"   🔧 Adjustments: {home_adj} Home {form_analysis['home_factor']:.2f}x  |  {away_adj} Away {form_analysis['away_factor']:.2f}x")
    
    print()

    print("🎲 OUTCOME PROBABILITIES:")
    probs = result["probabilities"]
    sorted_ = sorted(probs.items(), key=lambda x: -x[1])
    medals = ["🥇", "🥈", "🥉"]
    for i, (outcome, prob) in enumerate(sorted_):
        bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
        medal = medals[i] if i < 3 else "  "
        label = outcome.replace("_", " ").title()
        print(f"   {medal} {label:<12} {prob:6.1%}  {bar}")

    xg = result.get("expected_goals", {})
    print()
    print(f"⚽ EXPECTED GOALS: {xg.get('home', 0):.2f} – {xg.get('away', 0):.2f}")
    print(f"🔥 CONFIDENCE:    {result['confidence']:.1%}")

    ba = result.get("betting_analysis")
    if ba:
        print()
        print("💰 BETTING ANALYSIS:")
        print("-" * 50)
        outcomes = ba.get("outcomes", ba)
        if isinstance(outcomes, dict):
            for outcome, data in outcomes.items():
                if not isinstance(data, dict):
                    continue
                if data.get("decimal_odds", 0) > 1.0:
                    print(f"\n  {outcome.replace('_',' ').title():<14}"
                          f" @ {data.get('decimal_odds', 0):.2f}"
                          f"  Model: {data.get('model_prob', 0):.1%}"
                          f"  Edge: {data.get('edge', 0):+.1%}")
                    verdict = data.get("verdict") or ("✅ VALUE" if data.get("recommended") else "➖ NO VALUE")
                    print(f"  {'':14}  {verdict}")
                    if data.get("recommended"):
                        kelly = data.get("kelly_stake_pct", 0)
                        ev = data.get("expected_value", 0)
                        print(f"  {'':14}  Kelly: {kelly:.2f}% of bankroll  |  EV: {ev:+.3f}")

        best = ba.get("best_bet")
        if best:
            print()
            print(f"  ⭐ BEST BET: {best['outcome'].replace('_', ' ').title()}"
                  f" @ {best.get('decimal_odds', 0):.2f}"
                  f"  Edge: {best.get('edge', 0):+.1%}"
                  f"  Kelly: {best.get('kelly_stake_pct', 0):.2f}%")
        else:
            print()
            print("  ❌ No value bets at current odds")

    if verbose and "report" in result:
        print()
        print(result["report"])

    print()
    print("🤖 Bet Neural v4 — Calibrated Elo + Deterministic MC + xG + Kelly")


def cmd_predict(args: argparse.Namespace) -> None:
    from bet_neural import BetNeuralPredictor
    home, away = _parse_match(args.match)
    league = _league_key(args.league)
    odds = _parse_odds(args.odds) if args.odds else None
    full_rep = getattr(args, "report", False) or getattr(args, "analytics", False)

    predictor = BetNeuralPredictor()
    result = predictor.predict_match(home, away, league, odds=odds,
                                     match_date=getattr(args, "date", None),
                                     full_report=full_rep)
    _display_prediction(result, verbose=full_rep)


def cmd_analytics(args: argparse.Namespace) -> None:
    args.report = True
    cmd_predict(args)


def cmd_gameweek(args: argparse.Namespace) -> None:
    from bet_neural import BetNeuralPredictor
    from scraper import FootballDataClient

    league = _league_key(args.league)
    predictor = BetNeuralPredictor()

    fixtures = []
    try:
        client = FootballDataClient()
        fixtures_data = client.fetch_matches(league, status='SCHEDULED', days_back=0, limit=30)
        fixtures = [(f['home_team'], f['away_team']) for f in fixtures_data[:8]]
    except Exception:
        pass

    if not fixtures:
        fixtures = [
            ("Arsenal", "Chelsea"), ("Liverpool", "Man Utd"), ("Man City", "Spurs"),
            ("Newcastle", "Brighton"), ("West Ham", "Aston Villa"),
        ]

    print(f"\n🗓️  GAMEWEEK PREDICTIONS — {args.league.upper()}")
    print("=" * 60)

    total_conf = 0.0
    n_value_bets = 0

    for home, away in fixtures:
        try:
            odds = None
            result = predictor.predict_match(home, away, league, odds=odds)
            best_out = max(result["probabilities"], key=result["probabilities"].get)
            best_prob = result["probabilities"][best_out]
            total_conf += result["confidence"]

            engine_tag = "🤖" if "ml" in result.get("prediction_engine", "") else "📊"
            print(f"\n{engine_tag} {home} vs {away}")
            print(f"   Prediction: {best_out.replace('_',' ').title()} ({best_prob:.1%})")
            xg = result["expected_goals"]
            print(f"   xG: {xg['home']:.2f} – {xg['away']:.2f}")

            ba = result.get("betting_analysis")
            if ba:
                best = ba.get("best_bet")
                if best:
                    n_value_bets += 1
                    print(f"   💰 Value: {best['outcome'].replace('_',' ').title()}"
                          f" @ {best.get('decimal_odds', 0):.2f}"
                          f"  edge {best.get('edge', 0):+.1%}")
        except Exception as e:
            print(f"\n❌ {home} vs {away}: {e}")

    n = len(fixtures)
    if n > 0:
        print(f"\n{'─'*60}")
        print(f"📊 Gameweek Summary:")
        print(f"   Matches predicted: {n}")
        print(f"   Average confidence: {total_conf / n:.1%}")
        print(f"   Value bets found: {n_value_bets}")


def cmd_scrape(args: argparse.Namespace) -> None:
    from scraper import BetNeuralScraper, DatahubScraper, LEAGUE_META

    league = _league_key(args.league)
    leagues = list(LEAGUE_META.keys()) if getattr(args, "all", False) else [league]
    season = getattr(args, "season", "2024-2025")

    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print("⚠️  No FOOTBALL_DATA_API_KEY set. Using mock data fallback.")

    scraper = BetNeuralScraper(api_key=api_key, cache_dir=str(BASE_DIR / "data" / "cache"))
    dh_scraper = DatahubScraper()

    for lg in leagues:
        print(f"\n🔄 Scraping {lg} ({season})...")
        
        # Scrape API for recent season
        result = scraper.scrape_league(lg, season)
        summary = result.get("summary", {})
        print(f"✅ {lg} API:")
        print(f"   Match records:  {summary.get('matches', 0)}")
        print(f"   Teams in table: {summary.get('standings', 0)}")
        print(f"   Mock data:      {'Yes' if summary.get('mock_data') else 'No'}")
        
        # Also fetch historical data from datahub.io for better training
        # Fetch last 3 seasons for more training data
        historical = []
        try:
            current_year = int(season.split("-")[0])
            for y in range(current_year - 3, current_year):
                hist_season = f"{y}-{y+1}"
                hist_matches = dh_scraper.fetch_season(lg, hist_season)
                historical.extend(hist_matches)
        except Exception:
            pass
        
        if historical:
            print(f"   Historical:     {len(historical)} matches from 3 seasons")


def cmd_train(args: argparse.Namespace) -> None:
    from bet_neural import train_league
    from scraper import LEAGUE_META

    league = _league_key(args.league)
    leagues = list(LEAGUE_META.keys()) if getattr(args, "all", False) else [league]
    season = getattr(args, "season", "2024-2025")

    for lg in leagues:
        print(f"\n{'='*60}")
        print(f"🧠 Training ML models for {lg} ({season})...")
        print(f"{'='*60}")
        result = train_league(lg, season)

        if "error" in result:
            print(f"❌ {lg} training failed:")
            print(f"   {result['error']}")
            print(f"   Run: python3 bet_neural_cli.py scrape --league {lg}")
            continue

        weights = result.get('weights', {})
        total_matches = result.get('total_matches', result.get('n_train', 0))
        print(f"\n✅ {lg} training complete!")
        print(f"   Total training matches: {total_matches}")
        print(f"\n📊 Validation Metrics:")
        print(f"   Accuracy:    {result.get('accuracy', 0):>7.3f}")
        print(f"   Log-loss:    {result.get('log_loss', 0):>7.4f}")
        print(f"   Brier:       {result.get('brier_score', 0):>7.4f}")
        print(f"   ROC AUC:     {result.get('roc_auc', 0):>7.4f}")
        print(f"\n⚖️  Model Weights:")
        for model, w in weights.items():
            print(f"   {model.upper():<8} {float(w):>7.2%}")
        print(f"\n📊 Training Stats:")
        print(f"   Train samples: {result.get('n_train', 0):>5}")
        print(f"   Val samples:   {result.get('n_val', 0):>5}")


def cmd_standings(args: argparse.Namespace) -> None:
    """Show current league standings from football-data.org API."""
    from scraper import FootballDataClient
    league = _league_key(args.league)
    try:
        client = FootballDataClient()
        standings = client.fetch_standings(league)
    except Exception as e:
        print(f"❌ Could not fetch standings: {e}")
        print("   Check FOOTBALL_DATA_API_KEY is set in .env")
        return

    if not standings:
        print(f"❌ No standings data returned for {args.league}")
        return

    print(f"\n🏆 {args.league} — Current Standings")
    print("  Pos  Team                    Pld   W   D   L    GD  Pts")
    print("  " + "─" * 60)
    for s in standings:
        print(
            f"  {s['position']:>3}  {s['team']:<25} "
            f"{s['played']:>3}  {s['won']:>3}  {s['drawn']:>3}  {s['lost']:>3}  "
            f"{s['gd']:>+4}  {s['points']:>3}"
        )


def cmd_portfolio(args: argparse.Namespace) -> None:
    print("📈 Portfolio management coming soon")
    print("   Use: ./bet predict to generate betting recommendations")


def cmd_leagues(args: argparse.Namespace) -> None:
    from bet_neural import LEAGUES
    print("\n🏆 SUPPORTED EUROPEAN LEAGUES")
    print("=" * 60)
    print(f"  {'Key':<20} {'Name':<18} {'Country':<13} {'Strength'}")
    print("  " + "-" * 56)
    for key, meta in LEAGUES.items():
        print(f"  {key:<20} {meta['name']:<18} {meta['country']:<13} {meta['strength']:.0%}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from bet_neural import BetNeuralPredictor
    import time

    print("\n🧪 BET NEURAL BENCHMARK")
    print("=" * 50)

    test_matches = [
        ("Arsenal", "Chelsea", "premier_league"),
        ("Real Madrid", "Barcelona", "la_liga"),
        ("Bayern Munich", "Borussia Dortmund", "bundesliga"),
        ("Inter Milan", "Juventus", "serie_a"),
    ]

    predictor = BetNeuralPredictor()
    confidences = []

    for home, away, league in test_matches:
        t0 = time.time()
        try:
            result = predictor.predict_match(home, away, league)
            elapsed = time.time() - t0
            confidence = result["confidence"]
            engine = result.get("prediction_engine", "elo")
            confidences.append(confidence)
            print(f"✅ {home:<20} vs {away:<22} {confidence:.1%}  [{engine}]  {elapsed*1000:.0f}ms")
        except Exception as e:
            print(f"❌ {home} vs {away}: {e}")
            confidences.append(0.5)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    rating = "🟢 Excellent" if avg_conf > 0.7 else "🟡 Good" if avg_conf > 0.6 else "🔴 Needs data"

    print(f"\n📈 SYSTEM STATUS:")
    print(f"   Average confidence:  {avg_conf:.1%}")
    print(f"   Overall rating:      {rating}")
    print(f"   Engine:              {predictor._model_manager is not None and 'ML' or 'Elo fallback'}")


def cmd_status(args: argparse.Namespace) -> None:
    import time
    from scraper import RAW_DIR

    print("\n📂 BET NEURAL DATA STATUS")
    print("=" * 60)

    if RAW_DIR.exists():
        for subdir in sorted(RAW_DIR.iterdir()):
            if subdir.is_dir():
                files = list(subdir.glob("*.json"))
                if files:
                    total_kb = sum(f.stat().st_size for f in files) // 1024
                    print(f"\n  📁 {subdir.name}/ ({len(files)} files, {total_kb}KB)")
                    for f in sorted(files)[:5]:
                        age_h = (time.time() - f.stat().st_mtime) / 3600
                        kb = f.stat().st_size // 1024
                        print(f"     {f.name:<50} {kb:>5}KB  {age_h:.1f}h ago")
                    if len(files) > 5:
                        print(f"     ... and {len(files) - 5} more")
    else:
        print("  No scraped data yet. Run: python3 bet_neural_cli.py scrape --all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bet_neural_cli",
        description="Bet Neural v4 — Calibrated European Football Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # predict
    p_pred = sub.add_parser("predict", help="Predict a match outcome")
    p_pred.add_argument("match", help='Match: "Home vs Away"')
    p_pred.add_argument("--league", "-l", default="Premier League")
    p_pred.add_argument("--odds", "-o", help="Decimal odds: home,draw,away")
    p_pred.add_argument("--date", "-d", help="Match date (YYYY-MM-DD)")
    p_pred.add_argument("--report", "-r", action="store_true", help="Full analytics report")

    # analytics
    p_ana = sub.add_parser("analytics", help="Full match analytics & value betting")
    p_ana.add_argument("match", help='Match: "Home vs Away"')
    p_ana.add_argument("--league", "-l", default="Premier League")
    p_ana.add_argument("--odds", "-o", help="Decimal odds: home,draw,away")
    p_ana.add_argument("--date", "-d", help="Match date (YYYY-MM-DD)")

    # gameweek
    p_gw = sub.add_parser("gameweek", help="Predict a full gameweek")
    p_gw.add_argument("--league", "-l", default="Premier League")

    # scrape
    p_scr = sub.add_parser("scrape", help="Scrape data from football-data.org API")
    p_scr.add_argument("--league", "-l", default="premier_league")
    p_scr.add_argument("--season", "-s", default="2024-2025")
    p_scr.add_argument("--all", "-a", action="store_true", help="Scrape all leagues")

    # train
    p_tr = sub.add_parser("train", help="Train ML models on scraped data")
    p_tr.add_argument("--league", "-l", default="premier_league")
    p_tr.add_argument("--season", "-s", default="2024-2025")
    p_tr.add_argument("--all", "-a", action="store_true")

    # standings (was "odds" — renamed for clarity)
    p_standings = sub.add_parser("standings", help="Show current league standings")
    p_standings.add_argument("--league", "-l", default="Premier League",
                             help="League name (e.g. 'Premier League', 'La Liga')")
    # Keep 'odds' as a hidden alias so existing scripts don't break
    p_odds_alias = sub.add_parser("odds", help=argparse.SUPPRESS)
    p_odds_alias.add_argument("--league", "-l", default="Premier League")

    # portfolio
    p_port = sub.add_parser("portfolio", help="Manage betting portfolio")
    p_port.add_argument("subcommand", nargs="?", default="summary",
                        choices=["summary", "reset"])

    # leagues
    sub.add_parser("leagues", help="List supported leagues")

    # benchmark
    sub.add_parser("benchmark", help="Run system benchmark")

    # status
    sub.add_parser("status", help="Show data cache status")

    return parser


def main():
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    command = args.command

    dispatch = {
        "predict": cmd_predict,
        "analytics": cmd_analytics,
        "gameweek": cmd_gameweek,
        "scrape": cmd_scrape,
        "train": cmd_train,
        "standings": cmd_standings,
        "odds": cmd_standings,   # alias — same handler
        "portfolio": cmd_portfolio,
        "leagues": cmd_leagues,
        "benchmark": cmd_benchmark,
        "status": cmd_status,
    }

    handler = dispatch.get(command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
