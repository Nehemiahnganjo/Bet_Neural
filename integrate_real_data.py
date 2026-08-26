#!/usr/bin/env python3
"""
integrate_real_data.py — Load real match data and rebuild Elo ratings
=======================================================================

Usage:
    python integrate_real_data.py
    python integrate_real_data.py --test-predict "Arsenal" "Chelsea" "Premier League"
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from bet_neural import BetNeuralPredictor
from features import MatchHistory


def load_validated_data() -> List[Dict]:
    """Load validated match data from pickle."""
    pkl_path = Path("data/processed/matches.pkl")
    if not pkl_path.exists():
        print(f"❌ No validated data at {pkl_path}")
        raise FileNotFoundError(f"Run load_match_data.py first")

    with open(pkl_path, "rb") as fh:
        matches = pickle.load(fh)

    print(f"✅ Loaded {len(matches)} validated matches from {pkl_path}")
    return matches


def transform_to_features_format(matches: List[Dict]) -> List[Dict]:
    """
    Transform raw API matches to features.py format.
    
    Renames fields to match MatchHistory.load() expectations:
    - date_utc → date (ISO string)
    - home_goals/away_goals already canonical
    """
    transformed = []
    for m in matches:
        # Skip if season is out of range (2026-2027 future data)
        season = m.get("season")
        if season not in ("2023-2024", "2024-2025", "2025-2026"):
            continue

        # Convert UTC date to local ISO format (remove Z)
        date_utc = m.get("date_utc", "")
        try:
            dt = datetime.fromisoformat(date_utc.replace("Z", "+00:00"))
            date_str = dt.date().isoformat()  # YYYY-MM-DD format
        except (ValueError, AttributeError):
            continue

        transformed.append({
            "id":            m.get("id"),
            "league":        m.get("league"),
            "season":        m.get("season"),
            "date":          date_str,          # MatchHistory expects this
            "date_utc":      date_utc,
            "home_team":     m.get("home_team"),
            "away_team":     m.get("away_team"),
            "home_goals":    m.get("home_goals"),
            "away_goals":    m.get("away_goals"),
            "result":        m.get("result"),
            "xg_home":       m.get("xg_home"),
            "xg_away":       m.get("xg_away"),
        })

    print(f"✅ Transformed {len(transformed)} matches to features format")
    return transformed


def group_by_league(matches: List[Dict]) -> Dict[str, List[Dict]]:
    """Group matches by league."""
    by_league = {}
    for m in matches:
        league = m.get("league", "unknown")
        if league not in by_league:
            by_league[league] = []
        by_league[league].append(m)

    print(f"\n📊 Matches by league:")
    for league in sorted(by_league.keys()):
        print(f"   {league:20s}: {len(by_league[league]):5d} matches")

    return by_league


def compute_elo_ratings(matches_by_league: Dict[str, List[Dict]]) -> Dict[str, float]:
    """
    Compute Elo ratings for all teams from match history.
    
    Uses the standard Elo formula:
      new_rating = old_rating + K * (actual_result - expected_result)
    """
    from bet_neural import _elo_update

    ratings = {}
    K = 32  # Standard Elo K-factor

    for league, matches in matches_by_league.items():
        print(f"\n🏆 Computing Elo for {league}...")
        league_ratings = {}

        # Sort by date to process chronologically
        sorted_matches = sorted(matches, key=lambda m: m.get("date", ""))

        for m in sorted_matches:
            home_team = m.get("home_team", "").lower()
            away_team = m.get("away_team", "").lower()
            result = m.get("result")

            if not home_team or not away_team or result not in ("H", "D", "A"):
                continue

            # Initialize teams at 1500 if first match
            if home_team not in league_ratings:
                league_ratings[home_team] = 1500.0
            if away_team not in league_ratings:
                league_ratings[away_team] = 1500.0

            # Compute new ratings
            h_old = league_ratings[home_team]
            a_old = league_ratings[away_team]

            # Expected result (from Elo formula)
            expected_home = 1.0 / (1.0 + 10.0 ** ((a_old - h_old) / 400.0))
            expected_away = 1.0 - expected_home

            # Actual result
            if result == "H":
                actual_home = 1.0
                actual_away = 0.0
            elif result == "D":
                actual_home = 0.5
                actual_away = 0.5
            else:  # A
                actual_home = 0.0
                actual_away = 1.0

            # Update ratings
            h_new = h_old + K * (actual_home - expected_home)
            a_new = a_old + K * (actual_away - expected_away)

            league_ratings[home_team] = h_new
            league_ratings[away_team] = a_new

        # Store with league suffix
        for team, rating in league_ratings.items():
            key = f"{team}_{league}"
            ratings[key] = rating

        print(f"   ✅ Rated {len(league_ratings)} teams")

    return ratings


def save_elo_to_system(ratings: Dict[str, float]) -> None:
    """Save computed Elo ratings to elo_ratings.json (not bet_neural.py directly)."""
    from pathlib import Path

    output_path = Path("elo_ratings.json")

    # Save to JSON file
    data = {"ratings": ratings, "timestamp": str(datetime.now())}
    with open(output_path, "w") as fh:
        json.dump(data, fh, indent=2)

    print(f"✅ Saved {len(ratings)} Elo ratings to {output_path}")
    print(f"   Restart Bet Neural to load these ratings")



def test_prediction(home: str, away: str, league: str) -> None:
    """Test a single prediction with real data."""
    print(f"\n🎯 Testing prediction: {home} vs {away} ({league})")
    print("=" * 70)

    predictor = BetNeuralPredictor()
    result = predictor.predict_match(home, away, league)

    print(f"\n📊 PREDICTION RESULT:")
    print(f"   Home Win:  {result['probs']['home_win']:.1%}")
    print(f"   Draw:      {result['probs']['draw']:.1%}")
    print(f"   Away Win:  {result['probs']['away_win']:.1%}")
    print(f"   xG:        {result['xg'][0]:.2f} - {result['xg'][1]:.2f}")
    print(f"   Confidence: {result['confidence']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Integrate real match data into Bet Neural")
    parser.add_argument("--test-predict", nargs=3, help="Test prediction: HOME AWAY LEAGUE")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🏟️  BET NEURAL — REAL DATA INTEGRATION")
    print("=" * 70)

    # 1. Load validated data
    print("\n[1/5] Loading validated data...")
    matches = load_validated_data()

    # 2. Transform to features format
    print("\n[2/5] Transforming to features format...")
    matches_transformed = transform_to_features_format(matches)

    # 3. Group by league
    print("\n[3/5] Grouping by league...")
    by_league = group_by_league(matches_transformed)

    # 4. Compute Elo ratings
    print("\n[4/5] Computing Elo ratings...")
    elo_ratings = compute_elo_ratings(by_league)
    print(f"\n✅ Total Elo-rated teams: {len(elo_ratings)}")

    # 5. Save to system
    print("\n[5/5] Saving to system...")
    save_elo_to_system(elo_ratings)

    print("\n" + "=" * 70)
    print("✅ INTEGRATION COMPLETE!")
    print("=" * 70)

    # Optional: test prediction
    if args.test_predict:
        home, away, league = args.test_predict
        test_prediction(home, away, league)


if __name__ == "__main__":
    main()
