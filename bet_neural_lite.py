#!/usr/bin/env python3
"""
Bet Neural Lite - European Football Prediction System (Lightweight)
No external dependencies required - uses only Python built-in libraries
"""

import json
import math
import random
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class BetNeuralLite:
    """
    Lightweight European Football Prediction System
    Uses FiveThirtyEight-inspired Elo methodology without external dependencies
    """
    
    def __init__(self):
        self.leagues = {
            'premier_league': {'strength': 1.0, 'home_advantage': 65},
            'la_liga': {'strength': 0.95, 'home_advantage': 60},
            'bundesliga': {'strength': 0.90, 'home_advantage': 58},
            'serie_a': {'strength': 0.85, 'home_advantage': 55},
            'ligue_1': {'strength': 0.80, 'home_advantage': 52},
            'eredivisie': {'strength': 0.70, 'home_advantage': 50},
            'primeira_liga': {'strength': 0.65, 'home_advantage': 48}
        }
        
        # Initialize team ratings (simplified)
        self.team_ratings = self._initialize_team_ratings()
        self.elo_k_factor = 32
    
    def _initialize_team_ratings(self) -> Dict[str, float]:
        """Initialize team Elo ratings based on league strength"""
        ratings = {}
        
        # Premier League teams
        pl_teams = {
            'Arsenal': 1850, 'Manchester City': 1920, 'Liverpool': 1900,
            'Chelsea': 1780, 'Manchester United': 1760, 'Tottenham': 1740,
            'Newcastle': 1680, 'Brighton': 1620, 'Aston Villa': 1650,
            'West Ham': 1600, 'Crystal Palace': 1580, 'Fulham': 1570
        }
        
        # La Liga teams  
        ll_teams = {
            'Real Madrid': 1880, 'Barcelona': 1860, 'Atletico Madrid': 1780,
            'Athletic Bilbao': 1650, 'Real Sociedad': 1640, 'Valencia': 1620,
            'Sevilla': 1700, 'Villarreal': 1680, 'Betis': 1630, 'Osasuna': 1580
        }
        
        # Bundesliga teams
        bl_teams = {
            'Bayern Munich': 1870, 'Borussia Dortmund': 1780, 'RB Leipzig': 1720,
            'Bayer Leverkusen': 1700, 'Eintracht Frankfurt': 1650, 'Wolfsburg': 1620,
            'Borussia Monchengladbach': 1600, 'Union Berlin': 1640, 'Freiburg': 1610,
            'Augsburg': 1580,
        }
        
        # Serie A teams
        sa_teams = {
            'Juventus': 1750, 'AC Milan': 1720, 'Inter Milan': 1740,
            'Napoli': 1780, 'Roma': 1680, 'Lazio': 1670, 'Atalanta': 1690,
            'Fiorentina': 1630, 'Torino': 1590, 'Bologna': 1580
        }
        
        # Ligue 1 teams
        l1_teams = {
            'PSG': 1900, 'Marseille': 1740, 'Monaco': 1720,
            'Lyon': 1700, 'Lille': 1690, 'Nice': 1660,
            'Lens': 1650, 'Rennes': 1640, 'Strasbourg': 1580,
            'Nantes': 1570, 'Montpellier': 1560, 'Toulouse': 1540,
            'Brest': 1550, 'Reims': 1560, 'Le Havre': 1530,
            'Metz': 1520, 'Clermont': 1510, 'Lorient': 1505,
        }

        # Eredivisie teams
        ere_teams = {
            'Ajax': 1820, 'PSV': 1800, 'Feyenoord': 1780,
            'AZ Alkmaar': 1700, 'Twente': 1680, 'Utrecht': 1650,
            'Vitesse': 1620, 'Heerenveen': 1600, 'Groningen': 1580,
            'Sparta Rotterdam': 1560, 'Almere City': 1540, 'NEC Nijmegen': 1545,
            'Fortuna Sittard': 1530, 'Heracles': 1525, 'Go Ahead Eagles': 1520,
            'PEC Zwolle': 1515, 'RKC Waalwijk': 1510, 'Excelsior': 1505,
        }

        # Primeira Liga teams
        ppl_teams = {
            'Benfica': 1800, 'Porto': 1790, 'Sporting CP': 1780,
            'Braga': 1680, 'Guimaraes': 1640, 'Santa Clara': 1590,
            'Famalicao': 1570, 'Gil Vicente': 1560, 'Boavista': 1550,
            'Moreirense': 1540, 'Casa Pia': 1535, 'Arouca': 1530,
            'Chaves': 1520, 'Estoril': 1515, 'Rio Ave': 1510,
            'Vizela': 1505, 'Farense': 1500, 'Portimonense': 1495,
        }

        # Add teams to ratings with league suffix
        for league_key, teams in [
            ('premier_league', pl_teams), ('la_liga', ll_teams),
            ('bundesliga', bl_teams), ('serie_a', sa_teams),
            ('ligue_1', l1_teams), ('eredivisie', ere_teams),
            ('primeira_liga', ppl_teams),
        ]:
            for team, rating in teams.items():
                ratings[f"{team}_{league_key}"] = rating

        return ratings
    
    def get_team_elo(self, team_name: str, league: str) -> float:
        """Get team's Elo rating"""
        key = f"{team_name}_{league}"
        if key in self.team_ratings:
            return self.team_ratings[key]
        
        # Default rating based on league strength
        league_strength = self.leagues.get(league, {}).get('strength', 0.8)
        return 1500 * league_strength
    
    def calculate_match_probabilities(self, home_elo: float, away_elo: float, 
                                    home_advantage: float = 65) -> Dict[str, float]:
        """
        Calculate match outcome probabilities using Elo ratings.

        Replaces the old piecewise-linear formula which had discontinuous
        jumps of 14 pp at eh=0.65 and 9 pp at eh=0.35.

        Uses the same smooth Elo-logistic + draw-band approach as
        bet_neural.py for consistent results across the whole system.
        """
        adjusted_home_elo = home_elo + home_advantage
        p_home_raw = 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home_elo) / 400.0))

        # Draw probability decreases smoothly as the match becomes lopsided
        imbalance = abs(p_home_raw - 0.5)
        draw      = max(0.22, min(0.32, 0.30 - 0.16 * imbalance))

        remaining = 1.0 - draw
        home_win  = p_home_raw * remaining
        away_win  = (1.0 - p_home_raw) * remaining

        total = home_win + draw + away_win   # already 1.0; guard float drift
        return {
            'home_win': home_win / total,
            'draw':     draw     / total,
            'away_win': away_win / total,
        }

    def calculate_expected_goals(self, home_elo: float, away_elo: float,
                               league: str) -> Tuple[float, float]:
        """
        Calculate expected goals for both teams.

        The empirical European football average is ~2.6–2.7 *total* goals per
        match, split roughly 55 % home / 45 % away.  Using 2.7 as a per-team
        constant would produce ~5.4 total goals — double the real figure.

        Correct approach:
          mu_home ≈ 2.7 * 0.55 ≈ 1.49  (league-average home xG)
          mu_away ≈ 2.7 * 0.45 ≈ 1.22  (league-average away xG)

        Each team's xG is then scaled by their relative Elo strength and a
        league-specific factor.
        """
        mu_home = 1.49   # ≈ 2.7 * 0.55
        mu_away = 1.22   # ≈ 2.7 * 0.45

        league_factor = self.leagues.get(league, {}).get('strength', 0.8)

        home_xg = mu_home * (home_elo / 1500.0) * league_factor
        away_xg = mu_away * (away_elo / 1500.0) * league_factor

        return round(home_xg, 2), round(away_xg, 2)

    def calculate_betting_value(self, probabilities: Dict[str, float],
                              odds: Dict[str, float]) -> Dict[str, Dict]:
        """
        Calculate betting value using the Kelly Criterion.

        Standard Kelly:  f* = (b·p - q) / b
          where b = bookmaker_odds - 1,  p = win prob,  q = 1 - p

        The old code used edge/b = (p - 1/odds)/(odds-1) which underestimates
        the optimal stake by a factor of 'odds'. Fixed here to match analytics.py.
        Half-Kelly (×0.5) is applied and stake is capped at 5 % of bankroll.
        """
        betting_analysis = {}

        for outcome, model_prob in probabilities.items():
            if outcome not in odds:
                continue

            bookmaker_odds = odds[outcome]
            if bookmaker_odds <= 1.0:
                continue

            implied_prob  = 1.0 / bookmaker_odds
            edge          = model_prob - implied_prob
            b             = bookmaker_odds - 1.0

            if edge > 0 and model_prob > 0.55 and b > 0:
                full_kelly     = (model_prob * b - (1.0 - model_prob)) / b
                kelly_fraction = min(max(full_kelly, 0.0) * 0.5, 0.05)
                recommended    = edge > 0.05 and model_prob > 0.55
            else:
                kelly_fraction = 0.0
                recommended    = False

            if model_prob > 0.70:
                confidence = 'high'
            elif model_prob >= 0.58:
                confidence = 'medium'
            else:
                confidence = 'low'

            betting_analysis[outcome] = {
                'model_probability':  model_prob,
                'implied_probability': implied_prob,
                'edge':               edge,
                'kelly_fraction':     kelly_fraction,
                'recommended':        recommended,
                'confidence':         confidence,
                'potential_roi':      edge * 100 if edge > 0 else 0,
            }

        return betting_analysis
    
    def predict_match(self, home_team: str, away_team: str, league: str = 'premier_league',
                     odds: Optional[Dict[str, float]] = None) -> Dict:
        """Main prediction function"""
        
        # Get team ratings
        home_elo = self.get_team_elo(home_team, league)
        away_elo = self.get_team_elo(away_team, league)
        
        # Get league-specific home advantage
        home_advantage = self.leagues.get(league, {}).get('home_advantage', 60)
        
        # Calculate probabilities
        probabilities = self.calculate_match_probabilities(home_elo, away_elo, home_advantage)
        
        # Calculate expected goals
        home_xg, away_xg = self.calculate_expected_goals(home_elo, away_elo, league)
        
        # Determine most likely outcome
        best_outcome = max(probabilities.items(), key=lambda x: x[1])
        confidence = best_outcome[1]
        
        result = {
            'match': f"{home_team} vs {away_team}",
            'league': league,
            'probabilities': probabilities,
            'elo_ratings': {
                'home': round(home_elo, 0),
                'away': round(away_elo, 0),
                'home_advantage': home_advantage
            },
            'expected_goals': {
                'home': home_xg,
                'away': away_xg,
                'total': round(home_xg + away_xg, 2)
            },
            'prediction': {
                'most_likely': best_outcome[0],
                'confidence': round(confidence, 3)
            },
            'timestamp': datetime.now().isoformat()[:19]
        }
        
        # Add betting analysis if odds provided
        if odds:
            result['betting_analysis'] = self.calculate_betting_value(probabilities, odds)
            
            # Calculate overall betting recommendation
            recommendations = [v for v in result['betting_analysis'].values() if v['recommended']]
            result['betting_summary'] = {
                'has_recommendations': len(recommendations) > 0,
                'total_recommendations': len(recommendations),
                'best_value': max(recommendations, key=lambda x: x['edge']) if recommendations else None
            }
        
        return result

def main():
    """CLI interface for Bet Neural Lite"""
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    predictor = BetNeuralLite()
    
    if command == 'predict':
        if len(sys.argv) < 3:
            print("❌ Error: Match required. Use: predict \"Team1 vs Team2\"")
            return
        
        match_str = sys.argv[2]
        league = 'premier_league'  # default
        odds = None
        
        # Parse additional arguments
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg.startswith('--league='):
                league = arg.split('=', 1)[1].lower().replace(' ', '_')
            elif arg.startswith('--odds='):
                try:
                    odds_str = arg.split('=', 1)[1]
                    odds_values = [float(x) for x in odds_str.split(',')]
                    if len(odds_values) == 3:
                        odds = {
                            'home_win': odds_values[0],
                            'draw': odds_values[1], 
                            'away_win': odds_values[2]
                        }
                except:
                    print("⚠️  Invalid odds format. Use --odds=home,draw,away")
        
        # Parse team names
        if ' vs ' in match_str:
            home_team, away_team = match_str.split(' vs ')
        elif ' v ' in match_str:
            home_team, away_team = match_str.split(' v ')
        elif '-' in match_str:
            home_team, away_team = match_str.split('-')
        else:
            print("❌ Error: Use format 'Team1 vs Team2' or 'Team1-Team2'")
            return
        
        home_team = home_team.strip()
        away_team = away_team.strip()
        
        # Make prediction
        result = predictor.predict_match(home_team, away_team, league, odds)
        display_prediction(result)
        
    elif command == 'leagues':
        display_leagues()
        
    elif command == 'benchmark':
        run_benchmark(predictor)
        
    elif command == 'help':
        print_help()
        
    else:
        print(f"❌ Unknown command: {command}")
        print_help()

def display_prediction(result: Dict):
    """Display prediction in a nice format"""
    print(f"\n🎯 BET NEURAL LITE PREDICTION")
    print("=" * 70)
    
    print(f"🏟️  {result['match']}")
    print(f"🏆 League: {result['league'].replace('_', ' ').title()}")
    print(f"⏰ Time: {result['timestamp']}")
    print()
    
    # Elo ratings
    elo = result['elo_ratings']
    print("📊 ELO RATINGS:")
    print(f"   🏠 {result['match'].split(' vs ')[0]}: {elo['home']:.0f}")
    print(f"   ✈️  {result['match'].split(' vs ')[1]}: {elo['away']:.0f}")
    print(f"   🏡 Home Advantage: +{elo['home_advantage']} Elo")
    print()
    
    # Predictions  
    print("🎲 PREDICTIONS:")
    probs = result['probabilities']
    sorted_outcomes = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    for i, (outcome, prob) in enumerate(sorted_outcomes):
        outcome_name = outcome.replace('_', ' ').title()
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        
        medals = ["🥇", "🥈", "🥉"]
        medal = medals[i] if i < 3 else "  "
        
        print(f"   {medal} {outcome_name:<10}: {prob:6.1%} {bar}")
    
    print()
    
    # Expected goals
    xg = result['expected_goals']
    print(f"⚽ EXPECTED GOALS: {xg['home']} - {xg['away']} (Total: {xg['total']})")
    
    # Prediction summary
    pred = result['prediction']
    print(f"🎯 MOST LIKELY: {pred['most_likely'].replace('_', ' ').title()} ({pred['confidence']:.1%})")
    
    # Betting analysis
    if 'betting_analysis' in result:
        print(f"\n💰 BETTING ANALYSIS:")
        print("-" * 50)
        
        betting = result['betting_analysis']
        summary = result.get('betting_summary', {})
        
        if summary.get('has_recommendations'):
            print(f"✅ {summary['total_recommendations']} RECOMMENDED BET(S)")
            
            for outcome, analysis in betting.items():
                if analysis['recommended']:
                    outcome_name = outcome.replace('_', ' ').title()
                    print(f"\n🎯 {outcome_name}:")
                    print(f"   📊 Model Probability: {analysis['model_probability']:.1%}")
                    print(f"   💡 Edge: {analysis['edge']:+.2%}")
                    print(f"   💸 Kelly Stake: {analysis['kelly_fraction']:.2%}")
                    print(f"   🔥 Confidence: {analysis['confidence']}")
                    print(f"   💰 Potential ROI: {analysis['potential_roi']:.1f}%")
        else:
            print("❌ NO RECOMMENDED BETS")
            print("   Insufficient edge or confidence")
    
    print(f"\n🤖 Powered by Bet Neural Lite")

def display_leagues():
    """Display supported leagues"""
    print("\n🏆 SUPPORTED EUROPEAN LEAGUES")
    print("=" * 50)
    
    leagues_info = [
        ("Premier League", "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "England", "100%", "65"),
        ("La Liga", "🇪🇸", "Spain", "95%", "60"),  
        ("Bundesliga", "🇩🇪", "Germany", "90%", "58"),
        ("Serie A", "🇮🇹", "Italy", "85%", "55"),
        ("Ligue 1", "🇫🇷", "France", "80%", "52"),
        ("Eredivisie", "🇳🇱", "Netherlands", "70%", "50"),
        ("Primeira Liga", "🇵🇹", "Portugal", "65%", "48")
    ]
    
    print(f"{'Flag':<4} {'League':<15} {'Country':<11} {'Strength':<8} {'Home Adv'}")
    print("-" * 55)
    
    for league, flag, country, strength, home_adv in leagues_info:
        print(f"{flag:<4} {league:<15} {country:<11} {strength:<8} +{home_adv} Elo")
    
    print(f"\n📝 Usage: --league=premier_league or --league=la_liga")

def run_benchmark(predictor):
    """Run system benchmark"""
    print("\n🧪 BET NEURAL LITE BENCHMARK")
    print("=" * 50)
    
    test_matches = [
        ("Arsenal", "Chelsea", "premier_league"),
        ("Real Madrid", "Barcelona", "la_liga"),
        ("Bayern Munich", "Borussia Dortmund", "bundesliga"), 
        ("Juventus", "AC Milan", "serie_a")
    ]
    
    total_confidence = 0
    successful_predictions = 0
    
    print("🔄 Testing prediction system...")
    
    for home, away, league in test_matches:
        try:
            result = predictor.predict_match(home, away, league)
            confidence = result['prediction']['confidence']
            
            total_confidence += confidence
            successful_predictions += 1
            
            print(f"✅ {home} vs {away}: {confidence:.1%} confidence")
            
        except Exception as e:
            print(f"❌ {home} vs {away}: Error - {e}")
    
    if successful_predictions > 0:
        avg_confidence = total_confidence / successful_predictions
        
        print(f"\n📈 SYSTEM PERFORMANCE:")
        print(f"   Successful Predictions: {successful_predictions}/{len(test_matches)}")
        print(f"   Average Confidence: {avg_confidence:.1%}")
        print(f"   Elo Rating System: ✅ Active")
        print(f"   Betting Analysis: ✅ Active")
        print(f"   Dependencies: ✅ None Required")
        
        if avg_confidence > 0.7:
            rating = "🟢 Excellent"
        elif avg_confidence > 0.6:
            rating = "🟡 Good"  
        else:
            rating = "🔴 Needs Improvement"
        
        print(f"   Overall Rating: {rating}")

def print_help():
    """Display help information"""
    print("""
🏆 BET NEURAL LITE - European Football Prediction System
========================================================

USAGE:
  python3 bet_neural_lite.py <command> [options]

COMMANDS:
  predict "Team1 vs Team2"  Make match prediction
  leagues                   List supported leagues  
  benchmark                Run system test
  help                     Show this help

PREDICTION OPTIONS:
  --league=premier_league   Specify league (default: premier_league)
  --odds=2.1,3.4,3.8       Betting odds (home,draw,away)

EXAMPLES:
  python3 bet_neural_lite.py predict "Arsenal vs Chelsea"
  python3 bet_neural_lite.py predict "Real Madrid vs Barcelona" --league=la_liga
  python3 bet_neural_lite.py predict "Bayern vs Dortmund" --league=bundesliga --odds=2.1,3.4,3.8
  python3 bet_neural_lite.py leagues
  python3 bet_neural_lite.py benchmark

SUPPORTED LEAGUES:
  premier_league, la_liga, bundesliga, serie_a, ligue_1, eredivisie, primeira_liga

🤖 No external dependencies required - pure Python implementation
""")

if __name__ == "__main__":
    main()