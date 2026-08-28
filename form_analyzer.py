"""
form_analyzer.py — Enhanced Team Form and Injury Analysis
=========================================================
Advanced analysis of team form, player availability, and performance trends
for more accurate match predictions.
"""

import json
import math
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class FormMetrics:
    """Comprehensive form analysis results."""
    recent_form_rating: float  # 0-100 scale
    goals_per_game: float
    goals_conceded_per_game: float
    xg_per_game: Optional[float]
    xa_per_game: Optional[float]  # Expected goals against
    win_percentage: float
    clean_sheet_percentage: float
    scoring_streak: int  # consecutive games with goals
    defensive_stability: float  # 0-100 scale
    home_away_factor: float  # boost/penalty for venue
    recent_results: List[str]  # ["W", "D", "L", "W", "L"]


@dataclass  
class InjuryReport:
    """Team injury and availability analysis."""
    key_players_out: List[str]
    injury_severity_score: float  # 0-100, higher = more impact
    estimated_team_strength_penalty: float  # 0-1 multiplier
    doubtful_players: List[str]
    suspension_count: int
    total_unavailable: int


class EnhancedFormAnalyzer:
    """
    Advanced team form analysis combining:
    - Recent match results with weighted importance
    - Goal scoring and defensive trends  
    - Expected goals (xG) performance where available
    - Home/away form differentiation
    - Injury and availability impact assessment
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self.cache_dir = self.data_dir / "cache" 
        self.match_history = {}  # Cached match data
        
        # Form calculation weights (more recent = higher weight)
        self.form_weights = [1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
        
        # Key player positions and their impact multipliers
        self.position_impact = {
            'GK': 0.8,    # Goalkeeper
            'CB': 0.6,    # Centre-back  
            'LB': 0.4, 'RB': 0.4,  # Full-backs
            'CDM': 0.7, 'CM': 0.6, 'CAM': 0.7,  # Midfielders
            'LW': 0.5, 'RW': 0.5,  # Wingers
            'ST': 0.8, 'CF': 0.8   # Strikers
        }
        
    def _load_match_history(self, team: str, league: str, days_back: int = 90) -> List[Dict]:
        """Load recent match history for a team."""
        cache_key = f"{team}_{league}_{days_back}"
        
        if cache_key in self.match_history:
            return self.match_history[cache_key]
            
        # Try to load from existing match data
        matches_file = self.data_dir / "processed" / "matches.pkl"
        if matches_file.exists():
            try:
                import pickle
                with open(matches_file, 'rb') as f:
                    matches_data = pickle.load(f)
                
                # Handle both list and DataFrame formats
                if isinstance(matches_data, list):
                    # Data is already a list of dictionaries
                    all_matches = matches_data
                else:
                    # Data is a DataFrame, convert to list of dictionaries
                    all_matches = matches_data.to_dict('records')
                
                # Filter for team and recent matches
                cutoff_date = datetime.now() - timedelta(days=days_back)
                team_matches = []
                team_lower = team.lower()
                
                for match in all_matches:
                    # Check if this team is in the match
                    home_team = str(match.get('home_team', '')).lower()
                    away_team = str(match.get('away_team', '')).lower()
                    
                    if team_lower not in home_team and team_lower not in away_team:
                        continue
                    
                    # Check date - handle both 'date' and 'date_utc' fields
                    match_date_str = match.get('date') or match.get('date_utc', '')
                    if match_date_str:
                        try:
                            # Handle ISO format with Z suffix
                            if match_date_str.endswith('Z'):
                                match_date_str = match_date_str[:-1] + '+00:00'
                            elif 'T' not in match_date_str:
                                # Just a date, assume midnight
                                match_date_str += 'T00:00:00'
                            
                            match_date = datetime.fromisoformat(match_date_str)
                            if match_date.date() < cutoff_date.date():
                                continue
                        except (ValueError, AttributeError):
                            continue
                    
                    team_matches.append(match)
                    
                    # Limit to reasonable number for performance
                    if len(team_matches) > 50:
                        break
                
                self.match_history[cache_key] = team_matches
                return team_matches
                
            except Exception as e:
                logger.warning(f"Could not load match history from pickle: {e}")
        
        # Try to load from JSONL file
        jsonl_file = self.data_dir.parent / "matches_2024_2026.jsonl"
        if jsonl_file.exists():
            try:
                matches = []
                cutoff_date = datetime.now() - timedelta(days=days_back)
                team_lower = team.lower()
                
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            match = json.loads(line)
                            
                            # Check if this team is in the match
                            home_team = match.get('home_team', '').lower()
                            away_team = match.get('away_team', '').lower()
                            
                            if team_lower not in home_team and team_lower not in away_team:
                                continue
                                
                            # Check date
                            match_date_str = match.get('date', '')
                            if match_date_str:
                                try:
                                    match_date = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
                                    if match_date < cutoff_date:
                                        continue
                                except:
                                    continue
                                    
                            matches.append(match)
                            
                            # Limit to reasonable number to avoid performance issues
                            if len(matches) > 50:
                                break
                                
                self.match_history[cache_key] = matches
                return matches
                
            except Exception as e:
                logger.warning(f"Could not load match history from JSONL: {e}")
        
        # Fallback to empty list
        self.match_history[cache_key] = []
        return []
        
    def _calculate_form_score(self, matches: List[Dict], team: str, venue: Optional[str] = None) -> float:
        """
        Calculate weighted form score based on recent results.
        Returns score 0-100 where 50 is average.
        """
        if not matches:
            return 50.0  # Average/neutral form
            
        team_lower = team.lower()
        form_points = 0.0
        total_weight = 0.0
        
        # Process matches in reverse chronological order (most recent first)
        recent_matches = sorted(matches, key=lambda m: m.get('date', ''), reverse=True)[:10]
        
        for i, match in enumerate(recent_matches):
            if i >= len(self.form_weights):
                break
                
            weight = self.form_weights[i]
            is_home = team_lower in match.get('home_team', '').lower()
            is_away = team_lower in match.get('away_team', '').lower()
            
            if not (is_home or is_away):
                continue
                
            # Skip if venue filter doesn't match
            if venue == 'home' and not is_home:
                continue
            elif venue == 'away' and not is_away:
                continue
                
            # Get match result from team's perspective
            home_score = match.get('home_goals', match.get('home_score', 0)) or 0
            away_score = match.get('away_goals', match.get('away_score', 0)) or 0
            
            if is_home:
                team_score, opp_score = home_score, away_score
            else:
                team_score, opp_score = away_score, home_score
                
            # Award points: Win=3, Draw=1, Loss=0
            if team_score > opp_score:
                points = 3.0  # Win
            elif team_score == opp_score:
                points = 1.0  # Draw
            else:
                points = 0.0  # Loss
                
            # Bonus/penalty for goal difference
            goal_diff = team_score - opp_score
            if goal_diff >= 3:
                points += 0.5  # Dominant win bonus
            elif goal_diff <= -3:
                points -= 0.5  # Heavy loss penalty
                
            form_points += points * weight
            total_weight += weight
            
        if total_weight == 0:
            return 50.0
            
        # Convert to 0-100 scale (3 points per match * weight = max possible)
        max_possible = 3.0 * total_weight
        normalized_score = (form_points / max_possible) * 100
        
        return min(100.0, max(0.0, normalized_score))
        
    def _calculate_attacking_metrics(self, matches: List[Dict], team: str) -> Tuple[float, float, int]:
        """Calculate goals per game, xG per game, and scoring streak."""
        if not matches:
            return 1.4, 1.2, 0  # League averages
            
        team_lower = team.lower()
        total_goals = 0.0
        total_xg = 0.0
        valid_matches = 0
        scoring_streak = 0
        current_streak = True
        
        # Process in reverse chronological order for streak
        recent_matches = sorted(matches, key=lambda m: m.get('date', ''), reverse=True)
        
        for match in recent_matches:
            is_home = team_lower in match.get('home_team', '').lower()
            is_away = team_lower in match.get('away_team', '').lower()
            
            if not (is_home or is_away):
                continue
                
            # Get team's goals and xG
            if is_home:
                goals = match.get('home_goals', match.get('home_score', 0)) or 0
                xg = match.get('xg_home', 0) or 0
            else:
                goals = match.get('away_goals', match.get('away_score', 0)) or 0
                xg = match.get('xg_away', 0) or 0
                
            total_goals += goals
            total_xg += xg
            valid_matches += 1
            
            # Update scoring streak (consecutive games with goals)
            if current_streak:
                if goals > 0:
                    scoring_streak += 1
                else:
                    current_streak = False
                    
        goals_per_game = total_goals / valid_matches if valid_matches > 0 else 1.4
        xg_per_game = total_xg / valid_matches if valid_matches > 0 and total_xg > 0 else None
        
        return goals_per_game, xg_per_game, scoring_streak
        
    def _calculate_defensive_metrics(self, matches: List[Dict], team: str) -> Tuple[float, float, float]:
        """Calculate goals conceded per game, clean sheets %, and defensive stability."""
        if not matches:
            return 1.2, 30.0, 50.0  # League averages
            
        team_lower = team.lower()
        total_conceded = 0.0
        clean_sheets = 0
        valid_matches = 0
        
        for match in matches:
            is_home = team_lower in match.get('home_team', '').lower()
            is_away = team_lower in match.get('away_team', '').lower()
            
            if not (is_home or is_away):
                continue
                
            # Get goals conceded
            if is_home:
                conceded = match.get('away_goals', match.get('away_score', 0)) or 0
            else:
                conceded = match.get('home_goals', match.get('home_score', 0)) or 0
                
            total_conceded += conceded
            if conceded == 0:
                clean_sheets += 1
            valid_matches += 1
            
        goals_conceded_per_game = total_conceded / valid_matches if valid_matches > 0 else 1.2
        clean_sheet_percentage = (clean_sheets / valid_matches * 100) if valid_matches > 0 else 30.0
        
        # Defensive stability: lower variance in goals conceded = higher stability
        if valid_matches >= 3:
            conceded_per_match = [
                (match.get('away_goals', 0) if team_lower in match.get('home_team', '').lower()
                 else match.get('home_goals', 0)) or 0
                for match in matches
                if team_lower in match.get('home_team', '').lower() or team_lower in match.get('away_team', '').lower()
            ]
            
            if len(conceded_per_match) > 1:
                variance = sum((x - goals_conceded_per_game) ** 2 for x in conceded_per_match) / len(conceded_per_match)
                stability = max(0, 100 - (variance * 25))  # Convert variance to stability score
            else:
                stability = 50.0
        else:
            stability = 50.0
            
        return goals_conceded_per_game, clean_sheet_percentage, stability
        
    def _get_recent_results_string(self, matches: List[Dict], team: str, n: int = 5) -> List[str]:
        """Get recent results as ['W', 'D', 'L', 'W', 'D'] format."""
        if not matches:
            return []
            
        team_lower = team.lower()
        results = []
        
        # Process in reverse chronological order
        recent_matches = sorted(matches, key=lambda m: m.get('date', ''), reverse=True)[:n]
        
        for match in recent_matches:
            is_home = team_lower in match.get('home_team', '').lower()
            is_away = team_lower in match.get('away_team', '').lower()
            
            if not (is_home or is_away):
                continue
                
            home_score = match.get('home_goals', match.get('home_score', 0)) or 0
            away_score = match.get('away_goals', match.get('away_score', 0)) or 0
            
            if is_home:
                if home_score > away_score:
                    results.append('W')
                elif home_score == away_score:
                    results.append('D')
                else:
                    results.append('L')
            else:
                if away_score > home_score:
                    results.append('W')
                elif away_score == home_score:
                    results.append('D')
                else:
                    results.append('L')
                    
        return results
        
    def analyze_team_form(self, team: str, league: str, venue: Optional[str] = None) -> FormMetrics:
        """
        Comprehensive form analysis for a team.
        
        Args:
            team: Team name
            league: League identifier  
            venue: 'home', 'away', or None for overall
            
        Returns:
            FormMetrics with comprehensive analysis
        """
        # Load match history
        matches = self._load_match_history(team, league, days_back=120)
        
        # Filter by venue if specified
        if venue:
            team_lower = team.lower()
            if venue == 'home':
                matches = [m for m in matches if team_lower in m.get('home_team', '').lower()]
            elif venue == 'away':
                matches = [m for m in matches if team_lower in m.get('away_team', '').lower()]
        
        # Calculate all metrics
        form_rating = self._calculate_form_score(matches, team, venue)
        goals_per_game, xg_per_game, scoring_streak = self._calculate_attacking_metrics(matches, team)
        goals_conceded_per_game, clean_sheet_percentage, defensive_stability = self._calculate_defensive_metrics(matches, team)
        
        # Calculate win percentage
        results = self._get_recent_results_string(matches, team, n=10)
        win_percentage = (results.count('W') / len(results) * 100) if results else 50.0
        
        # Home/away factor (boost or penalty based on venue performance vs overall)
        home_away_factor = 1.0
        if venue:
            overall_form = self._calculate_form_score(self._load_match_history(team, league), team)
            venue_form = form_rating
            home_away_factor = min(1.3, max(0.7, venue_form / max(1, overall_form)))
        
        return FormMetrics(
            recent_form_rating=form_rating,
            goals_per_game=goals_per_game,
            goals_conceded_per_game=goals_conceded_per_game,
            xg_per_game=xg_per_game,
            xa_per_game=None,  # Would need opponent xG data
            win_percentage=win_percentage,
            clean_sheet_percentage=clean_sheet_percentage,
            scoring_streak=scoring_streak,
            defensive_stability=defensive_stability,
            home_away_factor=home_away_factor,
            recent_results=results[:5]
        )
        
    def analyze_injuries_and_availability(self, team: str, league: str) -> InjuryReport:
        """
        Analyze team injury situation and player availability.
        Note: This is a simplified implementation. In production, this would
        integrate with injury databases, team news APIs, etc.
        """
        # Placeholder implementation - would integrate with:
        # - Premier Injuries API
        # - Team news feeds
        # - Suspension tracking
        # - Transfer database
        
        # For now, return neutral/average impact
        return InjuryReport(
            key_players_out=[],
            injury_severity_score=20.0,  # Low impact
            estimated_team_strength_penalty=0.95,  # 5% penalty (typical)
            doubtful_players=[],
            suspension_count=0,
            total_unavailable=2  # Average unavailable players
        )
        
    def get_form_adjustment_factor(self, team: str, league: str, venue: str) -> float:
        """
        Get form-based adjustment factor for prediction models.
        Returns multiplier (0.8 - 1.2) to adjust team strength.
        """
        form = self.analyze_team_form(team, league, venue)
        injury_report = self.analyze_injuries_and_availability(team, league)
        
        # Base adjustment from form rating (50 = neutral)
        form_factor = 0.9 + (form.recent_form_rating - 50) / 250  # 50->0.9, 75->1.0, 100->1.1
        
        # Attacking adjustment
        if form.goals_per_game > 2.0:
            form_factor += 0.05  # Good attack bonus
        elif form.goals_per_game < 1.0:
            form_factor -= 0.05  # Poor attack penalty
            
        # Defensive adjustment  
        if form.clean_sheet_percentage > 50:
            form_factor += 0.03  # Strong defense bonus
        elif form.goals_conceded_per_game > 2.0:
            form_factor -= 0.05  # Leaky defense penalty
            
        # Injury impact
        form_factor *= injury_report.estimated_team_strength_penalty
        
        # Home/away venue factor
        form_factor *= form.home_away_factor
        
        # Clamp to reasonable range
        return max(0.75, min(1.25, form_factor))
        
    def get_form_summary(self, team: str, league: str) -> str:
        """Get human-readable form summary for display."""
        overall_form = self.analyze_team_form(team, league)
        home_form = self.analyze_team_form(team, league, 'home')  
        away_form = self.analyze_team_form(team, league, 'away')
        
        # Form rating to emoji
        def form_emoji(rating: float) -> str:
            if rating >= 75:
                return "🔥"
            elif rating >= 60:
                return "📈"
            elif rating >= 40:
                return "➡️"
            else:
                return "📉"
        
        recent_results = "".join(overall_form.recent_results)
        form_icon = form_emoji(overall_form.recent_form_rating)
        
        return (f"{form_icon} {recent_results} | "
                f"⚽{overall_form.goals_per_game:.1f} 🥅{overall_form.goals_conceded_per_game:.1f} | "
                f"🏠{home_form.recent_form_rating:.0f}% 🛫{away_form.recent_form_rating:.0f}%")


def enhance_prediction_with_form(predictor, home_team: str, away_team: str, league: str) -> Dict[str, Any]:
    """
    Enhance base prediction with form analysis.
    This would be integrated into the main prediction pipeline.
    """
    analyzer = EnhancedFormAnalyzer()
    
    # Analyze form for both teams
    home_form = analyzer.analyze_team_form(home_team, league, 'home')
    away_form = analyzer.analyze_team_form(away_team, league, 'away')
    
    # Get adjustment factors
    home_factor = analyzer.get_form_adjustment_factor(home_team, league, 'home')
    away_factor = analyzer.get_form_adjustment_factor(away_team, league, 'away')
    
    # Get summaries
    home_summary = analyzer.get_form_summary(home_team, league)
    away_summary = analyzer.get_form_summary(away_team, league)
    
    return {
        'home_form': home_form,
        'away_form': away_form,
        'home_factor': home_factor,
        'away_factor': away_factor,
        'home_summary': home_summary,
        'away_summary': away_summary
    }