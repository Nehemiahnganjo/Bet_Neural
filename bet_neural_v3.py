"""
bet_neural_v3.py — DEPRECATED
================================
⚠️  This file is superseded by bet_neural.py (v4.1).

All 10 algorithm improvements originally listed here are now integrated
directly into BetNeuralPredictor in bet_neural.py:
  - MatchLevelCalibrator
  - DynamicWeightOptimizer
  - HomeAwayModelStack
  - ExponentialFormAggregator
  - SOSComputer
  - AvailabilityPenalty
  - WeatherFeatureExtractor
  - MonteCarloConfidence  (now properly stochastic — see algorithm_improvements.py)
  - StackingEnsemble
  - OverroundAdjustedKelly

Known issues in this file (not fixed — file is deprecated):
  - _get_ha_predictions() falls back to plain Elo — the HomeAwayModelStack
    is never actually used.
  - _get_stacked_predictions() also falls back to plain Elo.
  - ImprovedBetNeuralPredictor.predict_match() therefore produces identical
    output to the base BetNeuralPredictor, making the "v3 improvements" a no-op.

Use bet_neural.py directly.  This file is kept only for import compatibility
via the BetNeuralPredictorImproved alias at the bottom.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

# Import improvements
try:
    from algorithm_improvements import (
        MatchLevelCalibrator,
        DynamicWeightOptimizer,
        HomeAwayModelStack,
        ExponentialFormAggregator,
        SOSComputer,
        AvailabilityPenalty,
        WeatherFeatureExtractor,
        MonteCarloConfidence,
        StackingEnsemble,
        OverroundAdjustedKelly,
        ImprovedPredictor,
    )
    IMPROVEMENTS_AVAILABLE = True
except ImportError:
    IMPROVEMENTS_AVAILABLE = False
    logger.warning("algorithm_improvements.py not found — using v2 baseline")

# Re-export for compatibility
from bet_neural import LEAGUES, BetNeuralPredictor, _elo_probs, _elo_update, DEFAULT_RATINGS_PATH


class ImprovedBetNeuralPredictor(BetNeuralPredictor):
    """
    Enhanced predictor with all 10 algorithm improvements.
    Inherits from v2 BetNeuralPredictor.
    """

    def __init__(self, ratings_path: str = str(DEFAULT_RATINGS_PATH), auto_load_models: bool = True) -> None:
        super().__init__(ratings_path, auto_load_models)

        # Improvement components
        self.calibrator = MatchLevelCalibrator() if IMPROVEMENTS_AVAILABLE else None
        self.weight_optimizer = DynamicWeightOptimizer() if IMPROVEMENTS_AVAILABLE else None
        self.home_away_models: Dict[str, HomeAwayModelStack] = {}
        self.form_aggregator = ExponentialFormAggregator(decay=0.80)
        self.sos_computers: Dict[str, SOSComputer] = {}
        self.availability = AvailabilityPenalty() if IMPROVEMENTS_AVAILABLE else None
        self.weather = WeatherFeatureExtractor() if IMPROVEMENTS_AVAILABLE else None
        self.monte_carlo = MonteCarloConfidence(n_samples=1000) if IMPROVEMENTS_AVAILABLE else None
        self.stacking = StackingEnsemble(['xgb', 'lgb', 'mlp']) if IMPROVEMENTS_AVAILABLE else None
        self.kelly = OverroundAdjustedKelly(fractional=0.5, max_stake=0.05) if IMPROVEMENTS_AVAILABLE else None

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        league: str = "premier_league",
        odds: Optional[Dict[str, float]] = None,
        match_date: Optional[str] = None,
        full_report: bool = False,
    ) -> Dict[str, Any]:
        """
        Enhanced prediction with all 10 improvements.
        """
        warnings_out: List[str] = []

        # Resolve team names
        h_resolved, h_sim, h_exact = self.resolve_team_name(home_team, league)
        a_resolved, a_sim, a_exact = self.resolve_team_name(away_team, league)

        if not h_exact:
            warnings_out.append(f"⚠️  '{home_team}' resolved to '{h_resolved}'" if h_sim >= 0.6 else
                              f"⚠️  '{home_team}' not recognized — using league-average")

        if not a_exact:
            warnings_out.append(f"⚠️  '{away_team}' resolved to '{a_resolved}'" if a_sim >= 0.6 else
                              f"⚠️  '{away_team}' not recognized — using league-average")

        if match_date is None:
            match_date = datetime.now().strftime("%Y-%m-%d")

        h_elo = self.get_team_elo(h_resolved, league)
        a_elo = self.get_team_elo(a_resolved, league)

        # Get match history for SOS calculation
        fb = self._get_feature_builder(league)
        if fb:
            matches = fb.history._matches
        else:
            matches = []

        # Compute improvements features
        h_xg = 1.4 * (h_elo / 1500.0)
        a_xg = 1.2 * (a_elo / 1500.0)

        # SOS features
        sos_diff = 0.0
        if fb and IMPROVEMENTS_AVAILABLE:
            sos_comp = SOSComputer(self.team_ratings, league)
            sos_diff = sos_comp.compute_sos_diff(h_resolved, a_resolved, matches)

        # Player availability penalty
        home_penalty = 1.0
        away_penalty = 1.0
        if IMPROVEMENTS_AVAILABLE:
            home_penalty = self.availability.get_penalty(h_resolved)
            away_penalty = self.availability.get_penalty(a_resolved)

        # Weather/pitch features
        weather_data = {}
        if IMPROVEMENTS_AVAILABLE:
            weather_data = self.weather.get_weather(match_date, LEAGUES[league]['country'])
            travel_dist = self.weather.compute_travel_distance(h_resolved, a_resolved, league)

        # Monte Carlo confidence
        mc_probs = None
        if IMPROVEMENTS_AVAILABLE and self.monte_carlo:
            mc_probs = self.monte_carlo.simulate(h_elo, a_elo, LEAGUES[league].get('home_adv', 65), h_xg, a_xg)

        # Base probabilities
        base_probs = self.get_match_probability(h_elo, a_elo, league)

        # Dynamic weights
        weights = {'xgb': 0.40, 'lgb': 0.35, 'mlp': 0.25}  # default
        if IMPROVEMENTS_AVAILABLE and self.weight_optimizer:
            n_samples = len(matches) if matches else 100
            weights = self.weight_optimizer.compute_weights(league, n_samples)

        # Home/away specific models
        ha_probs = base_probs
        if IMPROVEMENTS_AVAILABLE and league in self.home_away_models:
            ha_probs = self._get_ha_predictions(h_resolved, a_resolved, league)

        # Stacking ensemble
        stacked_probs = base_probs
        if IMPROVEMENTS_AVAILABLE and self.stacking.is_trained:
            stacked_probs = self._get_stacked_predictions(h_resolved, a_resolved, league)

        # Final blend (weighted average)
        final_probs = self._blend_probabilities(
            base_probs, mc_probs or base_probs, ha_probs, stacked_probs, weights
        )

        # Final renormalize
        total = sum(final_probs.values())
        final_probs = {k: v / total for k, v in final_probs.items()}

        # Confidence from MC or max
        confidence = mc_probs['confidence'] if mc_probs else max(final_probs.values())

        # Expected goals
        exp_goals = self._expected_goals(h_resolved, a_resolved, league, final_probs)

        # Betting analysis with overround-adjusted Kelly
        betting_analysis = None
        if odds and IMPROVEMENTS_AVAILABLE:
            overround = self.kelly.compute_overround(odds)
            betting_analysis = self._analyse_betting_with_improvements(
                final_probs, odds, overround
            )

        result = {
            "match": f"{h_resolved} vs {a_resolved}",
            "match_input": f"{home_team} vs {away_team}",
            "league": league,
            "probabilities": final_probs,
            "elo_ratings": {"home": h_elo, "away": a_elo},
            "expected_goals": exp_goals,
            "confidence": confidence,
            "prediction_engine": "improved_v3",
            "prediction_time": datetime.now().isoformat(),
            "warnings": warnings_out,
            "improvements_used": {
                "calibration": bool(self.calibrator),
                "dynamic_weights": bool(self.weight_optimizer),
                "home_away_models": league in self.home_away_models,
                "form_decay": True,
                "sos": sos_diff != 0.0,
                "player_availability": bool(self.availability),
                "weather": bool(weather_data),
                "monte_carlo": bool(mc_probs),
                "stacking": self.stacking.is_trained if self.stacking else False,
                "kelly_adjusted": bool(self.kelly),
            },
        }

        if betting_analysis:
            result["betting_analysis"] = betting_analysis

        if odds:
            result["odds_used"] = odds

        return result

    def _get_ha_predictions(self, home: str, away: str, league: str) -> Dict[str, float]:
        """Get predictions from home/away specific models."""
        ha = self.home_away_models.get(league)
        if not ha or not ha.is_trained:
            return self.get_match_probability(
                self.get_team_elo(home, league),
                self.get_team_elo(away, league),
                league
            )

        # For now, use Elo fallback until models are trained
        return self.get_match_probability(
            self.get_team_elo(home, league),
            self.get_team_elo(away, league),
            league
        )

    def _get_stacked_predictions(self, home: str, away: str, league: str) -> Dict[str, float]:
        """Get predictions from stacking ensemble."""
        # Fallback to Elo for now
        return self.get_match_probability(
            self.get_team_elo(home, league),
            self.get_team_elo(away, league),
            league
        )

    def _blend_probabilities(
        self,
        base: Dict[str, float],
        mc: Dict[str, float],
        ha: Dict[str, float],
        stacked: Dict[str, float],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Blend probabilities with dynamic weights."""
        # Weights: base(0.15) + MC(0.15) + HA(0.30) + Stacked(0.40)
        blended = {}
        for outcome in ("home_win", "draw", "away_win"):
            blended[outcome] = (
                0.15 * base[outcome] +
                0.15 * mc[outcome] +
                0.30 * ha[outcome] +
                0.40 * stacked[outcome]
            )
        return blended

    def _analyse_betting_with_improvements(
        self,
        probs: Dict[str, float],
        odds: Dict[str, float],
        overround: float,
    ) -> Dict[str, Any]:
        """Betting analysis with overround-adjusted Kelly."""
        from analytics import ValueBetDetector
        from algorithm_improvements import OverroundAdjustedKelly

        vbd = ValueBetDetector()
        analysis = vbd.analyse_match(probs, odds)

        # Adjust Kelly stakes for overround
        if self.kelly:
            for outcome, data in analysis.get("outcomes", {}).items():
                if data.get("kelly_stake_pct", 0) > 0:
                    adjusted = self.kelly.adjust_kelly_for_margin(
                        data["kelly_stake_pct"] / 100.0, overround
                    ) * 100.0
                    data["kelly_stake_pct"] = round(adjusted, 2)
                    data["kelly_adjusted"] = True

        return analysis


# Backwards compatibility alias
BetNeuralPredictorImproved = ImprovedBetNeuralPredictor
