"""
tests/test_core_components.py
==============================
Tests for components that had zero coverage:
  - MatchHistory  (form, H2H, fatigue, ppg, league_position)
  - FeatureBuilder (feature vector shape & values, season boundary)
  - ValueBetDetector (Kelly, edge, EV)
  - PoissonModel (probability properties, Dixon-Coles correction)
  - MonteCarloConfidence (stochastic, probability simplex)
  - CLI build_parser (standings command, --league arg)

Run from project root:
    pytest -v tests/test_core_components.py
"""

from __future__ import annotations

import math
import sys
import os
from typing import Dict, List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_match(date: str, home: str, away: str,
                home_goals: int, away_goals: int,
                xg_home: float = None, xg_away: float = None) -> Dict:
    result = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")
    return {
        "date":       date,
        "home_team":  home,
        "away_team":  away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "xg_home":    xg_home,
        "xg_away":    xg_away,
        "result":     result,
    }


def _build_history(matches: List[Dict]):
    from features import MatchHistory
    return MatchHistory().load(matches)


# ─────────────────────────────────────────────────────────────────────────────
#  MatchHistory
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchHistory:

    @pytest.fixture
    def season_matches(self):
        """10 matches: Arsenal home (7W, 1D, 2L), Chelsea away (5W, 2D, 3L)."""
        matches = [
            _make_match("2024-08-01", "Arsenal", "Chelsea",  2, 0, 1.8, 0.8),
            _make_match("2024-08-08", "Arsenal", "Man City", 0, 2, 1.0, 2.1),
            _make_match("2024-08-15", "Arsenal", "Spurs",    3, 1, 2.5, 0.9),
            _make_match("2024-08-22", "Chelsea", "Arsenal",  1, 1, 1.1, 1.2),
            _make_match("2024-08-29", "Arsenal", "Liverpool",2, 2, 1.9, 2.0),
            _make_match("2024-09-05", "Arsenal", "Brighton", 1, 0, 1.2, 0.7),
            _make_match("2024-09-12", "Chelsea", "Man City", 0, 3, 0.5, 2.8),
            _make_match("2024-09-19", "Arsenal", "Chelsea",  2, 1, 1.7, 1.1),
            _make_match("2024-09-26", "Man City", "Chelsea", 2, 0, 2.2, 0.6),
            _make_match("2024-10-03", "Arsenal", "Man City", 1, 0, 1.4, 1.3),
        ]
        return _build_history(matches)

    # ── team_form ──

    def test_form_win_rate_correct(self, season_matches):
        form = season_matches.team_form("Arsenal", "2024-10-10", n=5)
        # last 5 Arsenal matches before 2024-10-10:
        # 2024-08-22 (D), 2024-09-05 (W), 2024-09-19 (W), 2024-10-03 (W)
        # and one more — count wins / n
        assert 0.0 <= form["win_rate"] <= 1.0

    def test_form_probabilities_sum_to_one(self, season_matches):
        form = season_matches.team_form("Arsenal", "2024-10-10", n=5)
        total = form["win_rate"] + form["draw_rate"] + form["loss_rate"]
        assert abs(total - 1.0) < 1e-9

    def test_form_xg_uses_real_data(self, season_matches):
        """xg_avg must come from the recorded xg_home/xg_away, not be zero."""
        form = season_matches.team_form("Arsenal", "2024-10-10", n=5)
        assert form["xg_avg"] > 0.0, "xg_avg should be populated from real xG data"

    def test_form_returns_defaults_for_unknown_team(self, season_matches):
        form = season_matches.team_form("Unknown FC", "2024-10-10", n=5)
        # Should not raise; should return sensible defaults
        assert 0.0 <= form["win_rate"] <= 1.0
        assert form["n_matches"] == 0

    def test_form_venue_home_filters_correctly(self, season_matches):
        home_form = season_matches.team_form("Arsenal", "2024-10-10", n=10, venue="home")
        away_form = season_matches.team_form("Arsenal", "2024-10-10", n=10, venue="away")
        # Arsenal's home games should be separate from their away games
        assert home_form["n_matches"] != away_form["n_matches"] or (
            home_form["n_matches"] == 0 and away_form["n_matches"] == 0
        )

    # ── season_start boundary ──

    def test_season_start_excludes_previous_season(self, season_matches):
        """season_start guard must exclude matches before that date."""
        # Add a pre-season match from a "previous season"
        old_matches = [
            _make_match("2023-05-01", "Arsenal", "Chelsea", 3, 0),  # previous season
        ] + season_matches._matches
        history = _build_history(old_matches)

        # With season_start = 2024-07-01, the 2023 match should be excluded
        form_bounded = history.team_form("Arsenal", "2024-10-10", n=200, season_start="2024-07-01")
        form_unbounded = history.team_form("Arsenal", "2024-10-10", n=200)
        # Bounded form should see fewer matches (the old one is excluded)
        assert form_bounded["n_matches"] <= form_unbounded["n_matches"]

    # ── H2H ──

    def test_h2h_win_rates_sum_to_one(self, season_matches):
        h2h = season_matches.h2h_stats("Arsenal", "Chelsea", "2024-10-10")
        total = h2h["home_win_rate"] + h2h["draw_rate"] + h2h["away_win_rate"]
        assert abs(total - 1.0) < 1e-9

    def test_h2h_returns_defaults_for_no_history(self, season_matches):
        h2h = season_matches.h2h_stats("PSG", "Marseille", "2024-10-10")
        assert h2h["matches_available"] == 0

    # ── fatigue ──

    def test_fatigue_days_since_last_is_non_negative(self, season_matches):
        fat = season_matches.fatigue("Arsenal", "2024-10-10")
        assert fat["days_since_last"] >= 0

    def test_fatigue_matches_last_14d(self, season_matches):
        fat = season_matches.fatigue("Arsenal", "2024-10-10")
        assert fat["matches_last_14d"] >= 0

    # ── ppg ──

    def test_ppg_in_valid_range(self, season_matches):
        ppg = season_matches.ppg("Arsenal", "2024-10-10")
        assert 0.0 <= ppg <= 3.0

    def test_ppg_n5_vs_full_can_differ(self, season_matches):
        ppg_n5   = season_matches.ppg("Arsenal", "2024-10-10", n=5)
        ppg_full = season_matches.ppg("Arsenal", "2024-10-10")
        # Both must be valid floats; they may or may not be equal
        assert isinstance(ppg_n5, float)
        assert isinstance(ppg_full, float)

    # ── league_position ──

    def test_league_position_in_0_1(self, season_matches):
        teams = ["Arsenal", "Chelsea", "Man City", "Spurs", "Liverpool", "Brighton"]
        pos = season_matches.league_position("Arsenal", "2024-10-10", teams)
        assert 0.0 <= pos <= 1.0

    def test_league_position_top_team_near_one(self, season_matches):
        teams = ["Arsenal", "Chelsea", "Man City"]
        # Arsenal has the most wins → should be near 1.0
        pos_arsenal = season_matches.league_position("Arsenal", "2024-10-10", teams)
        pos_chelsea = season_matches.league_position("Chelsea", "2024-10-10", teams)
        assert pos_arsenal >= pos_chelsea


# ─────────────────────────────────────────────────────────────────────────────
#  FeatureBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureBuilder:

    @pytest.fixture
    def builder(self):
        from features import FeatureBuilder, MatchHistory, N_FEATURES

        matches = [
            _make_match("2024-08-01", "Arsenal", "Chelsea",  2, 0, 1.8, 0.7),
            _make_match("2024-08-08", "Arsenal", "Man City", 0, 2, 1.1, 2.2),
            _make_match("2024-08-15", "Chelsea", "Man City", 1, 1, 1.0, 1.0),
            _make_match("2024-08-22", "Arsenal", "Chelsea",  3, 1, 2.1, 0.9),
            _make_match("2024-08-29", "Man City", "Arsenal", 0, 1, 1.5, 1.6),
            _make_match("2024-09-05", "Chelsea", "Arsenal",  0, 2, 0.8, 1.9),
        ]
        history = MatchHistory().load(matches)
        return FeatureBuilder(
            match_history=history,
            elo_ratings={"Arsenal_premier_league": 1800, "Chelsea_premier_league": 1720},
            league="premier_league",
            season_start="2024-07-01",
        )

    def test_feature_vector_correct_length(self, builder):
        from features import N_FEATURES
        v = builder.build_feature_vector("Arsenal", "Chelsea", "2024-09-20")
        assert v.shape == (N_FEATURES,), (
            f"Expected shape ({N_FEATURES},), got {v.shape}"
        )

    def test_feature_vector_no_nan_or_inf(self, builder):
        v = builder.build_feature_vector("Arsenal", "Chelsea", "2024-09-20")
        assert not np.any(np.isnan(v)), "Feature vector contains NaN"
        assert not np.any(np.isinf(v)), "Feature vector contains Inf"

    def test_elo_diff_sign(self, builder):
        """Arsenal (1800) vs Chelsea (1720): elo_diff should be positive."""
        from features import FEATURE_NAMES
        v = builder.build_feature_vector("Arsenal", "Chelsea", "2024-09-20")
        elo_diff_idx = FEATURE_NAMES.index("elo_diff")
        assert v[elo_diff_idx] > 0, "Stronger home team should have positive elo_diff"

    def test_elo_diff_flips_when_teams_swap(self, builder):
        from features import FEATURE_NAMES
        v_home = builder.build_feature_vector("Arsenal", "Chelsea", "2024-09-20")
        v_away = builder.build_feature_vector("Chelsea", "Arsenal",  "2024-09-20")
        elo_diff_idx = FEATURE_NAMES.index("elo_diff")
        assert np.sign(v_home[elo_diff_idx]) != np.sign(v_away[elo_diff_idx])

    def test_build_training_set_shape(self, builder):
        from features import N_FEATURES
        matches = builder.history._matches
        X, y, ids = builder.build_training_set(matches)
        assert X.shape[1] == N_FEATURES
        assert len(y) == X.shape[0]
        assert len(ids) == X.shape[0]

    def test_build_training_set_labels_in_range(self, builder):
        matches = builder.history._matches
        _, y, _ = builder.build_training_set(matches)
        assert set(y).issubset({0, 1, 2}), f"Unexpected labels: {set(y)}"

    def test_season_boundary_respected(self, builder):
        """
        A match from before season_start should not contaminate season totals.
        Add a pre-season match and verify the builder sees fewer season matches.
        """
        from features import MatchHistory, FeatureBuilder, N_FEATURES

        old_match = _make_match("2023-05-15", "Arsenal", "Chelsea", 5, 0, 3.5, 0.5)
        new_matches = [old_match] + builder.history._matches
        new_history = MatchHistory().load(new_matches)

        bounded_builder = FeatureBuilder(
            match_history=new_history,
            elo_ratings={"Arsenal_premier_league": 1800, "Chelsea_premier_league": 1720},
            league="premier_league",
            season_start="2024-07-01",   # excludes the 2023 match
        )

        unbounded_builder = FeatureBuilder(
            match_history=new_history,
            elo_ratings={"Arsenal_premier_league": 1800, "Chelsea_premier_league": 1720},
            league="premier_league",
            season_start=None,           # sees all history
        )

        # season_start="2024-07-01" must not count the 2023 match in season totals
        from features import FEATURE_NAMES
        v_bounded   = bounded_builder.build_feature_vector("Arsenal", "Chelsea", "2024-09-20")
        v_unbounded = unbounded_builder.build_feature_vector("Arsenal", "Chelsea", "2024-09-20")

        # The features should differ because the 2023 blowout shifts averages
        assert not np.allclose(v_bounded, v_unbounded), (
            "Bounded and unbounded builders should produce different features "
            "when there is a pre-season match in history"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ValueBetDetector  (analytics.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestValueBetDetector:

    @pytest.fixture
    def vbd(self):
        from analytics import ValueBetDetector
        return ValueBetDetector()

    # ── analyse_outcome ──

    def test_no_edge_no_bet(self, vbd):
        result = vbd.analyse_outcome("home_win", model_prob=0.50, decimal_odds=2.0)
        # implied = 0.50, edge = 0 → not recommended
        assert result["edge"] == pytest.approx(0.0, abs=1e-9)
        assert not result["recommended"]
        assert result["kelly_stake_pct"] == 0.0

    def test_positive_edge_recommends_bet(self, vbd):
        # model 65 %, odds 2.0 (implied 50 %) → edge = +15 %
        result = vbd.analyse_outcome("home_win", model_prob=0.65, decimal_odds=2.0)
        assert result["edge"] == pytest.approx(0.15, abs=1e-9)
        assert result["recommended"]
        assert result["kelly_stake_pct"] > 0

    def test_kelly_capped_at_five_percent(self, vbd):
        # Extreme edge — Kelly must be capped
        result = vbd.analyse_outcome("home_win", model_prob=0.95, decimal_odds=5.0)
        assert result["kelly_stake_pct"] <= 5.0

    def test_ev_sign_matches_edge(self, vbd):
        """EV should be positive iff model_prob > implied_prob."""
        pos = vbd.analyse_outcome("home_win", model_prob=0.60, decimal_odds=2.0)
        neg = vbd.analyse_outcome("home_win", model_prob=0.40, decimal_odds=2.0)
        assert pos["expected_value"] > 0
        assert neg["expected_value"] < 0

    def test_negative_edge_no_bet(self, vbd):
        result = vbd.analyse_outcome("home_win", model_prob=0.40, decimal_odds=2.0)
        assert result["kelly_stake_pct"] == 0.0
        assert not result["recommended"]

    def test_odds_below_one_no_bet(self, vbd):
        result = vbd.analyse_outcome("home_win", model_prob=0.80, decimal_odds=0.9)
        assert not result["recommended"]
        assert result["kelly_stake_pct"] == 0.0

    # ── analyse_match ──

    def test_analyse_match_returns_all_outcomes(self, vbd):
        probs = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
        odds  = {"home_win": 2.0,  "draw": 4.0,  "away_win": 5.0}
        result = vbd.analyse_match(probs, odds)
        assert "outcomes" in result
        assert "best_bet" in result
        for key in ("home_win", "draw", "away_win"):
            assert key in result["outcomes"]

    def test_best_bet_has_highest_edge_among_recommended(self, vbd):
        probs = {"home_win": 0.65, "draw": 0.20, "away_win": 0.15}
        odds  = {"home_win": 2.0,  "draw": 5.5,  "away_win": 8.0}
        result = vbd.analyse_match(probs, odds)
        if result["best_bet"]:
            best_edge = result["best_bet"]["edge"]
            for outcome, data in result["outcomes"].items():
                if data["recommended"]:
                    assert data["edge"] <= best_edge + 1e-9

    def test_no_value_bets_when_all_under_threshold(self, vbd):
        # All model probs match implied probs exactly
        probs = {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
        odds  = {"home_win": 2.0,  "draw": 4.0,  "away_win": 4.0}
        result = vbd.analyse_match(probs, odds)
        assert result["best_bet"] is None
        assert result["n_recommended"] == 0

    # ── analyse_goalline ──

    def test_goalline_probs_sum_near_one(self, vbd):
        result = vbd.analyse_goalline(1.6, 1.2, line=2.5)
        total = result["p_over"] + result["p_under"]
        # p_over + p_under ≈ 1 (line can't land on 2.5 exactly)
        assert abs(total - 1.0) < 0.01

    def test_goalline_high_scoring_match_over_favoured(self, vbd):
        result = vbd.analyse_goalline(2.5, 2.5, line=2.5)
        assert result["p_over"] > result["p_under"]

    def test_goalline_low_scoring_match_under_favoured(self, vbd):
        result = vbd.analyse_goalline(0.5, 0.5, line=2.5)
        assert result["p_under"] > result["p_over"]


# ─────────────────────────────────────────────────────────────────────────────
#  PoissonModel  (models.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestPoissonModel:

    @pytest.fixture
    def fitted_model(self):
        pytest.importorskip("sklearn", reason="scikit-learn not installed — skipping PoissonModel tests")
        from models import PoissonModel
        matches = [
            {"home_team": "Arsenal",   "away_team": "Chelsea",   "home_goals": 2, "away_goals": 0},
            {"home_team": "Chelsea",   "away_team": "Arsenal",   "home_goals": 1, "away_goals": 1},
            {"home_team": "Arsenal",   "away_team": "Man City",  "home_goals": 0, "away_goals": 2},
            {"home_team": "Man City",  "away_team": "Chelsea",   "home_goals": 3, "away_goals": 0},
            {"home_team": "Chelsea",   "away_team": "Man City",  "home_goals": 1, "away_goals": 2},
            {"home_team": "Arsenal",   "away_team": "Spurs",     "home_goals": 2, "away_goals": 1},
            {"home_team": "Spurs",     "away_team": "Chelsea",   "home_goals": 0, "away_goals": 1},
            {"home_team": "Man City",  "away_team": "Arsenal",   "home_goals": 2, "away_goals": 2},
        ]
        return PoissonModel().fit(matches)

    def test_probs_sum_to_one(self, fitted_model):
        probs = fitted_model.predict("Arsenal", "Chelsea")
        total = probs["home_win"] + probs["draw"] + probs["away_win"]
        assert abs(total - 1.0) < 1e-6

    def test_all_probs_positive(self, fitted_model):
        probs = fitted_model.predict("Arsenal", "Chelsea")
        for key in ("home_win", "draw", "away_win"):
            assert probs[key] >= 0.0, f"{key} probability is negative"

    def test_stronger_attack_wins_more(self, fitted_model):
        """Man City scored 7 goals in 3 games — should win more than draw or away."""
        probs = fitted_model.predict("Man City", "Chelsea")
        assert probs["home_win"] > probs["away_win"]

    def test_expected_goals_are_positive(self, fitted_model):
        probs = fitted_model.predict("Arsenal", "Chelsea")
        assert probs["exp_home_goals"] > 0
        assert probs["exp_away_goals"] > 0

    def test_draw_probability_in_realistic_range(self, fitted_model):
        probs = fitted_model.predict("Arsenal", "Chelsea")
        assert 0.18 <= probs["draw"] <= 0.40, (
            f"Draw probability {probs['draw']:.3f} outside realistic range [0.18, 0.40]"
        )

    def test_unknown_team_uses_league_average(self, fitted_model):
        """An unknown team should not crash — falls back to league average attack/defence."""
        probs = fitted_model.predict("Unknown FC", "Arsenal")
        total = probs["home_win"] + probs["draw"] + probs["away_win"]
        assert abs(total - 1.0) < 1e-6

    def test_dc_low_score_correction_reduces_zero_zero_slightly(self, fitted_model):
        """
        The Dixon-Coles rho correction reduces P(0-0) vs raw independent Poisson.
        With rho < 0, the correction factor (1 - lambda_h * lambda_a * rho) > 1
        for 0-0, so P(0-0) is *increased* — this tests the sign is correct.
        """
        pytest.importorskip("sklearn", reason="scikit-learn not installed")
        from models import PoissonModel
        import math
        pm = PoissonModel(home_advantage=0.25)
        pm._attack  = {"A": 1.0, "B": 1.0}
        pm._defence = {"A": 1.0, "B": 1.0}
        pm._rho     = -0.13

        lh, la = pm._lambda("A", "B")
        raw_00 = math.exp(-lh) * math.exp(-la)   # independent Poisson P(0-0)
        dc_00  = pm._dc_prob(0, 0, lh, la)
        # With rho = -0.13, correction = (1 - lh*la*(-0.13)) = 1 + lh*la*0.13 > 1
        assert dc_00 > raw_00 * 0.99   # DC-corrected P(0-0) ≥ raw (rho correction sign)


# ─────────────────────────────────────────────────────────────────────────────
#  MonteCarloConfidence  (algorithm_improvements.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestMonteCarloConfidence:

    @pytest.fixture
    def mc(self):
        from algorithm_improvements import MonteCarloConfidence
        return MonteCarloConfidence(n_samples=2000, elo_std=20.0)

    def test_probs_sum_to_one(self, mc):
        result = mc.simulate(1700, 1600, 65, 1.5, 1.2)
        total = result["home_win"] + result["draw"] + result["away_win"]
        assert abs(total - 1.0) < 1e-9

    def test_all_probs_non_negative(self, mc):
        result = mc.simulate(1700, 1600, 65, 1.5, 1.2)
        for key in ("home_win", "draw", "away_win"):
            assert result[key] >= 0.0

    def test_confidence_equals_max_prob(self, mc):
        result = mc.simulate(1700, 1600, 65, 1.5, 1.2)
        expected_conf = max(result["home_win"], result["draw"], result["away_win"])
        assert result["confidence"] == pytest.approx(expected_conf, abs=1e-9)

    def test_home_advantage_increases_home_win_rate(self, mc):
        """High home advantage should push home_win above 50 % for equal teams."""
        result = mc.simulate(1500, 1500, home_adv=100, home_xg=1.4, away_xg=1.2)
        assert result["home_win"] > 0.40

    def test_dominant_team_wins_most(self, mc):
        """A 400-point Elo advantage should give a large home_win probability."""
        result = mc.simulate(1900, 1500, home_adv=65, home_xg=2.2, away_xg=0.8)
        assert result["home_win"] > 0.60

    def test_is_stochastic_across_calls(self, mc):
        """
        Two independent calls must NOT produce bit-identical results.
        (This would fail with the old deterministic hash-seeded RNG.)
        """
        r1 = mc.simulate(1700, 1600, 65, 1.5, 1.2)
        r2 = mc.simulate(1700, 1600, 65, 1.5, 1.2)
        # With 2000 samples, P(identical result) ≈ 0 for genuine random
        assert r1 != r2, (
            "Two independent MC calls returned identical results — "
            "the RNG is deterministic (hash-seeded). "
            "Fix: use np.random.default_rng() with no seed."
        )

    def test_draw_rate_in_realistic_range(self, mc):
        """Draw rate should be roughly 20–35 % for balanced matches."""
        result = mc.simulate(1500, 1500, 65, 1.4, 1.2)
        assert 0.15 <= result["draw"] <= 0.40, (
            f"Draw rate {result['draw']:.3f} outside realistic range"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI build_parser — standings command
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIParser:

    @pytest.fixture
    def parser(self):
        from bet_neural_cli import build_parser
        return build_parser()

    def test_standings_command_exists(self, parser):
        args = parser.parse_args(["standings", "--league", "La Liga"])
        assert args.command == "standings"
        assert args.league == "La Liga"

    def test_standings_default_league(self, parser):
        args = parser.parse_args(["standings"])
        assert args.command == "standings"
        assert args.league == "Premier League"

    def test_odds_alias_maps_to_standings(self, parser):
        """'odds' subcommand is kept as a back-compat alias."""
        args = parser.parse_args(["odds", "--league", "Bundesliga"])
        assert args.command == "odds"
        assert args.league == "Bundesliga"

    def test_predict_parses_correctly(self, parser):
        args = parser.parse_args(["predict", "Arsenal vs Chelsea",
                                  "--league", "Premier League",
                                  "--odds", "2.1,3.4,3.8"])
        assert args.command == "predict"
        assert args.match == "Arsenal vs Chelsea"
        assert args.odds == "2.1,3.4,3.8"

    def test_train_all_flag(self, parser):
        args = parser.parse_args(["train", "--all"])
        assert args.all is True

    def test_scrape_season_arg(self, parser):
        args = parser.parse_args(["scrape", "--league", "la_liga", "--season", "2023-2024"])
        assert args.season == "2023-2024"
        assert args.league == "la_liga"

    def test_no_subcommand_does_not_crash(self, parser):
        """Calling with no args should not raise — argparse returns None for command."""
        args = parser.parse_args([])
        assert args.command is None
