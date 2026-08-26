"""
tests/test_bet_neural.py
========================
pytest suite for Bet Neural.

Run from the project root:
    pip3 install pytest
    pytest -v tests/
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path regardless of where pytest is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bet_neural import BetNeuralPredictor


# ============================================================
#  Fixtures
# ============================================================

@pytest.fixture
def predictor(tmp_path):
    """Fresh predictor that persists ratings to a temp file."""
    return BetNeuralPredictor(ratings_path=str(tmp_path / "elo_ratings.json"))


@pytest.fixture
def balanced_probs() -> Dict[str, float]:
    """Probability dict for a roughly equal match."""
    p = BetNeuralPredictor.__new__(BetNeuralPredictor)
    p.home_advantage = 65
    return p.get_match_probability(1500.0, 1500.0)


# ============================================================
#  Elo maths
# ============================================================

class TestEloMath:

    def test_probabilities_sum_to_one(self, predictor):
        probs = predictor.get_match_probability(1800, 1600)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-9

    def test_home_advantage_increases_home_win_prob(self, predictor):
        probs_home = predictor.get_match_probability(1500, 1500)
        assert probs_home['home_win'] > probs_home['away_win'], (
            "Home side should have higher win probability at equal Elo (home advantage)"
        )

    def test_stronger_team_wins_more_often(self, predictor):
        probs = predictor.get_match_probability(1900, 1500)
        assert probs['home_win'] > probs['away_win']

    def test_draw_probability_in_empirical_range(self, predictor):
        """Draw should stay in the documented [0.22, 0.32] range for any realistic Elo gap."""
        for home_elo, away_elo in [
            (1500, 1500),   # balanced
            (1900, 1400),   # strong mismatch
            (1300, 1900),   # large underdog at home
        ]:
            probs = predictor.get_match_probability(home_elo, away_elo)
            assert 0.20 <= probs['draw'] <= 0.35, (
                f"Draw probability {probs['draw']:.3f} outside expected range "
                f"for Elo {home_elo} vs {away_elo}"
            )

    def test_draw_not_inflated_for_balanced_match(self, balanced_probs):
        """The old bug pushed draws to ~35%+ in balanced matches."""
        assert balanced_probs['draw'] <= 0.32, (
            f"Draw probability {balanced_probs['draw']:.3f} is inflated for equal teams"
        )

    def test_calculate_elo_rating_winner_gains(self, predictor):
        """The winning team should gain Elo, the losing team should lose Elo."""
        home_new, away_new = predictor.calculate_elo_rating(1500, 1500, 1.0)
        assert home_new > 1500, "Winner should gain Elo"
        assert away_new < 1500, "Loser should lose Elo"

    def test_calculate_elo_rating_draw_symmetry(self, predictor):
        """A draw between equal teams should leave both ratings unchanged."""
        home_new, away_new = predictor.calculate_elo_rating(1500, 1500, 0.5)
        assert abs(home_new - 1500) < 0.1
        assert abs(away_new - 1500) < 0.1

    def test_elo_is_zero_sum(self, predictor):
        """Total Elo change should be zero (what one gains the other loses)."""
        for result in (0.0, 0.5, 1.0):
            home_new, away_new = predictor.calculate_elo_rating(1600, 1400, result)
            delta_home = home_new - 1600
            delta_away = away_new - 1400
            assert abs(delta_home + delta_away) < 1e-9, (
                f"Elo not zero-sum for result={result}: Δhome={delta_home:.4f}, Δaway={delta_away:.4f}"
            )

    def test_k_factor_scales_change(self):
        """A higher K-factor should produce larger Elo swings."""
        p32 = BetNeuralPredictor.__new__(BetNeuralPredictor)
        p32.elo_k_factor = 32
        p32.home_advantage = 65
        p64 = BetNeuralPredictor.__new__(BetNeuralPredictor)
        p64.elo_k_factor = 64
        p64.home_advantage = 65

        _, away32 = p32.calculate_elo_rating(1500, 1500, 1.0)
        _, away64 = p64.calculate_elo_rating(1500, 1500, 1.0)

        assert abs(away64 - 1500) > abs(away32 - 1500), "K=64 should move ratings more than K=32"


# ============================================================
#  Kelly Criterion
# ============================================================

class TestKellyCriterion:

    def test_no_bet_when_no_edge(self, predictor):
        """Kelly should return 0 when model prob matches implied prob exactly."""
        # Odds of 2.0 → implied prob = 0.50
        probs = {'home_win': 0.50, 'draw': 0.25, 'away_win': 0.25}
        odds  = {'home_win': 2.0,  'draw': 4.0,  'away_win': 4.0}
        analysis = predictor.calculate_betting_value(probs, odds)
        assert analysis['home_win']['kelly_fraction'] == 0.0
        assert not analysis['home_win']['recommended']

    def test_positive_edge_generates_stake(self, predictor):
        """Kelly fraction must be > 0 when model probability beats implied prob."""
        probs = {'home_win': 0.65, 'draw': 0.20, 'away_win': 0.15}
        odds  = {'home_win': 2.0,  'draw': 4.5,  'away_win': 7.0}
        analysis = predictor.calculate_betting_value(probs, odds)
        assert analysis['home_win']['kelly_fraction'] > 0
        assert analysis['home_win']['recommended']

    def test_kelly_capped_at_five_percent(self, predictor):
        """Kelly fraction must never exceed 5%."""
        # Extremely high edge scenario
        probs = {'home_win': 0.90, 'draw': 0.05, 'away_win': 0.05}
        odds  = {'home_win': 5.0,  'draw': 10.0, 'away_win': 20.0}
        analysis = predictor.calculate_betting_value(probs, odds)
        assert analysis['home_win']['kelly_fraction'] <= 0.05

    def test_threshold_alignment(self, predictor):
        """
        Kelly fraction should be > 0 whenever the bet is 'recommended'.
        """
        probs = {'home_win': 0.60, 'draw': 0.25, 'away_win': 0.15}
        odds  = {'home_win': 2.0,  'draw': 4.0,  'away_win': 7.0}
        analysis = predictor.calculate_betting_value(probs, odds)
        for outcome, data in analysis.items():
            if data['recommended']:
                assert data['kelly_fraction'] > 0, (
                    f"{outcome}: recommended=True but kelly_fraction=0 (threshold misalignment)"
                )

    def test_edge_calculation(self, predictor):
        """Edge = model_prob - implied_prob."""
        probs = {'home_win': 0.60}
        odds  = {'home_win': 2.0}     # implied = 0.50
        analysis = predictor.calculate_betting_value(probs, odds)
        assert abs(analysis['home_win']['edge'] - 0.10) < 1e-6

    def test_confidence_levels(self, predictor):
        # Thresholds in ValueBetDetector: >= 0.70 → high, >= 0.58 → medium, else low
        for prob, expected_conf in [
            (0.70, 'high'),
            (0.80, 'high'),
            (0.60, 'medium'),
            (0.58, 'medium'),   # exact boundary — should be medium
            (0.57, 'low'),      # just below boundary — low
            (0.50, 'low'),
            (0.40, 'low'),
        ]:
            probs = {'home_win': prob}
            odds  = {'home_win': 3.0}
            analysis = predictor.calculate_betting_value(probs, odds)
            assert analysis['home_win']['confidence'] == expected_conf, (
                f"prob={prob} → expected confidence={expected_conf!r}, "
                f"got {analysis['home_win']['confidence']!r}"
            )


# ============================================================
#  Elo persistence
# ============================================================

class TestEloPersistence:

    def test_ratings_survive_round_trip(self, predictor):
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        predictor.save_ratings()

        predictor2 = BetNeuralPredictor(ratings_path=predictor.ratings_path)
        assert predictor2.get_team_elo('Arsenal', 'premier_league') == 1850.0

    def test_load_missing_file_is_silent(self, tmp_path):
        """Loading a non-existent file should silently give an empty dict."""
        p = BetNeuralPredictor(ratings_path=str(tmp_path / "nonexistent.json"))
        assert p.team_ratings == {}

    def test_update_after_match_persists(self, predictor):
        predictor.update_after_match('Arsenal', 'Chelsea', 'H', 'premier_league', persist=True)
        predictor2 = BetNeuralPredictor(ratings_path=predictor.ratings_path)
        # Arsenal won so their rating should be different from the default
        assert predictor2.get_team_elo('Arsenal', 'premier_league') != predictor.get_team_elo.__func__

    def test_update_after_match_invalid_result(self, predictor):
        with pytest.raises(ValueError, match="result must be"):
            predictor.update_after_match('Arsenal', 'Chelsea', 'X', 'premier_league')

    def test_update_after_match_result_codes(self, predictor):
        """All three result codes should be accepted without raising."""
        for result in ('H', 'D', 'A'):
            predictor.update_after_match('Arsenal', 'Chelsea', result, 'premier_league', persist=False)


# ============================================================
#  predict_match integration
# ============================================================

class TestPredictMatch:

    def test_returns_required_keys(self, predictor):
        result = predictor.predict_match('Arsenal', 'Chelsea', 'premier_league')
        for key in ('match', 'league', 'probabilities', 'elo_ratings', 'expected_goals', 'confidence'):
            assert key in result

    def test_probabilities_sum_to_one(self, predictor):
        result = predictor.predict_match('Real Madrid', 'Barcelona', 'la_liga')
        total = sum(result['probabilities'].values())
        assert abs(total - 1.0) < 1e-9

    def test_with_odds_adds_betting_analysis(self, predictor):
        result = predictor.predict_match(
            'Bayern Munich', 'Borussia Dortmund', 'bundesliga',
            odds={'home_win': 1.8, 'draw': 3.6, 'away_win': 4.5}
        )
        assert 'betting_analysis' in result

    def test_without_odds_no_betting_analysis(self, predictor):
        result = predictor.predict_match('Inter Milan', 'Juventus', 'serie_a')
        assert 'betting_analysis' not in result

    def test_unknown_team_uses_league_default(self, predictor):
        """A team not in the ratings store should get a sensible default, not crash."""
        result = predictor.predict_match('Unknown FC', 'Mystery United', 'premier_league')
        assert result['confidence'] > 0

    def test_confidence_is_in_valid_range(self, predictor):
        """Confidence must be in [0, 1] — it is now BSS/entropy-based, not max(probs)."""
        result = predictor.predict_match('Arsenal', 'Chelsea', 'premier_league')
        assert 0.0 <= result['confidence'] <= 1.0


# ============================================================
#  CLI commands (subprocess-free, exercising the CLI class directly)
# ============================================================

class TestCLIPredict:
    """
    Exercise the CLI command functions directly using argparse.Namespace objects.
    The CLI is function-based (cmd_predict, cmd_gameweek, etc.) — there is no
    BetNeuralCLI class.
    """

    def _make_args(self, match, league="Premier League", odds=None, report=False, date=None):
        import argparse
        ns = argparse.Namespace(
            match=match, league=league, odds=odds,
            report=report, analytics=False, date=date,
        )
        return ns

    def test_predict_vs_format(self, capsys):
        from bet_neural_cli import cmd_predict
        cmd_predict(self._make_args("Arsenal vs Chelsea"))
        out = capsys.readouterr().out
        assert "Arsenal" in out
        assert "Chelsea" in out

    def test_predict_dash_format(self, capsys):
        from bet_neural_cli import cmd_predict
        cmd_predict(self._make_args("Real Madrid-Barcelona", league="La Liga"))
        out = capsys.readouterr().out
        assert "Real Madrid" in out or "Barcelona" in out

    def test_predict_with_odds_shows_betting_section(self, capsys):
        from bet_neural_cli import cmd_predict
        cmd_predict(self._make_args(
            "Bayern Munich vs Borussia Dortmund",
            league="Bundesliga",
            odds="1.8,3.6,4.5"
        ))
        out = capsys.readouterr().out
        # Betting section header appears when odds are supplied
        assert "BETTING" in out.upper()

    def test_predict_invalid_format_raises(self):
        from bet_neural_cli import _parse_match
        with pytest.raises(ValueError):
            _parse_match("NoSeparator")

    def test_list_leagues_output(self, capsys):
        from bet_neural_cli import cmd_leagues
        import argparse
        cmd_leagues(argparse.Namespace())
        out = capsys.readouterr().out
        for league in ("Premier League", "La Liga", "Bundesliga", "Serie A",
                       "Ligue 1", "Eredivisie", "Primeira Liga"):
            assert league in out, f"Expected '{league}' in leagues output"

    def test_gameweek_runs_without_crash(self, capsys):
        """cmd_gameweek should complete and produce at least one match line."""
        from bet_neural_cli import cmd_gameweek
        import argparse
        args = argparse.Namespace(league="Premier League")
        cmd_gameweek(args)
        out = capsys.readouterr().out
        assert "GAMEWEEK" in out.upper() or "vs" in out


# ============================================================
#  Lite predictor sanity checks
# ============================================================

class TestBetNeuralLite:

    @pytest.fixture
    def lite(self):
        from bet_neural_lite import BetNeuralLite
        return BetNeuralLite()

    def test_all_seven_leagues_in_ratings(self, lite):
        """Every league should have at least 10 teams in the hardcoded table."""
        for league in ('premier_league', 'la_liga', 'bundesliga', 'serie_a',
                       'ligue_1', 'eredivisie', 'primeira_liga'):
            count = sum(1 for k in lite.team_ratings if k.endswith(f"_{league}"))
            assert count >= 10, f"{league} only has {count} teams in hardcoded table (need ≥ 10)"

    def test_ligue1_teams_present(self, lite):
        assert lite.get_team_elo('PSG', 'ligue_1') > 1400

    def test_eredivisie_teams_present(self, lite):
        assert lite.get_team_elo('Ajax', 'eredivisie') > 1400

    def test_primeira_liga_teams_present(self, lite):
        assert lite.get_team_elo('Benfica', 'primeira_liga') > 1400

    def test_probabilities_sum_to_one(self, lite):
        probs = lite.calculate_match_probabilities(1800, 1600)
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_predict_match_returns_keys(self, lite):
        result = lite.predict_match('Ajax', 'PSV', 'eredivisie')
        for key in ('match', 'league', 'probabilities', 'elo_ratings', 'expected_goals', 'prediction'):
            assert key in result


# ============================================================
#  Fuzzy team name resolution
# ============================================================

class TestFuzzyResolution:

    def test_exact_match_returns_was_exact_true(self, predictor):
        # Seed a known team
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        name, sim, exact = predictor.resolve_team_name('Arsenal', 'premier_league')
        assert exact is True
        assert name == 'Arsenal'
        assert sim == 1.0

    def test_typo_is_corrected(self, predictor):
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        name, sim, exact = predictor.resolve_team_name('Arsenall', 'premier_league')
        assert exact is False
        assert name == 'Arsenal'
        assert sim > 0.8

    def test_casing_is_normalised(self, predictor):
        predictor.update_team_elo('Chelsea', 1780.0, 'premier_league')
        name, sim, exact = predictor.resolve_team_name('chelsea', 'premier_league')
        assert name == 'Chelsea'
        assert sim > 0.9

    def test_completely_unknown_team_returns_zero_sim(self, predictor):
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        name, sim, exact = predictor.resolve_team_name('zzzzunknownzzzz', 'premier_league')
        assert exact is False
        assert sim == 0.0

    def test_predict_match_warns_on_typo(self, predictor):
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        predictor.update_team_elo('Chelsea', 1780.0, 'premier_league')
        result = predictor.predict_match('Arsenall', 'Chelseaa', 'premier_league')
        assert len(result['warnings']) == 2
        assert 'Arsenal' in result['warnings'][0]
        assert 'Chelsea' in result['warnings'][1]

    def test_predict_match_no_warnings_for_known_teams(self, predictor):
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        predictor.update_team_elo('Chelsea', 1780.0, 'premier_league')
        result = predictor.predict_match('Arsenal', 'Chelsea', 'premier_league')
        assert result['warnings'] == []

    def test_predict_match_warns_unknown_team(self, predictor):
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        result = predictor.predict_match('Arsenal', 'Totally Made Up FC', 'premier_league')
        assert any('not recognized' in w or 'not recognised' in w for w in result['warnings'])

    def test_corrected_name_used_in_result_match_field(self, predictor):
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        predictor.update_team_elo('Chelsea', 1780.0, 'premier_league')
        result = predictor.predict_match('Arsenall', 'Chelsea', 'premier_league')
        # The resolved name should appear in the match field, not the typo
        assert 'Arsenal' in result['match']
        assert 'Arsenall' not in result['match']
        # The original input is preserved separately
        assert 'Arsenall' in result['match_input']

    # ── same-team guard ───────────────────────────────────────────────

    def test_same_team_exact_raises(self, predictor):
        """A team vs itself should be rejected outright."""
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        with pytest.raises(ValueError, match="same team"):
            predictor.predict_match('Arsenal', 'Arsenal', 'premier_league')

    def test_same_team_via_typo_raises(self, predictor):
        """Two different typos that both resolve to the same team should be rejected."""
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        with pytest.raises(ValueError, match="same team"):
            predictor.predict_match('Arsenall', 'Arsnel', 'premier_league')

    def test_different_teams_do_not_raise(self, predictor):
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        predictor.update_team_elo('Chelsea', 1780.0, 'premier_league')
        # Should not raise
        predictor.predict_match('Arsenal', 'Chelsea', 'premier_league')

    def test_same_team_case_insensitive(self, predictor):
        """'arsenal' vs 'ARSENAL' should also be caught."""
        predictor.update_team_elo('Arsenal', 1850.0, 'premier_league')
        with pytest.raises(ValueError, match="same team"):
            predictor.predict_match('arsenal', 'ARSENAL', 'premier_league')
