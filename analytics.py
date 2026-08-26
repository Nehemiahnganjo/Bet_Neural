"""
analytics.py — Bet Neural Intelligent Analytics & Value Betting Engine
========================================================================
Implements:

  1. ValueBetDetector
       - Compares model probabilities vs bookmaker implied probabilities
       - Computes edge, expected value, ROI estimates
       - Kelly Criterion stake sizing (full Kelly + fractional variants)
       - Poisson goal totals (over/under analysis)

  2. OddsAnalyzer
       - Multi-bookmaker comparison (best odds, line shopping)
       - Vig/overround calculation
       - Margin-free fair odds extraction
       - Closing line value (CLV) tracking

  3. PortfolioManager
       - Bankroll management across multiple bets
       - Diversification: max exposure per league/match
       - Drawdown protection

  4. MatchAnalytics
       - Full match narrative with stats
       - Confidence breakdown
       - Key storyline detection (form trends, H2H dominance, rest advantage)

Theory references:
  - Kelly (1956): logarithmic utility maximisation
  - Pinnacle: sharp bookmaker serves as probability calibration benchmark
  - Shin (1991): overround decomposition
  - Joseph et al. (2006): Bayesian network for football
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_EDGE          = 0.03    # minimum model edge to flag a bet (3%)
MIN_PROB          = 0.50    # minimum model probability (50%)
MIN_ODDS          = 1.20    # ignore very short odds
MAX_KELLY_FRAC    = 0.05    # hard cap on any single bet (5% of bankroll)
FRACTIONAL_KELLY  = 0.50    # use half-Kelly by default (lower variance)


# ── Odds utilities ────────────────────────────────────────────────────────────

def decimal_to_implied(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def margin_free_prob(raw_implied: Dict[str, float]) -> Dict[str, float]:
    """
    Remove bookmaker overround (Shin method).
    Proportionally normalises implied probs so they sum to 1.
    """
    total = sum(raw_implied.values())
    if total <= 0:
        return raw_implied
    return {k: v / total for k, v in raw_implied.items()}


def fair_odds(model_probs: Dict[str, float], margin: float = 0.05) -> Dict[str, float]:
    """
    Convert model probabilities to fair decimal odds with a small margin.
    Useful for publishing reference odds.
    """
    return {
        k: round(1.0 / max(v * (1 + margin), 0.01), 2)
        for k, v in model_probs.items()
    }


def overround(odds_dict: Dict[str, float]) -> float:
    """Calculate bookmaker overround (sum of implied probs - 1)."""
    return sum(decimal_to_implied(v) for v in odds_dict.values()) - 1.0


# ── Value bet detection ───────────────────────────────────────────────────────

class ValueBetDetector:
    """
    Detects positive expected-value bets by comparing model probabilities
    to bookmaker odds.

    Kelly Criterion (full):
        f* = (bp - q) / b  where b = odds-1, p = win prob, q = 1-p

    We use fractional Kelly (f* × 0.5) by default to reduce variance
    while preserving EV. Stakes are additionally capped at MAX_KELLY_FRAC.
    """

    def __init__(
        self,
        min_edge:        float = MIN_EDGE,
        min_prob:        float = MIN_PROB,
        min_odds:        float = MIN_ODDS,
        kelly_fraction:  float = FRACTIONAL_KELLY,
        max_stake_pct:   float = MAX_KELLY_FRAC,
    ) -> None:
        self.min_edge       = min_edge
        self.min_prob       = min_prob
        self.min_odds       = min_odds
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct  = max_stake_pct

    def analyse_outcome(
        self,
        outcome:      str,
        model_prob:   float,
        decimal_odds: float,
    ) -> Dict[str, Any]:
        """
        Full analysis of a single outcome (home_win / draw / away_win).

        Returns dict with:
          edge, implied_prob, kelly_stake, expected_value,
          recommended, confidence, verdict
        """
        if decimal_odds <= 1.0:
            return _no_bet(outcome, model_prob, 0.0, 0.0)

        implied  = decimal_to_implied(decimal_odds)
        edge     = model_prob - implied
        b        = decimal_odds - 1.0          # net profit per unit staked

        # Kelly Criterion (full Kelly):
        #   f* = (b·p - q) / b   where b = odds - 1,  p = win prob,  q = 1 - p
        #
        # Derivation:  f* = (b·p - q) / b
        #                 = (p·(b+1) - 1) / b
        #                 = (p·odds - 1) / (odds - 1)
        #
        # Common mistake: using (p - 1/odds) / b  which = (p·odds - 1) / (odds·(odds-1))
        # = f* / odds  — underestimates Kelly by a factor of 'odds'.
        #
        # We apply fractional Kelly (×0.5) and hard-cap at max_stake_pct.
        if edge > 0 and model_prob > 0 and b > 0:
            full_kelly = (model_prob * b - (1.0 - model_prob)) / b
            kelly = min(max(full_kelly, 0.0) * self.kelly_fraction, self.max_stake_pct)
        else:
            kelly = 0.0

        # Expected value per unit staked: EV = p*b - q
        ev = model_prob * b - (1.0 - model_prob)

        recommended = (
            edge   >= self.min_edge
            and model_prob >= self.min_prob
            and decimal_odds >= self.min_odds
        )

        # Confidence tier
        if model_prob >= 0.70:
            confidence = "high"
        elif model_prob >= 0.58:
            confidence = "medium"
        else:
            confidence = "low"

        # Verdict
        if recommended and confidence == "high":
            verdict = "⭐ STRONG VALUE"
        elif recommended and confidence == "medium":
            verdict = "✅ VALUE BET"
        elif edge > 0 and model_prob >= 0.45:
            verdict = "👀 MARGINAL EDGE"
        elif edge < -0.10:
            verdict = "❌ STRONGLY AVOID"
        else:
            verdict = "➖ NO VALUE"

        return {
            "outcome":          outcome,
            "model_prob":       round(model_prob, 4),
            "decimal_odds":     decimal_odds,
            "implied_prob":     round(implied, 4),
            "edge":             round(edge, 4),
            "expected_value":   round(ev, 4),
            "kelly_stake_pct":  round(kelly * 100, 2),
            "recommended":      recommended,
            "confidence":       confidence,
            "verdict":          verdict,
        }

    def analyse_match(
        self,
        model_probs: Dict[str, float],
        odds:        Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Analyse all outcomes for a match.
        Returns per-outcome analysis + best bet recommendation.
        """
        outcomes = {}
        key_map = {
            "home_win": ["home_win", "home_odds", "1"],
            "draw":     ["draw",     "draw_odds", "X"],
            "away_win": ["away_win", "away_odds", "2"],
        }

        for outcome, keys in key_map.items():
            prob = model_probs.get(outcome, 0.0)
            # Find odds under any known key name
            odd = 0.0
            for k in keys:
                if k in odds and odds[k] > 1.0:
                    odd = odds[k]
                    break
            outcomes[outcome] = self.analyse_outcome(outcome, prob, odd)

        # Best bet = highest EV among recommended bets
        recommended = [v for v in outcomes.values() if v["recommended"]]
        best = max(recommended, key=lambda x: x["edge"]) if recommended else None

        # Overround (if all three odds available)
        or_val = None
        if all(k in odds and odds[k] > 1.0 for k in ("home_win", "draw", "away_win")):
            or_val = round(overround({"h": odds["home_win"], "d": odds["draw"], "a": odds["away_win"]}), 4)

        return {
            "outcomes":            outcomes,
            "best_bet":            best,
            "n_recommended":       len(recommended),
            "bookmaker_overround": or_val,
        }

    def analyse_goalline(
        self,
        exp_home_goals: float,
        exp_away_goals: float,
        line:           float = 2.5,
        over_odds:      Optional[float] = None,
        under_odds:     Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Over/under goal line analysis using Poisson distribution.
        """
        from math import factorial, exp as mexp

        lh, la = exp_home_goals, exp_away_goals
        p_over = p_under = 0.0

        for i in range(15):
            for j in range(15):
                p_ij = (mexp(-lh) * lh**i / factorial(i)) * (mexp(-la) * la**j / factorial(j))
                total_goals = i + j
                if total_goals > line:
                    p_over += p_ij
                elif total_goals < line:
                    p_under += p_ij
                # exactly on the line (e.g. 2.5 can't land on)

        result = {
            "line":        line,
            "exp_goals":   round(lh + la, 2),
            "p_over":      round(p_over, 4),
            "p_under":     round(p_under, 4),
        }

        for label, prob, odd in [("over", p_over, over_odds), ("under", p_under, under_odds)]:
            if odd and odd > 1.0:
                implied = decimal_to_implied(odd)
                edge    = prob - implied
                b       = odd - 1.0
                kelly   = min(max(edge / b * self.kelly_fraction, 0.0), self.max_stake_pct)
                result[f"{label}_edge"]       = round(edge, 4)
                result[f"{label}_kelly_pct"]  = round(kelly * 100, 2)
                result[f"{label}_recommended"] = edge >= self.min_edge and prob >= 0.50

        return result


def _no_bet(outcome, prob, implied, edge):
    return {
        "outcome": outcome, "model_prob": prob,
        "decimal_odds": 0.0, "implied_prob": implied,
        "edge": edge, "expected_value": 0.0,
        "kelly_stake_pct": 0.0, "recommended": False,
        "confidence": "low", "verdict": "➖ NO ODDS",
    }


# ── Multi-bookmaker odds comparison ───────────────────────────────────────────

class OddsAnalyzer:
    """
    Compare odds across multiple bookmakers and find line value.
    """

    def __init__(self, odds_entries: List[Dict]) -> None:
        """
        odds_entries: list from OddsScraper (each has bookmaker, home_odds, draw_odds, away_odds)
        """
        self._entries = odds_entries

    def best_odds(self) -> Dict[str, float]:
        """Return the best (highest) odds available for each outcome."""
        best = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        for e in self._entries:
            best["home_win"] = max(best["home_win"], e.get("home_odds", 0) or 0)
            best["draw"]     = max(best["draw"],     e.get("draw_odds", 0) or 0)
            best["away_win"] = max(best["away_win"], e.get("away_odds", 0) or 0)
        return best

    def consensus_odds(self) -> Dict[str, float]:
        """Average odds across all bookmakers (trimmed mean — excludes outliers)."""
        result = {}
        for key, field in [("home_win", "home_odds"), ("draw", "draw_odds"), ("away_win", "away_odds")]:
            vals = sorted([e[field] for e in self._entries if e.get(field, 0) > 1.0])
            if not vals:
                result[key] = 0.0
                continue
            # Trim top/bottom 10%
            n_trim = max(1, len(vals) // 10)
            trimmed = vals[n_trim:-n_trim] if len(vals) > 2 * n_trim else vals
            result[key] = round(sum(trimmed) / len(trimmed), 3)
        return result

    def pinnacle_odds(self) -> Optional[Dict[str, float]]:
        """Return Pinnacle odds specifically (sharp reference line)."""
        for e in self._entries:
            if "pinnacle" in str(e.get("bookmaker", "")).lower():
                return {
                    "home_win": e.get("home_odds", 0.0),
                    "draw":     e.get("draw_odds", 0.0),
                    "away_win": e.get("away_odds", 0.0),
                }
        return None

    def fair_market_probs(self) -> Dict[str, float]:
        """Consensus odds → margin-free implied probabilities."""
        cons = self.consensus_odds()
        raw  = {
            k: decimal_to_implied(v) for k, v in cons.items() if v > 1.0
        }
        return margin_free_prob(raw) if raw else {}

    def overround_summary(self) -> Dict[str, float]:
        """Per-bookmaker overround."""
        summary = {}
        for e in self._entries:
            bm = e.get("bookmaker", "unknown")
            ho = e.get("home_odds", 0)
            do = e.get("draw_odds", 0)
            ao = e.get("away_odds", 0)
            if ho > 1 and do > 1 and ao > 1:
                summary[bm] = round(overround({"h": ho, "d": do, "a": ao}), 4)
        return summary

    def line_shopping_report(self) -> str:
        """Human-readable report showing best odds per outcome and which book offers them."""
        lines = ["📊 Line Shopping Report:"]
        for key, field in [("Home Win", "home_odds"), ("Draw", "draw_odds"), ("Away Win", "away_odds")]:
            best_odds = 0.0
            best_book = ""
            for e in self._entries:
                if e.get(field, 0) > best_odds:
                    best_odds = e[field]
                    best_book = e.get("bookmaker", "?")
            if best_odds > 1.0:
                lines.append(f"  {key:<10}: {best_odds:.3f}  ({best_book})")
        return "\n".join(lines)


# ── Portfolio & bankroll management ──────────────────────────────────────────

class PortfolioManager:
    """
    Manages a betting portfolio across multiple matches/leagues.

    Enforces:
    - Maximum exposure per match (never bet more than MAX_STAKE of bankroll)
    - Diversification limit (max N recommended bets active at once)
    - Kelly scaling when total allocation exceeds safe threshold
    """

    def __init__(
        self,
        bankroll:        float = 1000.0,
        max_match_stake: float = 0.05,     # 5% per match
        max_total_exposure: float = 0.25,  # max 25% of bankroll deployed at once
    ) -> None:
        self.bankroll           = bankroll
        self.max_match_stake    = max_match_stake
        self.max_total_exposure = max_total_exposure
        self._bets:   List[Dict] = []
        self._history: List[Dict] = []

    def add_bet(
        self,
        match:       str,
        outcome:     str,
        odds:        float,
        kelly_pct:   float,
        model_prob:  float,
        edge:        float,
        league:      str = "",
    ) -> Dict:
        """
        Add a bet to the portfolio. Returns the actual stake recommended
        after portfolio-level constraints.
        """
        raw_stake_pct = min(kelly_pct / 100.0, self.max_match_stake)

        # Scale down if total exposure would exceed limit
        current_exposure = sum(b["stake_amount"] for b in self._bets) / self.bankroll
        remaining        = self.max_total_exposure - current_exposure
        if remaining <= 0:
            return {"match": match, "stake_pct": 0.0, "stake_amount": 0.0,
                    "status": "PORTFOLIO_FULL", "reason": "Max total exposure reached"}

        actual_pct    = min(raw_stake_pct, remaining)
        stake_amount  = round(self.bankroll * actual_pct, 2)

        bet = {
            "match":        match,
            "outcome":      outcome,
            "odds":         odds,
            "model_prob":   model_prob,
            "edge":         edge,
            "league":       league,
            "stake_pct":    round(actual_pct * 100, 2),
            "stake_amount": stake_amount,
            "expected_profit": round(stake_amount * (odds - 1) * model_prob
                                     - stake_amount * (1 - model_prob), 2),
            "status":       "PENDING",
            "placed_at":    datetime.now().isoformat(),
        }
        self._bets.append(bet)
        return bet

    def settle_bet(self, match: str, outcome: str, actual_result: str) -> Optional[Dict]:
        """Mark a bet as won/lost and update bankroll."""
        for bet in self._bets:
            if bet["match"] == match and bet["outcome"] == outcome and bet["status"] == "PENDING":
                won = (outcome == actual_result)
                profit = bet["stake_amount"] * (bet["odds"] - 1) if won else -bet["stake_amount"]
                self.bankroll += profit
                bet["status"]   = "WON" if won else "LOST"
                bet["profit"]   = round(profit, 2)
                bet["settled_at"] = datetime.now().isoformat()
                self._history.append(bet)
                self._bets.remove(bet)
                return bet
        return None

    def summary(self) -> Dict:
        """Portfolio summary stats."""
        settled = self._history
        if not settled:
            return {
                "bankroll": self.bankroll,
                "active_bets": len(self._bets),
                "total_bets": 0,
                "roi": 0.0,
            }

        n     = len(settled)
        won   = sum(1 for b in settled if b["status"] == "WON")
        total_staked = sum(b["stake_amount"] for b in settled)
        total_profit = sum(b.get("profit", 0) for b in settled)
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0

        return {
            "bankroll":          round(self.bankroll, 2),
            "active_bets":       len(self._bets),
            "total_bets_settled": n,
            "wins":              won,
            "losses":            n - won,
            "win_rate":          round(won / n * 100, 1),
            "total_staked":      round(total_staked, 2),
            "total_profit":      round(total_profit, 2),
            "roi_pct":           round(roi, 2),
        }

    def save(self, path: Optional[str] = None) -> None:
        path = path or str(DATA_DIR / "portfolio.json")
        with open(path, "w") as fh:
            json.dump({
                "bankroll":    self.bankroll,
                "active_bets": self._bets,
                "history":     self._history,
                "saved_at":    datetime.now().isoformat(),
            }, fh, indent=2)

    def load(self, path: Optional[str] = None) -> bool:
        path = path or str(DATA_DIR / "portfolio.json")
        if not Path(path).exists():
            return False
        with open(path) as fh:
            data = json.load(fh)
        self.bankroll  = data.get("bankroll", self.bankroll)
        self._bets     = data.get("active_bets", [])
        self._history  = data.get("history", [])
        return True


# ── Match analytics narrative ─────────────────────────────────────────────────

class MatchAnalytics:
    """
    Generates a human-readable intelligence report for a match.
    Incorporates model probabilities, odds analysis, and historical patterns.
    """

    def __init__(self, value_detector: Optional[ValueBetDetector] = None) -> None:
        self.vbd = value_detector or ValueBetDetector()

    def generate_report(
        self,
        home_team:   str,
        away_team:   str,
        league:      str,
        model_probs: Dict[str, float],
        odds:        Optional[Dict[str, float]] = None,
        features:    Optional[Dict[str, float]] = None,
        elo_ratings: Optional[Dict[str, float]] = None,
        exp_goals:   Optional[Tuple[float, float]] = None,
        poisson_probs: Optional[Dict[str, float]] = None,
    ) -> str:
        """Full intelligence report as a formatted string."""

        lines = []
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"  🔎 MATCH INTELLIGENCE REPORT")
        lines.append(f"  {home_team} vs {away_team}  |  {league.replace('_', ' ').title()}")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 72)

        # ── Probabilities ──
        lines.append("")
        lines.append("📊 OUTCOME PROBABILITIES")
        lines.append("-" * 40)

        sorted_outcomes = sorted(model_probs.items(), key=lambda x: -x[1])
        medals = ["🥇", "🥈", "🥉"]
        for i, (outcome, prob) in enumerate(sorted_outcomes):
            bar_len = int(prob * 32)
            bar     = "█" * bar_len + "░" * (32 - bar_len)
            medal   = medals[i] if i < 3 else "  "
            label   = outcome.replace("_", " ").title()
            lines.append(f"  {medal} {label:<12} {prob:6.1%}  {bar}")

        # ── xG & Poisson ──
        if exp_goals:
            lh, la = exp_goals
            lines.append("")
            lines.append("⚽ EXPECTED GOALS (Dixon-Coles Poisson)")
            lines.append("-" * 40)
            lines.append(f"  {home_team:<25} xG: {lh:.2f}")
            lines.append(f"  {away_team:<25} xG: {la:.2f}")
            lines.append(f"  Total expected goals: {lh + la:.2f}")
            if poisson_probs:
                for outcome, prob in poisson_probs.items():
                    if outcome in ("home_win", "draw", "away_win"):
                        lines.append(f"  Poisson {outcome.replace('_', ' '):<15}: {prob:.1%}")

        # ── Elo ──
        if elo_ratings:
            h_elo = elo_ratings.get("home", 1500)
            a_elo = elo_ratings.get("away", 1500)
            lines.append("")
            lines.append("📈 ELO RATINGS")
            lines.append("-" * 40)
            lines.append(f"  🏠 {home_team:<25} {h_elo:.0f}")
            lines.append(f"  ✈️  {away_team:<25} {a_elo:.0f}")
            lines.append(f"  Differential: {h_elo - a_elo:+.0f} (home advantage not included)")

        # ── Feature highlights ──
        if features:
            lines.append("")
            lines.append("🔬 KEY STATS")
            lines.append("-" * 40)
            stat_map = {
                "h_form5_pts":              (f"{home_team} last 5 pts",  lambda v: f"{v:.0f}/15"),
                "a_form5_pts":              (f"{away_team} last 5 pts",  lambda v: f"{v:.0f}/15"),
                "h_form5_xg_avg":           (f"{home_team} xG/game",     lambda v: f"{v:.2f}"),
                "a_form5_xg_avg":           (f"{away_team} xG/game",     lambda v: f"{v:.2f}"),
                "h_form5_xga_avg":          (f"{home_team} xGA/game",    lambda v: f"{v:.2f}"),
                "a_form5_xga_avg":          (f"{away_team} xGA/game",    lambda v: f"{v:.2f}"),
                "h_home_win_rate":          (f"{home_team} home W%",     lambda v: f"{v:.0%}"),
                "a_away_win_rate":          (f"{away_team} away W%",     lambda v: f"{v:.0%}"),
                "h2h_home_win_rate":        ("H2H home win%",            lambda v: f"{v:.0%}"),
                "h2h_matches_available":    ("H2H meetings",             lambda v: f"{v:.0f}"),
                "h_days_since_last_match":  (f"{home_team} rest (norm)", lambda v: f"{v*14:.0f}d"),
                "a_days_since_last_match":  (f"{away_team} rest (norm)", lambda v: f"{v*14:.0f}d"),
                "h_momentum":               (f"{home_team} momentum",    lambda v: f"{v:+.2f}"),
                "a_momentum":               (f"{away_team} momentum",    lambda v: f"{v:+.2f}"),
                "market_value_ratio":       ("Market value ratio",        lambda v: f"{v:.2f}x"),
            }
            for feat_key, (label, fmt) in stat_map.items():
                if feat_key in features:
                    lines.append(f"  {label:<35} {fmt(features[feat_key])}")

        # ── Value betting analysis ──
        if odds:
            lines.append("")
            lines.append("💰 VALUE BETTING ANALYSIS")
            lines.append("-" * 40)
            analysis = self.vbd.analyse_match(model_probs, odds)
            or_val   = analysis.get("bookmaker_overround")

            if or_val is not None:
                lines.append(f"  Bookmaker overround: {or_val:.1%}")

            for outcome, data in analysis["outcomes"].items():
                if data.get("decimal_odds", 0) > 1.0:
                    lines.append(
                        f"\n  {outcome.replace('_', ' ').title():<12}"
                        f"  Odds: {data['decimal_odds']:.2f}"
                        f"  Model: {data['model_prob']:.1%}"
                        f"  Implied: {data['implied_prob']:.1%}"
                        f"  Edge: {data['edge']:+.1%}"
                    )
                    lines.append(f"  {'':12}  {data['verdict']}")
                    if data["recommended"]:
                        lines.append(
                            f"  {'':12}  Kelly stake: {data['kelly_stake_pct']:.2f}% of bankroll"
                            f"  |  EV: {data['expected_value']:+.3f}"
                        )

            best = analysis.get("best_bet")
            if best:
                lines.append("")
                lines.append(f"  ⭐ BEST BET: {best['outcome'].replace('_',' ').title()}"
                              f" @ {best['decimal_odds']:.2f}"
                              f"  (edge: {best['edge']:+.1%},"
                              f"  kelly: {best['kelly_stake_pct']:.2f}%)")
            else:
                lines.append("")
                lines.append("  ❌ No value bets found at current odds.")

        # ── Storylines ──
        storylines = self._detect_storylines(
            home_team, away_team, features or {}, model_probs
        )
        if storylines:
            lines.append("")
            lines.append("📰 KEY STORYLINES")
            lines.append("-" * 40)
            for s in storylines:
                lines.append(f"  • {s}")

        lines.append("")
        lines.append("=" * 72)
        lines.append("  🤖 Bet Neural — Powered by XGBoost + LightGBM + Dixon-Coles")
        lines.append("=" * 72)
        return "\n".join(lines)

    def _detect_storylines(
        self,
        home_team:   str,
        away_team:   str,
        features:    Dict[str, float],
        model_probs: Dict[str, float],
    ) -> List[str]:
        """Detect key narrative points from feature values."""
        stories = []

        # Form contrast
        h_pts = features.get("h_form5_pts", 7.5)
        a_pts = features.get("a_form5_pts", 7.5)
        if h_pts >= 12:
            stories.append(f"{home_team} in outstanding form: {h_pts:.0f}/15 pts last 5 games")
        elif h_pts <= 4:
            stories.append(f"{home_team} in poor form: only {h_pts:.0f}/15 pts last 5 games")
        if a_pts >= 12:
            stories.append(f"{away_team} arrive on a {a_pts:.0f}/15 pts run — strong away threat")
        elif a_pts <= 4:
            stories.append(f"{away_team} struggling: {a_pts:.0f}/15 pts last 5 games")

        # Momentum swings
        h_mom = features.get("h_momentum", 0.0)
        a_mom = features.get("a_momentum", 0.0)
        if h_mom > 0.15:
            stories.append(f"{home_team} trending upward — PPG improving in recent games")
        if a_mom < -0.15:
            stories.append(f"{away_team} fading — recent form below their season average")

        # Rest advantage
        h_rest = features.get("h_days_since_last_match", 0.5) * 14
        a_rest = features.get("a_days_since_last_match", 0.5) * 14
        if h_rest > a_rest + 3:
            stories.append(f"{home_team} have a {h_rest - a_rest:.0f}-day rest advantage")
        elif a_rest > h_rest + 3:
            stories.append(f"{away_team} fresher: {a_rest - h_rest:.0f} more days rest")

        # H2H dominance
        h2h_rate = features.get("h2h_home_win_rate", 0.4)
        h2h_n    = features.get("h2h_matches_available", 0)
        if h2h_n >= 4:
            if h2h_rate >= 0.6:
                stories.append(f"{home_team} dominant in H2H: won {h2h_rate:.0%} of last {h2h_n:.0f} meetings")
            elif h2h_rate <= 0.25:
                stories.append(f"{home_team} historically struggles vs {away_team}: only {h2h_rate:.0%} H2H win rate")

        # Market value disparity
        mv_ratio = features.get("market_value_ratio", 1.0)
        if mv_ratio > 2.0:
            stories.append(f"{home_team} have a significant squad value advantage ({mv_ratio:.1f}x higher value)")
        elif mv_ratio < 0.5:
            stories.append(f"{away_team} are the higher-value squad ({1/mv_ratio:.1f}x advantage)")

        # xG attack/defence
        h_xg  = features.get("h_form5_xg_avg", 1.4)
        a_xga = features.get("a_form5_xga_avg", 1.4)
        if h_xg > 2.0 and a_xga > 1.8:
            stories.append(f"High-scoring potential: {home_team} create {h_xg:.1f} xG/game vs {away_team}'s leaky defence ({a_xga:.1f} xGA/game)")

        # Confidence level
        best_prob = max(model_probs.values())
        best_out  = max(model_probs, key=model_probs.get)
        if best_prob >= 0.65:
            stories.append(f"Strong prediction signal: {best_out.replace('_', ' ').title()} at {best_prob:.0%} probability")
        elif best_prob < 0.45:
            stories.append("Highly uncertain match — all three outcomes remain competitive")

        return stories[:6]  # cap at 6 storylines


# ── Convenience: full match report from prediction result ────────────────────

def generate_full_report(
    home_team:   str,
    away_team:   str,
    league:      str,
    model_probs: Dict[str, float],
    odds:        Optional[Dict[str, float]] = None,
    feature_vector: Optional[np.ndarray]   = None,
    feature_names:  Optional[List[str]]    = None,
    elo_ratings:    Optional[Dict]         = None,
    exp_goals:      Optional[Tuple[float, float]] = None,
    poisson_probs:  Optional[Dict]         = None,
) -> str:
    """One-call convenience wrapper for generating a full report."""

    features_dict: Optional[Dict[str, float]] = None
    if feature_vector is not None and feature_names:
        features_dict = dict(zip(feature_names, feature_vector.tolist()))

    analytics = MatchAnalytics()
    return analytics.generate_report(
        home_team=home_team,
        away_team=away_team,
        league=league,
        model_probs=model_probs,
        odds=odds,
        features=features_dict,
        elo_ratings=elo_ratings,
        exp_goals=exp_goals,
        poisson_probs=poisson_probs,
    )
