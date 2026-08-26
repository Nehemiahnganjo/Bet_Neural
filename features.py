"""
features.py — Bet Neural Feature Engineering Pipeline
======================================================
Transforms raw scraped data into ML-ready feature vectors.

Features computed:
  • Team form (rolling window: last 5, 10 matches — wins, draws, goals, xG)
  • xG differential (attack vs defence quality)
  • Head-to-head history (H2H win rate, avg goals, last 5 meetings)
  • Player quality index (weighted squad xG contribution, key player availability)
  • Transfer impact score (net spend, key arrivals/departures)
  • Market value ratio (proxy for squad depth)
  • League position & points trajectory
  • Home/away splits
  • Fatigue index (matches in last 14/21 days)
  • Elo rating differential
  • Bookmaker implied probabilities (if odds available)
  • Season momentum (points-per-game trend over last 10 vs full season)
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ── Name normalisation ────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return name.lower().strip().replace(".", "").replace("'", "")


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> str:
    """Parse date string, handling ISO format (2026-02-27T20:00:00Z). Returns YYYY-MM-DD."""
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")
    # If already YYYY-MM-DD format, return as-is
    if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
        return date_str
    # Handle ISO format
    return date_str[:10]



# ── Feature vector definition (MUST stay in sync with model) ─────────────────

FEATURE_NAMES = [
    # ---- Form (home team, last 5) ----
    "h_form5_win_rate",
    "h_form5_draw_rate",
    "h_form5_goals_scored_avg",
    "h_form5_goals_conceded_avg",
    "h_form5_xg_avg",
    "h_form5_xga_avg",
    "h_form5_pts",
    # ---- Form (away team, last 5) ----
    "a_form5_win_rate",
    "a_form5_draw_rate",
    "a_form5_goals_scored_avg",
    "a_form5_goals_conceded_avg",
    "a_form5_xg_avg",
    "a_form5_xga_avg",
    "a_form5_pts",
    # ---- Form (last 10 season-wide) ----
    "h_form10_pts",
    "a_form10_pts",
    # ---- Season totals ----
    "h_season_xg_per_game",
    "h_season_xga_per_game",
    "h_season_npxg_per_game",
    "h_season_goals_per_game",
    "h_season_goals_conceded_per_game",
    "h_season_win_rate",
    "a_season_xg_per_game",
    "a_season_xga_per_game",
    "a_season_npxg_per_game",
    "a_season_goals_per_game",
    "a_season_goals_conceded_per_game",
    "a_season_win_rate",
    # ---- xG differential (attack - defence quality) ----
    "h_xg_diff",
    "a_xg_diff",
    "xg_diff_delta",            # (h_xg_diff - a_xg_diff)
    # ---- Home/away specific form ----
    "h_home_win_rate",
    "h_home_goals_avg",
    "h_home_xg_avg",
    "a_away_win_rate",
    "a_away_goals_avg",
    "a_away_xg_avg",
    # ---- Head-to-head ----
    "h2h_home_win_rate",        # % of H2H home team won
    "h2h_draw_rate",
    "h2h_avg_goals_home",
    "h2h_avg_goals_away",
    "h2h_last5_home_pts",       # pts in last 5 H2H
    "h2h_matches_available",    # how many H2H we have
    # ---- Player quality ----
    "h_squad_xg_index",         # sum of top 11 players' xG/90
    "a_squad_xg_index",
    "h_key_player_rating",      # avg sofascore/fbref rating of top 5 players
    "a_key_player_rating",
    "h_top_scorer_xg",          # top scorer's xG this season
    "a_top_scorer_xg",
    # ---- Transfer impact ----
    "h_transfer_net_spend_m",   # net spend in EUR millions (positive = bought)
    "a_transfer_net_spend_m",
    "h_market_value_m",         # total squad market value EUR millions
    "a_market_value_m",
    "market_value_ratio",       # h/a market value ratio
    # ---- Elo ----
    "elo_diff",                 # home_elo - away_elo (with home advantage baked in)
    "h_elo",
    "a_elo",
    # ---- Fatigue / schedule ----
    "h_days_since_last_match",
    "a_days_since_last_match",
    "h_matches_last_14d",
    "a_matches_last_14d",
    # ---- Bookmaker implied probs (if available) ----
    "bm_home_implied_prob",
    "bm_draw_implied_prob",
    "bm_away_implied_prob",
    "bm_home_odds",
    "bm_draw_odds",
    "bm_away_odds",
    # ---- Season momentum ----
    "h_ppg_last10",             # points per game last 10
    "a_ppg_last10",
    "h_ppg_full_season",
    "a_ppg_full_season",
    "h_momentum",               # ppg_last10 - ppg_full_season
    "a_momentum",
    # ---- League position ----
    "h_league_position",        # normalised 0-1 (1 = top)
    "a_league_position",
    "position_diff",
]

N_FEATURES = len(FEATURE_NAMES)


class MatchHistory:
    """
    Efficient rolling match history store for a single league/season.
    Supports fast queries for:
    - Team form over last N matches (overall / home-only / away-only)
    - H2H records
    - Fatigue calculations
    """

    def __init__(self) -> None:
        self._matches: List[Dict] = []  # sorted chronologically

    def load(self, matches: List[Dict]) -> "MatchHistory":
        """Load match list. Each match: {date, home_team, away_team, home_goals, away_goals, xg_home, xg_away, result}"""
        self._matches = sorted(
            [m for m in matches if m.get("result") in ("H", "D", "A") and m.get("date")],
            key=lambda m: m["date"],
        )
        return self

    def _matches_before(
        self,
        date_str: str,
        team: Optional[str] = None,
        season_start: Optional[str] = None,   # YYYY-MM-DD — hard lower bound
    ) -> List[Dict]:
        """
        All completed matches strictly before *date_str*, optionally filtered
        to a single team and/or bounded by a season start date.

        The season_start guard prevents leaking previous-season data when
        computing full-season averages with large n values (e.g. n=200).
        """
        result = [m for m in self._matches if m["date"] < date_str]
        if season_start:
            result = [m for m in result if m["date"] >= season_start]
        if team:
            t = _norm(team)
            result = [
                m for m in result
                if _norm(m["home_team"]) == t or _norm(m["away_team"]) == t
            ]
        return result

    def team_form(
        self,
        team: str,
        as_of: str,
        n: int = 5,
        venue: Optional[str] = None,   # "home", "away", or None for all
        season_start: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute form metrics for *team* over the last *n* matches before *as_of*.
        Pass season_start to restrict to the current season only (prevents
        cross-season contamination when n is large, e.g. n=200).
        Returns dict of stats or defaults if not enough history.
        """
        t = _norm(team)
        matches = self._matches_before(as_of, team, season_start=season_start)

        if venue == "home":
            matches = [m for m in matches if _norm(m["home_team"]) == t]
        elif venue == "away":
            matches = [m for m in matches if _norm(m["away_team"]) == t]

        recent = matches[-n:] if len(matches) >= n else matches
        if not recent:
            return _default_form()

        wins = draws = losses = 0
        goals_for = goals_against = xg_for = xg_against = 0.0

        for m in recent:
            is_home = _norm(m["home_team"]) == t
            # Support both field-name conventions:
            #   features.py canonical:  home_goals / away_goals
            #   scraper + football_data_api:  home_score / away_score
            gf  = m.get("home_goals")  if m.get("home_goals")  is not None else m.get("home_score",  0) or 0
            ga  = m.get("away_goals")  if m.get("away_goals")  is not None else m.get("away_score",  0) or 0
            xgf = m.get("xg_home",    0.0) or 0.0
            xga = m.get("xg_away",    0.0) or 0.0

            if not is_home:
                gf, ga   = ga, gf
                xgf, xga = xga, xgf

            goals_for     += gf
            goals_against += ga
            xg_for        += xgf
            xg_against    += xga

            res = m.get("result")
            if (is_home and res == "H") or (not is_home and res == "A"):
                wins += 1
            elif res == "D":
                draws += 1
            else:
                losses += 1

        n_played = len(recent)
        pts = wins * 3 + draws

        return {
            "win_rate":          wins / n_played,
            "draw_rate":         draws / n_played,
            "loss_rate":         losses / n_played,
            "goals_scored_avg":  goals_for / n_played,
            "goals_conceded_avg": goals_against / n_played,
            "xg_avg":            xg_for / n_played,
            "xga_avg":           xg_against / n_played,
            "pts":               pts,
            "n_matches":         n_played,
        }

    def h2h_stats(self, home_team: str, away_team: str, as_of: str, n: int = 10) -> Dict:
        """H2H record for home_team hosting away_team (and reverse) over last n."""
        ht = _norm(home_team)
        at = _norm(away_team)

        relevant = [
            m for m in self._matches_before(as_of)
            if ((_norm(m["home_team"]) == ht and _norm(m["away_team"]) == at)
                or (_norm(m["home_team"]) == at and _norm(m["away_team"]) == ht))
        ][-n:]

        if not relevant:
            return _default_h2h()

        home_wins = draws = away_wins = 0
        home_goals_total = away_goals_total = 0
        home_pts_from_perspective = 0

        for m in relevant:
            is_home = _norm(m["home_team"]) == ht
            hg = m.get("home_goals") if m.get("home_goals") is not None else m.get("home_score", 0) or 0
            ag = m.get("away_goals") if m.get("away_goals") is not None else m.get("away_score", 0) or 0
            res = m.get("result")

            home_goals_total += (hg if is_home else ag)
            away_goals_total += (ag if is_home else hg)

            if res == "H":
                if is_home:
                    home_wins += 1
                    home_pts_from_perspective += 3
                else:
                    away_wins += 1
            elif res == "D":
                draws += 1
                home_pts_from_perspective += 1
            else:  # "A"
                if is_home:
                    away_wins += 1
                else:
                    home_wins += 1
                    home_pts_from_perspective += 3

        n_played = len(relevant)
        return {
            "home_win_rate":       home_wins / n_played,
            "draw_rate":           draws / n_played,
            "away_win_rate":       away_wins / n_played,
            "avg_goals_home":      home_goals_total / n_played,
            "avg_goals_away":      away_goals_total / n_played,
            "last5_home_pts":      home_pts_from_perspective,
            "matches_available":   n_played,
        }

    def fatigue(self, team: str, as_of: str) -> Dict[str, float]:
        """Matches played in last 14 and days since last match."""
        t = _norm(team)
        team_matches = self._matches_before(as_of, team)
        if not team_matches:
            return {"days_since_last": 14.0, "matches_last_14d": 0}

        last_date = _parse_date(team_matches[-1]["date"])
        try:
            d1 = datetime.strptime(last_date, "%Y-%m-%d")
            d2 = datetime.strptime(as_of,    "%Y-%m-%d")
            days_since = max(0, (d2 - d1).days)
        except ValueError:
            days_since = 7

        cutoff = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
        recent_14d = [m for m in team_matches if _parse_date(m["date"]) >= cutoff]

        return {
            "days_since_last": float(days_since),
            "matches_last_14d": float(len(recent_14d)),
        }

    def ppg(self, team: str, as_of: str, n: Optional[int] = None, season_start: Optional[str] = None) -> float:
        """Points per game over last n matches (or full season if n=None)."""
        matches = self._matches_before(as_of, team, season_start=season_start)
        if n:
            matches = matches[-n:]
        if not matches:
            return 1.2  # league average fallback

        t = _norm(team)
        pts = 0
        for m in matches:
            is_home = _norm(m["home_team"]) == t
            res = m.get("result")
            if (is_home and res == "H") or (not is_home and res == "A"):
                pts += 3
            elif res == "D":
                pts += 1

        return pts / len(matches)

    def league_position(self, team: str, as_of: str, all_teams: List[str], season_start: Optional[str] = None) -> float:
        """
        Estimate normalised league position (1.0 = top, 0.0 = bottom)
        based on points accumulated up to as_of.
        """
        pts_map: Dict[str, int] = defaultdict(int)
        for m in self._matches_before(as_of, season_start=season_start):
            ht_norm = _norm(m["home_team"])
            at_norm = _norm(m["away_team"])
            res = m.get("result")
            if res == "H":
                pts_map[ht_norm] += 3
            elif res == "D":
                pts_map[ht_norm] += 1
                pts_map[at_norm] += 1
            elif res == "A":
                pts_map[at_norm] += 3

        sorted_teams = sorted(pts_map.keys(), key=lambda t: -pts_map[t])
        t = _norm(team)
        n = len(sorted_teams)
        if n == 0:
            return 0.5
        try:
            rank = sorted_teams.index(t) + 1
        except ValueError:
            rank = n // 2
        return 1.0 - (rank - 1) / max(n - 1, 1)


# ── Player quality index ──────────────────────────────────────────────────────

class PlayerQualityIndex:
    """
    Builds a per-team quality index from player stats.
    Main metric: sum of top 11 players' non-penalty xG per 90 minutes.
    """

    def __init__(self, player_stats: List[Dict]) -> None:
        self._by_team: Dict[str, List[Dict]] = defaultdict(list)
        for p in player_stats:
            team = _norm(p.get("team", ""))
            if team:
                self._by_team[team].append(p)

    def squad_xg_index(self, team: str) -> float:
        """Sum of top 11 starters' xG per 90."""
        players = self._by_team.get(_norm(team), [])
        if not players:
            return 1.4  # league average

        def xg_per90(p: Dict) -> float:
            xg  = p.get("npxg") or p.get("xg") or 0.0
            mins = p.get("minutes") or p.get("min") or 0.0
            if mins < 90:
                return 0.0
            return (xg / mins) * 90.0

        per90 = sorted([xg_per90(p) for p in players], reverse=True)
        top11 = per90[:11]
        return round(sum(top11), 3)

    def key_player_rating(self, team: str) -> float:
        """
        Average rating of the top 5 players by xG contribution.
        Proxied via xG+xA per 90 since we don't have a direct "rating" field.
        """
        players = self._by_team.get(_norm(team), [])
        if not players:
            return 7.0  # fallback Sofascore-ish rating

        def contribution(p: Dict) -> float:
            xg  = (p.get("npxg") or p.get("xg") or 0.0)
            xa  = p.get("xa") or 0.0
            mins = p.get("minutes") or p.get("min") or 0.0
            if mins < 90:
                return 0.0
            return ((xg + xa) / mins) * 90.0

        top5 = sorted([contribution(p) for p in players], reverse=True)[:5]
        if not top5 or max(top5) == 0:
            return 7.0

        # Scale to ~6.0-9.5 range (comparable to Sofascore)
        raw = sum(top5) / len(top5)
        return round(6.0 + min(raw * 8.0, 3.5), 2)

    def top_scorer_xg(self, team: str) -> float:
        """xG of the team's top scorer."""
        players = self._by_team.get(_norm(team), [])
        if not players:
            return 5.0

        xg_vals = [p.get("npxg") or p.get("xg") or 0.0 for p in players]
        return round(max(xg_vals) if xg_vals else 0.0, 3)


# ── Transfer impact ───────────────────────────────────────────────────────────

class TransferImpact:
    """Computes net spend and market value ratios."""

    def __init__(
        self,
        transfers:     List[Dict],
        market_values: List[Dict],
    ) -> None:
        self._net_spend: Dict[str, float] = defaultdict(float)
        self._mv:        Dict[str, float] = {}

        for t in transfers:
            # This is league-level so we need team info from the row itself
            fee    = t.get("fee_eur", 0.0) or 0.0
            team   = _norm(t.get("team", "") or t.get("to_team", "") or t.get("from_team", ""))
            direction = t.get("direction", "unknown")
            if direction == "in":
                self._net_spend[team] += fee
            elif direction == "out":
                self._net_spend[team] -= fee

        for mv in market_values:
            team = _norm(mv.get("team", ""))
            self._mv[team] = mv.get("total_market_value_eur", 0.0) or 0.0

    def net_spend_m(self, team: str) -> float:
        """Net transfer spend in EUR millions (positive = net buyer)."""
        return round(self._net_spend.get(_norm(team), 0.0) / 1_000_000, 2)

    def market_value_m(self, team: str) -> float:
        """Total squad market value in EUR millions."""
        return round(self._mv.get(_norm(team), 0.0) / 1_000_000, 2)


# ── Feature builder ───────────────────────────────────────────────────────────

class FeatureBuilder:
    """
    Main class: takes raw data and produces feature vectors for any match.
    """

    def __init__(
        self,
        match_history:    MatchHistory,
        player_quality:   Optional[PlayerQualityIndex] = None,
        transfer_impact:  Optional[TransferImpact] = None,
        elo_ratings:      Optional[Dict[str, float]] = None,
        home_advantage:   float = 65.0,
        league:           str = "premier_league",
        season_start:     Optional[str] = None,   # e.g. "2024-07-01"
    ) -> None:
        self.history         = match_history
        self.players         = player_quality
        self.transfers       = transfer_impact
        self.elo_ratings     = elo_ratings or {}
        self.home_advantage  = home_advantage
        self.league          = league
        # season_start bounds season-wide stats (n=200) to the current season.
        # Derived automatically from match history if not supplied.
        self.season_start    = season_start or self._infer_season_start()
        self._all_teams: List[str] = []   # filled by build_training_set

    def _infer_season_start(self) -> Optional[str]:
        """
        Heuristically derive the season start from the earliest match in history.
        Returns YYYY-MM-DD or None if history is empty.
        """
        if not self.history._matches:
            return None
        earliest = self.history._matches[0]["date"][:10]
        return earliest

    # ── public API ──

    def build_feature_vector(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        odds: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Build a 1-D feature vector of shape (N_FEATURES,) for a match.
        """
        v = np.zeros(N_FEATURES, dtype=np.float32)

        # Helpers
        def set_feat(name: str, val: float) -> None:
            if name in FEATURE_NAMES:
                v[FEATURE_NAMES.index(name)] = float(val)

        # ---- Home form (last 5, all venues) ----
        hf5 = self.history.team_form(home_team, match_date, n=5)
        set_feat("h_form5_win_rate",           hf5["win_rate"])
        set_feat("h_form5_draw_rate",          hf5["draw_rate"])
        set_feat("h_form5_goals_scored_avg",   hf5["goals_scored_avg"])
        set_feat("h_form5_goals_conceded_avg", hf5["goals_conceded_avg"])
        set_feat("h_form5_xg_avg",             hf5["xg_avg"])
        set_feat("h_form5_xga_avg",            hf5["xga_avg"])
        set_feat("h_form5_pts",                hf5["pts"])

        # ---- Away form (last 5) ----
        af5 = self.history.team_form(away_team, match_date, n=5)
        set_feat("a_form5_win_rate",           af5["win_rate"])
        set_feat("a_form5_draw_rate",          af5["draw_rate"])
        set_feat("a_form5_goals_scored_avg",   af5["goals_scored_avg"])
        set_feat("a_form5_goals_conceded_avg", af5["goals_conceded_avg"])
        set_feat("a_form5_xg_avg",             af5["xg_avg"])
        set_feat("a_form5_xga_avg",            af5["xga_avg"])
        set_feat("a_form5_pts",                af5["pts"])

        # ---- Form last 10 ----
        set_feat("h_form10_pts", self.history.team_form(home_team, match_date, n=10)["pts"])
        set_feat("a_form10_pts", self.history.team_form(away_team, match_date, n=10)["pts"])

        # ---- Season totals (all prior matches this season) ----
        # season_start guard prevents cross-season contamination (task #6 fix)
        hsa = self.history.team_form(home_team, match_date, n=200, season_start=self.season_start)
        asa = self.history.team_form(away_team, match_date, n=200, season_start=self.season_start)

        def season_feat(prefix: str, form: Dict) -> None:
            gp = max(form["n_matches"], 1)
            set_feat(f"{prefix}_season_xg_per_game",             form["xg_avg"])
            set_feat(f"{prefix}_season_xga_per_game",            form["xga_avg"])
            set_feat(f"{prefix}_season_npxg_per_game",           form["xg_avg"] * 0.88)  # approx npxg
            set_feat(f"{prefix}_season_goals_per_game",          form["goals_scored_avg"])
            set_feat(f"{prefix}_season_goals_conceded_per_game", form["goals_conceded_avg"])
            set_feat(f"{prefix}_season_win_rate",                form["win_rate"])

        season_feat("h", hsa)
        season_feat("a", asa)

        # ---- xG differential ----
        h_xg_diff = hsa["xg_avg"] - hsa["xga_avg"]
        a_xg_diff = asa["xg_avg"] - asa["xga_avg"]
        set_feat("h_xg_diff",     h_xg_diff)
        set_feat("a_xg_diff",     a_xg_diff)
        set_feat("xg_diff_delta", h_xg_diff - a_xg_diff)

        # ---- Home/away splits ----
        hh = self.history.team_form(home_team, match_date, n=10, venue="home")
        aa = self.history.team_form(away_team, match_date, n=10, venue="away")
        set_feat("h_home_win_rate",  hh["win_rate"])
        set_feat("h_home_goals_avg", hh["goals_scored_avg"])
        set_feat("h_home_xg_avg",    hh["xg_avg"])
        set_feat("a_away_win_rate",  aa["win_rate"])
        set_feat("a_away_goals_avg", aa["goals_scored_avg"])
        set_feat("a_away_xg_avg",    aa["xg_avg"])

        # ---- H2H ----
        h2h = self.history.h2h_stats(home_team, away_team, match_date, n=10)
        set_feat("h2h_home_win_rate",      h2h["home_win_rate"])
        set_feat("h2h_draw_rate",          h2h["draw_rate"])
        set_feat("h2h_avg_goals_home",     h2h["avg_goals_home"])
        set_feat("h2h_avg_goals_away",     h2h["avg_goals_away"])
        set_feat("h2h_last5_home_pts",     h2h["last5_home_pts"])
        set_feat("h2h_matches_available",  h2h["matches_available"])

        # ---- Player quality ----
        if self.players:
            set_feat("h_squad_xg_index",    self.players.squad_xg_index(home_team))
            set_feat("a_squad_xg_index",    self.players.squad_xg_index(away_team))
            set_feat("h_key_player_rating", self.players.key_player_rating(home_team))
            set_feat("a_key_player_rating", self.players.key_player_rating(away_team))
            set_feat("h_top_scorer_xg",     self.players.top_scorer_xg(home_team))
            set_feat("a_top_scorer_xg",     self.players.top_scorer_xg(away_team))
        else:
            set_feat("h_squad_xg_index",    1.4)
            set_feat("a_squad_xg_index",    1.4)
            set_feat("h_key_player_rating", 7.0)
            set_feat("a_key_player_rating", 7.0)

        # ---- Transfer impact ----
        if self.transfers:
            h_ns = self.transfers.net_spend_m(home_team)
            a_ns = self.transfers.net_spend_m(away_team)
            h_mv = self.transfers.market_value_m(home_team)
            a_mv = self.transfers.market_value_m(away_team)
            set_feat("h_transfer_net_spend_m", np.clip(h_ns / 200.0, -1, 1))   # normalise
            set_feat("a_transfer_net_spend_m", np.clip(a_ns / 200.0, -1, 1))
            set_feat("h_market_value_m",       np.log1p(h_mv) / 10.0)          # log-scale
            set_feat("a_market_value_m",       np.log1p(a_mv) / 10.0)
            set_feat("market_value_ratio",     h_mv / max(a_mv, 1.0))
        else:
            set_feat("market_value_ratio", 1.0)
            set_feat("h_market_value_m", 0.5)
            set_feat("a_market_value_m", 0.5)

        # ---- Elo ----
        league_key  = self.league
        h_elo = self.elo_ratings.get(f"{home_team}_{league_key}", 1500.0)
        a_elo = self.elo_ratings.get(f"{away_team}_{league_key}", 1500.0)
        set_feat("elo_diff", (h_elo + self.home_advantage - a_elo) / 400.0)
        set_feat("h_elo",    h_elo / 2000.0)
        set_feat("a_elo",    a_elo / 2000.0)

        # ---- Fatigue ----
        hfat = self.history.fatigue(home_team, match_date)
        afat = self.history.fatigue(away_team, match_date)
        set_feat("h_days_since_last_match", min(hfat["days_since_last"] / 14.0, 1.0))
        set_feat("a_days_since_last_match", min(afat["days_since_last"] / 14.0, 1.0))
        set_feat("h_matches_last_14d",      hfat["matches_last_14d"] / 4.0)
        set_feat("a_matches_last_14d",      afat["matches_last_14d"] / 4.0)

        # ---- Bookmaker implied probs ----
        if odds:
            ho = odds.get("home_win") or odds.get("home_odds") or 0.0
            do = odds.get("draw")     or odds.get("draw_odds") or 0.0
            ao = odds.get("away_win") or odds.get("away_odds") or 0.0
            if ho > 1.0 and do > 1.0 and ao > 1.0:
                margin = 1/ho + 1/do + 1/ao
                set_feat("bm_home_implied_prob", (1/ho) / margin)
                set_feat("bm_draw_implied_prob", (1/do) / margin)
                set_feat("bm_away_implied_prob", (1/ao) / margin)
                set_feat("bm_home_odds",  ho)
                set_feat("bm_draw_odds",  do)
                set_feat("bm_away_odds",  ao)
        else:
            # No odds: use elo as a prior
            elo_probs = _elo_probs(h_elo, a_elo, self.home_advantage)
            set_feat("bm_home_implied_prob", elo_probs[0])
            set_feat("bm_draw_implied_prob", elo_probs[1])
            set_feat("bm_away_implied_prob", elo_probs[2])

        # ---- Momentum ----
        h_ppg10    = self.history.ppg(home_team, match_date, n=10)
        a_ppg10    = self.history.ppg(away_team, match_date, n=10)
        h_ppg_full = self.history.ppg(home_team, match_date, season_start=self.season_start)
        a_ppg_full = self.history.ppg(away_team, match_date, season_start=self.season_start)
        set_feat("h_ppg_last10",     h_ppg10 / 3.0)
        set_feat("a_ppg_last10",     a_ppg10 / 3.0)
        set_feat("h_ppg_full_season", h_ppg_full / 3.0)
        set_feat("a_ppg_full_season", a_ppg_full / 3.0)
        set_feat("h_momentum",       (h_ppg10 - h_ppg_full) / 3.0)
        set_feat("a_momentum",       (a_ppg10 - a_ppg_full) / 3.0)

        # ---- League position ----
        all_teams = self._all_teams or []
        h_pos = self.history.league_position(home_team, match_date, all_teams, season_start=self.season_start)
        a_pos = self.history.league_position(away_team, match_date, all_teams, season_start=self.season_start)
        set_feat("h_league_position", h_pos)
        set_feat("a_league_position", a_pos)
        set_feat("position_diff",     h_pos - a_pos)

        return v

    def build_training_set(
        self,
        matches: List[Dict],
        odds_lookup: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build full X, y arrays from a list of completed matches.

        Returns:
            X: (n_samples, N_FEATURES) float32 array
            y: (n_samples,) int array — 0=home win, 1=draw, 2=away win
            match_ids: list of "date_home_away" strings
        """
        # Determine all teams for league position calculation
        self._all_teams = list({
            _norm(m["home_team"]) for m in matches
        } | {_norm(m["away_team"]) for m in matches})

        X, y, ids = [], [], []
        result_map = {"H": 0, "D": 1, "A": 2}

        for m in matches:
            res = m.get("result")
            if res not in result_map:
                continue

            ht = m["home_team"]
            at = m["away_team"]
            dt = _parse_date(m["date"])

            # Retrieve pre-match odds if available
            odds = None
            if odds_lookup:
                key = f"{_norm(ht)}_{_norm(at)}"
                odds = odds_lookup.get(key)

            feats = self.build_feature_vector(ht, at, dt, odds)
            X.append(feats)
            y.append(result_map[res])
            ids.append(f"{dt}_{ht}_{at}")

        if not X:
            return np.empty((0, N_FEATURES)), np.empty(0, dtype=int), []

        return np.array(X, dtype=np.float32), np.array(y, dtype=int), ids

    def save_features(self, X: np.ndarray, y: np.ndarray, ids: List[str], name: str) -> None:
        """Save features to disk as numpy .npz."""
        np.savez(FEAT_DIR / f"{name}.npz", X=X, y=y)
        with open(FEAT_DIR / f"{name}_ids.json", "w") as fh:
            json.dump(ids, fh)
        logger.info(f"Saved {len(ids)} feature rows → {FEAT_DIR / name}.npz")

    @staticmethod
    def load_features(name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load previously saved features."""
        arr = np.load(FEAT_DIR / f"{name}.npz")
        with open(FEAT_DIR / f"{name}_ids.json") as fh:
            ids = json.load(fh)
        return arr["X"], arr["y"], ids


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_form() -> Dict[str, float]:
    return {
        "win_rate": 0.33, "draw_rate": 0.28, "loss_rate": 0.39,
        "goals_scored_avg": 1.4, "goals_conceded_avg": 1.3,
        "xg_avg": 1.35, "xga_avg": 1.30,
        "pts": 3.5, "n_matches": 0,
    }


def _default_h2h() -> Dict:
    return {
        "home_win_rate": 0.40, "draw_rate": 0.28, "away_win_rate": 0.32,
        "avg_goals_home": 1.4, "avg_goals_away": 1.2,
        "last5_home_pts": 6.0, "matches_available": 0,
    }


def _elo_probs(h_elo: float, a_elo: float, home_adv: float = 65.0) -> Tuple[float, float, float]:
    adj = h_elo + home_adv
    p_home_raw = 1.0 / (1.0 + 10 ** ((a_elo - adj) / 400.0))
    imbalance  = abs(p_home_raw - 0.5)
    draw_prob  = max(0.22, min(0.32, 0.30 - 0.16 * imbalance))
    remaining  = 1.0 - draw_prob
    h = p_home_raw * remaining
    a = (1.0 - p_home_raw) * remaining
    total = h + draw_prob + a
    return h / total, draw_prob / total, a / total


# ── Factory: build from scraped data files ────────────────────────────────────

def build_feature_builder_from_cache(
    league: str,
    season: str = "2024-2025",
    elo_ratings: Optional[Dict] = None,
) -> FeatureBuilder:
    """
    Convenience function: reads all cached raw data for a league/season
    and returns a fully configured FeatureBuilder.
    """
    us_season = season.split("-")[0]

    # Load match history
    matches: List[Dict] = []
    for candidate in [
        RAW_DIR / "api"       / f"{league}_{season}_matches.json",
        RAW_DIR / "understat" / f"{league}_{us_season}_matches.json",
        RAW_DIR / "fbref"     / f"{league}_{season}_matches.json",
    ]:
        if candidate.exists():
            with open(candidate) as fh:
                data = json.load(fh)
            # Normalize field names for API data (date_utc -> date)
            normalized = []
            for m in data:
                norm = dict(m)
                if "date_utc" in norm and "date" not in norm:
                    norm["date"] = norm.pop("date_utc")
                normalized.append(norm)
            matches.extend(normalized)

    history = MatchHistory().load(matches)

    # Player quality
    player_quality = None
    for candidate in [
        RAW_DIR / "api"       / f"{league}_{season}_players.json",
        RAW_DIR / "understat" / f"{league}_{us_season}_players.json",
        RAW_DIR / "fbref"     / f"{league}_{season}_players.json",
    ]:
        if candidate.exists():
            with open(candidate) as fh:
                players = json.load(fh)
            player_quality = PlayerQualityIndex(players)
            break

    # Transfer impact
    transfer_impact = None
    transfers_path = RAW_DIR / "transfermarkt" / f"{league}_{us_season}_transfers.json"
    mv_path        = RAW_DIR / "transfermarkt" / f"{league}_market_values.json"
    if transfers_path.exists() and mv_path.exists():
        with open(transfers_path) as fh:
            transfers = json.load(fh)
        with open(mv_path) as fh:
            mv = json.load(fh)
        transfer_impact = TransferImpact(transfers, mv)

    return FeatureBuilder(
        match_history=history,
        player_quality=player_quality,
        transfer_impact=transfer_impact,
        elo_ratings=elo_ratings or {},
        league=league,
        # Derive season start from the season string (e.g. "2024-2025" → "2024-07-01")
        # July 1 is a safe lower bound for all top European leagues.
        season_start=f"{us_season}-07-01",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    league = sys.argv[1] if len(sys.argv) > 1 else "premier_league"
    season = sys.argv[2] if len(sys.argv) > 2 else "2024-2025"

    builder = build_feature_builder_from_cache(league, season)
    matches = builder.history._matches

    if not matches:
        print(f"⚠️  No match data cached for {league} {season}. Run scraper first.")
        sys.exit(0)

    print(f"Building features for {len(matches)} matches in {league}...")
    X, y, ids = builder.build_training_set(matches)
    builder.save_features(X, y, ids, f"{league}_{season.replace('-', '_')}")

    class_counts = {0: int((y == 0).sum()), 1: int((y == 1).sum()), 2: int((y == 2).sum())}
    print(f"✅ Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"   Home wins: {class_counts[0]} | Draws: {class_counts[1]} | Away wins: {class_counts[2]}")
    print(f"   Saved → {FEAT_DIR}/{league}_{season.replace('-','_')}.npz")
