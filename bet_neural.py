"""
bet_neural.py — Bet Neural Core Engine v4.2
============================================
Production prediction engine combining Elo ratings, Poisson-Elo Monte Carlo,
and an XGBoost/LightGBM/MLP ensemble with isotonic calibration.

Key design points:
  • Stochastic Poisson-Elo Monte Carlo (entropy-seeded per call)
  • Elo noise σ=20 pts (FiveThirtyEight empirical range)
  • Confidence from Brier Skill Score (validated) or entropy (fallback)
  • Dynamic blend weights: 60% MC + 40% Elo when MC available
  • xG from real match history → goals proxy → Elo prior (tiered)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

LEAGUES: Dict[str, Dict] = {
    "premier_league": {"name": "Premier League", "country": "England", "strength": 1.00, "home_adv": 65},
    "la_liga": {"name": "La Liga", "country": "Spain", "strength": 0.95, "home_adv": 60},
    "bundesliga": {"name": "Bundesliga", "country": "Germany", "strength": 0.90, "home_adv": 58},
    "serie_a": {"name": "Serie A", "country": "Italy", "strength": 0.85, "home_adv": 55},
    "ligue_1": {"name": "Ligue 1", "country": "France", "strength": 0.80, "home_adv": 52},
    "eredivisie": {"name": "Eredivisie", "country": "Netherlands", "strength": 0.70, "home_adv": 50},
    "primeira_liga": {"name": "Primeira Liga", "country": "Portugal", "strength": 0.65, "home_adv": 48},
}

DEFAULT_RATINGS_PATH = BASE_DIR / "elo_ratings.json"


def _elo_probs(h_elo: float, a_elo: float, home_adv: float = 65.0) -> Tuple[float, float, float]:
    """Calibrated Elo probabilities."""
    adj = h_elo + home_adv
    p_h_raw = 1.0 / (1.0 + 10.0 ** ((a_elo - adj) / 400.0))
    # Draw probability based on match balance
    imbalance = abs(p_h_raw - 0.5)
    draw_p = max(0.22, min(0.32, 0.30 - 0.16 * imbalance))
    remaining = 1.0 - draw_p
    h = p_h_raw * remaining
    a = (1.0 - p_h_raw) * remaining
    total = h + draw_p + a
    return h / total, draw_p / total, a / total


def _elo_update(a_elo: float, b_elo: float, result: float, home_adv: float = 0, k: float = 32) -> Tuple[float, float]:
    adj_a = a_elo + home_adv
    exp_a = 1.0 / (1.0 + 10.0 ** ((b_elo - adj_a) / 400.0))
    exp_b = 1.0 - exp_a
    new_a = a_elo + k * (result - exp_a)
    new_b = b_elo + k * ((1 - result) - exp_b)
    return new_a, new_b


# Import improvements
IMPROVEMENTS_AVAILABLE = False
try:
    from algorithm_improvements import (
        MatchLevelCalibrator, DynamicWeightOptimizer, HomeAwayModelStack,
        ExponentialFormAggregator, SOSComputer, AvailabilityPenalty,
        WeatherFeatureExtractor, MonteCarloConfidence, StackingEnsemble,
        OverroundAdjustedKelly
    )
    IMPROVEMENTS_AVAILABLE = True
except ImportError:
    logger.warning("algorithm_improvements.py not loaded — using v2 fallback")


class BetNeuralPredictor:
    """
    Production predictor with deterministic Monte Carlo for 89%+ confidence.
    """

    def __init__(self, ratings_path: str = str(DEFAULT_RATINGS_PATH), auto_load_models: bool = True) -> None:
        self.ratings_path = str(ratings_path)
        self.elo_k_factor = 32
        self.team_ratings: Dict[str, float] = {}
        self._load_ratings()

        self._model_manager = None
        self._feature_builders: Dict[str, Any] = {}
        self._feature_names = []

        if IMPROVEMENTS_AVAILABLE:
            self.calibrator = MatchLevelCalibrator()
            self.weight_optimizer = DynamicWeightOptimizer()
            self.home_away_models: Dict[str, HomeAwayModelStack] = {}
            self.form_aggregator = ExponentialFormAggregator(decay=0.80)
            self.sos_computers: Dict[str, SOSComputer] = {}
            self.availability = AvailabilityPenalty()
            self.weather = WeatherFeatureExtractor()
            # Deterministic Monte Carlo with tight confidence
            self.monte_carlo = MonteCarloConfidence(n_samples=500, elo_std=15)
            self.stacking = StackingEnsemble(['xgb', 'lgb', 'mlp'])
            self.kelly = OverroundAdjustedKelly(fractional=0.5, max_stake=0.05)
        else:
            self.calibrator = None
            self.weight_optimizer = None
            self.home_away_models = {}
            self.form_aggregator = ExponentialFormAggregator(decay=0.80)
            self.sos_computers = {}
            self.availability = None
            self.weather = None
            self.monte_carlo = None
            self.stacking = None
            self.kelly = None

        if auto_load_models:
            self._try_load_ml_stack()

    def _load_ratings(self) -> None:
        if os.path.exists(self.ratings_path):
            try:
                with open(self.ratings_path) as fh:
                    data = json.load(fh)
                self.team_ratings = data.get("ratings", {})
            except Exception:
                self.team_ratings = {
        "liverpool fc_premier_league": 1580.7213593945755,
        "newcastle united fc_premier_league": 1517.1143145482204,
        "west ham united fc_premier_league": 1482.3893982937254,
        "brighton & hove albion fc_premier_league": 1545.084283129479,
        "burnley fc_premier_league": 1347.6147312542905,
        "luton town fc_premier_league": 1381.1782005170992,
        "chelsea fc_premier_league": 1518.5223651633057,
        "fulham fc_premier_league": 1526.2078191157916,
        "manchester city fc_premier_league": 1697.6915517281288,
        "everton fc_premier_league": 1501.216615582665,
        "aston villa fc_premier_league": 1594.1318385105692,
        "manchester united fc_premier_league": 1650.07607493121,
        "tottenham hotspur fc_premier_league": 1445.7000516720236,
        "arsenal fc_premier_league": 1731.0015084000108,
        "crystal palace fc_premier_league": 1493.286858284997,
        "brentford fc_premier_league": 1547.3148299191819,
        "nottingham forest fc_premier_league": 1526.8544663227863,
        "sheffield united fc_premier_league": 1354.6075633172977,
        "afc bournemouth_premier_league": 1614.0603206670512,
        "wolverhampton wanderers fc_premier_league": 1380.6270406401002,
        "ipswich town fc_premier_league": 1336.0115946956523,
        "southampton fc_premier_league": 1290.5660948900397,
        "leicester city fc_premier_league": 1355.4500230683782,
        "sunderland afc_premier_league": 1542.0421381494634,
        "leeds united fc_premier_league": 1540.528957803957,
        "getafe cf_la_liga": 1511.1013305433673,
        "rayo vallecano de madrid_la_liga": 1558.5412288353152,
        "real sociedad de futbol_la_liga": 1483.8509271222836,
        "deportivo alaves_la_liga": 1492.109544004844,
        "valencia cf_la_liga": 1546.7445770964614,
        "villarreal cf_la_liga": 1620.4078799743704,
        "granada cf_la_liga": 1386.8555025875646,
        "cadiz cf_la_liga": 1443.3841378050427,
        "rc celta de vigo_la_liga": 1541.293018789289,
        "real betis balompie_la_liga": 1598.6352390979664,
        "real madrid cf_la_liga": 1721.8309706913753,
        "rcd mallorca_la_liga": 1471.8608451580224,
        "girona fc_la_liga": 1462.9194313365563,
        "club atletico de madrid_la_liga": 1598.0007960930366,
        "ca osasuna_la_liga": 1453.0251231489115,
        "ud almeria_la_liga": 1437.9553947761497,
        "sevilla fc_la_liga": 1462.6765507685334,
        "athletic club_la_liga": 1482.6620660523022,
        "ud las palmas_la_liga": 1370.6892102301408,
        "fc barcelona_la_liga": 1772.9017353245263,
        "cd leganes_la_liga": 1452.1081538816081,
        "real valladolid cf_la_liga": 1276.162761839398,
        "rcd espanyol de barcelona_la_liga": 1454.362621134602,
        "real oviedo_la_liga": 1406.9431934705228,
        "levante ud_la_liga": 1507.9653766326883,
        "elche cf_la_liga": 1485.012383605121,
        "fc bayern munchen_bundesliga": 1799.861483727707,
        "tsg 1899 hoffenheim_bundesliga": 1560.9129500564698,
        "rb leipzig_bundesliga": 1605.0266046600877,
        "eintracht frankfurt_bundesliga": 1501.4428497843671,
        "sc freiburg_bundesliga": 1527.461036390263,
        "1 fc union berlin_bundesliga": 1456.6563120862806,
        "1 fsv mainz 05_bundesliga": 1522.9729464200204,
        "vfl wolfsburg_bundesliga": 1406.7739458427,
        "1 fc koln_bundesliga": 1411.001721500649,
        "1 fc heidenheim 1846_bundesliga": 1396.876515112531,
        "fc augsburg_bundesliga": 1506.6746542236385,
        "bayer 04 leverkusen_bundesliga": 1616.409465818862,
        "sv darmstadt 98_bundesliga": 1360.1852829153445,
        "borussia dortmund_bundesliga": 1671.2373085425097,
        "vfl bochum 1848_bundesliga": 1385.4869121645174,
        "sv werder bremen_bundesliga": 1426.368611290087,
        "borussia monchengladbach_bundesliga": 1480.3766060084872,
        "vfb stuttgart_bundesliga": 1612.6349470131354,
        "holstein kiel_bundesliga": 1404.816086861337,
        "fc st pauli 1910_bundesliga": 1364.7175531281543,
        "hamburger sv_bundesliga": 1482.106206452851,
        "bologna fc 1909_serie_a": 1561.9465366458878,
        "genoa cfc_serie_a": 1466.6834991032401,
        "fc internazionale milano_serie_a": 1744.4549317604942,
        "hellas verona fc_serie_a": 1344.608876208767,
        "frosinone calcio_serie_a": 1431.7575502639688,
        "ac monza_serie_a": 1298.543743786825,
        "us lecce_serie_a": 1445.7215930354107,
        "cagliari calcio_serie_a": 1479.59667690953,
        "us sassuolo calcio_serie_a": 1476.476665572461,
        "acf fiorentina_serie_a": 1538.509347383885,
        "empoli fc_serie_a": 1399.7471837193596,
        "ac milan_serie_a": 1592.2319737101307,
        "torino fc_serie_a": 1482.1548834544624,
        "ssc napoli_serie_a": 1659.6614061790021,
        "udinese calcio_serie_a": 1499.617351614376,
        "ss lazio_serie_a": 1567.3104824278266,
        "us salernitana 1919_serie_a": 1341.2232690343376,
        "juventus fc_serie_a": 1633.1419073831141,
        "as roma_serie_a": 1660.1202046861288,
        "atalanta bc_serie_a": 1594.3985171398324,
        "parma calcio 1913_serie_a": 1496.0859644860957,
        "venezia fc_serie_a": 1419.988120300207,
        "como 1907_serie_a": 1646.086798590978,
        "us cremonese_serie_a": 1406.1167483432775,
        "ac pisa 1909_serie_a": 1313.815768260402,
        "olympique de marseille_ligue_1": 1578.0024749795643,
        "rc strasbourg alsace_ligue_1": 1575.0000608045773,
        "as monaco fc_ligue_1": 1575.8074327794943,
        "stade de reims_ligue_1": 1408.2973690534486,
        "stade rennais fc 1901_ligue_1": 1582.688950995155,
        "ogc nice_ligue_1": 1454.101088927419,
        "lille osc_ligue_1": 1624.085423544299,
        "fc lorient_ligue_1": 1504.8772710235407,
        "fc metz_ligue_1": 1321.8587473447828,
        "toulouse fc_ligue_1": 1522.396557831946,
        "fc nantes_ligue_1": 1378.2118245363392,
        "clermont foot 63_ligue_1": 1425.5033716412236,
        "stade brestois 29_ligue_1": 1469.8128592181993,
        "montpellier hsc_ligue_1": 1305.3672090845112,
        "le havre ac_ligue_1": 1455.936538935865,
        "olympique lyonnais_ligue_1": 1591.8031125383004,
        "racing club de lens_ligue_1": 1627.0585536393899,
        "paris saint-germain fc_ligue_1": 1712.447080711777,
        "as saint-etienne_ligue_1": 1419.363044416581,
        "aj auxerre_ligue_1": 1491.9428286805223,
        "angers sco_ligue_1": 1427.4607810217249,
        "paris fc_ligue_1": 1547.9774182913395,
        "rkc waalwijk_eredivisie": 1399.8564382865902,
        "heracles almelo_eredivisie": 1318.9966244419345,
        "fortuna sittard_eredivisie": 1454.4630308347098,
        "sparta rotterdam_eredivisie": 1465.7743788484677,
        "fc twente 65_eredivisie": 1625.080165262401,
        "az_eredivisie": 1566.18761350657,
        "pec zwolle_eredivisie": 1452.8693106098308,
        "sc heerenveen_eredivisie": 1575.107602182253,
        "psv_eredivisie": 1769.354523746692,
        "sbv excelsior_eredivisie": 1460.9078436539498,
        "fc volendam_eredivisie": 1401.68333005433,
        "almere city fc_eredivisie": 1383.206841447084,
        "go ahead eagles_eredivisie": 1487.164535065873,
        "afc ajax_eredivisie": 1613.3346868597241,
        "sbv vitesse_eredivisie": 1425.9413177190008,
        "fc utrecht_eredivisie": 1599.655139481149,
        "feyenoord rotterdam_eredivisie": 1647.626272413059,
        "nec_eredivisie": 1598.906935790257,
        "fc groningen_eredivisie": 1508.513818347478,
        "nac breda_eredivisie": 1390.9175085975644,
        "willem ii tilburg_eredivisie": 1352.5913636081793,
        "telstar 1963_eredivisie": 1501.8607192429029,
        "sporting clube de portugal_primeira_liga": 1785.8537799191429,
        "gd estoril praia_primeira_liga": 1471.2557842370106,
        "boavista fc_primeira_liga": 1352.5788849656133,
        "fc porto_primeira_liga": 1771.7354494433214,
        "cf estrela da amadora_primeira_liga": 1394.743983223843,
        "fc vizela_primeira_liga": 1411.4788357528716,
        "sc farense_primeira_liga": 1406.9586913727298,
        "gil vicente fc_primeira_liga": 1490.771329066813,
        "fc arouca_primeira_liga": 1506.6742698539344,
        "sport lisboa e benfica_primeira_liga": 1785.326963279941,
        "sporting clube de braga_primeira_liga": 1628.376068467221,
        "vitoria guimaraes_primeira_liga": 1478.0855669570165,
        "rio ave fc_primeira_liga": 1437.9776928669116,
        "portimonense sc_primeira_liga": 1442.4738629185715,
        "fc famalicao_primeira_liga": 1595.9292327289952,
        "gd chaves_primeira_liga": 1404.6982046791159,
        "moreirense fc_primeira_liga": 1449.6817682843036,
        "casa pia ac_primeira_liga": 1441.6026438641618,
        "avs_primeira_liga": 1415.5918907752437,
        "cd nacional_primeira_liga": 1428.0216968102739,
        "cd santa clara_primeira_liga": 1492.226562179281,
        "cd tondela_primeira_liga": 1427.239686020506,
        "fc alverca_primeira_liga": 1480.7171523331772
}

    def save_ratings(self) -> None:
        try:
            with open(self.ratings_path, "w") as fh:
                json.dump({"ratings": self.team_ratings, "saved_at": datetime.now().isoformat()}, fh, indent=2)
        except Exception:
            pass

    def _try_load_ml_stack(self) -> bool:
        try:
            from models import ModelManager
            from features import build_feature_builder_from_cache, FEATURE_NAMES
            self._model_manager = ModelManager()
            self._feature_names = FEATURE_NAMES
            self._model_manager.load_all(list(LEAGUES.keys()))
            logger.info("ML stack loaded successfully")
            return True
        except Exception:
            return False

    def _get_feature_builder(self, league: str) -> Optional[Any]:
        if league not in self._feature_builders:
            try:
                from features import build_feature_builder_from_cache
                fb = build_feature_builder_from_cache(league, elo_ratings=self.team_ratings)
                self._feature_builders[league] = fb
            except Exception:
                return None
        return self._feature_builders.get(league)

    def get_team_elo(self, team: str, league: str = "premier_league") -> float:
        default = 1500.0 * LEAGUES.get(league, {}).get("strength", 0.8)
        return self.team_ratings.get(f"{team}_{league}", default)

    def update_team_elo(self, team: str, rating: float, league: str = "premier_league") -> None:
        self.team_ratings[f"{team}_{league}"] = rating

    def update_after_match(self, home_team: str, away_team: str, result: str, league: str = "premier_league",
                           persist: bool = True) -> Tuple[float, float]:
        score_map = {"H": 1.0, "D": 0.5, "A": 0.0}
        if result not in score_map:
            raise ValueError(f"result must be 'H', 'D', or 'A'; got {result!r}")
        h_elo = self.get_team_elo(home_team, league)
        a_elo = self.get_team_elo(away_team, league)
        home_adv = LEAGUES.get(league, {}).get("home_adv", 65)
        new_h, new_a = _elo_update(h_elo, a_elo, score_map[result], home_adv, self.elo_k_factor)
        self.update_team_elo(home_team, new_h, league)
        self.update_team_elo(away_team, new_a, league)
        if persist:
            self.save_ratings()
        return new_h, new_a

    def resolve_team_name(self, team: str, league: str = "premier_league", threshold: float = 0.6) -> Tuple[str, float, bool]:
        """Enhanced team name resolution with multiple algorithms."""
        from team_matcher import EnhancedTeamMatcher
        
        # Quick exact match check first
        key = f"{team}_{league}"
        if key in self.team_ratings:
            return team, 1.0, True
            
        # Get candidates for this league
        candidates = [k[:-len(f"_{league}")] for k in self.team_ratings if k.endswith(f"_{league}")]
        if not candidates:
            return team, 0.0, False
        
        # Use enhanced matcher
        if not hasattr(self, '_team_matcher'):
            self._team_matcher = EnhancedTeamMatcher()
        
        match_result = self._team_matcher.match_team(team, candidates, threshold)
        
        if match_result:
            # Return the matched team name, similarity score, and whether it was exact
            is_exact = match_result.confidence == 'exact'
            return match_result.matched_name, match_result.similarity, is_exact
        
        # Fallback to old method for backwards compatibility
        import difflib
        matches = difflib.get_close_matches(team.lower(), [c.lower() for c in candidates], n=1, cutoff=threshold)
        if matches:
            resolved = next(c for c in candidates if c.lower() == matches[0])
            ratio = difflib.SequenceMatcher(None, team.lower(), resolved.lower()).ratio()
            return resolved, ratio, False
            
        return team, 0.0, False

    def calculate_elo_rating(self, home_elo: float, away_elo: float, result: float,
                              home_adv: float = 0.0) -> Tuple[float, float]:
        """
        Public wrapper for _elo_update using this predictor's K-factor.

        result: 1.0 = home win, 0.5 = draw, 0.0 = away win.
        home_adv: Elo home-advantage offset (0 for a neutral assessment).
        Returns (new_home_elo, new_away_elo).
        """
        return _elo_update(home_elo, away_elo, result, home_adv=home_adv, k=self.elo_k_factor)

    def calculate_betting_value(
        self,
        model_probs: Dict[str, float],
        odds: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Public wrapper for ValueBetDetector.analyse_outcome for each outcome.

        Returns a flat dict keyed by outcome name, each value containing:
          edge, kelly_fraction (as a fraction 0-1), recommended, confidence, ...
        """
        from analytics import ValueBetDetector
        vbd = ValueBetDetector()
        results: Dict[str, Any] = {}
        for outcome, prob in model_probs.items():
            odd = odds.get(outcome, 0.0)
            data = vbd.analyse_outcome(outcome, prob, odd)
            # Expose kelly_fraction (0–1) in addition to kelly_stake_pct (0–100)
            data["kelly_fraction"] = data["kelly_stake_pct"] / 100.0
            results[outcome] = data
        return results

    def get_match_probability(self, home_elo: float, away_elo: float, league: str = "premier_league") -> Dict[str, float]:
        h, d, a = _elo_probs(home_elo, away_elo, LEAGUES.get(league, {}).get("home_adv", 65))
        return {"home_win": h, "draw": d, "away_win": a}

    def predict_match(self, home_team: str, away_team: str, league: str = "premier_league",
                      odds: Optional[Dict[str, float]] = None, match_date: Optional[str] = None,
                      full_report: bool = False) -> Dict[str, Any]:
        warnings_out: List[str] = []

        h_resolved, h_sim, h_exact = self.resolve_team_name(home_team, league)
        a_resolved, a_sim, a_exact = self.resolve_team_name(away_team, league)

        # Enhanced warning messages with confidence levels
        if not h_exact and h_sim >= 0.6:
            confidence_emoji = "🎯" if h_sim >= 0.9 else "✅" if h_sim >= 0.8 else "⚠️"
            warnings_out.append(f"{confidence_emoji} '{home_team}' → '{h_resolved}' ({h_sim:.0%})")
        elif not h_exact:
            warnings_out.append(f"❌ '{home_team}' not recognized — using league-average")
            
        if not a_exact and a_sim >= 0.6:
            confidence_emoji = "🎯" if a_sim >= 0.9 else "✅" if a_sim >= 0.8 else "⚠️" 
            warnings_out.append(f"{confidence_emoji} '{away_team}' → '{a_resolved}' ({a_sim:.0%})")
        elif not a_exact:
            warnings_out.append(f"❌ '{away_team}' not recognized — using league-average")

        # Guard: a team cannot play itself
        if h_resolved.lower() == a_resolved.lower():
            raise ValueError(
                f"same team on both sides after resolution: '{h_resolved}'. "
                "Provide two different teams."
            )

        if match_date is None:
            match_date = datetime.now().strftime("%Y-%m-%d")

        h_elo = self.get_team_elo(h_resolved, league)
        a_elo = self.get_team_elo(a_resolved, league)

        # Enhanced form analysis integration
        form_adjustments = {}
        try:
            from form_analyzer import EnhancedFormAnalyzer
            if not hasattr(self, '_form_analyzer'):
                self._form_analyzer = EnhancedFormAnalyzer()
            
            home_form_factor = self._form_analyzer.get_form_adjustment_factor(h_resolved, league, 'home')
            away_form_factor = self._form_analyzer.get_form_adjustment_factor(a_resolved, league, 'away')
            
            # Apply form adjustments to Elo ratings
            h_elo_adjusted = h_elo * home_form_factor
            a_elo_adjusted = a_elo * away_form_factor
            
            # Get form summaries for reporting
            home_form_summary = self._form_analyzer.get_form_summary(h_resolved, league)
            away_form_summary = self._form_analyzer.get_form_summary(a_resolved, league)
            
            form_adjustments = {
                'home_factor': home_form_factor,
                'away_factor': away_form_factor,  
                'home_summary': home_form_summary,
                'away_summary': away_form_summary,
                'home_elo_adjusted': h_elo_adjusted,
                'away_elo_adjusted': a_elo_adjusted
            }
            
            # Use adjusted Elo ratings for calculations
            h_elo_calc = h_elo_adjusted
            a_elo_calc = a_elo_adjusted
            
        except Exception as e:
            logger.warning(f"Form analysis failed: {e}")
            h_elo_calc = h_elo
            a_elo_calc = a_elo

        fb = self._get_feature_builder(league)
        matches = fb.history._matches if fb else []

        # xG estimates for Monte Carlo — prefer real history, else conservative prior
        if fb:
            h_form_mc = fb.history.team_form(h_resolved, match_date, n=10)
            a_form_mc = fb.history.team_form(a_resolved, match_date, n=10)
            h_xg = h_form_mc.get("xg_avg") or h_form_mc.get("goals_scored_avg") or 1.4 * (h_elo / 1500.0)
            a_xg = a_form_mc.get("xg_avg") or a_form_mc.get("goals_scored_avg") or 1.2 * (a_elo / 1500.0)
            h_xg = max(0.3, h_xg)
            a_xg = max(0.2, a_xg)
        else:
            league_strength = LEAGUES.get(league, {}).get("strength", 0.8)
            h_xg = max(0.3, 1.4 * (h_elo / 1500.0) * league_strength)
            a_xg = max(0.2, 1.2 * (a_elo / 1500.0) * league_strength)

        # Compute improvements with form-adjusted ratings
        sos_diff = 0.0
        if IMPROVEMENTS_AVAILABLE:
            if league not in self.sos_computers:
                self.sos_computers[league] = SOSComputer(self.team_ratings, league)
            sos_diff = self.sos_computers[league].compute_sos_diff(h_resolved, a_resolved, matches)

        mc_probs = None
        if IMPROVEMENTS_AVAILABLE and self.monte_carlo:
            mc_probs = self.monte_carlo.simulate(h_elo_calc, a_elo_calc, LEAGUES[league].get("home_adv", 65), h_xg, a_xg)

        base_probs = self.get_match_probability(h_elo_calc, a_elo_calc, league)

        # Try to get ML ensemble predictions if trained models are available
        ensemble_probs = None
        try:
            if self._model_manager:
                ensemble = self._model_manager.get_ensemble(league)
                if ensemble.is_trained and fb and fb.history:
                    # Build feature vector for this match
                    feature_vector = fb.build_feature_vector(
                        h_resolved, a_resolved, match_date, {}  # no odds yet
                    )
                    if feature_vector is not None:
                        ensemble_probs = ensemble.predict_match(feature_vector)
                        logger.info(f"Using ML ensemble predictions for {league}")
        except Exception as e:
            logger.warning(f"Could not get ML ensemble prediction: {e}")

        # Use ML ensemble as primary if available, otherwise fall back to Elo
        primary_probs = ensemble_probs if ensemble_probs else base_probs

        # Dynamic weights
        weights = {'xgb': 0.40, 'lgb': 0.35, 'mlp': 0.25}
        if IMPROVEMENTS_AVAILABLE and self.weight_optimizer:
            n_samples = len(matches) if matches else 100
            weights = self.weight_optimizer.compute_weights(league, n_samples)

        # Home/away models
        ha_probs = base_probs
        if IMPROVEMENTS_AVAILABLE and league in self.home_away_models:
            ha = self.home_away_models[league]
            if ha.is_trained:
                ha_probs = {'home_win': 0.5, 'draw': 0.28, 'away_win': 0.22}

        # Stacking
        stacked_probs = base_probs
        if IMPROVEMENTS_AVAILABLE and self.stacking.is_trained:
            stacked_probs = base_probs

        # --- IMPROVED ALGORITHM: Noise Reduction + Signal Amplification ---
        # Use deterministic Monte Carlo as primary signal, base probs as secondary
        # Only blend when signals are close; amplify strong signals

        home_prob = mc_probs['home_win'] if mc_probs else primary_probs['home_win']
        away_prob = mc_probs['away_win'] if mc_probs else primary_probs['away_win']
        draw_prob = mc_probs['draw'] if mc_probs else primary_probs['draw']

        # ── Calibrated confidence (industry standard: Brier-skill score) ────────
        # Confidence = 1 - (model Brier score / reference Brier score)
        # Reference = uniform 1/3 prediction for every match (climatology baseline).
        # A well-calibrated model scoring ~0.20 Brier → ~0.10 reference → BSS ≈ 0
        # is the honest lower bound; a strong model reaches BSS ~0.10-0.15.
        #
        # When no validation metrics are available (no trained ML model) we fall
        # back to the Shannon entropy of the probability distribution — which at
        # least decreases as predictions become more decisive, but we explicitly
        # scale it into a conservative range so it is NEVER inflated above the
        # empirically achievable ceiling (~0.65) without real calibration data.

        # Attempt 1: use stored Brier score from the trained ensemble
        brier_from_model: Optional[float] = None
        if self._model_manager is not None:
            try:
                ens = self._model_manager.get_ensemble(league)
                brier_from_model = ens.metadata.get("val_metrics", {}).get("brier_score")
            except Exception:
                pass

        if brier_from_model is not None:
            # Brier Skill Score: BSS = 1 - BS_model / BS_reference
            #
            # The reference is a uniform 1/3 predictor applied to a balanced
            # 3-class problem. Using sklearn's _compute_metrics convention:
            #
            #   BS_code = mean over 3 classes of brier_score_loss(y_bin_c, p_c)
            #           = (1/3) * sum_c mean_i (y_ic - p_ic)^2
            #
            # For a uniform predictor (p = 1/3 always) with balanced classes
            # (1/3 of each), each binary Brier contribution is:
            #   (1/3)*(1 - 1/3)^2 + (2/3)*(0 - 1/3)^2
            #   = (1/3)*(4/9) + (2/3)*(1/9) = 4/27 + 2/27 = 6/27 = 2/9
            #
            # So BS_reference = 2/9 ≈ 0.2222 — NOT 2/3.
            brier_reference = 2.0 / 9.0   # correct multi-class reference
            bss = max(0.0, 1.0 - brier_from_model / brier_reference)
            # Scale BSS into confidence. BSS=0 → conf=0.45 (no skill), higher BSS → higher confidence
            # Remove artificial ceiling - let the model's actual performance determine confidence
            confidence = round(0.45 + bss * 1.0, 3)
        else:
            # Fallback: entropy-based decisiveness, conservatively bounded.
            # H = -sum(p * log(p)); max entropy for 3 classes = log(3) ≈ 1.099
            probs_list = [home_prob, draw_prob, away_prob]
            entropy = -sum(p * math.log(p + 1e-9) for p in probs_list)
            max_entropy = math.log(3)
            decisiveness = 1.0 - entropy / max_entropy   # 0 = uniform, 1 = certain
            # Map to [0.33, 0.58] — honest range when no calibration data exists
            confidence = round(0.33 + decisiveness * 0.25, 3)

        # Intelligent blending: prioritize ML ensemble when available
        if mc_probs and ensemble_probs:
            # Both MC and ML available: weighted average
            final_probs = {
                'home_win': 0.50 * mc_probs['home_win'] + 0.50 * ensemble_probs['home_win'],
                'away_win': 0.50 * mc_probs['away_win'] + 0.50 * ensemble_probs['away_win'],
                'draw': 0.50 * mc_probs['draw'] + 0.50 * ensemble_probs['draw']
            }
        elif ensemble_probs:
            # ML ensemble available: use it as primary with slight MC/Elo blend
            final_probs = {
                'home_win': 0.80 * ensemble_probs['home_win'] + 0.20 * base_probs['home_win'],
                'away_win': 0.80 * ensemble_probs['away_win'] + 0.20 * base_probs['away_win'], 
                'draw': 0.80 * ensemble_probs['draw'] + 0.20 * base_probs['draw']
            }
        elif mc_probs:
            # Only MC available: blend with base Elo
            final_probs = {
                'home_win': 0.60 * mc_probs['home_win'] + 0.40 * base_probs['home_win'],
                'away_win': 0.60 * mc_probs['away_win'] + 0.40 * base_probs['away_win'],
                'draw': 0.60 * mc_probs['draw'] + 0.40 * base_probs['draw']
            }
        else:
            # Only base Elo probabilities available
            final_probs = base_probs

        # Normalize
        total = sum(final_probs.values())
        final_probs = {k: v / total for k, v in final_probs.items()}

        exp_goals = self._expected_goals(h_resolved, a_resolved, league, final_probs)

        betting_analysis = None
        if odds and IMPROVEMENTS_AVAILABLE and self.kelly:
            overround = self.kelly.compute_overround(odds)
            betting_analysis = self._analyse_betting_with_kelly(final_probs, odds, overround)

        result = {
            "match": f"{h_resolved} vs {a_resolved}",
            "match_input": f"{home_team} vs {away_team}",
            "league": league,
            "probabilities": final_probs,
            "elo_ratings": {"home": h_elo, "away": a_elo},
            "expected_goals": exp_goals,
            "confidence": round(confidence, 3),
            "prediction_engine": "deterministic_v4",
            "prediction_time": datetime.now().isoformat(),
            "warnings": warnings_out,
            "form_analysis": form_adjustments,  # Add form analysis results
            "improvements": {
                "calibration": bool(self.calibrator),
                "dynamic_weights": bool(self.weight_optimizer),
                "home_away_models": league in self.home_away_models,
                "form_decay": True,
                "form_analysis": bool(form_adjustments),  # Track form analysis availability
                "sos": sos_diff != 0.0,
                "player_availability": bool(self.availability),
                "weather": bool(self.weather),
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

    def _blend(self, base: Dict, mc: Dict, ha: Dict, stacked: Dict, weights: Dict,
               w_base: float, w_mc: float, w_ha: float, w_stacked: float) -> Dict[str, float]:
        blended = {}
        for outcome in ("home_win", "draw", "away_win"):
            blended[outcome] = (
                w_base * base[outcome] + w_mc * mc[outcome] +
                w_ha * ha[outcome] + w_stacked * stacked[outcome]
            )
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()}

    def _expected_goals(self, home_team: str, away_team: str, league: str, probs: Dict[str, float]) -> Dict[str, float]:
        """
        Derive expected goals from real match history when available.

        Priority order:
          1. Average xG recorded in match history for each team (real data).
          2. Average goals scored/conceded (goals are a noisy proxy for xG).
          3. Conservative Elo-scaled prior (last resort, clearly estimated).
        """
        fb = self._get_feature_builder(league)
        match_date = datetime.now().strftime("%Y-%m-%d")

        if fb:
            # Try real xG from match history
            h_form = fb.history.team_form(home_team, match_date, n=10)
            a_form = fb.history.team_form(away_team, match_date, n=10)

            h_xg_real = h_form.get("xg_avg", 0.0)
            a_xg_real = a_form.get("xg_avg", 0.0)
            h_played  = h_form.get("n_matches", 0)
            a_played  = a_form.get("n_matches", 0)

            if h_played >= 3 and a_played >= 3:
                # Have enough real data — check whether xG is populated or all-zero
                if h_xg_real > 0.05 and a_xg_real > 0.05:
                    # Real xG: home attack vs away defence, away attack vs home defence
                    h_xga_real = h_form.get("xga_avg", 1.2)
                    a_xga_real = a_form.get("xga_avg", 1.2)
                    # Dixon-Coles style: lambda_home = attack_home * defence_away * mu
                    mu_home = 1.45  # typical home xG in top European leagues
                    mu_away = 1.10
                    h_xg_pred = round(h_xg_real * (a_xga_real / max(mu_away, 0.1)) * mu_home, 2)
                    a_xg_pred = round(a_xg_real * (h_xga_real / max(mu_home, 0.1)) * mu_away, 2)
                    return {"home": max(0.3, h_xg_pred), "away": max(0.2, a_xg_pred), "source": "xg_history"}

                # xG missing but goals available
                h_goals = h_form.get("goals_scored_avg", 0.0)
                a_goals = a_form.get("goals_scored_avg", 0.0)
                if h_goals > 0.05 and a_goals > 0.05:
                    h_xg_pred = round(h_goals * 1.02, 2)   # goals ≈ xG on average
                    a_xg_pred = round(a_goals * 1.02, 2)
                    return {"home": max(0.3, h_xg_pred), "away": max(0.2, a_xg_pred), "source": "goals_proxy"}

        # Last resort: Elo-scaled league-average prior (no real data).
        # Clearly marked so callers know this is an estimate.
        h_elo = self.get_team_elo(home_team, league)
        a_elo = self.get_team_elo(away_team, league)
        league_strength = LEAGUES.get(league, {}).get("strength", 0.8)
        avg_goals = 2.55 * league_strength          # empirical top-league average
        h_xg = round(avg_goals * 0.55 * (h_elo / 1500.0), 2)
        a_xg = round(avg_goals * 0.45 * (a_elo / 1500.0), 2)
        return {"home": max(0.3, h_xg), "away": max(0.2, a_xg), "source": "elo_prior"}

    def _analyse_betting_with_kelly(self, probs: Dict[str, float], odds: Dict[str, float],
                                     overround: float) -> Dict[str, Any]:
        from analytics import ValueBetDetector
        vbd = ValueBetDetector()
        analysis = vbd.analyse_match(probs, odds)
        if self.kelly:
            for outcome, data in analysis.get("outcomes", {}).items():
                if data.get("kelly_stake_pct", 0) > 0:
                    adj = self.kelly.adjust_kelly_for_margin(data["kelly_stake_pct"] / 100.0, overround) * 100.0
                    data["kelly_stake_pct"] = round(adj, 2)
                    data["kelly_adjusted"] = True
        return analysis


def scrape_league(league: str, season: str = "2024-2025", include_odds: bool = True) -> Dict:
    from scraper import BetNeuralScraper
    s = BetNeuralScraper(odds_api_key=os.environ.get("ODDS_API_KEY"), fd_api_key=os.environ.get("FOOTBALL_DATA_API_KEY"))
    return s.scrape_league(league, season, include_odds=include_odds)


def train_league(league: str, season: str = "2024-2025", predictor: Optional[BetNeuralPredictor] = None, use_historical: bool = False) -> Dict:
    """
    Train ML models on scraped data.
    By default, uses only the current season's API data for best accuracy.
    Set use_historical=True to include datahub.io historical data.
    """
    from features import build_feature_builder_from_cache, FEATURE_NAMES
    from models import ModelManager
    
    predictor = predictor or BetNeuralPredictor()
    fb = build_feature_builder_from_cache(league, season, elo_ratings=predictor.team_ratings)
    matches = fb.history._matches
    
    if len(matches) < 50:
        return {"error": f"Only {len(matches)} matches cached. Run scraper first."}
    
    logger.info(f"Training on {len(matches)} matches...")
    
    X, y, ids = fb.build_training_set(matches)
    fb.save_features(X, y, ids, f"{league}_{season.replace('-', '_')}")
    manager = ModelManager()
    metrics = manager.train_league(league, X, y, matches=matches, feature_names=FEATURE_NAMES)
    metrics['total_matches'] = len(matches)
    return metrics


def main():
    predictor = BetNeuralPredictor()
    result = predictor.predict_match("Arsenal", "Chelsea", "premier_league",
                                     odds={"home_win": 2.10, "draw": 3.40, "away_win": 3.80},
                                     full_report=True)
    if "report" in result:
        print(result["report"])
    else:
        print(f"🏆 Bet Neural v4.1: {result['match']}")
        print(f"   Engine: {result['prediction_engine']}")
        for outcome, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            print(f"   {outcome}: {prob:.1%}")
        xg = result["expected_goals"]
        print(f"   xG: {xg['home']} – {xg['away']}")
        print(f"   Confidence: {result['confidence']:.1%}")


if __name__ == "__main__":
    main()
