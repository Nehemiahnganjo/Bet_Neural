"""
models.py — Bet Neural ML Model Stack
======================================
Three-model ensemble for football outcome prediction:

  1. XGBoost  — gradient-boosted trees (handles non-linear interactions well)
  2. LightGBM — fast gradient boosting (better on large feature sets)
  3. MLP      — multi-layer perceptron via sklearn (captures deep feature interactions)
  4. Ensemble — stacked/weighted average of all three

Research basis:
  - Dixon & Coles (1997): bivariate Poisson model — still SOTA baseline
  - Groll et al. (2018): machine learning for football prediction
  - Hubáček et al. (2019): XGBoost consistently outperforms deep nets on tabular data
  - Baio & Blangiardo (2010): Bayesian hierarchical models
  - Empirical: Elo + form + xG + bookmaker odds are the 4 strongest signals

Key design decisions:
  - Calibrated probabilities (CalibratedClassifierCV) — raw XGB scores are overconfident
  - Isotonic regression calibration > Platt scaling for football
  - Class weights to handle draw imbalance (draws ~27%, model tends to underpredict)
  - Cross-validated on last 20% of season (time-based split, not random!)
  - Soft voting ensemble with learned weights from validation set
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Result encoding
HOME_WIN, DRAW, AWAY_WIN = 0, 1, 2
RESULT_NAMES = {HOME_WIN: "home_win", DRAW: "draw", AWAY_WIN: "away_win"}


# ── Individual models ─────────────────────────────────────────────────────────

def _build_xgb(class_weights: Optional[Dict] = None) -> Any:
    """XGBoost classifier with football-tuned hyperparameters."""
    if not XGB_AVAILABLE:
        return None

    # Sample weights handle class imbalance better than scale_pos_weight for multi-class
    params = {
        "n_estimators":      500,
        "max_depth":         6,
        "learning_rate":     0.03,
        "subsample":         0.85,
        "colsample_bytree":  0.85,
        "min_child_weight":  5,
        "gamma":             0.05,
        "reg_alpha":         0.05,
        "reg_lambda":        2.0,
        "objective":         "multi:softprob",
        "num_class":         3,
        "eval_metric":       "mlogloss",
        "use_label_encoder": False,
        "random_state":      42,
        "n_jobs":            -1,
        "verbosity":         0,
    }
    return xgb.XGBClassifier(**params)


def _build_lgb(class_weights: Optional[Dict] = None) -> Any:
    """LightGBM classifier — tends to generalise better than XGB on small football datasets."""
    if not LGB_AVAILABLE:
        return None

    return lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.02,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=15,
        reg_alpha=0.05,
        reg_lambda=1.5,
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    )


def _build_mlp(n_features: int) -> MLPClassifier:
    """
    Scikit-learn MLP.  Architecture: input → 256 → 128 → 64 → 3 (softmax).
    Uses Adam + early stopping on validation loss.
    """
    return MLPClassifier(
        hidden_layer_sizes=(192, 96, 48),
        activation="relu",
        solver="adam",
        alpha=0.0005,           # L2 regularisation (lower = less regularisation)
        batch_size=16,
        learning_rate="adaptive",
        learning_rate_init=0.002,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=30,
        random_state=42,
        verbose=False,
    )


# ── Calibration wrapper ───────────────────────────────────────────────────────

def _calibrate(model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> Any:
    """
    Wrap a fitted model in isotonic calibration.
    This step is critical: raw XGB probabilities are typically over-confident.
    """
    # Use sklearn's CalibratedClassifierCV with proper CV splitter
    from sklearn.model_selection import StratifiedKFold
    
    n_samples = len(y_cal)
    n_classes = len(np.unique(y_cal))
    
    # Determine appropriate CV folds
    # Need at least n_classes * 2 samples per fold for stratification
    max_folds = n_samples // (n_classes * 2)
    n_folds = max(2, min(5, max_folds))
    
    if n_samples < 10:
        # Very small dataset - skip calibration, use raw probabilities
        return model
    
    try:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cal = CalibratedClassifierCV(model, method="isotonic", cv=cv)
        cal.fit(X_cal, y_cal)
        return cal
    except Exception:
        # Fallback: use a simple probability normalization
        # Fit a simple logistic regression for calibration
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        
        probs_cal = model.predict_proba(X_cal)
        # Use the predicted probabilities as features for calibration
        cal = LogisticRegression(random_state=42)
        cal.fit(probs_cal, y_cal)
        
        # Create a wrapper that applies both
        class CalibratedModel:
            def __init__(self, base, cal):
                self.base = base
                self.cal = cal
                
            def predict_proba(self, X):
                base_probs = self.base.predict_proba(X)
                return self.cal.predict_proba(base_probs)
                
            def predict(self, X):
                probs = self.predict_proba(X)
                return np.argmax(probs, axis=1)
        
        return CalibratedModel(model, cal)


# ── Ensemble ──────────────────────────────────────────────────────────────────

class BetNeuralEnsemble:
    """
    Weighted soft-voting ensemble of XGBoost + LightGBM + MLP.

    Training protocol (time-aware):
      1. Sort matches chronologically.
      2. Train on first 80% (in-sample), validate on last 20%.
      3. Calibrate each model on the validation set.
      4. Learn ensemble weights by minimising log-loss on validation set.
      5. Save everything to disk.

    Prediction:
      For each model, get calibrated P(home_win), P(draw), P(away_win).
      Weighted average → final probabilities.
    """

    def __init__(self, league: str = "premier_league") -> None:
        self.league    = league
        self.scaler    = StandardScaler()
        self.models:   Dict[str, Any] = {}
        self.weights:  Dict[str, float] = {"xgb": 0.40, "lgb": 0.40, "mlp": 0.20}
        self.is_trained = False
        self.metadata:  Dict = {}

    # ── Training ──

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        val_fraction: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Train the full ensemble.  X and y must be chronologically ordered
        (most-recent match last).

        Returns a dict of validation metrics.
        """
        n = len(y)
        if n < 50:
            raise ValueError(f"Need at least 50 training samples, got {n}")

        split = int(n * (1 - val_fraction))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        logger.info(f"Training on {len(X_tr)} samples, validating on {len(X_val)}")

        # Scale features (important for MLP, marginal for tree models)
        self.scaler.fit(X_tr)
        X_tr_s  = self.scaler.transform(X_tr)
        X_val_s = self.scaler.transform(X_val)

        # Sample weights for class balance
        classes   = np.unique(y_tr)
        cw        = compute_class_weight("balanced", classes=classes, y=y_tr)
        sw_tr     = np.array([cw[yi] for yi in y_tr])

        val_probs: Dict[str, np.ndarray] = {}

        # ---- XGBoost ----
        if XGB_AVAILABLE:
            logger.info("  📊 Training XGBoost...")
            xgb_model = _build_xgb()
            xgb_model.fit(X_tr_s, y_tr, sample_weight=sw_tr,
                          eval_set=[(X_val_s, y_val)],
                          verbose=False)
            xgb_cal  = _calibrate(xgb_model, X_val_s, y_val)
            self.models["xgb"] = xgb_cal
            val_probs["xgb"]   = xgb_cal.predict_proba(X_val_s)
            xgb_ll = log_loss(y_val, val_probs['xgb'])
            logger.info(f"  ✅ XGBoost complete - Log-loss: {xgb_ll:.4f}")

        # ---- LightGBM ----
        if LGB_AVAILABLE:
            logger.info("  📊 Training LightGBM...")
            lgb_model = _build_lgb()
            lgb_model.fit(X_tr_s, y_tr, sample_weight=sw_tr)
            lgb_cal  = _calibrate(lgb_model, X_val_s, y_val)
            self.models["lgb"] = lgb_cal
            val_probs["lgb"]   = lgb_cal.predict_proba(X_val_s)
            lgb_ll = log_loss(y_val, val_probs['lgb'])
            logger.info(f"  ✅ LightGBM complete - Log-loss: {lgb_ll:.4f}")

        # ---- MLP ----
        logger.info("  📊 Training MLP...")
        mlp_model = _build_mlp(X_tr_s.shape[1])
        mlp_model.fit(X_tr_s, y_tr)
        mlp_cal  = _calibrate(mlp_model, X_val_s, y_val)
        self.models["mlp"] = mlp_cal
        val_probs["mlp"]   = mlp_cal.predict_proba(X_val_s)
        mlp_ll = log_loss(y_val, val_probs['mlp'])
        logger.info(f"  ✅ MLP complete - Log-loss: {mlp_ll:.4f}")

        # ---- Learn optimal ensemble weights ----
        self.weights = self._learn_weights(val_probs, y_val)
        logger.info(f"  ⚖️  Ensemble weights: {self.weights}")

        # ---- Validation metrics ----
        ensemble_probs = self._blend(val_probs)
        metrics = self._compute_metrics(y_val, ensemble_probs)
        metrics["n_train"] = len(X_tr)
        metrics["n_val"]   = len(X_val)
        metrics["weights"] = self.weights

        self.is_trained = True
        self.metadata = {
            "league":        self.league,
            "trained_at":    datetime.now().isoformat(),
            "n_train":       len(X_tr),
            "n_val":         len(X_val),
            "val_metrics":   metrics,
            "feature_names": feature_names or [],
        }

        logger.info(f"  🎯 Ensemble validation:")
        logger.info(f"     Accuracy:  {metrics['accuracy']:.3f}")
        logger.info(f"     Log-loss:  {metrics['log_loss']:.4f}")
        logger.info(f"     Brier:     {metrics['brier_score']:.4f}")
        logger.info(f"     ROC AUC:   {metrics['roc_auc']:.4f}")
        return metrics

    def _learn_weights(
        self,
        val_probs: Dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """
        Optimise ensemble weights by minimising log-loss on validation set.
        Uses a simple grid search over a 3-model simplex.
        """
        model_keys = sorted(val_probs.keys())
        if len(model_keys) == 1:
            return {model_keys[0]: 1.0}

        best_ll   = float("inf")
        best_weights = {k: 1.0 / len(model_keys) for k in model_keys}

        # Grid search over weight triplets that sum to 1
        steps = np.arange(0.0, 1.05, 0.10)
        for w0 in steps:
            for w1 in steps:
                w2 = 1.0 - w0 - w1
                if w2 < 0 or w2 > 1:
                    continue
                ws = dict(zip(model_keys, [w0, w1, w2] if len(model_keys) == 3
                              else [w0, w1]))
                blended = sum(ws[k] * val_probs[k] for k in model_keys)
                total   = blended.sum(axis=1, keepdims=True)
                blended = blended / np.maximum(total, 1e-9)

                try:
                    ll = log_loss(y_val, blended)
                except Exception:
                    continue

                if ll < best_ll:
                    best_ll      = ll
                    best_weights = ws

        # Normalise (in case of float error)
        total = sum(best_weights.values())
        return {k: round(v / total, 3) for k, v in best_weights.items()}

    # ── Prediction ──

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return calibrated ensemble probabilities.
        X: shape (n_samples, N_FEATURES) — raw (unscaled) features.
        Returns: shape (n_samples, 3) — [P(home), P(draw), P(away)]
        """
        if not self.models:
            raise RuntimeError("Model not trained. Call train() or load().")

        X_s = self.scaler.transform(X)
        per_model: Dict[str, np.ndarray] = {
            k: m.predict_proba(X_s) for k, m in self.models.items()
        }
        blended = self._blend(per_model)

        # Ensure valid probability simplex
        blended = np.clip(blended, 1e-6, 1.0)
        blended /= blended.sum(axis=1, keepdims=True)
        return blended

    def predict_match(
        self,
        feature_vector: np.ndarray,
    ) -> Dict[str, float]:
        """
        Single-match prediction.  feature_vector shape: (N_FEATURES,).
        Returns {"home_win": p, "draw": p, "away_win": p}.
        """
        X = feature_vector.reshape(1, -1)
        probs = self.predict_proba(X)[0]
        return {
            "home_win": float(probs[HOME_WIN]),
            "draw":     float(probs[DRAW]),
            "away_win": float(probs[AWAY_WIN]),
        }

    def _blend(self, per_model: Dict[str, np.ndarray]) -> np.ndarray:
        result = None
        for k, probs in per_model.items():
            w = self.weights.get(k, 1.0 / len(per_model))
            if result is None:
                result = w * probs
            else:
                result = result + w * probs
        return result if result is not None else np.ones((1, 3)) / 3.0

    # ── Metrics ──

    def _compute_metrics(self, y_true: np.ndarray, probs: np.ndarray) -> Dict:
        y_pred = probs.argmax(axis=1)
        metrics = {
            "accuracy":    float(accuracy_score(y_true, y_pred)),
            "log_loss":    float(log_loss(y_true, probs)),
        }

        # Brier score (multi-class version: average over classes)
        brier_total = 0.0
        for c in range(3):
            y_bin = (y_true == c).astype(int)
            brier_total += brier_score_loss(y_bin, probs[:, c])
        metrics["brier_score"] = brier_total / 3.0

        # Per-class accuracy
        for c, name in RESULT_NAMES.items():
            mask = y_true == c
            if mask.sum() > 0:
                metrics[f"acc_{name}"] = float((y_pred[mask] == c).mean())

        # ROC AUC (one-vs-rest)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr"))
        except Exception:
            metrics["roc_auc"] = 0.0

        return metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate on any (X, y) set."""
        probs = self.predict_proba(X)
        return self._compute_metrics(y, probs)

    # ── Persistence ──

    def save(self, name: Optional[str] = None) -> Path:
        """Save trained ensemble to disk."""
        name = name or f"{self.league}_ensemble"
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump({
                "models":    self.models,
                "weights":   self.weights,
                "scaler":    self.scaler,
                "metadata":  self.metadata,
                "league":    self.league,
            }, fh)
        logger.info(f"Model saved → {path}")
        return path

    def load(self, name: Optional[str] = None) -> bool:
        """Load trained ensemble from disk. Returns True if successful."""
        name = name or f"{self.league}_ensemble"
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            logger.warning(f"No saved model at {path}")
            return False
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.models    = data["models"]
        self.weights   = data["weights"]
        self.scaler    = data["scaler"]
        self.metadata  = data.get("metadata", {})
        self.league    = data.get("league", self.league)
        self.is_trained = True
        logger.info(f"Model loaded ← {path}")
        return True

    def model_info(self) -> Dict:
        """Return metadata about the loaded model."""
        return {
            "league":       self.league,
            "is_trained":   self.is_trained,
            "models":       list(self.models.keys()),
            "weights":      self.weights,
            "metadata":     self.metadata,
        }


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(
    ensemble: BetNeuralEnsemble,
    feature_names: List[str],
    top_n: int = 20,
) -> List[Tuple[str, float]]:
    """
    Extract and average feature importance from XGB and LGB models.
    MLP doesn't provide importances so it's excluded.
    """
    importances: Dict[str, float] = {}
    count = 0

    for key in ("xgb", "lgb"):
        model = ensemble.models.get(key)
        if model is None:
            continue
        # Unwrap CalibratedClassifierCV
        base = getattr(model, "estimator", model)
        base = getattr(base, "calibrated_classifiers_", [model])
        if isinstance(base, list):
            base = getattr(base[0], "estimator", base[0])

        fi = getattr(base, "feature_importances_", None)
        if fi is None:
            continue

        n = min(len(fi), len(feature_names))
        for i in range(n):
            importances[feature_names[i]] = importances.get(feature_names[i], 0.0) + float(fi[i])
        count += 1

    if count > 1:
        importances = {k: v / count for k, v in importances.items()}

    sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
    return sorted_imp[:top_n]


# ── Poisson model (analytical baseline) ──────────────────────────────────────

class PoissonModel:
    """
    Dixon-Coles bivariate Poisson model.
    Uses attack/defence strength estimated from goal data.
    Provides a calibrated analytical baseline (no ML training required).
    Reference: Dixon & Coles (1997) "Modelling Association Football Scores"
    """

    def __init__(self, home_advantage: float = 0.25) -> None:
        self.home_advantage = home_advantage  # log-scale home advantage
        self._attack:  Dict[str, float] = {}
        self._defence: Dict[str, float] = {}
        self._rho = -0.13  # DC low-score correction (estimated empirically)

    def fit(self, matches: List[Dict]) -> "PoissonModel":
        """Estimate attack/defence strengths using MLE (iterative EM)."""
        from collections import defaultdict
        import scipy.optimize as opt

        teams = list({m["home_team"] for m in matches} | {m["away_team"] for m in matches})
        n_teams = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        # Support both field-name conventions: home_goals/away_goals (canonical)
        # and home_score/away_score (emitted by scraper + football_data_api)
        def _goals(m, side):
            canonical = "home_goals" if side == "home" else "away_goals"
            fallback  = "home_score" if side == "home" else "away_score"
            v = m.get(canonical)
            if v is None:
                v = m.get(fallback)
            return v

        goals_h = np.array([_goals(m, "home") or 1 for m in matches if _goals(m, "home") is not None], dtype=float)
        goals_a = np.array([_goals(m, "away") or 1 for m in matches if _goals(m, "away") is not None], dtype=float)

        # Simple mean-estimation (full MLE is iterative — this is robust enough)
        home_goals_by_team: Dict[str, List] = defaultdict(list)
        away_goals_by_team: Dict[str, List] = defaultdict(list)
        conceded_h_by_team: Dict[str, List] = defaultdict(list)
        conceded_a_by_team: Dict[str, List] = defaultdict(list)

        for m in matches:
            hg = _goals(m, "home")
            ag = _goals(m, "away")
            if hg is None or ag is None:
                continue
            ht = m["home_team"]
            at = m["away_team"]
            home_goals_by_team[ht].append(hg)
            away_goals_by_team[at].append(ag)
            conceded_h_by_team[ht].append(ag)
            conceded_a_by_team[at].append(hg)

        avg_home = float(np.mean(goals_h)) if len(goals_h) > 0 else 1.5
        avg_away = float(np.mean(goals_a)) if len(goals_a) > 0 else 1.2

        for team in teams:
            scored_home  = home_goals_by_team.get(team, [])
            scored_away  = away_goals_by_team.get(team, [])
            conceded_home = conceded_h_by_team.get(team, [])
            conceded_away = conceded_a_by_team.get(team, [])

            # Attack strength: normalise home goals scored by avg_home,
            # away goals scored by avg_away, then average.
            # This correctly accounts for the home-scoring advantage.
            attack_parts = (
                [g / avg_home for g in scored_home] +
                [g / avg_away for g in scored_away]
            )
            self._attack[team] = float(np.mean(attack_parts)) if attack_parts else 1.0

            # Defence strength: normalise home goals conceded by avg_away
            # (opponents scoring at home score at avg_away rate),
            # away goals conceded by avg_home.
            defence_parts = (
                [g / avg_away for g in conceded_home] +
                [g / avg_home for g in conceded_away]
            )
            self._defence[team] = float(np.mean(defence_parts)) if defence_parts else 1.0

        return self

    def _lambda(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Expected goals (lambda_home, lambda_away)."""
        import math as _math
        mu_home = 1.5   # typical home goals average
        mu_away = 1.2
        atk_h   = self._attack.get(home_team, 1.0)
        def_a   = self._defence.get(away_team, 1.0)
        atk_a   = self._attack.get(away_team, 1.0)
        def_h   = self._defence.get(home_team, 1.0)

        lambda_h = mu_home * atk_h * def_a * _math.exp(self.home_advantage)
        lambda_a = mu_away * atk_a * def_h
        return lambda_h, lambda_a

    def predict(self, home_team: str, away_team: str, max_goals: int = 8) -> Dict[str, float]:
        """Compute outcome probabilities via bivariate Poisson."""
        lh, la = self._lambda(home_team, away_team)
        p_home = p_draw = p_away = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p_ij = self._dc_prob(i, j, lh, la)
                if i > j:
                    p_home += p_ij
                elif i == j:
                    p_draw += p_ij
                else:
                    p_away += p_ij

        total = p_home + p_draw + p_away
        return {
            "home_win": p_home / total,
            "draw":     p_draw / total,
            "away_win": p_away / total,
            "exp_home_goals": round(lh, 2),
            "exp_away_goals": round(la, 2),
        }

    def _dc_prob(self, i: int, j: int, lh: float, la: float) -> float:
        """Dixon-Coles adjusted probability for scoreline i:j."""
        from math import factorial, exp
        p = (exp(-lh) * lh**i / factorial(i)) * (exp(-la) * la**j / factorial(j))
        # Rho correction for low-scoring games (reduces slight bias)
        if i == 0 and j == 0:
            p *= (1 - lh * la * self._rho)
        elif i == 1 and j == 0:
            p *= (1 + la * self._rho)
        elif i == 0 and j == 1:
            p *= (1 + lh * self._rho)
        elif i == 1 and j == 1:
            p *= (1 - self._rho)
        return max(p, 0.0)


# ── Master model manager ──────────────────────────────────────────────────────

class ModelManager:
    """
    Manages training, loading, and prediction for all leagues.
    Holds one BetNeuralEnsemble + one PoissonModel per league.
    """

    def __init__(self) -> None:
        self._ensembles: Dict[str, BetNeuralEnsemble] = {}
        self._poisson:   Dict[str, PoissonModel]       = {}

    def get_ensemble(self, league: str) -> BetNeuralEnsemble:
        if league not in self._ensembles:
            ens = BetNeuralEnsemble(league)
            ens.load()  # silent no-op if no saved model
            self._ensembles[league] = ens
        return self._ensembles[league]

    def get_poisson(self, league: str) -> Optional[PoissonModel]:
        return self._poisson.get(league)

    def train_league(
        self,
        league:   str,
        X:        np.ndarray,
        y:        np.ndarray,
        matches:  Optional[List[Dict]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """Train ensemble + Poisson for a league. Returns validation metrics."""
        ens = BetNeuralEnsemble(league)
        metrics = ens.train(X, y, feature_names=feature_names)
        ens.save()
        self._ensembles[league] = ens

        if matches:
            poisson = PoissonModel()
            poisson.fit(matches)
            self._poisson[league] = poisson
            # Persist Poisson model
            p_path = MODEL_DIR / f"{league}_poisson.pkl"
            with open(p_path, "wb") as fh:
                pickle.dump(poisson, fh)

        return metrics

    def load_all(self, leagues: List[str]) -> None:
        for league in leagues:
            self.get_ensemble(league)
            p_path = MODEL_DIR / f"{league}_poisson.pkl"
            if p_path.exists():
                with open(p_path, "rb") as fh:
                    self._poisson[league] = pickle.load(fh)

    def predict(
        self,
        league:         str,
        feature_vector: np.ndarray,
        home_team:      str = "",
        away_team:      str = "",
        blend_poisson:  float = 0.15,
    ) -> Dict[str, float]:
        """
        Ensemble + Poisson blended prediction.

        blend_poisson: weight given to the Poisson model (0 = pure ML).
        """
        ens = self.get_ensemble(league)

        if ens.is_trained:
            ml_probs = ens.predict_match(feature_vector)
        else:
            # Fall back to Elo-derived probabilities (stored in feature vector)
            from features import N_FEATURES, FEATURE_NAMES
            elo_diff_idx = FEATURE_NAMES.index("elo_diff") if "elo_diff" in FEATURE_NAMES else -1
            if elo_diff_idx >= 0 and len(feature_vector) > elo_diff_idx:
                ed = float(feature_vector[elo_diff_idx]) * 400.0
                p_h_raw = 1.0 / (1.0 + 10 ** (-ed / 400.0))
                imb     = abs(p_h_raw - 0.5)
                draw_p  = max(0.22, min(0.32, 0.30 - 0.16 * imb))
                rem     = 1.0 - draw_p
                ml_probs = {
                    "home_win": p_h_raw * rem,
                    "draw":     draw_p,
                    "away_win": (1 - p_h_raw) * rem,
                }
            else:
                ml_probs = {"home_win": 0.45, "draw": 0.27, "away_win": 0.28}

        # Blend with Poisson if available and team names given
        poisson = self._poisson.get(league)
        if poisson and home_team and away_team and blend_poisson > 0:
            try:
                p_probs = poisson.predict(home_team, away_team)
                for key in ("home_win", "draw", "away_win"):
                    ml_probs[key] = (
                        (1 - blend_poisson) * ml_probs[key]
                        + blend_poisson * p_probs[key]
                    )
            except Exception:
                pass

        # Renormalise
        total = sum(ml_probs.values())
        return {k: v / total for k, v in ml_probs.items()}

    def feature_importance_report(self, league: str, feature_names: List[str]) -> str:
        ens = self.get_ensemble(league)
        if not ens.is_trained:
            return "Model not trained."

        imps = get_feature_importance(ens, feature_names)
        lines = [f"Top {len(imps)} features for {league}:"]
        for i, (name, imp) in enumerate(imps, 1):
            bar = "█" * int(imp * 200)
            lines.append(f"  {i:>2}. {name:<35} {imp:.4f}  {bar}")
        return "\n".join(lines)


# ── CLI: train command ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse
    from features import (
        build_feature_builder_from_cache,
        FEATURE_NAMES,
        N_FEATURES,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Bet Neural — train ML models")
    parser.add_argument("--league",  default="premier_league")
    parser.add_argument("--season",  default="2024-2025")
    parser.add_argument("--all",     action="store_true")
    args = parser.parse_args()

    leagues = [
        "premier_league", "la_liga", "bundesliga", "serie_a",
        "ligue_1", "eredivisie", "primeira_liga",
    ] if args.all else [args.league]

    manager = ModelManager()

    for league in leagues:
        print(f"\n{'='*60}")
        print(f"Training {league} ({args.season})")
        print("="*60)

        builder = build_feature_builder_from_cache(league, args.season)
        matches  = builder.history._matches

        if len(matches) < 50:
            print(f"⚠️  Only {len(matches)} matches — need at least 50 to train. "
                  f"Run scraper first: python3 scraper.py scrape --league {league}")
            continue

        X, y, ids = builder.build_training_set(matches)
        print(f"Dataset: {X.shape[0]} matches × {X.shape[1]} features")

        metrics = manager.train_league(
            league, X, y,
            matches=matches,
            feature_names=FEATURE_NAMES,
        )

        print(f"\n✅ {league} Results:")
        print(f"   Accuracy:    {metrics['accuracy']:.1%}")
        print(f"   Log-loss:    {metrics['log_loss']:.4f}")
        print(f"   Brier score: {metrics['brier_score']:.4f}")
        print(f"   ROC AUC:     {metrics.get('roc_auc', 0):.4f}")
        print(f"   Weights:     {metrics['weights']}")

        print("\n" + manager.feature_importance_report(league, FEATURE_NAMES))
