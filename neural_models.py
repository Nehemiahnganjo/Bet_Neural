"""
neural_models.py — LEGACY / UNUSED
====================================
⚠️  THIS FILE IS NOT USED BY THE PRODUCTION SYSTEM.

The live prediction stack is in:
  - models.py          (XGBoost + LightGBM + MLP ensemble — actual production models)
  - algorithm_improvements.py  (calibration, Monte Carlo, stacking)

Why this file exists:
  This was the original TensorFlow LSTM proof-of-concept written before the
  sklearn/XGB ensemble was adopted.  It is kept for historical reference only.

Do NOT import from this file in new code.  The TensorFlow dependency is optional
and the LSTM architecture here is not used in predictions.

Known issues (not fixed because file is unused):
  - build_betting_lstm: first LSTM uses return_sequences=True but second also uses
    return_sequences=True — feeding into a third LSTM with return_sequences=False.
    This is architecturally valid but the output shape mismatch with batch-norm
    layers would require careful shape handling.
  - create_betting_features: fallback xG/xGA defaults are hardcoded magic numbers
    rather than league-specific priors.
  - No connection to the live MatchHistory / FeatureBuilder pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class BetNeuralModels:
    """
    Collection of neural network models for football prediction
    """
    
    def __init__(self):
        self.models = {}
        self.model_configs = {
            'premier_league': {
                'lstm_units': 128,
                'dense_units': 64,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'home_advantage_factor': 1.15
            },
            'la_liga': {
                'lstm_units': 96,
                'dense_units': 48,
                'dropout_rate': 0.25,
                'learning_rate': 0.0008,
                'home_advantage_factor': 1.10
            },
            'bundesliga': {
                'lstm_units': 112,
                'dense_units': 56,
                'dropout_rate': 0.35,
                'learning_rate': 0.0012,
                'home_advantage_factor': 1.12
            },
            'serie_a': {
                'lstm_units': 88,
                'dense_units': 44,
                'dropout_rate': 0.28,
                'learning_rate': 0.0009,
                'home_advantage_factor': 1.08
            }
        }
    
    def build_betting_lstm(self, league: str, input_shape: Tuple[int, int]) -> Optional[object]:
        """
        Build LSTM model optimized for betting predictions
        """
        if not TENSORFLOW_AVAILABLE:
            return None
        
        config = self.model_configs.get(league, self.model_configs['premier_league'])
        
        model = keras.Sequential([
            layers.Input(shape=input_shape, name=f'{league}_input'),
            
            # First LSTM layer with return sequences
            layers.LSTM(config['lstm_units'], return_sequences=True,
                       recurrent_dropout=0.2, name=f'{league}_lstm1'),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout_rate']),
            
            # Second LSTM layer
            layers.LSTM(config['lstm_units'] // 2, return_sequences=True,
                       name=f'{league}_lstm2'),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout_rate']),
            
            # Final LSTM layer
            layers.LSTM(config['lstm_units'] // 4, name=f'{league}_lstm3'),
            layers.BatchNormalization(),
            
            # Dense layers for betting-specific features
            layers.Dense(config['dense_units'], activation='relu', name=f'{league}_dense1'),
            layers.Dropout(config['dropout_rate']),
            layers.Dense(config['dense_units'] // 2, activation='relu', name=f'{league}_dense2'),
            layers.Dropout(0.2),
            
            # Output layer - 3 outcomes (home, draw, away)
            layers.Dense(3, activation='softmax', name=f'{league}_output')
        ], name=f'{league.title()}_Betting_LSTM')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_ensemble_model(self, input_shape: Tuple[int, int]) -> Optional[object]:
        """
        Build ensemble model combining multiple league-specific models
        """
        if not TENSORFLOW_AVAILABLE:
            return None
        
        inputs = keras.Input(shape=input_shape, name='ensemble_input')
        
        # League-specific branches
        pl_branch = self._build_league_branch(inputs, 'premier_league', 64)
        ll_branch = self._build_league_branch(inputs, 'la_liga', 48)
        bl_branch = self._build_league_branch(inputs, 'bundesliga', 56)
        sa_branch = self._build_league_branch(inputs, 'serie_a', 44)
        
        # Combine branches
        combined = layers.Concatenate(name='ensemble_combine')([
            pl_branch, ll_branch, bl_branch, sa_branch
        ])
        
        # Meta-learner
        x = layers.Dense(128, activation='relu', name='ensemble_meta1')(combined)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu', name='ensemble_meta2')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu', name='ensemble_meta3')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(3, activation='softmax', name='ensemble_output')(x)
        
        model = keras.Model(inputs, outputs, name='European_Betting_Ensemble')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0008),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_league_branch(self, inputs, league: str, units: int):
        """Build a branch for a specific league"""
        x = layers.LSTM(units, name=f'{league}_branch_lstm')(inputs)
        x = layers.Dense(units // 2, activation='relu', name=f'{league}_branch_dense')(x)
        return x
    
    def create_betting_features(self, team_stats: Dict, opponent_stats: Dict, 
                               h2h_history: List[Dict]) -> np.ndarray:
        """
        Create feature vector for neural network prediction
        """
        features = []
        
        # Team offensive metrics
        features.extend([
            team_stats.get('goals_per_game', 1.5),
            team_stats.get('shots_per_game', 12),
            team_stats.get('shots_on_target_per_game', 4),
            team_stats.get('xg_per_game', 1.4),
            team_stats.get('possession_avg', 50),
        ])
        
        # Team defensive metrics  
        features.extend([
            team_stats.get('goals_conceded_per_game', 1.2),
            team_stats.get('shots_conceded_per_game', 10),
            team_stats.get('xga_per_game', 1.1),
            team_stats.get('clean_sheets_pct', 0.3),
        ])
        
        # Opponent metrics (similar structure)
        features.extend([
            opponent_stats.get('goals_per_game', 1.5),
            opponent_stats.get('goals_conceded_per_game', 1.2),
            opponent_stats.get('xg_per_game', 1.4),
            opponent_stats.get('xga_per_game', 1.1),
        ])
        
        # Head-to-head features
        if h2h_history:
            recent_h2h = h2h_history[-5:]  # Last 5 meetings
            team_wins = sum(1 for match in recent_h2h if match.get('winner') == 'team')
            draws = sum(1 for match in recent_h2h if match.get('winner') == 'draw')
            features.extend([
                team_wins / len(recent_h2h),
                draws / len(recent_h2h),
                np.mean([m.get('total_goals', 2) for m in recent_h2h])
            ])
        else:
            features.extend([0.33, 0.33, 2.5])  # Default values
        
        # Form features (last 5 games)
        team_form = team_stats.get('form_points', 7.5) / 15  # Normalize to 0-1
        opponent_form = opponent_stats.get('form_points', 7.5) / 15
        features.extend([team_form, opponent_form])
        
        return np.array(features, dtype=np.float32)
    
    def predict_with_neural_network(self, league: str, team_features: np.ndarray) -> Dict[str, float]:
        """
        Make prediction using trained neural network
        """
        if not TENSORFLOW_AVAILABLE or league not in self.models:
            # Fallback to statistical prediction
            return self._statistical_prediction(team_features)
        
        model = self.models[league]
        
        # Reshape for LSTM input (batch_size, timesteps, features)
        if len(team_features.shape) == 1:
            team_features = team_features.reshape(1, 1, -1)
        
        prediction = model.predict(team_features, verbose=0)[0]
        
        return {
            'home_win': float(prediction[0]),
            'draw': float(prediction[1]),
            'away_win': float(prediction[2])
        }
    
    def _statistical_prediction(self, features: np.ndarray) -> Dict[str, float]:
        """
        Fallback statistical prediction when neural networks aren't available
        """
        # Simple statistical model based on feature differences
        team_strength = np.mean(features[:5])  # Offensive features
        team_defense = 2.0 - np.mean(features[5:9])  # Defensive features (inverted)
        opponent_strength = np.mean(features[9:13])
        form_diff = features[-2] - features[-1]
        
        # Calculate relative strength
        strength_diff = (team_strength + team_defense) - opponent_strength + form_diff
        
        # Convert to probabilities
        home_win = max(0.1, min(0.8, 0.45 + strength_diff * 0.3))
        away_win = max(0.1, min(0.8, 0.25 - strength_diff * 0.2))
        draw = 1.0 - home_win - away_win
        
        # Normalize
        total = home_win + draw + away_win
        return {
            'home_win': home_win / total,
            'draw': draw / total,
            'away_win': away_win / total
        }
    
    def train_model(self, league: str, training_data: List[Dict], epochs: int = 50):
        """
        Train neural network model for specific league
        """
        if not TENSORFLOW_AVAILABLE:
            print("⚠️  TensorFlow not available. Cannot train neural network models.")
            return False
        
        # Prepare training data
        X, y = self._prepare_training_data(training_data)
        
        if len(X) < 10:  # Need minimum data for training
            print(f"⚠️  Insufficient training data for {league} ({len(X)} samples)")
            return False
        
        # Build and train model
        input_shape = (1, X.shape[1])  # (timesteps, features)
        model = self.build_betting_lstm(league, input_shape)
        
        if model is None:
            return False
        
        # Reshape data for LSTM
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        
        # Train model
        history = model.fit(
            X_reshaped, y,
            epochs=epochs,
            batch_size=self.model_configs[league].get('batch_size', 32),
            validation_split=0.2,
            verbose=1
        )
        
        self.models[league] = model
        print(f"✅ Model trained for {league}")
        return True
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for neural network
        """
        X, y = [], []
        
        for match in training_data:
            # Create features
            features = self.create_betting_features(
                match.get('home_team_stats', {}),
                match.get('away_team_stats', {}),
                match.get('h2h_history', [])
            )
            
            # Create target (one-hot encoded result)
            result = match.get('result')
            if result == 'H':  # Home win
                target = [1, 0, 0]
            elif result == 'D':  # Draw
                target = [0, 1, 0]
            elif result == 'A':  # Away win
                target = [0, 0, 1]
            else:
                continue  # Skip invalid results
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        if not TENSORFLOW_AVAILABLE:
            return False
        
        model_data = {
            'leagues': list(self.models.keys()),
            'configs': self.model_configs,
            'saved_at': datetime.now().isoformat()
        }
        
        # Save model architectures and weights
        for league, model in self.models.items():
            model.save(f"{filepath}_{league}_model.h5")
        
        # Save metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✅ Models saved to {filepath}")
        return True
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        if not TENSORFLOW_AVAILABLE:
            return False
        
        try:
            # Load metadata
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Load models
            for league in metadata['leagues']:
                model_path = f"{filepath}_{league}_model.h5"
                self.models[league] = keras.models.load_model(model_path)
            
            print(f"✅ Models loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

# Example usage
if __name__ == "__main__":
    models = BetNeuralModels()
    
    if TENSORFLOW_AVAILABLE:
        print("🧠 Neural Network Models Available")
        
        # Create sample model
        input_shape = (1, 20)  # 1 timestep, 20 features
        model = models.build_betting_lstm('premier_league', input_shape)
        
        if model:
            print(f"✅ Premier League model created with {model.count_params()} parameters")
    else:
        print("📊 Using statistical prediction models (TensorFlow not available)")
    
    # Test feature creation
    sample_team_stats = {
        'goals_per_game': 2.1,
        'shots_per_game': 15,
        'xg_per_game': 1.8,
        'form_points': 12
    }
    
    sample_opponent_stats = {
        'goals_per_game': 1.3,
        'goals_conceded_per_game': 1.1,
        'xg_per_game': 1.2,
        'form_points': 8
    }
    
    features = models.create_betting_features(sample_team_stats, sample_opponent_stats, [])
    print(f"📊 Generated {len(features)} features for prediction")