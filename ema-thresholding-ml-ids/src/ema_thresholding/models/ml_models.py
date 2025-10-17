"""
Machine learning models for intrusion detection
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from typing import Dict, Any, Optional, Tuple
import logging
import pickle

logger = logging.getLogger(__name__)

class MLModels:
    """Collection of ML models for intrusion detection"""

    def __init__(self):
        self.models = {}
        self.calibrated_models = {}

    def create_random_forest(
        self, 
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Create Random Forest model with default IDS-optimized parameters

        Parameters:
        -----------
        n_estimators : int
            Number of trees
        max_depth : int or None
            Maximum tree depth
        random_state : int
            Random seed
        **kwargs : dict
            Additional RandomForest parameters

        Returns:
        --------
        model : RandomForestClassifier
            Configured Random Forest model
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )

        self.models['random_forest'] = model
        logger.info(f"Created Random Forest: n_estimators={n_estimators}, max_depth={max_depth}")
        return model

    def create_xgboost(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs
    ) -> xgb.XGBClassifier:
        """
        Create XGBoost model with default IDS-optimized parameters

        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        random_state : int
            Random seed
        **kwargs : dict
            Additional XGBoost parameters

        Returns:
        --------
        model : XGBClassifier
            Configured XGBoost model
        """
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='logloss',
            verbosity=0,
            **kwargs
        )

        self.models['xgboost'] = model
        logger.info(f"Created XGBoost: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
        return model

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        calibrate: bool = True,
        cv_folds: int = 3
    ) -> object:
        """
        Train a model and optionally calibrate it

        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train, y_train : arrays
            Training data
        calibrate : bool
            Whether to calibrate the model for better probability estimates
        cv_folds : int
            Number of CV folds for calibration

        Returns:
        --------
        model : object
            Trained (and possibly calibrated) model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")

        model = self.models[model_name]

        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)

        # Calibrate model for better probability estimates
        if calibrate:
            logger.info(f"Calibrating {model_name} with {cv_folds}-fold CV...")
            calibrated_model = CalibratedClassifierCV(model, cv=cv_folds)
            calibrated_model.fit(X_train, y_train)
            self.calibrated_models[model_name] = calibrated_model

            logger.info(f"Model {model_name} trained and calibrated")
            return calibrated_model

        logger.info(f"Model {model_name} trained")
        return model

    def get_model(self, model_name: str, calibrated: bool = True) -> object:
        """
        Get trained model

        Parameters:
        -----------
        model_name : str
            Name of the model
        calibrated : bool
            Whether to return calibrated version

        Returns:
        --------
        model : object
            Trained model
        """
        if calibrated and model_name in self.calibrated_models:
            return self.calibrated_models[model_name]
        elif model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found")

    def predict_proba(self, model_name: str, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        """
        Get probability predictions

        Parameters:
        -----------
        model_name : str
            Name of the model
        X : array
            Input features
        calibrated : bool
            Whether to use calibrated model

        Returns:
        --------
        probabilities : array
            Probability estimates (positive class)
        """
        model = self.get_model(model_name, calibrated)
        proba = model.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class

    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        calibrated: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Parameters:
        -----------
        model_name : str
            Name of the model
        X_test, y_test : arrays
            Test data
        calibrated : bool
            Whether to use calibrated model

        Returns:
        --------
        metrics : dict
            Dictionary of performance metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        model = self.get_model(model_name, calibrated)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }

        logger.info(f"{model_name} evaluation:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def cross_validate(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        scoring: str = 'f1'
    ) -> Dict[str, float]:
        """
        Perform cross-validation

        Parameters:
        -----------
        model_name : str
            Name of the model
        X, y : arrays
            Data for cross-validation
        cv_folds : int
            Number of CV folds
        scoring : str
            Scoring metric

        Returns:
        --------
        cv_results : dict
            Cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]

        logger.info(f"Cross-validating {model_name} with {cv_folds} folds...")
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)

        results = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }

        logger.info(f"CV results for {model_name}: {results['mean']:.4f} ± {results['std']:.4f}")
        return results

    def save_model(self, model_name: str, filepath: str, calibrated: bool = True) -> None:
        """Save trained model"""
        model = self.get_model(model_name, calibrated)

        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Model {model_name} saved to {filepath}")

    def load_model(self, filepath: str, model_name: str, calibrated: bool = True) -> object:
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        if calibrated:
            self.calibrated_models[model_name] = model
        else:
            self.models[model_name] = model

        logger.info(f"Model {model_name} loaded from {filepath}")
        return model

    def get_available_models(self) -> Dict[str, list]:
        """Get list of available models"""
        return {
            'models': list(self.models.keys()),
            'calibrated_models': list(self.calibrated_models.keys())
        }

def create_default_models(random_state: int = 42) -> MLModels:
    """
    Create ML models with default IDS-optimized hyperparameters

    Parameters:
    -----------
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    ml_models : MLModels
        MLModels instance with pre-configured models
    """
    ml_models = MLModels()

    # Random Forest with IDS-optimized parameters
    ml_models.create_random_forest(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    )

    # XGBoost with IDS-optimized parameters  
    ml_models.create_xgboost(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )

    logger.info("Created default models: Random Forest, XGBoost")
    return ml_models

def get_hyperparameters() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter grids for tuning"""
    return {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    }
