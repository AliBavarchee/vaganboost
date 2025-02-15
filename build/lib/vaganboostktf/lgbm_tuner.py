import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import pickle
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional, Dict, Any
import warnings
# Import modules from VaganBoost

from .data_preprocessor import DataPreprocessor
#from .lgbm_classifier import main as lgbm_pipeline_main

from .data_preprocessor import DataPreprocessor  # Standardized preprocessing
from .lgbm_classifier import train_lgbm_pipeline  # Custom LGBM pipeline
from .utils import DecompositionSwitcher  # PCA/LDA/TruncatedSVD switcher

class LightGBMTuner(BaseEstimator, ClassifierMixin):
    """
    Hyperparameter tuning and training for LGBMClassifier with preprocessing and feature selection.
    """

    def __init__(self, 
                 input_path: str,
                 output_path: str = "trained_models",
                 n_iter: int = 30,
                 cv: int = 3,
                 random_state: int = 42,
                 verbose: int = 0,
                 early_stopping_rounds: int = 20):
        """
        Initialize LightGBM tuner with pipeline.

        Args:
            input_path (str): Path to training data.
            output_path (str): Path to save model and artifacts.
            n_iter (int): Number of parameter combinations for tuning.
            cv (int): Cross-validation folds.
            random_state (int): Seed for reproducibility.
            verbose (int): Verbosity level.
            early_stopping_rounds (int): Rounds for early stopping.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.best_model_ = None
        self.best_params_ = None
        self.class_weights_ = None

    def tune(self):
        """
        Uses `train_lgbm_pipeline()` from `lgbm_classifier.py` to train the best LGBM model.
        """
        train_lgbm_pipeline(self.input_path, self.output_path)

        # Load the best trained model
        self.best_model_ = joblib.load(f"{self.output_path}/optimized_model.joblib")

        # Load best parameters if available
        best_params_path = Path(self.output_path) / "best_parameters.txt"
        if best_params_path.exists():
            with open(best_params_path, "r") as f:
                self.best_params_ = eval(f.read())  # Convert text to dictionary

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the best trained pipeline.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.best_model_ is None:
            raise ValueError("Model not trained. Call `tune()` first.")
        return self.best_model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted probabilities for each class.
        """
        if self.best_model_ is None:
            raise ValueError("Model not trained. Call `tune()` first.")
        return self.best_model_.predict_proba(X)

    def save(self, file_path: str) -> None:
        """
        Save the best trained model.

        Args:
            file_path (str): Path to save the model.
        """
        if self.best_model_ is None:
            raise ValueError("No trained model to save.")

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'best_model': self.best_model_, 'best_params': self.best_params_}, file_path)

    @classmethod
    def load(cls, file_path: str) -> 'LightGBMTuner':
        """
        Load a saved model.

        Args:
            file_path (str): Path to the saved model.

        Returns:
            LightGBMTuner: Loaded instance.
        """
        data = joblib.load(file_path)
        tuner = cls()
        tuner.best_model_ = data['best_model']
        tuner.best_params_ = data['best_params']
        return tuner
