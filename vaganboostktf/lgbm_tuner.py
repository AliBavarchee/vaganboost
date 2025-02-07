import lightgbm as lgb
import numpy as np
import joblib
from pathlib import Path
import pickle
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional, Dict, Any
import warnings
from .data_preprocessor import DataPreprocessor

class LightGBMTuner(BaseEstimator, ClassifierMixin):
    """LightGBM classifier with integrated hyperparameter tuning and class weighting"""
    
    def __init__(self, 
                 n_iter: int = 30,
                 cv: int = 3,
                 random_state: int = 42,
                 verbose: int = 0,
                 early_stopping_rounds: int = 20):
        """
        Initialize LightGBM tuner
        
        Args:
            n_iter (int): Number of parameter combinations to try
            cv (int): Number of cross-validation folds
            random_state (int): Seed for reproducibility
            verbose (int): Controls verbosity of output
            early_stopping_rounds (int): Early stopping rounds for final training
        """
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.class_weights_ = None
        self.best_model_ = None
        self.best_params_ = None
        self._param_distributions = {
            'num_leaves': sp_randint(20, 150),
            'learning_rate': sp_uniform(0.01, 0.3),
            'n_estimators': sp_randint(50, 500),
            'max_depth': sp_randint(3, 12),
            'min_child_samples': sp_randint(10, 100)
        }
        
        self.lgb_model = lgb.LGBMClassifier(objective='multiclass',
            num_class=4,
            random_state=42,
            force_col_wise=True,
            verbosity=-1  # LightGBM-specific verbosity (separate from sklearn)
        )

        self.random_search = RandomizedSearchCV(
            estimator=self.lgb_model,
            param_distributions=self._param_distributions,
            n_iter=n_iter,
            cv=cv,
            verbose=0,  # Valid values: 0 (silent), 1-50 (increasing verbosity)
            random_state=random_state,
            n_jobs=-1
        )
        

    def _compute_class_weights(self, y: np.ndarray):
        """Calculate balanced class weights"""
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y
        )
        self.class_weights_ = dict(zip(classes, weights))

    def tune(self, 
            X: np.ndarray, 
            y: np.ndarray,
            eval_set: Optional[list] = None) -> None:
        """
        Perform hyperparameter tuning with RandomizedSearchCV
        
        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Target labels
            eval_set (list, optional): Validation data in format [(X_val, y_val)]
        """
        self._compute_class_weights(y)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            base_model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=len(np.unique(y)),
                class_weight=self.class_weights_,
                random_state=self.random_state,
                verbosity=self.verbose,
                force_col_wise=True
            )

            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self._param_distributions,
                n_iter=self.n_iter,
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=self.verbose
            )

            search.fit(X, y, eval_set=eval_set, eval_metric='multi_logloss')
            
            self.best_params_ = search.best_params_
            self.best_model_ = search.best_estimator_

    def final_fit(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 eval_set: Optional[list] = None) -> None:
        """
        Train final model with best parameters and early stopping
        
        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Target labels
            eval_set (list, optional): Validation data in format [(X_val, y_val)]
        """
        if self.best_params_ is None:
            raise ValueError("Must run tune() before final_fit()")

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model"""
        if self.best_model_ is None:
            raise ValueError("Model not trained. Call tune() or final_fit() first.")
        return self.best_model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the best model"""
        if self.best_model_ is None:
            raise ValueError("Model not trained. Call tune() or final_fit() first.")
        return self.best_model_.predict_proba(X)

    def save(self, file_path: str) -> None:
        if self.best_model_ is None:
            raise ValueError("No model to save")
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'best_model': self.best_model_,
            'best_params': self.best_params_,
            'class_weights': self.class_weights_
        }, file_path)

    @classmethod
    def load(cls, file_path: str) -> 'LightGBMTuner':
        data = joblib.load(file_path)
        tuner = cls()
        tuner.best_model_ = data['best_model']
        tuner.best_params_ = data['best_params']
        tuner.class_weights_ = data['class_weights']
        return tuner
        
    # tune_lgbm Method:
    def tune_lgbm(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              num_iterations: Optional[int] = None):
        # Optionally update n_iter if num_iterations is provided
        if num_iterations is not None:
            self.n_iter = num_iterations
            # Perform tuning
            self.tune(X_train, y_train, eval_set=[(X_val, y_val)])
            # Perform final fitting
            self.final_fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return self.best_model_




# ==============================

    # def save(self, file_path: str) -> None:
        # """
        # Save tuner state including best model and parameters
        
        # Args:
            # file_path (str): Path to save the tuner state
        # """
        # if self.best_model_ is None:
            # raise ValueError("No model to save")
        
        ##Create directory if it doesn't exist
        # Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        ##Save all necessary components
        # joblib.dump({
            # 'best_model': self.best_model_,
            # 'best_params': self.best_params_,
            # 'class_weights': self.class_weights_
        # }, file_path)

    # @classmethod
    # def load(cls, file_path: str) -> 'LightGBMTuner':
        # """
        # Load saved tuner state
        
        # Args:
            # file_path (str): Path to saved tuner state
            
        # Returns:
            # LightGBMTuner: Loaded tuner instance
        # """
        # data = joblib.load(file_path)
        # tuner = cls()
        # tuner.best_model_ = data['best_model']
        # tuner.best_params_ = data['best_params']
        # tuner.class_weights_ = data['class_weights']
        # return tuner

# ==============================
        
    
# Predict methods
    # def predict(self, X):
        # if self.best_model_ is None:
            # raise ValueError("Model not trained")
        # return self.best_model_.predict(X)

    # def predict_proba(self, X):
        # if self.best_model_ is None:
            # raise ValueError("Model not trained")
        # return self.best_model_.predict_proba(X)
