import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Handles complete data preprocessing pipeline"""
    
    def __init__(self, test_size=0.2, random_state=42, stratify=True):
        """
        Initialize preprocessing parameters
        
        Args:
            test_size (float): Proportion of data for testing (default: 0.2)
            random_state (int): Random seed (default: 42)
            stratify (bool): Stratified splitting (default: True)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.scaler = StandardScaler()
        self.class_weights = None
        self.feature_columns = None
        self.target_column = None  # Now properly initialized

    def __reduce__(self):
        return (self.__class__, 
                (self.test_size, self.random_state, self.stratify))

    def fit(self, X, y=None):
        """Standard sklearn-compatible fit method"""
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        """Standard sklearn-compatible transform method"""
        return self.scaler.transform(X)

    def prepare_data(self, dataframe, feature_columns, target_column):
        """
        Complete preprocessing workflow
        
        Args:
            dataframe: Input DataFrame
            feature_columns: List of feature column names
            target_column: Name of target column
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self._validate_input(dataframe, feature_columns, target_column)

        X = dataframe[feature_columns].values
        y = dataframe[target_column].values

        return self._process_data(X, y)

    def _validate_input(self, dataframe, feature_columns, target_column):
        """Validate input data integrity"""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        missing = [col for col in feature_columns + [target_column] 
                  if col not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _process_data(self, X, y):
        """Handle data splitting and scaling"""
        stratify = y if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )

        # Fit and transform scaler
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Calculate class weights
        self._calculate_class_weights(y_train)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def _calculate_class_weights(self, y_train):
        """Compute balanced class weights"""
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )
        self.class_weights = dict(zip(classes, weights))

    def save_pipeline(self, file_path):
        """Save preprocessing artifacts"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'class_weights': self.class_weights
        }, file_path)

    @classmethod
    def load_pipeline(cls, file_path):
        """Load preprocessing artifacts"""
        artifacts = joblib.load(file_path)
        preprocessor = cls()
        preprocessor.scaler = artifacts['scaler']
        preprocessor.feature_columns = artifacts['feature_columns']
        preprocessor.target_column = artifacts['target_column']
        preprocessor.class_weights = artifacts['class_weights']
        return preprocessor

    def get_feature_names(self):
        """Get original feature names"""
        return self.feature_columns

    def __getstate__(self):
        """Custom serialization for pickle"""
        state = self.__dict__.copy()
        # Exclude non-pickleable objects
        del state['scaler']
        return state

    def __setstate__(self, state):
        """Custom deserialization for pickle"""
        self.__dict__.update(state)
        # Reinitialize scaler
        self.scaler = StandardScaler()
        
    def save(self, path):
        """Save preprocessor with joblib"""
        joblib.dump({
            'state': self.__dict__,
            'scaler_mean_': self.scaler.mean_,
            'scaler_scale_': self.scaler.scale_
        }, path)

    @classmethod
    def load(cls, path):
        """Load preprocessor with joblib"""
        data = joblib.load(path)
        obj = cls()
        obj.__dict__.update(data['state'])
        obj.scaler.mean_ = data['scaler_mean_']
        obj.scaler.scale_ = data['scaler_scale_']
        return obj