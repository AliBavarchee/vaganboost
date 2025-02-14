import os
import json
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Union, Optional, List
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from tensorflow.keras.models import Model
import lightgbm as lgb
from .data_preprocessor import DataPreprocessor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, TruncatedSVD
import lightgbm as lgb
import tensorflow as tf

# Visualization settings
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
COLORS = sns.color_palette()

__all__ = [
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_pr_curves',
    'save_model_artifacts',
    'load_model_artifacts',
    'create_directory',
    'set_random_seeds',
    'DecompositionSwitcher'
]

### ==============================
### 1. DECOMPOSITION SWITCHER CLASS
### ==============================
class DecompositionSwitcher(BaseEstimator, TransformerMixin):
    """Dynamic decomposition method selector: PCA, LDA, or TruncatedSVD."""
    def __init__(self, estimator=PCA(n_components=20)):
        self.estimator = estimator
        
    def fit(self, X, y=None):
        if isinstance(self.estimator, LDA):
            self.estimator.fit(X, y)  # LDA requires class labels (y)
        else:
            self.estimator.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.estimator.transform(X)

### ==============================
### 2. MODEL PERFORMANCE VISUALIZATION
### ==============================
def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         classes: List[str],
                         title: str = 'Confusion Matrix',
                         normalize: bool = False,
                         output_path: Optional[str] = None) -> plt.Figure:
    """Plot confusion matrix with optional normalization."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if output_path:
        create_directory(Path(output_path).parent)
        plt.savefig(output_path, bbox_inches='tight')
        
    return plt.gcf()

def plot_roc_curves(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   classes: List[str],
                   title: str = 'ROC Curves',
                   output_path: Optional[str] = None) -> plt.Figure:
    """Plot multi-class ROC curves."""
    plt.figure()
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
        auc_score = roc_auc_score(y_true == i, y_proba[:, i])
        plt.plot(fpr, tpr, 
                 label=f'{class_name} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    
    if output_path:
        create_directory(Path(output_path).parent)
        plt.savefig(output_path, bbox_inches='tight')
        
    return plt.gcf()

def plot_pr_curves(y_true: np.ndarray,
                  y_proba: np.ndarray,
                  classes: List[str],
                  title: str = 'Precision-Recall Curves',
                  output_path: Optional[str] = None) -> plt.Figure:
    """Plot multi-class Precision-Recall curves."""
    plt.figure()
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_proba[:, i])
        plt.plot(recall, precision, label=f'{class_name}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    
    if output_path:
        create_directory(Path(output_path).parent)
        plt.savefig(output_path, bbox_inches='tight')
        
    return plt.gcf()

### ==============================
### 3. MODEL ARTIFACT HANDLING
### ==============================
def save_model_artifacts(model: Union[tf.keras.Model, lgb.Booster, Any],
                        path: str,
                        metadata: Optional[Dict] = None) -> None:
    """Save model artifacts with proper format detection."""
    create_directory(Path(path).parent)
    
    if isinstance(model, tf.keras.Model):
        model.save(path)  # Save in native Keras format (.keras)
    elif isinstance(model, lgb.Booster):
        model.save_model(path)
    else:
        joblib.dump({'model': model, 'metadata': metadata}, path)

def load_model_artifacts(path: str) -> Dict[str, Any]:
    """Load saved model artifacts."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No model found at {path}")
        
    if path.endswith('.keras'):
        return {'model': tf.keras.models.load_model(path)}
    elif path.endswith('.txt'):
        return {'model': lgb.Booster(model_file=path)}
    else:
        return joblib.load(path)

### ==============================
### 4. UTILITY FUNCTIONS
### ==============================
def create_directory(path: Union[str, Path]) -> None:
    """Safely create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
