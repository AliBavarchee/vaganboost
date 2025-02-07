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
    'validate_data_shape',
    'calculate_class_weights',
    'load_config'
]

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         classes: List[str],
                         title: str = 'Confusion Matrix',
                         normalize: bool = False,
                         output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix with optional normalization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        title: Plot title
        normalize: Whether to normalize values
        output_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
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
    """
    Plot multi-class ROC curves
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        classes: List of class names
        title: Plot title
        output_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
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
    """
    Plot multi-class Precision-Recall curves
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        classes: List of class names
        title: Plot title
        output_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
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

def save_model_artifacts(model: Union[Model, lgb.Booster, Any],
                        path: str,
                        metadata: Optional[Dict] = None) -> None:
    """
    Save model artifacts with proper format detection
    
    Args:
        model: Model object to save
        path: Output path for saving
        metadata: Additional metadata to save
    """
    create_directory(Path(path).parent)
    
    if isinstance(model, Model):
        model.save(path)
    elif isinstance(model, lgb.Booster):
        model.save_model(path)
    else:
        joblib.dump({'model': model, 'metadata': metadata}, path)

def load_model_artifacts(path: str) -> Dict[str, Any]:
    """
    Load saved model artifacts
    
    Args:
        path: Path to saved model
        
    Returns:
        Dictionary containing loaded model and metadata
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"No model found at {path}")
        
    if path.endswith('.keras') or path.endswith('.h5'):
        return {'model': tf.keras.models.load_model(path)}
    elif path.endswith('.txt'):
        return {'model': lgb.Booster(model_file=path)}
    else:
        return joblib.load(path)

def create_directory(path: Union[str, Path]) -> None:
    """
    Safely create directory if it doesn't exist
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def validate_data_shape(data: np.ndarray,
                       expected_shape: tuple,
                       data_name: str = 'input data') -> None:
    """
    Validate data shape matches expectations
    
    Args:
        data: Numpy array to validate
        expected_shape: Tuple of expected shape
        data_name: Name for error messages
    """
    if data.shape != expected_shape:
        raise ValueError(f"{data_name} shape {data.shape} "
                         f"doesn't match expected {expected_shape}")

def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary of class weights
    """
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    return dict(zip(classes, weights))

def load_config(config_path: str = None) -> Dict:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary with configuration parameters
    """
    if config_path:
        with open(config_path) as f:
            return json.load(f)
    return {
        'latent_dim': 8,
        'num_classes': 4,
        'cvae_epochs': 50,
        'cgan_epochs': 100,
        'samples_per_class': 100
    }