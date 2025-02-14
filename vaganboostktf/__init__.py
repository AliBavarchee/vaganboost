"""
VAE-GAN Boost Keras & TF: Hybrid Generative Modeling with LightGBM Integration
"""

from .data_preprocessor import DataPreprocessor
from .cvae import CVAE
from .cgan import CGAN
from .lgbm_classifier import train_lgbm_pipeline
from .lgbm_tuner import LightGBMTuner

from .trainer import HybridModelTrainer
from .utils import (
    DecompositionSwitcher,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    save_model_artifacts,
    load_model_artifacts,
    create_directory,
    set_random_seeds
)

#from .config import DEFAULT_PARAMS

__version__ = "0.9.5"
__all__ = [
    'DataPreprocessor',
    'CVAE',
    'CGAN',
    'train_lgbm_pipeline',
    'LightGBMTuner',
    'HybridModelTrainer',
    'DecompositionSwitcher',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_pr_curves',
    'save_model_artifacts',
    'load_model_artifacts',
    'create_directory',
    'set_random_seeds'
]