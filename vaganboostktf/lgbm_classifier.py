import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, TruncatedSVD
import umap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import joblib
from joblib import Memory, dump
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, label_binarize
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from lightgbm import LGBMClassifier
from pathlib import Path
from imblearn.pipeline import Pipeline as imbPipeline

# Import from VaganBoost package
from .utils import plot_confusion_matrix, plot_roc_curves, plot_pr_curves, DecompositionSwitcher
from .data_preprocessor import DataPreprocessor

# CUDA Warning Suppression Edition
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

import logging
logging.getLogger('lightgbm').setLevel(logging.WARNING)
import warnings
from numba import cuda
try:
    cuda.close()
except:
    pass

warnings.filterwarnings('ignore')

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, TruncatedSVD
import umap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import joblib
from joblib import Memory, dump
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, label_binarize
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from lightgbm import LGBMClassifier
from pathlib import Path
from imblearn.pipeline import Pipeline as imbPipeline

# Import from VaganBoost package
from .utils import plot_confusion_matrix, plot_roc_curves, plot_pr_curves, DecompositionSwitcher
from .data_preprocessor import DataPreprocessor

# CUDA Warning Suppression Edition
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'


import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, TruncatedSVD
import umap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import joblib
from joblib import Memory, dump
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, label_binarize
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from lightgbm import LGBMClassifier
from pathlib import Path
from imblearn.pipeline import Pipeline as imbPipeline

# Import from VaganBoost package
from .utils import plot_confusion_matrix, plot_roc_curves, plot_pr_curves, DecompositionSwitcher
from .data_preprocessor import DataPreprocessor

# CUDA Warning Suppression Edition
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

import logging
logging.getLogger('lightgbm').setLevel(logging.WARNING)
import warnings
from numba import cuda
try:
    cuda.close()
except:
    pass

warnings.filterwarnings('ignore')


def generate_sampling_strategies(y_train):
    """
    Generate sampling strategies for balancing data in a generic way.
    
    For binary classification, returns a list of float ratios (minority/majority).
    For multi-class classification, returns a list of dictionaries mapping each class label
    to a desired target count.
    
    This function works for any number of classes (from 2 to even 1000+ classes).

    Args:
        y_train (array-like or pd.Series): Training labels.
        
    Returns:
        list: A list of sampling strategies.
    """
    # Ensure y_train is a pandas Series
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    
    class_counts = y_train.value_counts().to_dict()
    n_classes = len(class_counts)
    
    strategies = []
    
    if n_classes == 2:
        # For binary classification, generate some float ratios.
        # These ratios represent the desired ratio of minority to majority samples.
        strategies.extend([0.5, 0.75, 1.0, 1.25, 1.5])
    else:
        # For multi-class classification, generate dictionary strategies.
        # Identify the majority class and its count.
        majority_class = max(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]
        
        # Define multipliers to scale minority classes.
        multipliers = [1.0, 1.5, 2.0, 3.0]
        
        for m in multipliers:
            strat = {}
            # For the majority class, keep its original count.
            strat[majority_class] = majority_count
            # For all other classes, compute target counts.
            for cls, count in class_counts.items():
                if cls != majority_class:
                    # Upsample the minority class by the multiplier but do not exceed the majority count.
                    target = int(min(majority_count, count * m))
                    # Ensure the target is at least the original count.
                    target = max(target, count)
                    strat[cls] = target
            strategies.append(strat)
    return strategies


def train_lgbm_pipeline(input_path: str, output_path: str, dim_reducer="pca"):
    """
    Train an LGBMClassifier using a full pipeline with preprocessing, feature selection, SMOTE balancing, 
    and dimensionality reduction (PCA, LDA, or TruncatedSVD).

    Args:
        input_path: Path to input CSV file.
        output_path: Directory to save results.
        dim_reducer: Dimensionality reduction method. Options: ["pca", "lda", "svd"]
    """
    os.makedirs(output_path, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_path)
    feature_columns = [col for col in df.columns if col != "label"]
    target_column = "label"

    # Initialize DataPreprocessor and get train/test splits
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, feature_columns, target_column)
    
    # Select dimensionality reduction method (pass actual estimator objects)
    if dim_reducer.lower() == "pca":
        reducer = DecompositionSwitcher(estimator=PCA(n_components=20))
    elif dim_reducer.lower() == "lda":
        reducer = DecompositionSwitcher(estimator=LDA())
    elif dim_reducer.lower() == "svd":
        reducer = DecompositionSwitcher(estimator=TruncatedSVD(n_components=20))
    else:
        raise ValueError("Invalid dim_reducer. Choose from 'pca', 'lda', or 'svd'.")

    # Define the model pipeline
    pipeline = imbPipeline([
        ('scaler', RobustScaler()),  # Normalize features
        ('feature_selector', SelectKBest(mutual_info_classif, k=20)),  # Select top 20 features
        ('dim_reducer', reducer),  # Apply dimensionality reduction
        ('sampler', SMOTE(random_state=42)),  # Handle class imbalance
        ('classifier', LGBMClassifier(
            objective='multiclass',
            num_class=len(np.unique(y_train)),
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ))
    ])
    
    # Generate sampling strategies generically from y_train
    sampling_strategies = generate_sampling_strategies(y_train)
    
    # Hyperparameter tuning: override the decomposition step and sampler strategy
    param_dist = {
        'dim_reducer__estimator': [PCA(), LDA(), TruncatedSVD()],  # Tune best decomposition method
        'classifier__num_leaves': [127, 255],
        'feature_selector__k': [500, 1000, 1500],
        'sampler__sampling_strategy': sampling_strategies,  # Include sampling strategies
        'sampler__k_neighbors': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__n_estimators': [100, 300],
        'classifier__max_depth': [7, 10]
    }
    
    # Hyperparameter tuning using RandomizedSearchCV
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    
    # Use the best estimator (fitted pipeline)
    final_model = search.best_estimator_

    # Save best model
    joblib.dump(final_model, os.path.join(output_path, "optimized_model.joblib"))

    # Generate evaluation reports
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)

    # Save classification report
    with open(os.path.join(output_path, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred))
    
    # Visualize model performance
    class_names = [str(i) for i in np.unique(y_test)]
    plot_confusion_matrix(y_test, y_pred, class_names, output_path=os.path.join(output_path, "confusion_matrix.png"))
    plot_roc_curves(y_test, y_proba, class_names, output_path=os.path.join(output_path, "roc_curve.png"))
    plot_pr_curves(y_test, y_proba, class_names, output_path=os.path.join(output_path, "pr_curve.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM with a full pipeline")
    parser.add_argument("--input_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", required=True, help="Output directory for model and results")
    parser.add_argument("--dim_reducer", default="pca", help="Dimensionality reduction method: 'pca', 'lda', or 'svd'")
    args = parser.parse_args()

    train_lgbm_pipeline(args.input_path, args.output_path, dim_reducer=args.dim_reducer)
