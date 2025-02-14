# vaganboostktf: VAE-GAN Boost by TF

=============================================<p align="Center">![vaganboostktf](https://teal-broad-gecko-650.mypinata.cloud/ipfs/bafybeiew72v2a7okbxdawl6febkahxeulpxkjylizttacz5xfreq4d675q)</p>=============================================
=====

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Hybrid generative-classification framework combining Conditional VAE, Conditional GAN, and LightGBM for handling class-imbalanced classification tasks.

## Features

- 🧬 Hybrid architecture combining generative and discriminative models
- ⚖️ Effective handling of class imbalance through synthetic data generation
- 🔄 Iterative training process with automatic model refinement
- 📊 Comprehensive evaluation metrics and visualizations
- 💾 Model persistence and reproducibility features
- 🖥️ Command-line interface for easy operation

## Installation

```bash
pip install vaganboostktf
```

For development installation:
```bash
git clone https://github.com/yourusername/vaganboostktf.git
cd vaganboostktf
pip install -e .
```

## Usage

### Basic Python API

```python
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import dill
from vaganboostktf.data_preprocessor import DataPreprocessor
from vaganboostktf.cvae import CVAE
from vaganboostktf.cgan import CGAN
from vaganboostktf.lgbm_tuner import LightGBMTuner
from vaganboostktf.trainer import HybridModelTrainer
from vaganboostktf.utils import plot_confusion_matrix, plot_roc_curves, plot_pr_curves

# Load and prepare data
df = pd.read_csv("Input.csv")

# Identify features and target
feature_columns = [f"ClE{i}" for i in range(1, 26)]  # 25 features
target_column = "label"

# Initialize data preprocessor
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.prepare_data(
    df,
    feature_columns=[f"ClE{i}" for i in range(1, 26)],
    target_column="label"
)

# Prepare scaled datasets
X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.prepare_data(
    df,
    feature_columns=feature_columns,
    target_column=target_column
)

# Initialize and train hybrid model
trainer = HybridModelTrainer(config={
    'num_classes': 4,
    'cvae_params': {
        'input_dim': 25,
        'latent_dim': 8,
        'num_classes': 4,
        'learning_rate': 0.01
    },
    'cgan_params': {
        'input_dim': 25,
        'latent_dim': 8,
        'num_classes': 4,
        'generator_lr': 0.0002,
        'discriminator_lr': 0.0002
    },
    'model_dir': 'trained_models',
    'cvae_epochs': 10,
    'cgan_epochs': 10,
    'lgbm_iterations': 10,
    'samples_per_class': 50
})
trainer.training_loop(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    iterations=5
)

print("Training completed! Best models saved in 'trained_models' directory")

# Visualization of the metrics and results

# Load trained LightGBM model
lgbm_model = joblib.load("trained_models/lgbm_model.pkl")

# Get predictions
y_pred = lgbm_model.predict(X_test_scaled)
y_proba = lgbm_model.predict_proba(X_test_scaled)

# Define class names
class_names = [str(i) for i in np.unique(y_test)]

# Plot and save confusion matrix
conf_matrix_fig = plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True)
conf_matrix_fig.savefig("trained_models/confusion_matrix.png")

# Plot and save ROC curves
roc_curve_fig = plot_roc_curves(y_test, y_proba, classes=class_names)
roc_curve_fig.savefig("trained_models/roc_curves.png")

# Plot and save Precision-Recall curves
pr_curve_fig = plot_pr_curves(y_test, y_proba, classes=class_names)
pr_curve_fig.savefig("trained_models/pr_curves.png")

# Extract feature importances and corresponding feature names
feature_importance = lgbm_model.feature_importances_
feature_names = [f"ClE{i}" for i in range(1, 26)]  # 25 features

# Create a DataFrame for sorting
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)  # Top 10 features

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='Blues_r')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 10 Most Important Features")
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Save and show plot
plt.savefig("trained_models/top_10_features.png", bbox_inches='tight')
plt.show()
```

## Architecture

```mermaid
graph TD
    A[Input Data] --> B[Data Preprocessing]
    B --> C[CVAE Training]
    B --> D[CGAN Training]
    C --> E[Synthetic Data Generation]
    D --> E
    E --> F[Data Augmentation]
    F --> G[LightGBM Training]
    G --> H[Evaluation]
    H --> I[Model Persistence]
```

## Key Components

- **Conditional VAE**: Generates class-conditioned synthetic samples
- **Conditional GAN**: Produces additional class-specific synthetic data
- **LightGBM Tuner**: Optimized gradient boosting with automated hyperparameter search
- **Hybrid Trainer**: Orchestrates iterative training process

```mermaid
graph TD;
    A[Raw Data (CSV)] --> B[DataPreprocessor];
    B --> C[Preprocessed Data];
    C --> D[CVAE];
    C --> E[CGAN];
    D --> F[Synthetic Data (CVAE)];
    E --> G[Synthetic Data (CGAN)];
    F --> H[Combined Real & Synthetic Data];
    G --> H;
    H --> I[LightGBM Classifier Pipeline];
    I --> J[Evaluation (Confusion Matrix, ROC, PR Curves)];
    J --> K[Best Models Saved];

## Configuration

Default parameters can be modified through:
- Command-line arguments
- JSON configuration files
- Python API parameters

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

=============================================<p align="Center">![ALI BAVARCHIEE](https://teal-broad-gecko-650.mypinata.cloud/ipfs/bafkreif332ra4lrdjfzaiowc2ikhl65uflok37e7hmuxomwpccracarqpy)</p>=============================================
=====
| https://github.com/AliBavarchee/ |
----
| https://www.linkedin.com/in/ali-bavarchee-qip/ |

