# vaganboostktf: VAE-GAN Boost by TF

=============================================<p align="Center">![vaganboostktf](https://teal-broad-gecko-650.mypinata.cloud/ipfs/bafybeiew72v2a7okbxdawl6febkahxeulpxkjylizttacz5xfreq4d675q)</p>=============================================
=====

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# VaganBoostKFT

VaganBoostKFT is a hybrid machine learning package that integrates generative modeling (using CVAE and CGAN) with an advanced LightGBM classifier pipeline. It provides robust data preprocessing, custom sampling strategies for imbalanced data, automated hyperparameter tuning (including dimensionality reduction via PCA, LDA, or TruncatedSVD), and built-in visualization of model evaluation metrics (confusion matrices, ROC curves, and precision-recall curves).


## Features

- 🧬 Hybrid architecture combining generative and discriminative models
- ⚖️ Effective handling of class imbalance through synthetic data generation
- 🔄 Iterative training process with automatic model refinement
- 📊 Comprehensive evaluation metrics and visualizations
- 💾 Model persistence and reproducibility features
- 🖥️ Command-line interface for easy operation

## Installation

Install the required dependencies:

```bash
pip install dill
pip install dask[dataframe]
pip install umap-learn
```

Additional dependencies (if not already installed) include:
- scikit-learn
- imbalanced-learn
- lightgbm
- tensorflow
- seaborn
- matplotlib
- joblib

```bash
pip install vaganboostktf
```

For development installation:
```bash
git clone https://github.com/yourusername/vaganboostktf.git
cd vaganboostktf
pip install -e .
```

## Modules

- **data_preprocessor.py:** Provides consistent data preprocessing (scaling, handling missing values, and encoding).
- **trainer.py:** Orchestrates the hybrid training workflow combining generative models (CVAE, CGAN) and the LightGBM classifier.
- **lgbm_tuner.py:** Implements hyperparameter tuning for the advanced LightGBM pipeline.
- **lgbm_classifier.py:** Contains the full LightGBM classifier pipeline that integrates preprocessing, feature selection, dimensionality reduction, SMOTE balancing (with custom sampling strategies), and hyperparameter tuning.
- **utils.py:** Provides utility functions for visualization (confusion matrix, ROC curves, precision-recall curves) and helper classes like `DecompositionSwitcher`.

## Usage Example

Below is a sample script demonstrating how to use VaganBoostKFT:

```python
import pandas as pd
import numpy as np
from vaganboostktf.data_preprocessor import DataPreprocessor
from vaganboostktf.trainer import HybridModelTrainer
from vaganboostktf.lgbm_tuner import LightGBMTuner
from vaganboostktf.utils import plot_confusion_matrix, plot_roc_curves, plot_pr_curves

# ===========================
# 1. Load and Prepare Data
# ===========================
df = pd.read_csv("input.csv")

# Identify features and target
feature_columns = [col for col in df.columns if col != "label"]
target_column = "label"

# Initialize data preprocessor
preprocessor = DataPreprocessor()

# Preprocess data (handling missing values, scaling, encoding)
X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.prepare_data(
    df, feature_columns, target_column
)

# ===========================
# 2. Train Hybrid Model (CVAE, CGAN + LGBM)
# ===========================
trainer = HybridModelTrainer(config={
    'num_classes': 4,
    'cvae_params': {
        'input_dim': 25,
        'latent_dim': 10,
        'num_classes': 4,
        'learning_rate': 0.01
    },
    'cgan_params': {
        'input_dim': 25,
        'latent_dim': 10,
        'num_classes': 4,
        'generator_lr': 0.0002,
        'discriminator_lr': 0.0002
    },
	'input_path': 'input.csv',
    'model_dir': 'trained_models',
    'cvae_epochs': 100,
    'cgan_epochs': 100,
    'lgbm_iterations': 100,
    'samples_per_class': 50
})

# Run hybrid training (Generative + LGBM)
trainer.training_loop(X_train_scaled, y_train, X_test_scaled, y_test, iterations=5)
print("\nHybrid training completed! Models saved in 'trained_models/'")

# ===========================
# 3. Load and Evaluate LightGBM Model
# ===========================
lgbm_tuner = LightGBMTuner(input_path="input.csv", output_path="trained_models")

# Train the LightGBM model (already tuned within `lgbm_classifier`)
lgbm_tuner.tune()

# Predict on test data
y_pred = lgbm_tuner.predict(X_test_scaled)
y_proba = lgbm_tuner.predict_proba(X_test_scaled)

# ===========================
# 4. Visualize Results
# ===========================
class_names = [str(i) for i in np.unique(y_test)]

# Plot Confusion Matrix
conf_matrix_fig = plot_confusion_matrix(y_test, y_pred, class_names, normalize=True)
conf_matrix_fig.savefig("trained_models/confusion_matrix.png")

# Plot ROC Curves
roc_curve_fig = plot_roc_curves(y_test, y_proba, class_names)
roc_curve_fig.savefig("trained_models/roc_curve.png")

# Plot Precision-Recall Curves
pr_curve_fig = plot_pr_curves(y_test, y_proba, class_names)
pr_curve_fig.savefig("trained_models/pr_curve.png")

print("\nEvaluation completed! Check 'trained_models/' for plots.")
```

## Architecture

```mermaid
graph TD
    A["Raw Data (CSV)"] --> B["DataPreprocessor"]
    B --> C["Preprocessed Data"]
    C --> D["CVAE"]
    C --> E["CGAN"]
    D --> F["Synthetic Data (CVAE)"]
    E --> G["Synthetic Data (CGAN)"]
    F --> H["Combined Real & Synthetic Data"]
    G --> H
    H --> I["LightGBM Classifier Pipeline"]
    I --> J["Evaluation (Confusion Matrix, ROC, PR Curves)"]
    J --> K["Best Models Saved"]
```

## Key Components

- **Conditional VAE**: Generates class-conditioned synthetic samples
- **Conditional GAN**: Produces additional class-specific synthetic data
- **LightGBM Tuner**: Optimized gradient boosting with automated hyperparameter search
- **Hybrid Trainer**: Orchestrates iterative training process


## Additional Information

- **Hybrid Workflow:** The training loop in `trainer.py` first trains generative models (CVAE and CGAN) to create synthetic data, which is then combined with real data to train a robust LightGBM classifier.
- **Custom Sampling Strategies:** `lgbm_classifier.py` integrates a function to generate sampling strategies for SMOTE to address severe class imbalance.
- **Visualization:** Evaluation plots are generated and saved in the output directory to help assess model performance.


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

