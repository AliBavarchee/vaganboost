# VaganBoostKFT

VaganBoostKFT is a hybrid machine learning package that integrates generative modeling (using CVAE and CGAN) with an advanced LightGBM classifier pipeline. It provides robust data preprocessing, custom sampling strategies for imbalanced data, automated hyperparameter tuning (including dimensionality reduction via PCA, LDA, or TruncatedSVD), and built-in visualization of model evaluation metrics (confusion matrices, ROC curves, and precision-recall curves).

## Features

- **Data Preprocessing:** Handle missing values, scale features, and encode categorical variables with the `DataPreprocessor` module.
- **Generative Modeling:** Train CVAE and CGAN models to augment your training data.
- **LightGBM Classifier Pipeline:** Incorporates feature selection, custom dimensionality reduction, and SMOTE-based balancing with custom sampling strategies.
- **Hyperparameter Tuning:** Uses RandomizedSearchCV to automatically tune the pipeline parameters.
- **Visualization:** Generates and saves evaluation plots (confusion matrix, ROC curves, and precision-recall curves) via the utilities in `utils.py`.

## Architecture

Below is an overview of the hybrid generative model architecture implemented in VaganBoostKFT:

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
