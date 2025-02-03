# VaganBoost: Hybrid VAE-GAN + LightGBM for Advanced Classification 0.7.7

![VAGANBoost Logo](https://teal-broad-gecko-650.mypinata.cloud/ipfs/bafybeicfrxxm3kmvh4sswyqtcueqj3rxx3sknwnypriu7vfg2umzkwkihu)

## Introduction
VAGANBoost is a hybrid generative model combining Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN) with boosting techniques to enhance high-energy gamma-ray analysis.

## Outlines
- Implements VAE+GAN and LGBM models
- Designed for high-energy physics applications
- Utilizes deep learning and gradient boosting techniques





## Key Features

- **Hybrid Architecture**: Combines deep generative models with gradient boosting
- **VAE-GAN Integration**: Joint latent space learning for improved feature representation
- **LightGBM Classifier**: State-of-the-art gradient boosting for final classification
- **Automatic Feature Fusion**: Combines VAE latent features with GAN-generated features
- **Visualization Tools**: Built-in metrics visualization and feature analysis
- **PyTorch Backend**: GPU-accelerated training with seamless CUDA support


## Key Features Table

| Feature | Description | Benefit |
|---------|-------------|---------|
| **VAE-GAN Fusion** | Combines reconstruction power of VAEs with GANs' generative capabilities | Enhanced feature learning |
| **LightGBM Integration** | Gradient boosting on learned features | Superior classification performance |
| **Automatic GPU Support** | Seamless CUDA integration | Faster training on supported hardware |
| **Dynamic Feature Fusion** | Combines latent and generated features | Improved representation learning |
| **Visualization Suite** | Built-in metrics plotting | Easy model evaluation |

## Troubleshooting

**Common Issues:**
1. **CUDA Out of Memory**: Reduce batch size or input dimensions
2. **Poor Classification Performance**: 
   - Increase VAE latent dimensions
   - Adjust GAN-LightGBM feature ratio
3. **Training Instability**:
   ```python
   model = VaganBoost(
       ...,
       vae_kl_weight=0.5,  # Adjust KL loss weight
       gan_gp_weight=10.0  # Add gradient penalty
   )

## Installation

### Prerequisites
- Python 3.6+
- NVIDIA GPU (recommended) with CUDA 11.0+

### Install via pip
```bash
pip install vaganboost
```

### From source
```bash
git clone https://github.com/AliBavarchee/vaganboost.git
cd vaganboost
pip install -e .
```

## Quick Start

### Basic Usage
```python
from vaganboost import VaganBoost, load_data, split_data, normalize_data

# Prepare data
X, y = load_data("data.csv", target_column="label")
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
X_train_norm, X_test_norm = normalize_data(X_train, X_test)

# Initialize model
model = VaganBoost(
    vae_input_dim=X_train_norm.shape[1],
    vae_latent_dim=64,
    gan_input_dim=100,
    num_class=4,
    device="cuda"
)

# Train components
model.train_vae(X_train_norm, epochs=100)
model.train_gan(X_train_norm, epochs=50)
model.train_lgbm(X_train_norm, y_train)

# Evaluate
accuracy = model.evaluate(X_test_norm, y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

### Advanced Configuration
```python
# Custom LightGBM parameters
lgbm_params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'num_leaves': 63,
    'learning_rate': 0.1,
    'feature_fraction': 0.7
}

model = VaganBoost(
    vae_input_dim=128,
    vae_latent_dim=64,
    gan_input_dim=100,
    num_class=4,
    lgbm_params=lgbm_params,
    device="cuda"
)
```

## Documentation

### Core Components
| Module | Description |
|--------|-------------|
| `data_utils` | Data loading, splitting, and normalization |
| `models` | VAE, GAN, and LightGBM implementations |
| `train` | Joint training procedures |
| `utils` | Visualization and evaluation tools |

## Dependencies
See `requirements.txt` for required packages.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License.

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

Contact
Ali Bavarchee - ali.bavarchee@gmail.com

Project Link: https://github.com/AliBavarchee/vaganboost


=============================================<p align="Center">![ALI BAVARCHIEE](https://teal-broad-gecko-650.mypinata.cloud/ipfs/bafkreif332ra4lrdjfzaiowc2ikhl65uflok37e7hmuxomwpccracarqpy)</p>=============================================
=====
----
| https://www.linkedin.com/in/ali-bavarchee-qip/ |
