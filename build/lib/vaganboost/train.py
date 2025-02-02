import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from torch import nn

# Import key modules from vaganboost
from .data_utils import load_data, split_data, normalize_data, to_tensor, create_dataloader, preprocess_data
from .models import VAE, Generator, Discriminator, LightGBMModel, VaganBoost, VAE_GAN
#from .utils import save_model, load_model, normalize_data, split_data, calculate_accuracy, get_latent_features, early_stopping_train


import torch
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import os
from sklearn.manifold import TSNE
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from sklearn.manifold import TSNE
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import sys
import glob
import json
import joblib
from rich.jupyter import print
from tqdm import tqdm
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from torchvision.utils import save_image
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import loguniform
import time
import csv
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from torchvision.utils import save_image
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import loguniform
import time
import csv

import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report


%matplotlib inline
import imageio
#import plotly.express as px


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
from sklearn import metrics, manifold
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers, models
#import keras_tuner as kt
from lightgbm import LGBMClassifier
import logging
from multiprocessing import Manager, Process
import logging
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report


%matplotlib inline
import imageio
#import plotly.express as px


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
from sklearn import metrics, manifold
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers, models
#import keras_tuner as kt
from lightgbm import LGBMClassifier
import logging
from multiprocessing import Manager, Process
import logging

import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from torch import nn

# Import key modules from vaganboost
from .data_utils import load_data, split_data, normalize_data, to_tensor, create_dataloader, preprocess_data
from .models import VAE, Generator, Discriminator, LightGBMModel, VaganBoost
#from .utils import save_model, load_model, normalize_data, split_data, calculate_accuracy, get_latent_features, early_stopping_train


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import label_binarize

# Import key modules from vaganboost
from .data_utils import load_data, split_data, normalize_data, to_tensor, create_dataloader, preprocess_data
from .models import VAE, Generator, Discriminator, LightGBMModel, VaganBoost, VAE_GAN
from .utils import save_model, load_model, normalize_data, split_data, calculate_accuracy, get_latent_features, early_stopping_train


def train_vaganboost(args):
    """
    Train a VAE-GAN model followed by a LightGBM classifier.
    """
    # Load dataset
    X, y = load_data(args.data_path, args.target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model setup
    vae = VAE(input_dim=X_train.shape[1], latent_dim=args.latent_dim).to(args.device)
    discriminator = Discriminator(input_dim=args.latent_dim).to(args.device)
    model = VAE_GAN(vae, discriminator).to(args.device)
    
    vae_optimizer = optim.Adam(vae.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    vae_criterion = nn.MSELoss()
    disc_criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    
    best_val_loss = float('inf')
    early_stop_count = 0

    train_losses = []  # Initialize list to store training losses
    val_losses = []  # Initialize list to store validation losses

    for epoch in range(args.num_epochs):
        vae.train()
        discriminator.train()
        running_vae_loss = 0.0
        running_disc_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(args.device)
            labels = labels.to(args.device)
            batch_size = data.size(0)
            
            # Train VAE
            vae_optimizer.zero_grad()
            recon_batch, mu, log_var = vae(data)
            vae_loss = vae_criterion(recon_batch, data) + vae.kl_divergence(mu, log_var)
            vae_loss.backward()
            vae_optimizer.step()
            
            # Train Discriminator
            discriminator_optimizer.zero_grad()
            latent, _ = vae.encode(data)
            real_labels = torch.ones(batch_size, 1).to(args.device)
            fake_labels = torch.zeros(batch_size, 1).to(args.device)
            
            real_loss = disc_criterion(discriminator(latent), real_labels)
            fake_loss = disc_criterion(discriminator(torch.randn_like(latent)), fake_labels)
            disc_loss = (real_loss + fake_loss) / 2
            disc_loss.backward()
            discriminator_optimizer.step()
            
            running_vae_loss += vae_loss.item()
            running_disc_loss += disc_loss.item()
        
        train_losses.append(running_vae_loss / len(train_loader))  # Store training loss
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | VAE Loss: {running_vae_loss / len(train_loader):.4f} | Discriminator Loss: {running_disc_loss / len(train_loader):.4f}")
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(args.device)
                recon_batch, mu, log_var = vae(data)
                val_loss += (vae_criterion(recon_batch, data) + vae.kl_divergence(mu, log_var)).item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)  # Store validation loss
        print(f"Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            save_model(vae, os.path.join(args.model_save_path, "vae_best.pth"))
            save_model(discriminator, os.path.join(args.model_save_path, "discriminator_best.pth"))
            print("Model saved with best validation loss!")
        else:
            early_stop_count += 1

        if early_stop_count > args.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Extract latent space representations
    vae.eval()
    with torch.no_grad():
        latent_train = vae.encode(torch.tensor(X_train).to(args.device))[0].cpu().numpy()
        latent_test = vae.encode(torch.tensor(X_test).to(args.device))[0].cpu().numpy()
    
    # Train LightGBM on latent space
    lgb_train = lgb.Dataset(latent_train, label=y_train)
    lgb_test = lgb.Dataset(latent_test, label=y_test, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    gbm = lgb.train(params, lgb_train, valid_sets=[lgb_test], num_boost_round=100)
    
    # Perform early stopping
    early_stopping_result = gbm.best_iteration
    
    # Evaluate model
    y_pred = gbm.predict(latent_test, num_iteration=early_stopping_result)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred_binary == y_test)
    print(f"LightGBM Classification Accuracy: {accuracy:.4f}")

    return train_losses, val_losses, y_test, y_pred_binary  # Return results for visualization

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the VAE-GAN with LightGBM model")
    
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--latent_dim', type=int, default=64, help="Dimension of latent space")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for optimizers")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument('--model_save_path', type=str, default="./models", help="Directory to save the trained models")
    
    args = parser.parse_args()
    os.makedirs(args.model_save_path, exist_ok=True)
    
    train_losses, val_losses, y_test, y_pred_binary = train_vaganboost(args)

    # Visualization
    import matplotlib.pyplot as plt

    def plot_losses(train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()

    def plot_classification_results(y_test, y_pred_binary):
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(y_test)), y_test, label='True Labels')
        plt.scatter(range(len(y_test)), y_pred_binary, alpha=0.5, label='Predicted Labels')
        plt.xlabel('Sample Index')
        plt.ylabel('Label')
        plt.title('True vs Predicted Labels')
        plt.legend()
        plt.show()

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses)

    # Plot the classification results
    plot_classification_results(y_test, y_pred_binary)
