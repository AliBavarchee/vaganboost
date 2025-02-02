import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

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

import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report

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

# ---- Variational Autoencoder (VAE) ----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
# Import key modules from vaganboost
from .models import VAE, Generator, Discriminator, LightGBMModel, VaganBoost, VAE_GAN
#from .train import train_vaganboost
from .data_utils import load_data, split_data, normalize_data, to_tensor, create_dataloader, preprocess_data




# Function to save a model's state dictionary (weights)
def save_model(model, path):
    """
    Save the model's state dictionary to a file.
    This stores the learned parameters so that the model can be reloaded later.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load a model from a saved state dictionary
def load_model(model, path):
    """
    Load the model's state dictionary from a file.
    This restores the model's learned parameters for use.
    """
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode (important for dropout, batchnorm, etc.)
    print(f"Model loaded from {path}")
    return model

# Function to normalize a numpy array (common preprocessing step for neural networks)
def normalize_data(X):
    """
    Normalize the data to have a mean of 0 and a standard deviation of 1.
    This is often done so that the model training process isn't biased by variable scales.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    normalized_data = (X - mean) / std
    return normalized_data

# Function to split the data into training and testing sets
def split_data(X, y, test_size=0.2):
    """
    Split the data into training and testing sets.
    Think of this like dividing a deck of cards into two piles: one to train the model (training),
    and one to check how well it learned (testing).
    """
    total_samples = X.shape[0]
    test_samples = int(total_samples * test_size)
    train_samples = total_samples - test_samples

    # Shuffle the data first
    indices = np.random.permutation(total_samples)
    X_shuffled, y_shuffled = X[indices], y[indices]

    # Split
    X_train, X_test = X_shuffled[:train_samples], X_shuffled[train_samples:]
    y_train, y_test = y_shuffled[:train_samples], y_shuffled[train_samples:]
    
    return X_train, X_test, y_train, y_test

# Function to calculate accuracy for model predictions
def calculate_accuracy(predictions, labels):
    """
    Calculate the accuracy of the model's predictions.
    It's like checking how many times the model 'guessed' the right card from a deck.
    """
    return accuracy_score(labels, predictions)

# Function to generate latent features from a trained VAE model
def get_latent_features(model, data_loader, device='cpu'):
    """
    Extract the latent representations of the data using a trained VAE model.
    This step is like compressing your data into a smaller, more informative format.
    """
    model.eval()  # Set to evaluation mode to disable things like dropout
    latent_features = []

    with torch.no_grad():  # We don't need to calculate gradients here
        for data in data_loader:
            inputs = data[0].to(device)
            latent, _ = model.encode(inputs)  # Get the latent features from the encoder
            latent_features.append(latent.cpu().numpy())  # Move to CPU if necessary
    
    return np.concatenate(latent_features, axis=0)

# Function to train a model with early stopping based on validation performance
def early_stopping_train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10, device='cpu'):
    """
    Train the model and stop early if the validation performance stops improving.
    This is like running a race and stopping if you stop seeing improvements in your speed after a while.
    """
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, targets = data[0].to(device), data[1].to(device)

            # Zero the gradients, perform the forward pass, compute the loss, and backpropagate
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the counter if validation loss improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break  # Stop training early

    return model
