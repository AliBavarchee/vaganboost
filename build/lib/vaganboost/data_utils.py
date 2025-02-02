import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# ---- Variational Autoencoder (VAE) ----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
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
#from .models import VAE, Generator, Discriminator, LightGBMModel, VaganBoost
#from .train import train_vaganboost
#from .utils import save_model, load_model, normalize_data, split_data, calculate_accuracy, get_latent_features, early_stopping_train


def load_data(file_path, target_column):
    """
    Load data from a CSV file and split it into features (X) and target (y).
    Args:
        file_path (str): Path to the CSV file.
        target_column (str): The column name for the target variable.
    Returns:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
    """
    # Load the dataset
    #data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    df = pd.read_csv(file_path)
    
    # Split into features and target
    #X = data[:, :-1]  # assuming the last column is the target
    #y = data[:, -1]   # target column (for classification/regression)
    X = df.drop(target_column, axis=1).values  # Features
    y = df[target_column].values               # Target
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    Args:
        X (numpy.ndarray): Features matrix.
        y (numpy.ndarray): Target vector.
        test_size (float): Proportion of the data to be used as the test set.
        random_state (int): Random seed for reproducibility.
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    """
    Normalize the data using StandardScaler.
    Args:
        X_train (numpy.ndarray): Training feature matrix.
        X_test (numpy.ndarray): Testing feature matrix.
    Returns:
        X_train_normalized, X_test_normalized
    """
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized

def to_tensor(X, y=None, device='cpu'):
    """
    Convert data to PyTorch tensors.
    Args:
        X (numpy.ndarray): Features matrix.
        y (numpy.ndarray or None): Target vector. If None, only X will be converted.
        device (str): Device to load tensors to ('cpu' or 'cuda').
    Returns:
        X_tensor (torch.Tensor): Feature tensor.
        y_tensor (torch.Tensor or None): Target tensor.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)
        return X_tensor, y_tensor
    else:
        return X_tensor

def create_dataloader(X, y, batch_size=64, shuffle=True, device='cpu'):
    """
    Create a PyTorch DataLoader from the dataset.
    Args:
        X (numpy.ndarray): Features matrix.
        y (numpy.ndarray): Target vector.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.
        device (str): Device to load tensors to ('cpu' or 'cuda').
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    X_tensor, y_tensor = to_tensor(X, y, device=device)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def preprocess_data(file_path, target_column, test_size=0.2, batch_size=64, device='cpu'):
    """
    Full data preprocessing pipeline: load, split, normalize, and create DataLoader.
    Args:
        file_path (str): Path to the data file (CSV).
        target_column (str): Column name of the target variable.
        test_size (float): Proportion of the data to be used as the test set.
        batch_size (int): Batch size for DataLoader.
        device (str): Device to load tensors to ('cpu' or 'cuda').
    Returns:
        train_loader, test_loader
    """
    # Load data
    X, y = load_data(file_path, target_column)
    
    # Split data into training and testing
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    
    # Normalize data
    X_train_normalized, X_test_normalized = normalize_data(X_train, X_test)
    
    # Create DataLoader
    train_loader = create_dataloader(X_train_normalized, y_train, batch_size, device=device)
    test_loader = create_dataloader(X_test_normalized, y_test, batch_size, device=device)
    
    return train_loader, test_loader

