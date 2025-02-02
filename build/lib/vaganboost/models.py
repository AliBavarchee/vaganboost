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
from .data_utils import load_data, split_data, normalize_data, to_tensor, create_dataloader, preprocess_data
#from .train import train_vaganboost
#from .utils import save_model, load_model, normalize_data, split_data, calculate_accuracy, get_latent_features, early_stopping_train



# Normalize data function
def normalize_data(X):
    return (X - X.min()) / (X.max() - X.min())

# Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_mu = nn.Linear(256, latent_dim)
        self.fc3_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.fc4 = nn.Linear(latent_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, input_dim)

    def encode(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc3_mu(x)
        logvar = self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = torch.relu(self.fc4(z))
        z = torch.relu(self.fc5(z))
        return self.fc6(z)  # No sigmoid for real-valued data

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def kl_divergence(self, mu, logvar):
        return 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# VAE loss function using MSELoss
def vae_loss(recon_x, x, mu, logvar):
    MSE = nn.MSELoss(reduction='sum')(recon_x, x)
    KL = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE - KL  # KL is negative in ELBO
    

#Training function
# def train_vae(self, X_train, epochs=50, batch_size=32, learning_rate=1e-3):
    # dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    # dataloader = torch.utils.data.DataLoader(
        # dataset, 
        # batch_size=batch_size,  # Use the parameter here
        # shuffle=True
    # )
    # X, y = load_data(args.data_path)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_test = normalize_data(X_train), normalize_data(X_test)
    
    # train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    # test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # vae = VAE(input_dim=X_train.shape[1], latent_dim=args.latent_dim).to(args.device)
    # optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    
    # for epoch in range(args.num_epochs):
        # vae.train()
        # running_loss = 0.0
        
        # for data, _ in train_loader:
            # data = data.to(args.device)
            # optimizer.zero_grad()
            # recon_batch, mu, logvar = vae(data)
            # loss = vae_loss(recon_batch, data, mu, logvar)
            # loss.backward()
            # optimizer.step()
            # running_loss += loss.item()
        
        # print(f"Epoch [{epoch+1}/{args.num_epochs}] | VAE Loss: {running_loss/len(train_loader):.4f}")
    
    # return vae

# Generative Adversarial Network (GAN)
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, output_dim)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(z))

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

# LightGBM Model Wrapper
class LightGBMModel:
    def __init__(self, num_class=None, params=None):
        self.params = params if params is not None else {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 4,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
        }
        self.model = None

    def fit(self, X_train, y_train, num_boost_round=100):
        train_data = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(self.params, train_data, num_boost_round=num_boost_round)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_class(self, X_test):
        return np.argmax(self.predict(X_test), axis=1)

    def get_importance(self):
        return self.model.feature_importance()

class VAE_GAN(nn.Module):
    def __init__(self, vae, discriminator):
        super(VAE_GAN, self).__init__()
        self.vae = vae
        self.discriminator = discriminator

    def forward(self, x):
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        recon_x = self.vae.decode(z)
        disc_output = self.discriminator(z)
        return recon_x, disc_output, mu, logvar

# Combined Model (VAE + GAN + LightGBM)
class VaganBoost:
    def __init__(self, vae_input_dim, vae_latent_dim, gan_input_dim, gan_output_dim, num_class, lgbm_params=None, device="cpu"):
        self.device = device
        self.vae = VAE(vae_input_dim, vae_latent_dim).to(device)
        self.generator = Generator(gan_input_dim, gan_output_dim).to(device)
        self.discriminator = Discriminator(gan_output_dim).to(device)
        self.num_class = num_class
        self.lgbm = LightGBMModel(num_class=num_class, params=lgbm_params)

    def train_vae(self, X_train, epochs=50, batch_size=32, learning_rate=1e-3):
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)
        self.vae.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x_batch = batch[0].to(self.device)
                optimizer.zero_grad()

                recon_x, mu, logvar = self.vae(x_batch)
                loss = vae_loss(recon_x, x_batch, mu, logvar)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    def train_gan(self, train_loader, epochs=10, lr=1e-3):
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        for epoch in range(epochs):
            for batch_idx, (data, _) in enumerate(train_loader):
                real_data = data.view(data.size(0), -1).to(self.device)
                batch_size = real_data.size(0)

                # Train Discriminator
                optimizer_d.zero_grad()
                z = torch.randn(batch_size, 100).to(self.device)
                fake_data = self.generator(z)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                real_loss = torch.nn.BCELoss()(self.discriminator(real_data), real_labels)
                fake_loss = torch.nn.BCELoss()(self.discriminator(fake_data.detach()), fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                g_loss = torch.nn.BCELoss()(self.discriminator(fake_data), real_labels)
                g_loss.backward()
                optimizer_g.step()

            print(f'Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    def train_lgbm(self, X_train, y_train):
        self.lgbm.fit(X_train, y_train)

    def predict(self, X_test):
        with torch.no_grad():
            vae_latent = self.vae.encode(torch.tensor(X_test, dtype=torch.float32).to(self.device))[0].cpu().numpy()
            gan_output = self.generator(torch.randn(X_test.shape[0], 100).to(self.device)).cpu().numpy()

        combined_features = np.concatenate([vae_latent, gan_output], axis=1)
        return self.lgbm.predict_class(combined_features)

