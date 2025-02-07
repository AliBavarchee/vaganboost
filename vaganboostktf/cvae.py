import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple
import pickle
import numpy as np
#from .data_preprocessor import DataPreprocessor

class CVAE(Model):
    """Conditional Variational Autoencoder for synthetic data generation"""
    
    def __init__(self, input_dim: int = 25, latent_dim: int = 8, num_classes: int = 4, learning_rate: float = 0.001, **kwargs):
        """
        Initialize CVAE model
        
        Args:
            input_dim (int): Dimension of input features
            latent_dim (int): Dimension of latent space
            num_classes (int): Number of class labels
        """
        super(CVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # Encoder architecture
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim + num_classes,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(2 * latent_dim)  # Both mean and log variance
        ])

        # Decoder architecture
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim + num_classes,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Forward pass through the CVAE
        
        Args:
            inputs (tuple): (input features, labels)
            
        Returns:
            tf.Tensor: Reconstructed input
        """
        x, y = inputs
        y_onehot = tf.one_hot(y, depth=self.num_classes)
        
        # Concatenate features with class labels
        x_y = tf.concat([x, y_onehot], axis=1)
        
        # Encode to latent space parameters
        encoder_output = self.encoder(x_y)
        mean, log_var = tf.split(encoder_output, 2, axis=1)
        
        # Reparameterization trick
        z = self.reparameterize(mean, log_var)
        
        # Concatenate latent vector with labels
        z_y = tf.concat([z, y_onehot], axis=1)
        
        # Decode to reconstructed input
        x_recon = self.decoder(z_y)
        
        # Add KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_var - tf.square(mean) - tf.exp(log_var)
        )
        self.add_loss(kl_loss)
        
        return x_recon

    def reparameterize(self, 
                      mean: tf.Tensor, 
                      log_var: tf.Tensor) -> tf.Tensor:
        """
        Reparameterization trick for sampling from latent space
        
        Args:
            mean (tf.Tensor): Mean of latent distribution
            log_var (tf.Tensor): Log variance of latent distribution
            
        Returns:
            tf.Tensor: Sampled latent vector
        """
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(log_var * 0.5) + mean

    def generate(self, 
                class_label: int, 
                num_samples: int = 100) -> np.ndarray:
        """
        Generate synthetic samples for specified class
        
        Args:
            class_label (int): Target class for generation
            num_samples (int): Number of samples to generate
            
        Returns:
            np.ndarray: Generated samples
        """
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        y = tf.fill((num_samples,), class_label)
        y_onehot = tf.one_hot(y, depth=self.num_classes)
        z_y = tf.concat([z, y_onehot], axis=1)
        return self.decoder(z_y).numpy()

    def get_config(self) -> dict:
        """Get model configuration for serialization"""
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'num_classes': self.num_classes
        }

    @classmethod
    def from_config(cls, config: dict):
        """Create model from configuration"""
        return cls(**config)