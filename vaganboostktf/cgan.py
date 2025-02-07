import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Optional
import numpy as np
import pickle
import os
#from .data_preprocessor import DataPreprocessor

class CGAN:
    """Conditional Generative Adversarial Network for synthetic data generation"""
    
    def __init__(self, 
                 input_dim: int = 25,
                 latent_dim: int = 8,
                 num_classes: int = 4,
                 generator_lr: float = 0.0002,
                 discriminator_lr: float = 0.0002):
        """
        Initialize CGAN model
        
        Args:
            input_dim (int): Dimension of input features
            latent_dim (int): Dimension of latent space
            num_classes (int): Number of class labels
            generator_lr (float): Learning rate for generator
            discriminator_lr (float): Learning rate for discriminator
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        # Build components
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Compile models
        self._compile_models()

    def _build_generator(self) -> Model:
        """Construct generator network"""
        noise_input = layers.Input(shape=(self.latent_dim,))
        label_input = layers.Input(shape=(self.num_classes,))
        
        x = layers.concatenate([noise_input, label_input])
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(self.input_dim)(x)
        
        return Model(inputs=[noise_input, label_input], outputs=x)

    def _build_discriminator(self) -> Model:
        """Construct discriminator network"""
        data_input = layers.Input(shape=(self.input_dim,))
        label_input = layers.Input(shape=(self.num_classes,))
        
        x = layers.concatenate([data_input, label_input])
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=[data_input, label_input], outputs=x)

    def _compile_models(self):
        """Configure model optimizers and loss functions"""
        optimizer_gen = tf.keras.optimizers.Adam(self.generator_lr, beta_1=0.5)
        optimizer_dis = tf.keras.optimizers.Adam(self.discriminator_lr, beta_1=0.5)

        # Discriminator
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer_dis,
            metrics=['accuracy']  # Add accuracy metric
        )

        # Combined model (stacked generator and discriminator)
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(self.num_classes,))
        generated_data = self.generator([noise, label])
        validity = self.discriminator([generated_data, label])
        
        self.combined = Model([noise, label], validity)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer_gen
        )

    def train(self, 
             X: np.ndarray, 
             y: np.ndarray,
             epochs: int = 1000,
             batch_size: int = 32,
             sample_interval: int = 100,
             output_dir: str = "cgan_models"):
        """
        Train CGAN model
        
        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            sample_interval (int): Interval for saving model checkpoints
            output_dir (str): Directory to save model checkpoints
        """
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        os.makedirs(output_dir, exist_ok=True)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select random real samples
            idx = np.random.randint(0, X.shape[0], batch_size)
            real_data = X[idx]
            real_labels = tf.one_hot(y[idx], depth=self.num_classes)

            # Generate fake samples
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_labels = tf.one_hot(
                np.random.randint(0, self.num_classes, batch_size),
                depth=self.num_classes
            )
            fake_data = self.generator.predict([noise, gen_labels], verbose=0)

            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(
                [real_data, real_labels], valid
            )
            d_loss_fake = self.discriminator.train_on_batch(
                [fake_data, gen_labels], fake
            )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_labels = tf.one_hot(
                np.random.randint(0, self.num_classes, batch_size),
                depth=self.num_classes
            )

            # Train generator (to have discriminator label samples as valid)
            g_loss = self.combined.train_on_batch([noise, gen_labels], valid)

            # Print progress and save models
            if epoch % sample_interval == 0:
                # Safely handle d_loss structure
                if isinstance(d_loss, (list, np.ndarray)) and len(d_loss) >= 2:
                    d_loss_val = d_loss[0]
                    d_acc_val = d_loss[1]
                else:
                    d_loss_val = d_loss
                    d_acc_val = 0.0

                print(f"Epoch {epoch} [D loss: {d_loss_val:.3f}, acc: {100*d_acc_val:.1f}%] "
                      f"[G loss: {g_loss:.3f}]")
                
                # Save model checkpoints
                self.save_models(
                    output_dir=output_dir,
                    epoch=epoch
                )

    def generate_samples(self, 
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
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        labels = tf.one_hot(
            np.full(num_samples, class_label), 
            depth=self.num_classes
        )
        return self.generator.predict([noise, labels], verbose=0)

    def save_models(self, 
                   output_dir: str = "cgan_models",
                   epoch: Optional[int] = None):
        """
        Save generator and discriminator models
        
        Args:
            output_dir (str): Directory to save models
            epoch (int, optional): Epoch number for filename
        """
        suffix = f"_{epoch:04d}" if epoch is not None else ""
        os.makedirs(output_dir, exist_ok=True)
        
        self.generator.save(
            os.path.join(output_dir, f"generator{suffix}.h5")
        )
        self.discriminator.save(
            os.path.join(output_dir, f"discriminator{suffix}.h5")
        )

    @classmethod
    def load_models(cls, 
                   generator_path: str,
                   discriminator_path: str,
                   **kwargs):
        """
        Load pretrained CGAN models
        
        Args:
            generator_path (str): Path to generator model
            discriminator_path (str): Path to discriminator model
            **kwargs: Initialization parameters
            
        Returns:
            CGAN: Loaded CGAN instance
        """
        instance = cls(**kwargs)
        instance.generator = tf.keras.models.load_model(generator_path)
        instance.discriminator = tf.keras.models.load_model(discriminator_path)
        instance._compile_models()
        return instance