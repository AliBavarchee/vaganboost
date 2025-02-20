#!pip instrall dill
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import callbacks
from typing import Dict, Tuple, Optional
import pickle
from .data_preprocessor import DataPreprocessor
from .cvae import CVAE
from .cgan import CGAN
from .lgbm_tuner import LightGBMTuner
from .utils import plot_confusion_matrix, plot_roc_curves, plot_pr_curves

#!pip instrall dill
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import callbacks
from typing import Dict, Tuple, Optional
import pickle
from .data_preprocessor import DataPreprocessor
from .cvae import CVAE
from .cgan import CGAN
from .lgbm_tuner import LightGBMTuner
from .utils import plot_confusion_matrix, plot_roc_curves, plot_pr_curves

class HybridModelTrainer:
    """
    Orchestrates the hybrid workflow: trains generative models (CVAE, CGAN) and an LGBM classifier.
    """

    def __init__(self, 
                 config: Optional[Dict] = None,
                 random_state: int = 42):
        """
        Initialize hybrid model trainer.

        Args:
            config (dict): Training configuration parameters.
            random_state (int): Seed for reproducibility.
        """
        self.config = config or self._default_config() 
        self.components = {
            'cvae': None,
            'cgan': None,
            'lgb_tuner': None,
            'scaler': None
        }
        self.feature_columns = None
        self.target_column = None
        self._create_dirs()
        self.random_state = random_state
        self.best_score = 0.0

        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    def _default_config(self) -> Dict:
        """Return default training configuration."""
        return {
            'latent_dim': 8,
            'num_classes': 4,
            'cvae_epochs': 50,
            'cgan_epochs': 100,
            'lgbm_iterations': 30,
            'samples_per_class': 100,
            'model_dir': 'trained_models',
            'input_path': 'input.csv',  # Generic input path; user can override this
            'cvae_params': {
                'input_dim': 25,
                'latent_dim': 8,
                'num_classes': 4,
                'learning_rate': 0.001
            },
            'cgan_params': {
                'input_dim': 25,
                'latent_dim': 8,
                'num_classes': 4,
                'generator_lr': 0.0002,
                'discriminator_lr': 0.0002
            }
        }
        
    def _create_dirs(self):
        os.makedirs(f"{self.config['model_dir']}/cvae", exist_ok=True)
        os.makedirs(f"{self.config['model_dir']}/cgan", exist_ok=True)
        os.makedirs(f"{self.config['model_dir']}/lgbm", exist_ok=True)

    def initialize_components(self, X_train: np.ndarray, y_train: np.ndarray):
        """Initialize all model components with proper configuration."""
        # Initialize data preprocessor
        self.components['scaler'] = DataPreprocessor()
        self.feature_columns = self.components['scaler'].feature_columns
        self.target_column = self.components['scaler'].target_column       

        # Initialize CVAE
        self.components['cvae'] = CVAE(**self.config['cvae_params'])
        self.components['cvae'].compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['cvae_params']['learning_rate']),
          loss='mse'
        )
        
        # Initialize CGAN
        self.components['cgan'] = CGAN(**self.config['cgan_params'])

        # Initialize LightGBM Tuner with a generic input path from the config
        self.components['lgb_tuner'] = LightGBMTuner(
            input_path=self.config['input_path'],
            output_path=self.config['model_dir']
        )

    def train_cvae(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train CVAE with checkpointing."""
        checkpoint = callbacks.ModelCheckpoint(
            f"{self.config['model_dir']}/cvae/best_cvae.keras",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
        
        self.components['cvae'].fit(
            (X_train, y_train), X_train,
            epochs=self.config['cvae_epochs'],
            batch_size=32,
            validation_split=0.1,
            callbacks=[checkpoint],
            verbose=1
        )
        
        # Load best model
        self.components['cvae'] = tf.keras.models.load_model(
            f"{self.config['model_dir']}/cvae/best_cvae.keras",
            custom_objects={'CVAE': CVAE}
        )

    def train_cgan(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train CGAN with periodic checkpointing."""
        self.components['cgan'].train(
            X_train, y_train,
            epochs=self.config['cgan_epochs'],
            output_dir=f"{self.config['model_dir']}/cgan"
        )

    def generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data using both CVAE and CGAN."""
        synthetic_data = []
        samples_per_class = self.config['samples_per_class']

        # Generate from CVAE
        for label in range(self.config['num_classes']):
            synthetic = self.components['cvae'].generate(label, samples_per_class)
            synthetic_data.append((synthetic, np.full(samples_per_class, label)))
        
        # Generate from CGAN
        for label in range(self.config['num_classes']):
            synthetic = self.components['cgan'].generate_samples(label, samples_per_class)
            synthetic_data.append((synthetic, np.full(samples_per_class, label)))

        X_syn = np.concatenate([d[0] for d in synthetic_data])
        y_syn = np.concatenate([d[1] for d in synthetic_data])
        return X_syn, y_syn

    def train_lightgbm(self):
        """Train and tune LightGBM classifier."""
        self.components['lgb_tuner'].tune()

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate current model and return accuracy."""
        y_pred = self.components['lgb_tuner'].predict(X_test)
        accuracy = (y_pred == y_test).mean()
        return accuracy

    def training_loop(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     iterations: int = 5):
        """
        Complete training workflow.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.
            iterations (int): Number of hybrid training iterations.
        """
        self.initialize_components(X_train, y_train)

        for iteration in range(iterations):
            print(f"\n=== Training Iteration {iteration+1}/{iterations} ===")

            # 1. Train generative models
            self.train_cvae(X_train, y_train)
            self.train_cgan(X_train, y_train)

            # 2. Generate synthetic data
            X_syn, y_syn = self.generate_synthetic_data()
            X_combined = np.vstack([X_train, X_syn])
            y_combined = np.concatenate([y_train, y_syn])

            # 3. Train LightGBM
            self.train_lightgbm()

            # 4. Evaluate model
            current_score = self.evaluate_model(X_test, y_test)
            print(f"Iteration {iteration+1}: Accuracy = {current_score:.4f}")

            # 5. Save best model if improved
            if current_score > self.best_score:
                print(f"New best score: {current_score:.4f} (previous: {self.best_score:.4f})")
                self.best_score = current_score

        # Optionally, retrieve and print feature weights from the LightGBM tuner
        feature_weights = self.components['lgb_tuner'].get_feature_weights()
        if feature_weights is not None:
            print("Per-class feature weights:")
            print(feature_weights)

print("Training completed! Best models saved in 'trained_models' directory.")
