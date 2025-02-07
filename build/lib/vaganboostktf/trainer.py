#!pip instrall dill
import dill
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



class HybridModelTrainer:
    """Orchestrates hybrid training workflow"""
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 random_state: int = 42):
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
        self.current_iteration = 0
        """
        Initialize hybrid model trainer
        
        Args:
            config (dict): Training configuration parameters
            random_state (int): Seed for reproducibility
        """


        
     
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    def _default_config(self) -> Dict:
        """Return default training configuration"""
        return {
            'latent_dim': 8,
            'num_classes': 4,
            'cvae_epochs': 50,
            'cgan_epochs': 100,
            'lgbm_iterations': 30,
            'samples_per_class': 100,
            'model_dir': 'best_models',
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
        """Initialize all model components with proper configuration"""
        # Initialize data preprocessor
        self.components['scaler'] = DataPreprocessor()
        # After data preparation
        self.feature_columns = self.components['scaler'].feature_columns
        self.target_column = self.components['scaler'].target_column       
        # Initialize CVAE
        self.components['cvae'] = CVAE(**self.config['cvae_params'])
        self.components['cvae'].compile(
          optimizer=tf.keras.optimizers.Adam(**self.config.get('cvae_optimizer', {})),
          loss='mse'
        )
        
        # Initialize CGAN
        self.components['cgan'] = CGAN(**{k:v for k,v in self.config['cgan_params'].items() if k not in ['generator_lr', 'discriminator_lr']}, generator_lr=self.config['cgan_params'].get('generator_lr', 0.0002), discriminator_lr=self.config['cgan_params'].get('discriminator_lr', 0.0002))

        
        # Initialize LightGBM Tuner
        self.components['lgb_tuner'] = LightGBMTuner(
            n_iter=self.config['lgbm_iterations'],
            random_state=self.random_state
        )

    def train_cvae(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train CVAE with checkpointing"""
        os.makedirs(f"{self.config['model_dir']}/cvae", exist_ok=True)
        
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
        
        # Load best weights
        self.components['cvae'] = tf.keras.models.load_model(
            f"{self.config['model_dir']}/cvae/best_cvae.keras",
            custom_objects={'CVAE': CVAE}
        )

    def train_cgan(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train CGAN with periodic checkpointing"""
        self.components['cgan'].train(
            X_train, y_train,
            epochs=self.config['cgan_epochs'],
            output_dir=f"{self.config['model_dir']}/cgan"
        )

    def generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data using both CVAE and CGAN"""
        synthetic_data = []
        samples_per_class = self.config['samples_per_class']
        
        # Generate from CVAE
        for label in range(self.config['num_classes']):
            synthetic = self.components['cvae'].generate(
                class_label=label,
                num_samples=samples_per_class
            )
            synthetic_data.append((synthetic, np.full(samples_per_class, label)))
        
        # Generate from CGAN
        for label in range(self.config['num_classes']):
            synthetic = self.components['cgan'].generate_samples(
                class_label=label,
                num_samples=samples_per_class
            )
            synthetic_data.append((synthetic, np.full(samples_per_class, label)))
            
        X_syn = np.concatenate([d[0] for d in synthetic_data])
        y_syn = np.concatenate([d[1] for d in synthetic_data])
        return X_syn, y_syn

    def train_lightgbm(self, X_combined: np.ndarray, y_combined: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray):
        """Train and tune LightGBM classifier"""
        self.components['lgb_tuner'].tune(
            X_combined, y_combined,
            eval_set=[(X_val, y_val)]
        )
        
        self.components['lgb_tuner'].final_fit(
            X_combined, y_combined,
            eval_set=[(X_val, y_val)]
        )

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate current model and return accuracy"""
        #y_pred = self.components['lgb_tuner'].predict(X_test)
        y_pred = self.components['lgb_tuner'].predict(X_test)
        accuracy = (y_pred == y_test).mean()
        return accuracy

    # def save_best_models(self):
        # self.components['lgb_tuner'].save(
            # f"{self.config['model_dir']}/lgbm/best_lgbm.pkl"
        # )
        # joblib.dump(
            # self.components['scaler'],
            # f"{self.config['model_dir']}/scaler.pkl"
        # )

    # def save_best_models(self):
        # """Save all model components using joblib"""
        # os.makedirs(f"{self.config['model_dir']}/lgbm", exist_ok=True)
        #Save DataPreprocessor with joblib
         # joblib.dump(
            # {
               # 'preprocessor': self.components['scaler'],
               # 'feature_columns': self.feature_columns,
               # 'target_column': self.target_column
            # },
            # f"{self.config['model_dir']}/preprocessor.pkl"
        # )
        
    def training_loop(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     iterations: int = 5):
        """
        Complete training workflow
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            iterations (int): Number of hybrid training iterations
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
            self.components['lgbm_model'] = self.components['lgb_tuner'].tune_lgbm(
                    X_combined, y_combined,
                    X_test, y_test,
                    num_iterations=self.config.get('lgbm_iterations', 100)
                )
            self.train_lightgbm(X_combined, y_combined, X_test, y_test)
            
            # 4. Evaluate
            current_score = self.evaluate_model(X_test, y_test)
            
            # 5. Save best models
            if current_score > self.best_score:
                print(f"New best score: {current_score:.4f} (previous: {self.best_score:.4f})")
                self.best_score = current_score
                self.save_best_models()


    def _generate_synthetic_data(self, samples_per_class=100):
        """Generate synthetic samples"""
        synthetic_data = []
        for label in range(4):
            # CVAE samples
            cvae_samples = self.components['cvae'].generate(label, samples_per_class)
            synthetic_data.append((cvae_samples, np.full(samples_per_class, label)))
            
            # CGAN samples
            cgan_samples = self.components['cgan'].generate_samples(label, samples_per_class)
            synthetic_data.append((cgan_samples, np.full(samples_per_class, label)))
            
        return (
            np.concatenate([d[0] for d in synthetic_data]),
            np.concatenate([d[1] for d in synthetic_data])
        )                


    def save_best_models(self):
        """Saves the best performing models (CVAE, CGAN, and LightGBM)."""
        os.makedirs(f"{self.config['model_dir']}", exist_ok=True)

        # Use dill to dump objects that might not be pickle-able
        with open(os.path.join(self.config['model_dir'], 'preprocessor.dill'), 'wb') as f:
            dill.dump(self.components['scaler'], f)  

        with open(os.path.join(self.config['model_dir'], 'cvae.dill'), 'wb') as f:
            dill.dump(self.components['cvae'], f)  

        with open(os.path.join(self.config['model_dir'], 'cgan.dill'), 'wb') as f:
            dill.dump(self.components['cgan'], f) 

        # Use joblib for LightGBM since it's usually well-supported
        joblib.dump(self.components['lgbm_model'], os.path.join(self.config['model_dir'], 'lgbm_model.pkl'))




    # Save the Best Model
    # def save_best_models(self):
        # """Save all components using joblib"""
        # os.makedirs(f"{self.config['model_dir']}", exist_ok=True)
    
        # joblib.dump(
            # {
                 # 'preprocessor': self.components['scaler'],
                 # 'features': self.feature_columns,
                 # 'target': self.target_column
            # },
            # f"{self.config['model_dir']}/preprocessor.pkl"
        # )

    # Load the best Model
    @classmethod
    def load_best_models(cls, model_dir):
      """Load all components"""
      data = joblib.load(f"{model_dir}/preprocessor.pkl")
      trainer = cls()

      trainer.feature_columns = data['features']
      trainer.target_column = data['target']
      return trainer