"""
Configuration file for Arabic Hate Speech Detection project.
Contains all hyperparameters, paths, and settings.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration class for the Arabic Hate Speech Detection project."""
    
    # Model Configuration
    model_name: str = "UBC-NLP/MARBERT"
    num_labels: int = 2  # Binary classification: hate speech or not
    max_length: int = 128
    
    # Training Configuration
    batch_size: int = 12
    learning_rate: float = 9e-6
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Data Configuration
    dataset_name: str = "manueltonneau/arabic-hate-speech-superset"
    train_split: str = "train"
    test_split: str = "test"
    validation_split: float = 0.1
    
    # Paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(project_root, "data")
    models_dir: str = os.path.join(project_root, "models")
    results_dir: str = os.path.join(project_root, "results")
    
    # Model saving
    model_save_path: str = os.path.join(models_dir, "marbert_hate_speech_model")
    best_model_path: str = os.path.join(models_dir, "best_marbert_model")
    
    # Logging
    log_file: str = os.path.join(results_dir, "training.log")
    loss_plot_path: str = os.path.join(results_dir, "loss_curve.png")
    metrics_file: str = os.path.join(results_dir, "metrics.json")
    
    # Training settings
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    early_stopping_patience: int = 3
    
    # Device Configuration
    device: str = "auto"  # auto, cuda, cpu, or specific GPU like "cuda:0"
    force_cpu: bool = False  # Set to True to force CPU usage even if CUDA is available
    mixed_precision: bool = True  # Enable mixed precision training for better performance
    dataloader_pin_memory: bool = True  # Pin memory for faster GPU transfer
    
    # Random seed for reproducibility
    seed: int = 42
    
    # New configuration options
    loss_function: str = "focal"  # focal
    threshold: float = 0.3  # Probability threshold for classification
    use_sampler: bool = True  # Use WeightedRandomSampler for oversampling
    focal_alpha: float = 0.7  # Alpha parameter for focal loss
    focal_gamma: float = 3  # Gamma parameter for focal loss
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    @classmethod
    def from_json(cls, config_path: str = "config.json"):
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            Config instance loaded from JSON
        """
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Create config instance with default values
            config = cls()
            
            # Update with values from JSON
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"Warning: Unknown configuration key '{key}' in {config_path}")
            
            return config
        else:
            print(f"Warning: Config file {config_path} not found, using default values")
            return cls()

# Global config instance
config = Config.from_json()
