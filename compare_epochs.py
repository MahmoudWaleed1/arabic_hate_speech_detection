"""
Script to compare performance of different epoch checkpoints.
"""

import torch
from src.config import Config
from src.dataset import DataProcessor
from src.evaluate import Evaluator
from src.model import ModelManager
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def compare_epochs():
    """Compare performance of different epoch checkpoints."""
    
    # Load configuration
    config = Config()
    
    # Load data
    print("Loading dataset...")
    data_processor = DataProcessor(config)
    train_dataset, val_dataset, test_dataset = data_processor.load_dataset()
    train_loader, val_loader, test_loader = data_processor.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Check which epoch files exist
    epoch_files = []
    for epoch in [1, 2, 3]:
        model_path = f"{config.models_dir}/marbert_hate_speech_model_epoch_{epoch}.pt"
        if os.path.exists(model_path):
            epoch_files.append((epoch, model_path))
    
    # Also check for best model
    best_model_path = f"{config.models_dir}/best_marbert_model.pt"
    if os.path.exists(best_model_path):
        epoch_files.append(("best", best_model_path))
    
    if not epoch_files:
        print("No model checkpoints found!")
        return
    
    # Evaluate each model
    results = {}
    for epoch_name, model_path in epoch_files:
        print(f"\n{'='*50}")
        print(f"Evaluating {epoch_name} model...")
        print(f"{'='*50}")
        
        try:
            # Create fresh model
            model = evaluator.model_manager.create_model()
            
            # Load specific checkpoint
            evaluator.model_manager.load_model(model, model_path, load_optimizer=False)
            
            # Evaluate
            metrics = evaluator.evaluate_model(model, test_loader, threshold=0.3)
            results[epoch_name] = metrics
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Hate Precision: {metrics['per_class_metrics']['Hate Speech']['precision']:.4f}")
            print(f"Hate Recall: {metrics['per_class_metrics']['Hate Speech']['recall']:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {epoch_name}: {e}")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    for epoch_name, metrics in results.items():
        print(f"{epoch_name}: Acc={metrics['accuracy']:.4f}, "
              f"Hate-P={metrics['per_class_metrics']['Hate Speech']['precision']:.4f}, "
              f"Hate-R={metrics['per_class_metrics']['Hate Speech']['recall']:.4f}")

if __name__ == "__main__":
    compare_epochs()