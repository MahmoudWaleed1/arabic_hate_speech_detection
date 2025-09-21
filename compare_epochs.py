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
    for epoch in [1, 2, 3, 4, 5]:
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
            
            # Evaluate with threshold 0.3
            metrics = evaluator.evaluate_model(model, test_loader, threshold=0.3)
            results[epoch_name] = metrics
            
            # Print in the exact format you want
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            
            # Print metrics for BOTH classes
            for class_name in ['Hate Speech', 'Not Hate Speech']:
                if class_name in metrics['per_class_metrics']:
                    class_metrics = metrics['per_class_metrics'][class_name]
                    print(f"{class_name} Precision: {class_metrics['precision']:.4f}")
                    print(f"{class_name} Recall: {class_metrics['recall']:.4f}")
                    print(f"{class_name} F1: {class_metrics.get('f1_score', class_metrics.get('f1', 0)):.4f}")
                    print()  # Empty line for readability
            
        except Exception as e:
            print(f"Error evaluating {epoch_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison table with ALL metrics
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Print header
    print(f"{'Model':<8} | {'Acc':<6} | {'Hate-P':<6} | {'Hate-R':<6} | {'Hate-F1':<6} | {'NonHate-P':<8} | {'NonHate-R':<8} | {'NonHate-F1':<8}")
    print('-' * 80)
    
    for epoch_name, metrics in results.items():
        hate_metrics = metrics['per_class_metrics']['Hate Speech']
        non_hate_metrics = metrics['per_class_metrics']['Not Hate Speech']
        
        print(f"{epoch_name:<8} | {metrics['accuracy']:.4f} | "
              f"{hate_metrics['precision']:.4f} | "
              f"{hate_metrics['recall']:.4f} | "
              f"{hate_metrics.get('f1_score', hate_metrics.get('f1', 0)):.4f} | "
              f"{non_hate_metrics['precision']:.4f} | "
              f"{non_hate_metrics['recall']:.4f} | "
              f"{non_hate_metrics.get('f1_score', non_hate_metrics.get('f1', 0)):.4f}")

if __name__ == "__main__":
    compare_epochs()