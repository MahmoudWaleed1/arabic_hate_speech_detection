# [file name]: evaluate_with_threshold_tuning.py
# [file content begin]
"""
Script to evaluate Arabic Hate Speech Detection model with different thresholds.
Helps find the optimal threshold for classification.
"""

import torch
import json
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import Config
from src.dataset import DataProcessor
from src.evaluate import Evaluator
from src.model import ModelManager
from src.utils import setup_logging, get_device

def evaluate_thresholds():
    """Evaluate model performance across different thresholds."""
    
    # Setup logging
    logger = setup_logging("threshold_tuning.log")
    logger.info("Starting threshold tuning evaluation...")
    
    # Load configuration
    config = Config()
    
    # Load data
    logger.info("Loading dataset...")
    data_processor = DataProcessor(config)
    train_dataset, val_dataset, test_dataset = data_processor.load_dataset()
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_processor.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Load best model
    logger.info("Loading best model...")
    model = evaluator.load_best_model()
    
    # Get true labels and probabilities from test set
    logger.info("Getting predictions from test set...")
    all_labels, all_probabilities = get_predictions(model, test_loader, evaluator.device)
    
    # Define threshold range to test
    thresholds = np.arange(0.3, 0.91, 0.05)  # 0.1 to 0.7 in steps of 0.05
    logger.info(f"Testing thresholds: {thresholds}")
    
    # Evaluate each threshold
    results = {}
    for threshold in thresholds:
        logger.info(f"Evaluating threshold: {threshold:.2f}")
        metrics = evaluate_with_threshold(all_labels, all_probabilities, threshold)
        results[threshold] = metrics
    
    # Find best threshold based on F1-score
    best_threshold = None
    best_f1 = 0.0
    for threshold, metrics in results.items():
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = threshold
    
    logger.info(f"Best threshold: {best_threshold:.2f} (F1-score: {best_f1:.4f})")
    
    # Generate detailed analysis
    analysis = generate_analysis(all_labels, all_probabilities, results, best_threshold)
    
    # Save results
    save_results(results, analysis, config.results_dir)
    
    # Plot results
    plot_results(results, config.results_dir)
    
    # Print summary
    print_summary(results, best_threshold, config.results_dir)
    
    return results, best_threshold

def get_predictions(model, test_loader, device):
    """Get true labels and prediction probabilities from test set."""
    model.eval()
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs['logits'], dim=-1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probabilities)

def evaluate_with_threshold(true_labels, probabilities, threshold):
    """Evaluate predictions using a specific threshold."""
    # Get hate speech probabilities (class 1)
    hate_probs = probabilities[:, 1]
    
    # Apply threshold
    predictions = (hate_probs > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': predictions.tolist()
    }

def generate_analysis(true_labels, probabilities, results, best_threshold):
    """Generate comprehensive analysis of threshold performance."""
    # Precision-Recall curve
    hate_probs = probabilities[:, 1]
    precision, recall, thresholds_pr = precision_recall_curve(true_labels, hate_probs)
    
    # F1-score curve
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
    
    # Find threshold that maximizes F1-score from PR curve
    best_idx = np.argmax(f1_scores)
    best_threshold_pr = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 0.5
    
    analysis = {
        'precision_recall_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds_pr.tolist(),
            'f1_scores': f1_scores,
            'best_threshold_pr': float(best_threshold_pr),
            'best_f1_pr': float(f1_scores[best_idx])
        },
        'threshold_comparison': {
            'best_threshold_selected': float(best_threshold),
            'best_threshold_pr': float(best_threshold_pr),
            'threshold_difference': float(abs(best_threshold - best_threshold_pr))
        },
        'probability_statistics': {
            'mean_hate_prob': float(np.mean(probabilities[:, 1])),
            'std_hate_prob': float(np.std(probabilities[:, 1])),
            'min_hate_prob': float(np.min(probabilities[:, 1])),
            'max_hate_prob': float(np.max(probabilities[:, 1])),
            'median_hate_prob': float(np.median(probabilities[:, 1]))
        }
    }
    
    return analysis

def save_results(results, analysis, results_dir):
    """Save results to JSON file."""
    output_path = f"{results_dir}/threshold_tuning_results.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    results_data = {
        'results': convert_numpy_types(results),
        'analysis': convert_numpy_types(analysis)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")

def plot_results(results, results_dir):
    """Plot threshold tuning results."""
    thresholds = list(results.keys())
    accuracies = [results[t]['accuracy'] for t in thresholds]
    precisions = [results[t]['precision'] for t in thresholds]
    recalls = [results[t]['recall'] for t in thresholds]
    f1_scores = [results[t]['f1_score'] for t in thresholds]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy vs Threshold
    ax1.plot(thresholds, accuracies, 'b-', marker='o', linewidth=2, markersize=6)
    ax1.set_title('Accuracy vs Threshold')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.1, 0.7)
    
    # Precision vs Threshold
    ax2.plot(thresholds, precisions, 'r-', marker='o', linewidth=2, markersize=6)
    ax2.set_title('Precision vs Threshold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Precision')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.1, 0.7)
    
    # Recall vs Threshold
    ax3.plot(thresholds, recalls, 'g-', marker='o', linewidth=2, markersize=6)
    ax3.set_title('Recall vs Threshold')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Recall')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.1, 0.7)
    
    # F1-Score vs Threshold
    ax4.plot(thresholds, f1_scores, 'm-', marker='o', linewidth=2, markersize=6)
    ax4.set_title('F1-Score vs Threshold')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('F1-Score')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.1, 0.7)
    
    # Mark best threshold
    best_threshold = max(results.keys(), key=lambda t: results[t]['f1_score'])
    best_f1 = results[best_threshold]['f1_score']
    ax4.axvline(x=best_threshold, color='orange', linestyle='--', 
                label=f'Best: {best_threshold:.2f} (F1: {best_f1:.3f})')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/threshold_tuning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined plot
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracies, 'b-', marker='o', label='Accuracy', linewidth=2, markersize=6)
    plt.plot(thresholds, precisions, 'r-', marker='o', label='Precision', linewidth=2, markersize=6)
    plt.plot(thresholds, recalls, 'g-', marker='o', label='Recall', linewidth=2, markersize=6)
    plt.plot(thresholds, f1_scores, 'm-', marker='o', label='F1-Score', linewidth=2, markersize=6)
    
    plt.axvline(x=best_threshold, color='orange', linestyle='--', 
                label=f'Best Threshold: {best_threshold:.2f}')
    
    plt.title('Model Performance vs Classification Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.1, 0.7)
    
    plt.savefig(f"{results_dir}/threshold_tuning_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {results_dir}/threshold_tuning_*.png")

def print_summary(results, best_threshold, results_dir):
    """Print summary of threshold tuning results."""
    best_metrics = results[best_threshold]
    
    print("\n" + "="*80)
    print("THRESHOLD TUNING SUMMARY")
    print("="*80)
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best F1-Score:  {best_metrics['f1_score']:.4f}")
    print(f"Accuracy:       {best_metrics['accuracy']:.4f}")
    print(f"Precision:      {best_metrics['precision']:.4f}")
    print(f"Recall:         {best_metrics['recall']:.4f}")
    print("="*80)
    
    # Print table header
    print(f"\n{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    
    # Print results for each threshold
    for threshold in sorted(results.keys()):
        metrics = results[threshold]
        print(f"{threshold:<10.2f} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    print("="*80)
    print(f"Detailed results saved to: {results_dir}/threshold_tuning_results.json")
    print(f"Plots saved to: {results_dir}/threshold_tuning_*.png")
    print("="*80)

if __name__ == "__main__":
    try:
        results, best_threshold = evaluate_thresholds()
        print(f"\n✅ Threshold tuning completed successfully!")
        print(f"🎯 Recommended threshold: {best_threshold:.2f}")
        
    except Exception as e:
        print(f"❌ Error during threshold tuning: {e}")
        import traceback
        traceback.print_exc()