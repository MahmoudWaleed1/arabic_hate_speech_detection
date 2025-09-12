"""
Evaluation pipeline for Arabic Hate Speech Detection.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import json
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)

from .config import Config
from .model import ArabicHateSpeechClassifier, ModelManager
from .utils import (
    setup_logging, get_device, plot_confusion_matrix, 
    print_classification_report, save_metrics
)

logger = setup_logging("evaluation.log")

class Evaluator:
    """
    Evaluator class for Arabic Hate Speech Detection model.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = get_device()
        self.model_manager = ModelManager(config)
        
        # Label names for classification report
        self.label_names = ['Not Hate Speech', 'Hate Speech']
        
        logger.info(f"Evaluator initialized on device: {self.device}")
    
    def load_best_model(self) -> ArabicHateSpeechClassifier:
        """
        Load the best trained model.
        
        Returns:
            Loaded model
        """
        model = self.model_manager.create_model()
        
        try:
            self.model_manager.load_best_model(model)
            logger.info("Best model loaded successfully")
        except FileNotFoundError:
            logger.warning("Best model not found, using untrained model")
        
        return model
    
    def evaluate_model(self, 
                      model: ArabicHateSpeechClassifier,
                      test_loader: DataLoader,
                      threshold: float = None) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            threshold: Probability threshold for classification (if None, uses argmax)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        
        logger.info("Starting evaluation...")
        if threshold is not None:
            logger.info(f"Using threshold: {threshold}")
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                logits = outputs['logits']
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Use threshold-based prediction if specified
                if threshold is not None:
                    # Use threshold for classification
                    hate_probs = probabilities[:, 1]  # Probability of hate speech
                    predictions = (hate_probs > threshold).long()
                else:
                    # Use argmax for classification
                    predictions = torch.argmax(logits, dim=-1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate precision, recall, f1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        # Create detailed per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.label_names):
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
        
        # Create metrics dictionary
        metrics = {
            'test_loss': float(avg_loss),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(support) if support is not None else 0,
            'threshold': threshold,
            'per_class_metrics': per_class_metrics,
            'predictions': [int(p) for p in all_predictions],  # Convert to int for JSON serialization
            'labels': [int(l) for l in all_labels],  # Convert to int for JSON serialization
            'probabilities': [[float(prob) for prob in probs] for probs
                               in all_probabilities]  # Convert to float for JSON serialization
        }
        
        # Log results
        logger.info(f"\n{'='*50}")
        logger.info("EVALUATION RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"Test Loss: {avg_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        if threshold is not None:
            logger.info(f"Threshold: {threshold:.4f}")
        logger.info(f"{'='*50}")
        
        return metrics
    
    def print_detailed_report(self, 
                             y_true: List[int], 
                             y_pred: List[int]) -> str:
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        logger.info("\nDetailed Classification Report:")
        logger.info("-" * 50)
        
        report = print_classification_report(y_true, y_pred, self.label_names)
        
        return report
    
    def plot_confusion_matrix_analysis(self, 
                                     y_true: List[int], 
                                     y_pred: List[int],
                                     save_path: str = None) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = f"{self.config.results_dir}/confusion_matrix.png"
        
        plot_confusion_matrix(y_true, y_pred, self.label_names, save_path)
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    def analyze_predictions(self, 
                           y_true: List[int], 
                           y_pred: List[int],
                           probabilities: List[List[float]],
                           texts: List[str] = None) -> Dict[str, Any]:
        """
        Analyze prediction results in detail.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            probabilities: Prediction probabilities
            texts: Original texts (optional)
            
        Returns:
            Analysis results dictionary
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        probabilities = np.array(probabilities)
        
        # Convert to numpy arrays
        if texts is not None:
            texts = np.array(texts)
        
        # Find correct and incorrect predictions
        correct_mask = y_true == y_pred
        incorrect_mask = y_true != y_pred
        
        analysis = {
            'total_samples': len(y_true),
            'correct_predictions': np.sum(correct_mask),
            'incorrect_predictions': np.sum(incorrect_mask),
            'accuracy': np.mean(correct_mask)
        }
        
        # Analyze confidence scores
        max_probs = np.max(probabilities, axis=1)
        analysis['confidence_stats'] = {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs)
        }
        
        # Analyze by class
        for class_idx, class_name in enumerate(self.label_names):
            class_mask = y_true == class_idx
            class_correct = correct_mask & class_mask
            class_incorrect = incorrect_mask & class_mask
            
            analysis[f'{class_name.lower().replace(" ", "_")}_analysis'] = {
                'total_samples': np.sum(class_mask),
                'correct_predictions': np.sum(class_correct),
                'incorrect_predictions': np.sum(class_incorrect),
                'accuracy': np.mean(class_correct) if np.sum(class_mask) > 0 else 0.0,
                'mean_confidence': np.mean(max_probs[class_mask]) if np.sum(class_mask) > 0 else 0.0
            }
        
        # Find most confident correct and incorrect predictions
        if texts is not None:
            # Most confident correct predictions
            correct_confidences = max_probs[correct_mask]
            correct_texts = texts[correct_mask]
            if len(correct_confidences) > 0:
                most_confident_correct_idx = np.argmax(correct_confidences)
                analysis['most_confident_correct'] = {
                    'text': correct_texts[most_confident_correct_idx],
                    'confidence': correct_confidences[most_confident_correct_idx],
                    'true_label': y_true[correct_mask][most_confident_correct_idx],
                    'predicted_label': y_pred[correct_mask][most_confident_correct_idx]
                }
            
            # Most confident incorrect predictions
            incorrect_confidences = max_probs[incorrect_mask]
            incorrect_texts = texts[incorrect_mask]
            if len(incorrect_confidences) > 0:
                most_confident_incorrect_idx = np.argmax(incorrect_confidences)
                analysis['most_confident_incorrect'] = {
                    'text': incorrect_texts[most_confident_incorrect_idx],
                    'confidence': incorrect_confidences[most_confident_incorrect_idx],
                    'true_label': y_true[incorrect_mask][most_confident_incorrect_idx],
                    'predicted_label': y_pred[incorrect_mask][most_confident_incorrect_idx]
                }
        
        return analysis
    
    def evaluate_multiple_thresholds(self, 
                                    model: ArabicHateSpeechClassifier,
                                    test_loader: DataLoader,
                                    thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Evaluate model with multiple thresholds for threshold tuning.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            thresholds: List of thresholds to test (if None, uses default range)
            
        Returns:
            Dictionary containing results for all thresholds
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        logger.info(f"Evaluating with thresholds: {thresholds}")
        
        results = {}
        for threshold in thresholds:
            logger.info(f"Evaluating with threshold: {threshold}")
            metrics = self.evaluate_model(model, test_loader, threshold)
            results[f"threshold_{threshold}"] = metrics
        
        # Find best threshold based on F1-score
        best_threshold = None
        best_f1 = 0.0
        
        for threshold in thresholds:
            f1 = results[f"threshold_{threshold}"]['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        results['best_threshold'] = best_threshold
        results['best_f1_score'] = best_f1
        
        logger.info(f"Best threshold: {best_threshold} (F1-score: {best_f1:.4f})")
        
        return results
    
    def evaluate_and_save_results(self, 
                                 test_loader: DataLoader,
                                 texts: List[str] = None) -> Dict[str, Any]:
        """
        Complete evaluation pipeline.
        
        Args:
            test_loader: Test data loader
            texts: Original texts (optional)
            
        Returns:
            Complete evaluation results
        """
        # Load best model
        model = self.load_best_model()
        
        # Evaluate model with configured threshold
        threshold = getattr(self.config, 'threshold', 0.5)
        if threshold == 0.5:
            # Use argmax if threshold is 0.5 (default)
            metrics = self.evaluate_model(model, test_loader, threshold=None)
        else:
            # Use specified threshold
            metrics = self.evaluate_model(model, test_loader, threshold=threshold)
        
        # Print detailed report
        self.print_detailed_report(metrics['labels'], metrics['predictions'])
        
        # Plot confusion matrix
        self.plot_confusion_matrix_analysis(
            metrics['labels'], 
            metrics['predictions']
        )
        
        # Analyze predictions
        analysis = self.analyze_predictions(
            metrics['labels'], 
            metrics['predictions'],
            metrics['probabilities'],
            texts
        )
        
        # Combine all results
        complete_results = {
            'metrics': metrics,
            'analysis': analysis,
            'config': self.config.__dict__
        }
        
        # Save results with proper JSON serialization
        results_file = f"{self.config.results_dir}/evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Complete evaluation results saved to: {results_file}")
        
        return complete_results
