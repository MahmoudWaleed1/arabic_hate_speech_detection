"""
Main entry point for Arabic Hate Speech Detection project.
"""

import argparse
import sys
import os
import logging
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.dataset import DataProcessor
from src.train import Trainer
from src.evaluate import Evaluator
from src.utils import setup_logging, set_seed

def train_model(config: Config, 
                freeze_bert: bool = False,
                dropout_rate: float = 0.1) -> None:
    """
    Train the Arabic Hate Speech Detection model.
    
    Args:
        config: Configuration object
        freeze_bert: Whether to freeze BERT parameters
        dropout_rate: Dropout rate for the classifier
    """
    logger = setup_logging(config.log_file)
    logger.info("Starting training process...")
    
    # Set random seed
    set_seed(config.seed)
    
    try:
        # Load and prepare data
        logger.info("Loading dataset...")
        data_processor = DataProcessor(config)
        train_dataset, val_dataset, test_dataset = data_processor.load_dataset()
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_processor.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Print dataset information
        data_processor.print_dataset_info(train_dataset, val_dataset, test_dataset)
        
        # Compute class weights if using weighted loss
        class_weights = None
        if config.loss_function == "weighted":
            # Get training labels for computing class weights
            train_labels = [train_dataset[i]['labels'].item() for i in range(len(train_dataset))]
            class_weights = data_processor.compute_class_weights(train_labels)
            logger.info(f"Using weighted loss with class weights: {class_weights}")
        
        # Initialize trainer
        trainer = Trainer(config)
        
        # Train model
        logger.info("Starting model training...")
        training_results = trainer.train(
            train_loader, 
            val_loader, 
            freeze_bert=freeze_bert,
            dropout_rate=dropout_rate,
            class_weights=class_weights
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def evaluate_model(config: Config) -> None:
    """
    Evaluate the trained Arabic Hate Speech Detection model.
    
    Args:
        config: Configuration object
    """
    logger = setup_logging(config.log_file)
    logger.info("Starting evaluation process...")
    
    try:
        # Load and prepare data
        logger.info("Loading dataset...")
        data_processor = DataProcessor(config)
        train_dataset, val_dataset, test_dataset = data_processor.load_dataset()
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_processor.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Initialize evaluator
        evaluator = Evaluator(config)
        
        # Evaluate model
        logger.info("Starting model evaluation...")
        evaluation_results = evaluator.evaluate_and_save_results(test_loader)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def predict_text(config: Config, text: str) -> dict:
    """
    Predict hate speech for a single text.
    
    Args:
        config: Configuration object
        text: Text to classify
        
    Returns:
        Prediction results
    """
    logger = setup_logging(config.log_file)
    
    try:
        from src.model import ModelManager
        from transformers import AutoTokenizer
        
        # Load model and tokenizer
        model_manager = ModelManager(config)
        model = model_manager.load_best_model()
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Preprocess text
        from src.utils import clean_text
        cleaned_text = clean_text(text)
        
        # Tokenize
        encoding = tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=config.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        device = model.device if hasattr(model, 'device') else 'cpu'
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs['logits'], dim=-1)
            
            # Use threshold-based prediction if specified
            if hasattr(config, 'threshold') and config.threshold != 0.5:
                # Use threshold for classification
                hate_prob = probabilities[0][1].item()  # Probability of hate speech
                prediction = torch.tensor([1 if hate_prob > config.threshold else 0])
            else:
                # Use argmax for classification
                prediction = torch.argmax(outputs['logits'], dim=-1)
        
        # Format results
        label_names = ['Not Hate Speech', 'Hate Speech']
        predicted_label = label_names[prediction.item()]
        confidence = probabilities[0][prediction.item()].item()
        
        results = {
            'text': text,
            'cleaned_text': cleaned_text,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                'not_hate_speech': probabilities[0][0].item(),
                'hate_speech': probabilities[0][1].item()
            }
        }
        
        logger.info(f"Prediction: {predicted_label} (confidence: {confidence:.4f})")
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Arabic Hate Speech Detection using Marbert"
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'evaluate', 'predict'],
        required=True,
        help='Mode to run: train, evaluate, or predict'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Text to classify (required for predict mode)'
    )
    
    parser.add_argument(
        '--freeze-bert',
        action='store_true',
        help='Freeze BERT parameters during training'
    )
    
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.1,
        help='Dropout rate for the classifier (default: 0.1)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training/evaluation'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.epochs:
        config.num_epochs = args.epochs
    
    # Run based on mode
    if args.mode == 'train':
        print("🚀 Starting training...")
        train_model(config, args.freeze_bert, args.dropout_rate)
        print("✅ Training completed!")
        
    elif args.mode == 'evaluate':
        print("📊 Starting evaluation...")
        evaluate_model(config)
        print("✅ Evaluation completed!")
        
    elif args.mode == 'predict':
        if not args.text:
            print("❌ Error: --text is required for predict mode")
            sys.exit(1)
        
        print(f"🔍 Predicting: '{args.text}'")
        results = predict_text(config, args.text)
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Original text: {results['text']}")
        print(f"Cleaned text: {results['cleaned_text']}")
        print(f"Prediction: {results['predicted_label']}")
        print(f"Confidence: {results['confidence']:.4f}")
        print(f"Probabilities:")
        print(f"  Not Hate Speech: {results['probabilities']['not_hate_speech']:.4f}")
        print(f"  Hate Speech: {results['probabilities']['hate_speech']:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()
