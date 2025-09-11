"""
Training pipeline for Arabic Hate Speech Detection.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional  

# Import for mixed precision training
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from .config import Config
from .model import ArabicHateSpeechClassifier, ModelManager
from .utils import (
    setup_logging, set_seed, get_device, format_time, 
    plot_training_curves, save_metrics, count_parameters, print_device_info
)

logger = setup_logging("training.log")

class Trainer:
    """
    Trainer class for Arabic Hate Speech Detection model.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = get_device(config.device, config.force_cpu)
        self.model_manager = ModelManager(config)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
        self.early_stopping_counter = 0
        
        # Mixed precision training
        self.use_amp = (config.mixed_precision and 
                       AMP_AVAILABLE and 
                       self.device.type == 'cuda')
        self.scaler = GradScaler() if self.use_amp else None
        
        # Print device information
        print_device_info()
        if self.use_amp:
            logger.info("Mixed precision training enabled")
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_model_and_optimizer(self, 
                                 freeze_bert: bool = False,
                                 dropout_rate: float = 0.1,
                                 class_weights: Optional[torch.Tensor] = None) -> Tuple[ArabicHateSpeechClassifier, 
                                                                   torch.optim.Optimizer,
                                                                   torch.optim.lr_scheduler.LRScheduler]:
        """
        Setup model, optimizer, and scheduler.
        
        Args:
            freeze_bert: Whether to freeze BERT parameters
            dropout_rate: Dropout rate for the classifier
            class_weights: Class weights for weighted loss
            
        Returns:
            Tuple of (model, optimizer, scheduler)
        """
        # Create model
        model = self.model_manager.create_model(freeze_bert, dropout_rate, class_weights)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Model parameters: {count_parameters(model)}")
        logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate})")
        logger.info(f"Scheduler: Linear with warmup ({self.config.warmup_steps} steps)")
        
        return model, optimizer, scheduler
    
    def train_epoch(self, 
                   model: ArabicHateSpeechClassifier,
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler.LRScheduler) -> float:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.use_amp:
                with autocast():
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
            else:
                # Standard training without mixed precision
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
            
            # Update loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # Log training progress
            if (batch_idx + 1) % self.config.logging_steps == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches} - "
                    f"Loss: {loss.item():.4f} - "
                    f"Avg Loss: {total_loss / (batch_idx + 1):.4f}"
                )
        
        return total_loss / num_batches
    
    def validate_epoch(self, 
                      model: ArabicHateSpeechClassifier) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            model: Model to validate
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with mixed precision if enabled
                if self.use_amp:
                    with autocast():
                        outputs = model(input_ids, attention_mask, labels)
                else:
                    outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                logits = outputs['logits']
                
                # Update loss
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                # Update progress bar
                accuracy = correct_predictions / total_predictions
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'accuracy': f'{accuracy:.4f}'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              freeze_bert: bool = False,
              dropout_rate: float = 0.1,
              class_weights: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            freeze_bert: Whether to freeze BERT parameters
            dropout_rate: Dropout rate for the classifier
            class_weights: Class weights for weighted loss
            
        Returns:
            Training results dictionary
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set random seed
        set_seed(self.config.seed)
        
        # Setup model and optimizer
        model, optimizer, scheduler = self.setup_model_and_optimizer(freeze_bert, dropout_rate, class_weights)
        
        logger.info("Starting training...")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*50}")
            
            # Training
            train_loss = self.train_epoch(model, optimizer, scheduler)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_accuracy = self.validate_epoch(model)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1} Results:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Accuracy: {val_accuracy:.4f}")
            logger.info(f"  Epoch Time: {format_time(epoch_time)}")
            
            # Check if this is the best model
            is_best = val_accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = val_accuracy
                self.early_stopping_counter = 0
                logger.info(f"New best model! Accuracy: {val_accuracy:.4f}")
            else:
                self.early_stopping_counter += 1
            
            # Save model checkpoint
            self.model_manager.save_model(
                model, optimizer, epoch + 1, val_loss, val_accuracy, is_best
            )
            
            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Calculate total training time
        total_time = time.time() - start_time
        
        # Plot training curves
        plot_training_curves(
            self.train_losses, 
            self.val_losses, 
            self.val_accuracies,
            self.config.loss_plot_path
        )
        
        # Save training metrics
        training_results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_accuracy': self.best_accuracy,
            'total_epochs': len(self.train_losses),
            'total_time': total_time,
            'config': self.config.__dict__
        }
        
        save_metrics(training_results, self.config.metrics_file)
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Total time: {format_time(total_time)}")
        logger.info(f"Best validation accuracy: {self.best_accuracy:.4f}")
        logger.info(f"Training curves saved to: {self.config.loss_plot_path}")
        
        return training_results
