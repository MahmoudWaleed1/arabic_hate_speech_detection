"""
Model definition for Arabic Hate Speech Detection using MarBERT.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Any
import logging

from .config import Config
from .utils import setup_logging, count_parameters
from .losses import get_loss_function

logger = setup_logging("model.log")

class ArabicHateSpeechClassifier(nn.Module):
    """
    Arabic Hate Speech Detection model based on MarBERT.
    """
    
    def __init__(self, 
                 model_name: str = "UBC-NLP/MARBERT", 
                 num_labels: int = 2,
                 dropout_rate: float = 0.1,
                 freeze_bert: bool = False,
                 loss_function: str = "ce",
                 class_weights: Optional[torch.Tensor] = None,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the pre-trained model
            num_labels: Number of classification labels
            dropout_rate: Dropout rate for the classifier
            freeze_bert: Whether to freeze BERT parameters
            loss_function: Type of loss function ('ce', 'weighted', 'focal')
            class_weights: Class weights for weighted loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super(ArabicHateSpeechClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.freeze_bert = freeze_bert
        self.loss_function = loss_function
        self.class_weights = class_weights
        
        # Load pre-trained model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("BERT parameters frozen")
        
        # Get hidden size from config
        self.hidden_size = self.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize classifier weights
        self._init_classifier_weights()
        
        # Initialize loss function
        self.loss_fn = get_loss_function(
            loss_function, 
            class_weights=class_weights,
            alpha=focal_alpha,
            gamma=focal_gamma
        )
        
        logger.info(f"Model initialized with {count_parameters(self)} parameters")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        logger.info(f"Loss function: {loss_function}")
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            
        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get pooled output (CLS token representation)
        pooled_output = bert_outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            outputs["loss"] = loss
        
        return outputs
    
    def get_embeddings(self, 
                      input_ids: torch.Tensor, 
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get BERT embeddings for input text.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            BERT embeddings
        """
        with torch.no_grad():
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return bert_outputs.last_hidden_state
    
    def predict(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probabilities = torch.softmax(outputs["logits"], dim=-1)
            return probabilities
    
    def predict_classes(self, 
                       input_ids: torch.Tensor, 
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for input data.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Predicted class labels
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(outputs["logits"], dim=-1)
            return predictions

class ModelManager:
    """
    Model manager for saving, loading, and managing model checkpoints.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def create_model(self, 
                    freeze_bert: bool = False,
                    dropout_rate: float = 0.1,
                    class_weights: Optional[torch.Tensor] = None) -> ArabicHateSpeechClassifier:
        """
        Create a new model instance.
        
        Args:
            freeze_bert: Whether to freeze BERT parameters
            dropout_rate: Dropout rate for the classifier
            class_weights: Class weights for weighted loss
            
        Returns:
            Model instance
        """
        model = ArabicHateSpeechClassifier(
            model_name=self.config.model_name,
            num_labels=self.config.num_labels,
            dropout_rate=dropout_rate,
            freeze_bert=freeze_bert,
            loss_function=self.config.loss_function,
            class_weights=class_weights,
            focal_alpha=self.config.focal_alpha,
            focal_gamma=self.config.focal_gamma
        )
        
        model.to(self.device)
        logger.info(f"Model created and moved to {self.device}")
        
        return model
    
    def save_model(self, 
                   model: ArabicHateSpeechClassifier, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   accuracy: float,
                   is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss
            accuracy: Current accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, f"{self.config.model_save_path}_epoch_{epoch}.pt")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, f"{self.config.best_model_path}.pt")
            logger.info(f"Best model saved at epoch {epoch} with accuracy {accuracy:.4f}")
        
        logger.info(f"Model checkpoint saved for epoch {epoch}")
    
    def load_model(self, 
                   model: ArabicHateSpeechClassifier,
                   checkpoint_path: str,
                   load_optimizer: bool = True) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Dictionary containing loaded information
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        loaded_info = {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'accuracy': checkpoint['accuracy']
        }
        
        if load_optimizer:
            loaded_info['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        
        logger.info(f"Model loaded from {checkpoint_path}")
        logger.info(f"Epoch: {loaded_info['epoch']}, Loss: {loaded_info['loss']:.4f}, Accuracy: {loaded_info['accuracy']:.4f}")
        
        return loaded_info
    
    def load_best_model(self, 
                       model: ArabicHateSpeechClassifier) -> Dict[str, Any]:
        """
        Load the best model.
        
        Args:
            model: Model to load weights into
            
        Returns:
            Dictionary containing loaded information
        """
        best_model_path = f"{self.config.best_model_path}.pt"
        return self.load_model(model, best_model_path)
