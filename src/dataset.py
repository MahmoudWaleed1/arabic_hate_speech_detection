"""
Dataset handling for Arabic Hate Speech Detection.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datasets import load_dataset
import logging
import numpy as np
import os
from .config import Config
from .utils import clean_text, setup_logging

logger = setup_logging("dataset.log")

class ArabicHateSpeechDataset(Dataset):
    """
    Custom Dataset class for Arabic Hate Speech Detection.
    """
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int], 
                 tokenizer: AutoTokenizer,
                 max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate inputs
        assert len(self.texts) == len(self.labels), \
            "Number of texts and labels must be equal"
        
        logger.info(f"Dataset initialized with {len(self.texts)} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing text and label (tokenization happens in collate_fn)
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Clean the text
        text = clean_text(text)
        
        # Return raw text for batch processing (tokenization happens in collate_fn)
        return {
            'text': text,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataProcessor:
    """
    Data processor for loading and preparing the Arabic Hate Speech dataset.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded: {config.model_name}")
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for efficient batch tokenization.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Batch of tokenized samples
        """
        texts = [item['text'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        
        # Tokenize the entire batch at once (much more efficient!)
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels
        }
    
    def load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load and preprocess the dataset from our cleaned CSV file.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Loading cleaned dataset from CSV file...")
        
        try:
            
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Cleaned_arabic_hate_speech.csv") 
            df = pd.read_csv(data_path)
            
            # Extract texts and labels
            all_texts = df['text'].astype(str).tolist()
            all_labels = df['labels'].astype(int).tolist()
            
            # Validate data
            if len(all_texts) == 0:
                raise ValueError("No text data found in dataset")
            if len(all_labels) == 0:
                raise ValueError("No label data found in dataset")
            if len(all_texts) != len(all_labels):
                raise ValueError(f"Mismatch between text ({len(all_texts)}) and labels ({len(all_labels)}) count")
            
            logger.info(f"Successfully loaded {len(all_texts)} samples from CSV")
            
            # Split the data into train, validation, and test sets
            total_size = len(all_texts)
            train_size = int(total_size * 0.7)  # 70% for training
            val_size = int(total_size * 0.15)   # 15% for validation
            test_size = total_size - train_size - val_size  # 15% for testing
            
            logger.info(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
            
            # Split the data
            train_texts = all_texts[:train_size]
            train_labels = all_labels[:train_size]
            
            val_texts = all_texts[train_size:train_size + val_size]
            val_labels = all_labels[train_size:train_size + val_size]
            
            test_texts = all_texts[train_size + val_size:]
            test_labels = all_labels[train_size + val_size:]
            
            logger.info(f"After split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
            
            # Log label distribution
            train_dist = self.get_label_distribution(train_labels)
            val_dist = self.get_label_distribution(val_labels)
            test_dist = self.get_label_distribution(test_labels)
            
            logger.info(f"Train label distribution: {train_dist}")
            logger.info(f"Validation label distribution: {val_dist}")
            logger.info(f"Test label distribution: {test_dist}")
            
            # Create datasets
            train_dataset = ArabicHateSpeechDataset(
                train_texts, train_labels, self.tokenizer, self.config.max_length
            )
            
            val_dataset = ArabicHateSpeechDataset(
                val_texts, val_labels, self.tokenizer, self.config.max_length
            )
            
            test_dataset = ArabicHateSpeechDataset(
                test_texts, test_labels, self.tokenizer, self.config.max_length
            )
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset from CSV: {str(e)}")
            raise
    
    def create_dataloaders(self, 
                          train_dataset: Dataset, 
                          val_dataset: Dataset, 
                          test_dataset: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for training, validation, and testing.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Determine pin_memory based on device configuration
        pin_memory = self.config.dataloader_pin_memory and torch.cuda.is_available()
        
        # Create train loader with optional weighted sampling
        if self.config.use_sampler:
            train_sampler = self._create_weighted_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                sampler=train_sampler,
                num_workers=0,  # Set to 0 for Windows compatibility
                pin_memory=pin_memory,
                collate_fn=self.collate_fn  # Add custom collate function
            )
            logger.info("Using WeightedRandomSampler for training")
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for Windows compatibility
                pin_memory=pin_memory,
                collate_fn=self.collate_fn  # Add custom collate function
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn  # Add custom collate function
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn  # Add custom collate function
        )
        
        logger.info(f"DataLoaders created - Batch size: {self.config.batch_size}")
        
        return train_loader, val_loader, test_loader
    
    def _create_weighted_sampler(self, dataset: Dataset) -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler for handling class imbalance.
        
        Args:
            dataset: Training dataset
            
        Returns:
            WeightedRandomSampler instance
        """
        # Get all labels from the dataset
        labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
        
        # Count occurrences of each class
        class_counts = np.bincount(labels)
        
        # Compute class weights (inverse frequency)
        class_weights = 1.0 / class_counts
        
        # Assign weights to each sample
        sample_weights = [class_weights[label] for label in labels]
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        
        logger.info(f"Created WeightedRandomSampler with class weights: {class_weights}")
        logger.info(f"Class distribution: {class_counts}")
        
        return sampler
    
    def compute_class_weights(self, labels: List[int]) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.
        
        Args:
            labels: List of class labels
            
        Returns:
            Class weights tensor
        """
        # Count occurrences of each class
        class_counts = np.bincount(labels)
        
        # Avoid division by zero
        class_counts = np.clamp(class_counts, 1.0, None)
        
        # Compute inverse frequency weights
        total_samples = len(labels)
        num_classes = len(class_counts)
        class_weights = total_samples / (num_classes * class_counts)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * num_classes
        
        logger.info(f"Class weights computed: {class_weights}")
        logger.info(f"Class distribution: {class_counts}")
        
        return torch.FloatTensor(class_weights)
    
    def get_label_distribution(self, labels: List[int]) -> Dict[int, int]:
        """
        Get the distribution of labels in the dataset.
        
        Args:
            labels: List of labels
            
        Returns:
            Dictionary with label counts
        """
        from collections import Counter
        return dict(Counter(labels))
    
    def print_dataset_info(self, 
                          train_dataset: Dataset, 
                          val_dataset: Dataset, 
                          test_dataset: Dataset) -> None:
        """
        Print information about the datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
        """
        logger.info("=" * 50)
        logger.info("DATASET INFORMATION")
        logger.info("=" * 50)
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        logger.info(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
        logger.info(f"Max sequence length: {self.config.max_length}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info("=" * 50)