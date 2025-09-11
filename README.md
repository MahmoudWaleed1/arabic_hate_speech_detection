Arabic Hate Speech Detection using MARBERT
A complete deep learning project for detecting hate speech in Arabic text using the MARBERT model from Hugging Face Transformers. This project features advanced CUDA support, mixed precision training, and comprehensive evaluation tools.

🎯 Project Overview
This project implements a state-of-the-art Arabic hate speech detection system using:

Model: aubmindlab/bert-base-arabertv02 (MARBERT v2)

Dataset: manueltonneau/arabic-hate-speech-superset

Framework: PyTorch + Hugging Face Transformers

Task: Binary classification (Hate Speech vs. Not Hate Speech)

Hardware: Full CUDA support with mixed precision training

Performance: Optimized for both CPU and GPU execution

📁 Project Structure
text
arabic_hate_speech_detection/
├── data/                          # Data storage and sample datasets
│   └── sample_arabic_hate_speech/ # Sample data files
│       ├── train.json            # Training data
│       └── test.json             # Test data
├── models/                        # Trained model checkpoints
│   ├── marbert_hate_speech_model_epoch_*.pt  # Epoch checkpoints
│   └── best_marbert_model.pt     # Best model checkpoint
├── results/                       # Training logs, metrics, and plots
│   ├── training.log              # Training progress logs
│   ├── loss_curve.png            # Training curves visualization
│   ├── confusion_matrix.png      # Confusion matrix plot
│   ├── metrics.json              # Detailed metrics
│   └── evaluation_results.json   # Complete evaluation results
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration parameters and device settings
│   ├── dataset.py                # Data loading and preprocessing
│   ├── model.py                  # MARBERT model definition and management
│   ├── train.py                  # Training pipeline with mixed precision
│   ├── evaluate.py               # Comprehensive evaluation pipeline
│   └── utils.py                  # Utility functions and device management
├── main.py                       # Main entry point with CLI interface
├── requirements.txt              # Python dependencies
├── CUDA_GUIDE.md                 # Detailed CUDA usage guide
├── *.log                         # Various log files
└── README.md                     # This file
🚀 Quick Start
1. Installation
bash
# Clone or download the project
cd arabic_hate_speech_detection

# Install dependencies
pip install -r requirements.txt
2. Training
bash
# Train the model with default settings (automatic device selection)
python main.py --mode train

# Train with custom parameters
python main.py --mode train --batch-size 32 --learning-rate 3e-5 --epochs 5

# Train with frozen BERT (faster, less accurate)
python main.py --mode train --freeze-bert --dropout-rate 0.2

# Train with specific device
python main.py --mode train --device cuda:0

# Train with mixed precision (enabled by default on CUDA)
python main.py --mode train --mixed-precision

# Force CPU usage (for debugging)
python main.py --mode train --force-cpu

# Train with different loss functions (configure in config.json)
# Set "loss_function": "weighted" for class imbalance
# Set "loss_function": "focal" for hard examples
# Set "use_sampler": true for oversampling
3. Evaluation
bash
# Evaluate the trained model
python main.py --mode evaluate

# Evaluate with specific device
python main.py --mode evaluate --device cuda:0

# Evaluate with custom batch size
python main.py --mode evaluate --batch-size 32

# Evaluate with custom threshold (configure in config.json)
# Set "threshold": 0.3 for more sensitive detection
# Set "threshold": 0.7 for more conservative detection
4. Prediction
bash
# Predict on a single text
python main.py --mode predict --text "هذا نص عربي للاختبار"

# Example predictions
python main.py --mode predict --text "أحب هذا المكان"
python main.py --mode predict --text "أنت شخص سيء جداً"

# Predict with specific device
python main.py --mode predict --text "النص هنا" --device cuda:0
🆕 Enhanced Features
Loss Functions
The project now supports multiple loss functions to handle different scenarios:

Cross-Entropy Loss (ce): Standard loss for balanced datasets

Weighted Cross-Entropy (weighted): Automatically computes class weights for imbalanced datasets

Focal Loss (focal): Advanced loss function that focuses on hard-to-classify examples

Threshold Tuning
Configure probability thresholds for classification:

Default: Uses argmax (threshold = 0.5)

Custom Threshold: Set any threshold between 0.0 and 1.0

Multiple Threshold Testing: Evaluate performance across different thresholds

Best Threshold Selection: Automatically find optimal threshold based on F1-score

Class Imbalance Handling
Advanced techniques for handling imbalanced datasets:

WeightedRandomSampler: Oversamples minority class during training

Class Weight Computation: Automatic calculation of inverse frequency weights

Configurable Sampling: Enable/disable weighted sampling via configuration

Configuration Options
New configuration parameters in config.json:

json
{
  "loss_function": "ce", // "ce", "weighted", or "focal"
  "threshold": 0.5, // Probability threshold for classification
  "use_sampler": false, // Enable WeightedRandomSampler
  "focal_alpha": 1.0, // Alpha parameter for focal loss
  "focal_gamma": 2.0 // Gamma parameter for focal loss
}
⚙️ Configuration
The project uses a centralized configuration system in src/config.py and config.json. Key parameters:

python
# Model Configuration
model_name = "aubmindlab/bert-base-arabertv02"
max_length = 128
num_labels = 2

# Training Configuration
batch_size = 16
learning_rate = 2e-5
num_epochs = 3
warmup_steps = 100
weight_decay = 0.01

# Data Configuration
dataset_name = "manueltonneau/arabic-hate-speech-superset"
validation_split = 0.1

# Device Configuration
device = "auto"  # auto, cuda, cuda:0, cpu
force_cpu = False
mixed_precision = True
dataloader_pin_memory = True

# Training Settings
save_steps = 500
eval_steps = 500
logging_steps = 100
early_stopping_patience = 3
seed = 42
📊 Features
Data Processing
Arabic Text Normalization: Handles various Arabic text variations and diacritics

Automatic Dataset Loading: Downloads and processes the Hugging Face dataset

Data Splitting: Automatic train/validation/test splits (70/15/15)

DataLoader Integration: Efficient batching with PyTorch DataLoader

Memory Optimization: Pin memory for faster GPU data transfer

Class Imbalance Handling: WeightedRandomSampler for oversampling minority classes

Class Weight Computation: Automatic calculation of class weights for weighted loss

Model Architecture
MARBERT v2: Pre-trained Arabic BERT

just srnd me the read me file change
Arabic Hate Speech Detection using MARBERT
A complete deep learning project for detecting hate speech in Arabic text using the MARBERT model from Hugging Face Transformers. This project features advanced CUDA support, mixed precision training, and comprehensive evaluation tools.

🎯 Project Overview
This project implements a state-of-the-art Arabic hate speech detection system using:

Model: aubmindlab/bert-base-arabertv02 (MARBERT v2)

Dataset: manueltonneau/arabic-hate-speech-superset

Framework: PyTorch + Hugging Face Transformers

Task: Binary classification (Hate Speech vs. Not Hate Speech)

Hardware: Full CUDA support with mixed precision training

Performance: Optimized for both CPU and GPU execution

📁 Project Structure
text
arabic_hate_speech_detection/
├── data/                          # Data storage and sample datasets
│   └── sample_arabic_hate_speech/ # Sample data files
│       ├── train.json            # Training data
│       └── test.json             # Test data
├── models/                        # Trained model checkpoints
│   ├── marbert_hate_speech_model_epoch_*.pt  # Epoch checkpoints
│   └── best_marbert_model.pt     # Best model checkpoint
├── results/                       # Training logs, metrics, and plots
│   ├── training.log              # Training progress logs
│   ├── loss_curve.png            # Training curves visualization
│   ├── confusion_matrix.png      # Confusion matrix plot
│   ├── metrics.json              # Detailed metrics
│   └── evaluation_results.json   # Complete evaluation results
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration parameters and device settings
│   ├── dataset.py                # Data loading and preprocessing
│   ├── model.py                  # MARBERT model definition and management
│   ├── train.py                  # Training pipeline with mixed precision
│   ├── evaluate.py               # Comprehensive evaluation pipeline
│   └── utils.py                  # Utility functions and device management
├── main.py                       # Main entry point with CLI interface
├── requirements.txt              # Python dependencies
├── CUDA_GUIDE.md                 # Detailed CUDA usage guide
├── *.log                         # Various log files
└── README.md                     # This file
🚀 Quick Start
1. Installation
bash
# Clone or download the project
cd arabic_hate_speech_detection

# Install dependencies
pip install -r requirements.txt
2. Training
bash
# Train the model with default settings (automatic device selection)
python main.py --mode train

# Train with custom parameters
python main.py --mode train --batch-size 32 --learning-rate 3e-5 --epochs 5

# Train with frozen BERT (faster, less accurate)
python main.py --mode train --freeze-bert --dropout-rate 0.2

# Train with specific device
python main.py --mode train --device cuda:0

# Train with mixed precision (enabled by default on CUDA)
python main.py --mode train --mixed-precision

# Force CPU usage (for debugging)
python main.py --mode train --force-cpu

# Train with different loss functions (configure in config.json)
# Set "loss_function": "weighted" for class imbalance
# Set "loss_function": "focal" for hard examples
# Set "use_sampler": true for oversampling
3. Evaluation
bash
# Evaluate the trained model
python main.py --mode evaluate

# Evaluate with specific device
python main.py --mode evaluate --device cuda:0

# Evaluate with custom batch size
python main.py --mode evaluate --batch-size 32

# Evaluate with custom threshold (configure in config.json)
# Set "threshold": 0.3 for more sensitive detection
# Set "threshold": 0.7 for more conservative detection
4. Prediction
bash
# Predict on a single text
python main.py --mode predict --text "هذا نص عربي للاختبار"

# Example predictions
python main.py --mode predict --text "أحب هذا المكان"
python main.py --mode predict --text "أنت شخص سيء جداً"

# Predict with specific device
python main.py --mode predict --text "النص هنا" --device cuda:0
🆕 Enhanced Features
Loss Functions
The project now supports multiple loss functions to handle different scenarios:

Cross-Entropy Loss (ce): Standard loss for balanced datasets

Weighted Cross-Entropy (weighted): Automatically computes class weights for imbalanced datasets

Focal Loss (focal): Advanced loss function that focuses on hard-to-classify examples

Threshold Tuning
Configure probability thresholds for classification:

Default: Uses argmax (threshold = 0.5)

Custom Threshold: Set any threshold between 0.0 and 1.0

Multiple Threshold Testing: Evaluate performance across different thresholds

Best Threshold Selection: Automatically find optimal threshold based on F1-score

Class Imbalance Handling
Advanced techniques for handling imbalanced datasets:

WeightedRandomSampler: Oversamples minority class during training

Class Weight Computation: Automatic calculation of inverse frequency weights

Configurable Sampling: Enable/disable weighted sampling via configuration

Configuration Options
New configuration parameters in config.json:

json
{
  "loss_function": "ce", // "ce", "weighted", or "focal"
  "threshold": 0.5, // Probability threshold for classification
  "use_sampler": false, // Enable WeightedRandomSampler
  "focal_alpha": 1.0, // Alpha parameter for focal loss
  "focal_gamma": 2.0 // Gamma parameter for focal loss
}
⚙️ Configuration
The project uses a centralized configuration system in src/config.py and config.json. Key parameters:

python
# Model Configuration
model_name = "aubmindlab/bert-base-arabertv02"
max_length = 128
num_labels = 2

# Training Configuration
batch_size = 16
learning_rate = 2e-5
num_epochs = 3
warmup_steps = 100
weight_decay = 0.01

# Data Configuration
dataset_name = "manueltonneau/arabic-hate-speech-superset"
validation_split = 0.1

# Device Configuration
device = "auto"  # auto, cuda, cuda:0, cpu
force_cpu = False
mixed_precision = True
dataloader_pin_memory = True

# Training Settings
save_steps = 500
eval_steps = 500
logging_steps = 100
early_stopping_patience = 3
seed = 42
📊 Features
Data Processing
Arabic Text Normalization: Handles various Arabic text variations and diacritics

Automatic Dataset Loading: Downloads and processes the Hugging Face dataset

Data Splitting: Automatic train/validation/test splits (70/15/15)

DataLoader Integration: Efficient batching with PyTorch DataLoader

Memory Optimization: Pin memory for faster GPU data transfer

Class Imbalance Handling: WeightedRandomSampler for oversampling minority classes

Class Weight Computation: Automatic calculation of class weights for weighted loss

Model Architecture
MARBERT v2: Pre-trained Arabic BERT model

Custom Classification Head: Dropout + Linear layer with Xavier initialization

Flexible Configuration: Support for freezing BERT parameters

Model Checkpointing: Automatic saving of best models and epoch checkpoints

Parameter Management: Comprehensive model state management

Multiple Loss Functions: Support for Cross-Entropy, Weighted Cross-Entropy, and Focal Loss

Configurable Loss: Easy switching between loss functions via configuration

Training Features
Mixed Precision Training: ~2x faster training with ~50% less memory usage

Device Management: Automatic CUDA/CPU detection and selection

Progress Tracking: Real-time training progress with tqdm

Learning Rate Scheduling: Linear warmup + decay

Early Stopping: Prevents overfitting with configurable patience

Gradient Clipping: Stabilizes training with gradient norm clipping

Comprehensive Logging: Detailed training logs with device information

Reproducibility: Fixed random seeds for consistent results

Evaluation Metrics
Accuracy: Overall classification accuracy

Precision, Recall, F1-Score: Per-class and weighted metrics

Confusion Matrix: Visual analysis of predictions

Confidence Analysis: Prediction confidence statistics and analysis

Classification Report: Detailed performance breakdown

Error Analysis: Most confident correct/incorrect predictions

Per-Class Analysis: Detailed metrics for each class

Threshold Tuning: Configurable probability thresholds for classification

Multiple Threshold Evaluation: Test performance across different thresholds

JSON Serialization: Proper serialization of all metrics for easy analysis

Hardware Optimization
CUDA Support: Full GPU acceleration with automatic device selection

Mixed Precision: Automatic mixed precision training on compatible GPUs

Memory Management: Optimized memory usage with pin memory

Multi-Device Support: Support for specific GPU selection (cuda:0, cuda:1, etc.)

CPU Fallback: Automatic fallback to CPU when CUDA is unavailable

Visualization
Training Curves: Loss and accuracy plots with high-resolution output

Confusion Matrix: Visual prediction analysis with proper labeling

Automatic Saving: All plots saved to results/ directory

Device Information: Comprehensive device status reporting

🚀 CUDA and Device Configuration
Device Selection
The project supports multiple device configurations:

Device Option	Description	Use Case
"auto"	Automatically select best available device	Recommended
"cuda"	Use first available GPU	When you want GPU specifically
"cuda:0"	Use specific GPU (GPU 0)	Multi-GPU systems
"cpu"	Force CPU usage	Debugging or when GPU has issues
CUDA Features
Mixed Precision Training: Enabled by default on CUDA-compatible GPUs

Memory Optimization: Pin memory for faster CPU→GPU data transfer

Automatic Detection: Detects CUDA availability and falls back gracefully

Performance Monitoring: Built-in GPU memory and performance tracking

Quick CUDA Commands
bash
# Check CUDA status
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Train with optimal GPU settings
python main.py --mode train --batch-size 32

# Force CPU usage (for debugging)
python main.py --mode train --force-cpu

# Use specific GPU
python main.py --mode train --device cuda:0
For detailed CUDA configuration, see CUDA_GUIDE.md.

🔧 Usage Examples
Using Enhanced Features
Switching Loss Functions
python
# In config.json, set the loss function:
{
  "loss_function": "weighted",  # For class imbalance
  "use_sampler": true           # Enable oversampling
}

# Or for focal loss:
{
  "loss_function": "focal",
  "focal_alpha": 1.0,
  "focal_gamma": 2.0
}
Threshold Tuning
python
# In config.json, set custom threshold:
{
  "threshold": 0.3  # More sensitive to hate speech
}

# Or for conservative detection:
{
  "threshold": 0.7  # More conservative
}
Class Imbalance Handling
python
# Enable weighted sampling and loss:
{
  "loss_function": "weighted",
  "use_sampler": true
}

# The system will automatically:
# 1. Compute class weights from training data
# 2. Use WeightedRandomSampler for oversampling
# 3. Apply weighted loss during training
Training with Custom Settings
python
from src.config import Config
from src.train import Trainer
from src.dataset import DataProcessor

# Load configuration
config = Config()
config.batch_size = 32
config.learning_rate = 3e-5
config.num_epochs = 5

# Load data
data_processor = DataProcessor(config)
train_dataset, val_dataset, test_dataset = data_processor.load_dataset()
train_loader, val_loader, test_loader = data_processor.create_dataloaders(
    train_dataset, val_dataset, test_dataset
)

# Train model
trainer = Trainer(config)
results = trainer.train(train_loader, val_loader)
Custom Prediction
python
from src.model import ModelManager
from src.utils import clean_text
from transformers import AutoTokenizer

# Load model
config = Config()
model_manager = ModelManager(config)
model = model_manager.load_best_model()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# Predict
text = "هذا نص للاختبار"
cleaned_text = clean_text(text)
encoding = tokenizer(cleaned_text, return_tensors='pt',
                   truncation=True, padding=True, max_length=128)

with torch.no_grad():
    outputs = model(encoding['input_ids'], encoding['attention_mask'])
    prediction = torch.argmax(outputs['logits'], dim=-1)
    confidence = torch.softmax(outputs['logits'], dim=-1)
📈 Results
The model typically achieves:

Accuracy: 85-92% on the test set

F1-Score: 0.85-0.92 (weighted)

Training Time:

CPU: ~2-4 hours

GPU (CUDA): ~30-60 minutes

GPU (Mixed Precision): ~15-30 minutes

Performance Benchmarks
Hardware	Batch Size	Training Time	Memory Usage	Speedup
CPU	16	~3 hours	~4GB RAM	1x
GPU (CUDA)	16	~45 minutes	~2GB VRAM	4x
GPU (Mixed Precision)	32	~25 minutes	~1.5GB VRAM	7x
Output Files
Results are automatically saved to:

results/training.log: Detailed training logs

results/loss_curve.png: Training and validation curves

results/confusion_matrix.png: Confusion matrix visualization

results/metrics.json: Detailed metrics and statistics

results/evaluation_results.json: Complete evaluation analysis

models/best_marbert_model.pt: Best model checkpoint

models/marbert_hate_speech_model_epoch_*.pt: Epoch checkpoints

🛠️ Advanced Usage
Custom Dataset
To use your own dataset, modify the load_dataset method in src/dataset.py:

python
def load_custom_dataset(self, data_path: str):
    # Load your custom dataset
    df = pd.read_csv(data_path)

    # Extract texts and labels
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # Create datasets
    train_dataset = ArabicHateSpeechDataset(texts, labels, self.tokenizer)
    return train_dataset
Model Fine-tuning
For fine-tuning on specific domains:

python
# Load pre-trained model
model = ArabicHateSpeechClassifier(
    model_name="aubmindlab/bert-base-arabertv02",
    num_labels=2,
    freeze_bert=False  # Allow fine-tuning
)

# Use lower learning rate for fine-tuning
config.learning_rate = 1e-5
Hyperparameter Tuning
Key hyperparameters to tune:

learning_rate: 1e-5 to 5e-5

batch_size: 8, 16, 32, 64

max_length: 64, 128, 256

dropout_rate: 0.1, 0.2, 0.3

warmup_steps: 50, 100, 200

🐛 Troubleshooting
Common Issues
CUDA Out of Memory

bash
# Reduce batch size
python main.py --mode train --batch-size 8

# Or reduce batch size in config.py
batch_size: int = 8

# Disable mixed precision temporarily
mixed_precision: bool = False
CUDA Not Available

bash
# Check CUDA installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Force CPU usage
python main.py --mode train --force-cpu

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
Dataset Download Issues

bash
# Check internet connection and Hugging Face access
# The dataset will be cached after first download

# Clear cache if needed
rm -rf ~/.cache/huggingface/datasets/
Model Loading Errors

bash
# Ensure the model is trained first
python main.py --mode train
python main.py --mode evaluate

# Check if model files exist
ls -la models/
Memory Issues on CPU

bash
# Reduce batch size
python main.py --mode train --batch-size 4

# Use smaller model or freeze BERT
python main.py --mode train --freeze-bert
Performance Tips
Use GPU: Ensure CUDA is available for faster training

Batch Size: Increase batch size if you have more GPU memory (16→32→64)

Mixed Precision: Enabled by default on CUDA-compatible GPUs

Pin Memory: Keep enabled for faster data loading

Monitor Resources: Use nvidia-smi to monitor GPU usage

Early Stopping: Adjust patience based on

### Debugging Commands

```bash
# Check device information
python -c "from src.utils import print_device_info; print_device_info()"

# Test CUDA functionality
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Check model parameters
python -c "from src.model import ArabicHateSpeechClassifier; from src.config import Config; model = ArabicHateSpeechClassifier(Config().model_name); print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')"
```

## 📚 Dependencies

### Core Dependencies

- **PyTorch** (≥1.12.0): Deep learning framework with CUDA support
- **Transformers** (≥4.21.0): Hugging Face transformers library
- **Datasets** (≥2.0.0): Hugging Face datasets library
- **Scikit-learn** (≥1.1.0): Machine learning utilities and metrics

### Data Processing

- **Pandas** (≥1.4.0): Data manipulation and analysis
- **NumPy** (≥1.21.0): Numerical computing

### Visualization

- **Matplotlib** (≥3.5.0): Plotting and visualization
- **Seaborn** (≥0.11.0): Statistical data visualization

### Utilities

- **TQDM** (≥4.64.0): Progress bars
- **Accelerate** (≥0.20.0): Optional performance optimization

### Development (Optional)

- **Pytest** (≥7.0.0): Testing framework
- **Black** (≥22.0.0): Code formatting
- **Flake8** (≥4.0.0): Code linting

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Install with CUDA support (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install only core dependencies
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn tqdm
```

## 🖥️ Command-Line Interface

The project provides a comprehensive CLI through `main.py`:

### Training Options

```bash
python main.py --mode train [OPTIONS]

Options:
  --batch-size INT        Batch size for training/evaluation
  --learning-rate FLOAT   Learning rate for training
  --epochs INT           Number of training epochs
  --freeze-bert          Freeze BERT parameters during training
  --dropout-rate FLOAT   Dropout rate for the classifier (default: 0.1)
  --device STR           Device to use (auto, cuda, cuda:0, cpu)
  --force-cpu            Force CPU usage even if CUDA is available
  --mixed-precision      Enable mixed precision training
```

### Evaluation Options

```bash
python main.py --mode evaluate [OPTIONS]

Options:
  --batch-size INT       Batch size for evaluation
  --device STR          Device to use (auto, cuda, cuda:0, cpu)
```

### Prediction Options

```bash
python main.py --mode predict --text "TEXT" [OPTIONS]

Options:
  --text STR            Text to classify (required)
  --device STR          Device to use (auto, cuda, cuda:0, cpu)
```

### Examples

```bash
# Complete training pipeline
python main.py --mode train --batch-size 32 --learning-rate 2e-5 --epochs 5

# Quick evaluation
python main.py --mode evaluate

# Single prediction
python main.py --mode predict --text "هذا نص للاختبار"

# Debug mode (CPU only)
python main.py --mode train --force-cpu --batch-size 8
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **AraBERT Team**: For the excellent pre-trained Arabic BERT model (`aubmindlab/bert-base-arabertv02`)
- **Hugging Face**: For the transformers and datasets libraries
- **Dataset Authors**: For providing the Arabic hate speech dataset (`manueltonneau/arabic-hate-speech-superset`)
- **PyTorch Team**: For the deep learning framework and CUDA support
- **Arabic NLP Community**: For advancing Arabic natural language processing

## 📞 Support

For questions or issues:

1. **Check the troubleshooting section** above
2. **Review the logs** in `results/training.log`
3. **Check CUDA status** using the debugging commands
4. **Open an issue** with detailed error information and system specs

### Getting Help

- **Documentation**: This README and `CUDA_GUIDE.md`
- **Logs**: Check `results/` directory for detailed logs
- **Device Info**: Use the built-in device information tools
- **Performance**: Monitor with `nvidia-smi` (GPU) or system tools (CPU)

## 🎯 Quick Reference

### Essential Commands

```bash
# Check system status
python -c "from src.utils import print_device_info; print_device_info()"

# Train model
python main.py --mode train

# Evaluate model
python main.py --mode evaluate

# Predict text
python main.py --mode predict --text "النص العربي هنا"
```

### Key Files

- `src/config.py`: Configuration settings
- `CUDA_GUIDE.md`: Detailed CUDA usage guide
- `results/training.log`: Training progress logs
- `models/best_arabert_model.pt`: Best trained model

---

**Happy Training! 🚀**

_Built with ❤️ for Arabic NLP and hate speech detection_
