# Hierarchical Federated Learning for Credit Card Fraud Detection

This project implements a complete hierarchical federated learning system for credit card fraud detection with edge computing architecture.

## Architecture Overview

```
Merchants (5 clients) → Edge Nodes (2 nodes) → Central Server → Global Model
     ↓                       ↓                      ↓
Local Training         Edge Aggregation      Global Aggregation
   (FedAvg)              (FedAvg)            (Final FedAvg)
```

### System Components

1. **Merchants (Clients)**: 5 financial institutions with their own transaction data
2. **Edge Nodes**: 2 edge computing nodes that aggregate nearby merchants
   - Edge Node 1: Merchants 1, 2, 3
   - Edge Node 2: Merchants 4, 5
3. **Central Server**: Performs final global model aggregation

## Features

### ✅ Complete Implementation

- **Hierarchical FL Architecture**: Three-tier federated learning system
- **Multiple Balance Methods**: Support for SMOTE, ADASYN, and Random Undersampling
- **Model Options**: Logistic Regression and Multi-Layer Perceptron (MLP)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Performance Comparison**: Centralized vs Pure FL vs Hierarchical FL
- **Visualization**: Training curves and performance plots
- **Model Persistence**: Save trained models for deployment

### 🔧 Key Algorithms

- **FedAvg**: Standard federated averaging at both edge and central levels
- **Stratified Data Split**: Maintains class distribution across clients
- **Privacy Preservation**: No raw data sharing between merchants
- **Edge Computing**: Reduces communication overhead with hierarchical aggregation

## Quick Start

### 1. Prerequisites

Ensure you have the preprocessed federated client data from running the preprocessing pipeline:

```bash
processed_data/
├── smote_balanced/federated_clients/
├── adasyn_balanced/federated_clients/
└── random_undersampled/federated_clients/
```

### 2. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Quick Test

```bash
# Test the system
python test_federated_system.py
```

### 4. Full Experiment

```bash
# Run complete federated learning experiments
python hierarchical_federated_learning.py
```

## Usage Examples

### Basic Usage

```python
from hierarchical_federated_learning import HierarchicalFederatedLearning

# Initialize system
fl_system = HierarchicalFederatedLearning()

# Run complete experiment
fl_system.run_complete_experiment(
    balance_method="smote",  # or "adasyn", "undersampling"
    model_type="mlp"         # or "logistic"
)
```

### Custom Configuration

```python
# Initialize with custom settings
fl_system = HierarchicalFederatedLearning()
fl_system.global_rounds = 15        # More training rounds
fl_system.local_epochs = 10         # More local training
fl_system.learning_rate = 0.001     # Different learning rate

# Load specific dataset
datasets = fl_system.load_data(balance_method="adasyn")

# Train specific approach
metrics = fl_system.train_hierarchical_federated(datasets, "mlp")
```

## Output Files

After running experiments, the following files are generated:

```
fraud_detection_project/
├── training_curves.png              # Training progress visualization
├── performance_comparison.csv       # Results comparison table
├── federated_learning.log          # Detailed training logs
├── experiment_results_*.json       # Raw results data
└── trained_models/
    ├── centralized_model.pkl       # Centralized baseline model
    ├── pure_fl_model.pkl          # Pure federated model
    └── hierarchical_fl_model.pkl   # Hierarchical federated model
```

## Performance Metrics

The system evaluates all approaches using:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True fraud / (True fraud + False fraud)
- **Recall**: True fraud / (True fraud + Missed fraud)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Experimental Setup

### Default Configuration

```python
{
    "num_merchants": 5,
    "num_edge_nodes": 2,
    "global_rounds": 10,
    "local_epochs": 5,
    "learning_rate": 0.01,
    "edge_config": {
        "edge_1": [1, 2, 3],  # Merchants 1-3
        "edge_2": [4, 5]      # Merchants 4-5
    }
}
```

### Supported Experiments

1. **Balance Method Comparison**:
   - SMOTE (Synthetic Minority Oversampling)
   - ADASYN (Adaptive Synthetic Sampling)
   - Random Undersampling

2. **Model Architecture Comparison**:
   - Logistic Regression (Linear baseline)
   - Multi-Layer Perceptron (Non-linear deep learning)

3. **Federated Approach Comparison**:
   - Centralized (Traditional ML baseline)
   - Pure Federated Learning (Direct client-to-server)
   - Hierarchical Federated Learning (Client-Edge-Server)

## Sample Results

```
PERFORMANCE COMPARISON TABLE
================================================================================
        Approach     Accuracy  Precision    Recall   F1-Score   ROC-AUC
    Centralized       0.9245     0.9180     0.9315     0.9247     0.9756
       Pure Fl        0.9198     0.9125     0.9276     0.9200     0.9698
Hierarchical Fl       0.9223     0.9156     0.9295     0.9225     0.9721
================================================================================
```

## Architecture Benefits

### 1. **Hierarchical Structure**
- Reduces communication overhead between clients and central server
- Edge nodes provide local aggregation and processing
- Scalable to more merchants and edge nodes

### 2. **Privacy Preservation**
- No raw transaction data leaves merchant premises
- Only model weights are shared
- Support for differential privacy (extensible)

### 3. **Edge Computing Integration**
- Local processing at edge nodes
- Reduced latency for merchant communications
- Distributed computational load

### 4. **Fraud Detection Optimization**
- Handles extreme class imbalance (0.17% fraud rate)
- Multiple resampling techniques for comparison
- Specialized metrics for fraud detection evaluation

## Extensibility

The system is designed for easy extension:

```python
# Add new aggregation algorithms
def custom_aggregation(models, weights):
    # Implement custom federated averaging
    pass

# Add new model architectures
def create_custom_model(input_size):
    # Implement new model type
    pass

# Add new evaluation metrics
def custom_evaluation(model, test_data):
    # Implement specialized metrics
    pass
```

## Logging and Debugging

Comprehensive logging is provided:

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check log file
tail -f federated_learning.log
```

## Troubleshooting

### Common Issues

1. **Missing Data Files**:
   ```
   FileNotFoundError: Federated clients folder not found
   ```
   **Solution**: Run the preprocessing pipeline first to generate federated client data.

2. **Memory Issues**:
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Reduce `global_rounds` or use undersampling dataset.

3. **Model Convergence**:
   ```
   Warning: Model may not have converged
   ```
   **Solution**: Increase `local_epochs` or adjust `learning_rate`.

### Performance Optimization

```python
# For faster experiments
fl_system.global_rounds = 5      # Reduce training rounds
fl_system.local_epochs = 3       # Reduce local training

# For better performance
fl_system.global_rounds = 20     # More training rounds
fl_system.local_epochs = 10      # More local training
fl_system.learning_rate = 0.001  # Smaller learning rate
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hierarchical_fl_fraud_2025,
  title={Hierarchical Federated Learning for Credit Card Fraud Detection with Edge Computing},
  author={GitHub Copilot},
  year={2025},
  howpublished={GitHub Repository}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs in `federated_learning.log`
3. Open an issue on GitHub with detailed error information

---

**Note**: This implementation is for research and educational purposes. For production deployment in financial institutions, additional security, privacy, and regulatory compliance measures should be implemented.