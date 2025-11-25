# Credit Card Fraud Detection Preprocessor

## 🎓 Thesis Project
**Title**: "Federated Learning for Fraud Detection in Credit Card Transaction using Edge Computing for Privacy-Preserving AI"

This preprocessing pipeline prepares credit card fraud data for federated learning implementation with edge computing constraints.

## 📁 Project Structure
```
fraud_detection_project/
├── credit_card_preprocessor.py    # Main preprocessing pipeline
├── generate_demo_data.py          # Demo dataset generator
├── requirements.txt               # Python dependencies
├── creditcard.csv                # Dataset (you need to provide this)
├── processed_data/               # Generated outputs
│   ├── smote/                   # SMOTE balanced data
│   ├── adasyn/                  # ADASYN balanced data
│   ├── undersampling/           # Under-sampled data
│   ├── federated_data/          # Client data for 5 edge nodes
│   ├── visualizations/          # Plots and charts
│   └── metadata/                # JSON reports
└── fraud_env/                   # Python virtual environment
```

## 🔧 Setup Instructions

### 1. Dataset Preparation

**Option A: Real Kaggle Dataset (Recommended for thesis)**
1. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Extract and place `creditcard.csv` in this folder
3. The file should be ~143MB with 284,807 transactions

**Option B: Demo Dataset (For testing)**
```powershell
# Activate virtual environment
.\fraud_env\Scripts\Activate.ps1

# Generate synthetic demo data
python generate_demo_data.py

# This creates a sample creditcard.csv for testing
```

### 2. Install Dependencies
```powershell
# Activate virtual environment
.\fraud_env\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 3. Run Preprocessing
```powershell
# Make sure virtual environment is active
.\fraud_env\Scripts\Activate.ps1

# Run the preprocessing pipeline
python credit_card_preprocessor.py
```

## 📊 What the Preprocessor Does

### 1. **Data Loading & Exploration**
- Handles multiple CSV encodings (UTF-8, Latin-1, CP1252)
- Comprehensive data quality analysis
- Class imbalance detection and reporting
- Statistical summaries and visualizations

### 2. **Feature Scaling**
- RobustScaler for `Amount` and `Time` features
- Preserves V1-V28 PCA components (already normalized)
- Handles outliers effectively

### 3. **Class Balancing (3 Methods)**
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **ADASYN**: Adaptive Synthetic Sampling  
- **Random Under-sampling**: Reduces majority class

### 4. **Federated Learning Simulation**
- Splits data across 5 edge nodes (banks/merchants)
- Maintains class distribution per client
- Prepares for privacy-preserving ML

### 5. **Comprehensive Outputs**
- Balanced datasets for each method
- High-resolution visualizations (300 DPI)
- Detailed JSON metadata and statistics
- Complete processing logs

## 🔐 Security & Privacy Features

✅ **Company Laptop Compliant**
- No external API calls
- All processing is local
- No internet dependency during processing
- Secure data handling

✅ **Privacy-Preserving Design**
- Federated data splits prevent central data aggregation
- Edge computing simulation
- Privacy-by-design architecture

## 📈 Generated Visualizations

1. **Class Distribution Analysis** - Shows extreme imbalance
2. **Transaction Amount Analysis** - Box plots and distributions
3. **Feature Correlation Heatmap** - Identifies important relationships
4. **Balancing Techniques Comparison** - Before/after balancing
5. **Federated Distribution** - Data across edge nodes

## 🎯 Thesis Integration

### Research Contributions
- **Federated Learning**: 5-client simulation for real-world deployment
- **Edge Computing**: Resource-constrained processing simulation  
- **Privacy Preservation**: No centralized data aggregation
- **Class Imbalance Handling**: Multiple techniques for comparison

### Academic Analysis Ready
- Compare SMOTE vs ADASYN vs Under-sampling
- Federated vs Centralized performance analysis
- Privacy-utility trade-off evaluation
- Edge computing efficiency metrics

## 🔍 Troubleshooting

### Common Issues

**1. Unicode/Encoding Errors**
- Fixed: Uses ASCII symbols instead of Unicode
- Handles multiple CSV encodings automatically

**2. Missing Dependencies**
```powershell
pip install -r requirements.txt
```

**3. Corrupted Dataset**
- Use `generate_demo_data.py` to create test data
- Download fresh copy from Kaggle

**4. Memory Issues (Large Dataset)**
- The real dataset is ~143MB, ensure sufficient RAM
- Close other applications if needed

### File Locations
- **Logs**: `preprocessing_log.txt`
- **Outputs**: `processed_data/` folder
- **Metadata**: `processed_data/metadata/`

## 📋 Next Steps for Thesis

After successful preprocessing:

1. **Implement Federated Learning Architecture**
   - Local model training on each client
   - Secure aggregation mechanism
   - Communication efficiency optimization

2. **Edge Computing Deployment**
   - Resource constraint simulation
   - Latency and bandwidth analysis
   - Real-time inference testing

3. **Privacy Analysis**
   - Differential privacy implementation
   - Communication security protocols
   - Privacy-utility trade-off evaluation

4. **Performance Evaluation**
   - Compare federated vs centralized approaches
   - Analyze each balancing technique
   - Document academic findings

## 📄 Dependencies

Core libraries used:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning preprocessing
- `imbalanced-learn` - Class balancing techniques
- `matplotlib` & `seaborn` - Visualization
- Standard Python libraries (logging, json, etc.)

## 🎓 Academic Notes

This preprocessing pipeline supports the thesis research on:
- **Federated Learning**: Decentralized ML with privacy preservation
- **Edge Computing**: Resource-constrained real-time processing
- **Fraud Detection**: Financial security and anomaly detection
- **Privacy-Preserving AI**: Maintaining data confidentiality

The generated datasets enable comprehensive comparison of different approaches while maintaining academic rigor and real-world applicability.

---

**🚨 Important**: For your actual thesis submission, always use the real Kaggle dataset. The demo generator is only for testing the preprocessing pipeline when the real data is unavailable.