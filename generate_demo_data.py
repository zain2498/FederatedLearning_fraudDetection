"""
Demo Credit Card Dataset Generator
Creates a synthetic dataset similar to Kaggle's Credit Card Fraud Detection dataset
for testing the preprocessing pipeline when the actual dataset is not available.

This is for DEMO/TESTING purposes only - use the real Kaggle dataset for actual thesis work.
"""

import pandas as pd
import numpy as np
import os

def generate_demo_creditcard_data(n_samples=10000, fraud_rate=0.00172):
    """
    Generate synthetic credit card transaction data for testing
    
    Args:
        n_samples: Total number of transactions to generate
        fraud_rate: Percentage of fraudulent transactions (default matches real dataset)
    
    Returns:
        pandas.DataFrame: Synthetic dataset matching creditcard.csv structure
    """
    np.random.seed(42)  # For reproducibility
    
    print(f"Generating {n_samples:,} synthetic transactions...")
    print(f"Target fraud rate: {fraud_rate:.4%}")
    
    # Calculate fraud and normal counts
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    print(f"Normal transactions: {n_normal:,}")
    print(f"Fraud transactions: {n_fraud:,}")
    
    # Generate Time feature (0 to 172792 seconds, about 48 hours)
    time_values = np.random.uniform(0, 172792, n_samples)
    
    # Generate V1-V28 features (PCA components, normally distributed)
    v_features = {}
    for i in range(1, 29):
        v_features[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Generate Amount feature (log-normal distribution to simulate real transaction amounts)
    amount_values = np.random.lognormal(mean=2, sigma=1.5, size=n_samples)
    amount_values = np.clip(amount_values, 0, 25000)  # Reasonable transaction limits
    
    # Create Class labels (0=Normal, 1=Fraud)
    class_labels = np.concatenate([
        np.zeros(n_normal, dtype=int),      # Normal transactions
        np.ones(n_fraud, dtype=int)         # Fraudulent transactions  
    ])
    
    # Shuffle the data
    shuffle_indices = np.random.permutation(n_samples)
    
    # Create DataFrame
    data = {
        'Time': time_values[shuffle_indices],
        **{f'V{i}': v_features[f'V{i}'][shuffle_indices] for i in range(1, 29)},
        'Amount': amount_values[shuffle_indices],
        'Class': class_labels[shuffle_indices]
    }
    
    df = pd.DataFrame(data)
    
    # Adjust V features slightly based on class to make some discrimination possible
    fraud_mask = df['Class'] == 1
    for i in [1, 2, 3, 4, 7, 10, 11, 12, 14, 16, 17, 18]:
        df.loc[fraud_mask, f'V{i}'] += np.random.normal(0.5, 0.3, fraud_mask.sum())
    
    # Adjust amount distribution for frauds (tend to be smaller amounts)
    df.loc[fraud_mask, 'Amount'] *= 0.3
    
    print(f"\nDataset generated successfully!")
    print(f"Final fraud rate: {df['Class'].mean():.4%}")
    print(f"Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
    
    return df


def main():
    """Generate demo dataset and save to creditcard.csv"""
    print("=" * 60)
    print("DEMO CREDIT CARD DATASET GENERATOR")
    print("For Testing Fraud Detection Preprocessing Pipeline")
    print("=" * 60)
    
    # Generate demo dataset
    demo_df = generate_demo_creditcard_data(n_samples=50000, fraud_rate=0.00172)
    
    # Save to creditcard.csv
    output_path = 'creditcard.csv'
    demo_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Demo dataset saved as: {output_path}")
    print(f"  Size: {len(demo_df):,} transactions")
    print(f"  Columns: {list(demo_df.columns)}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Normal transactions: {(demo_df['Class'] == 0).sum():,}")
    print(f"  Fraudulent transactions: {(demo_df['Class'] == 1).sum():,}")
    print(f"  Fraud rate: {demo_df['Class'].mean():.4%}")
    
    print(f"\n🚨 IMPORTANT NOTES:")
    print(f"  - This is SYNTHETIC data for testing purposes only")
    print(f"  - For your actual thesis, download the real Kaggle dataset:")
    print(f"    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print(f"  - Replace this demo file with the real creditcard.csv")
    print(f"  - The real dataset has 284,807 transactions with 492 frauds")
    
    print(f"\n✅ Ready to run: python credit_card_preprocessor.py")


if __name__ == "__main__":
    main()