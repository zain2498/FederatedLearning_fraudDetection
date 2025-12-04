"""
Test Script for Hierarchical Federated Learning System
====================================================

This script tests the federated learning system with a quick run
to verify all components work correctly.
"""

import os
import sys
from hierarchical_federated_learning import HierarchicalFederatedLearning
import logging

# Reduce logging verbosity for test
logging.getLogger().setLevel(logging.WARNING)

def test_system():
    """Test the federated learning system"""
    print("Testing Hierarchical Federated Learning System")
    print("=" * 50)
    
    # Initialize system
    fl_system = HierarchicalFederatedLearning()
    
    # Test configuration for quick run
    fl_system.global_rounds = 3  # Reduced for testing
    fl_system.local_epochs = 2   # Reduced for testing
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        datasets = fl_system.load_data(balance_method="smote")
        print(f"   [OK] Loaded data for {len(datasets)} merchants")
        
        # Test model creation
        print("2. Testing model creation...")
        sample_data = list(datasets.values())[0]
        model = fl_system.create_model("mlp", sample_data['X_train'].shape[1])
        print(f"   [OK] Created MLP model")
        
        # Test local training
        print("3. Testing local training...")
        local_model, loss = fl_system.train_local(sample_data, model_type="mlp")
        print(f"   [OK] Local training completed, loss: {loss:.4f}")
        
        # Test evaluation
        print("4. Testing evaluation...")
        metrics = fl_system.evaluate_model(local_model, sample_data, "Test Model")
        print(f"   [OK] Evaluation completed, accuracy: {metrics['accuracy']:.4f}")
        
        # Test quick centralized baseline
        print("5. Testing centralized baseline...")
        centralized_metrics = fl_system.train_centralized_baseline(datasets, "mlp")
        print(f"   [OK] Centralized training completed")
        
        print("\n" + "=" * 50)
        print("QUICK TEST RESULTS:")
        print("-" * 50)
        print(f"Centralized Accuracy: {centralized_metrics['accuracy']:.4f}")
        print(f"Centralized F1-Score: {centralized_metrics['f1_score']:.4f}")
        print(f"Centralized ROC-AUC: {centralized_metrics['roc_auc']:.4f}")
        print("=" * 50)
        
        print("\n[SUCCESS] All tests passed! System is ready for full experiments.")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_experiment():
    """Run a quick federated learning experiment"""
    print("\nRunning Quick Federated Learning Experiment")
    print("=" * 50)
    
    # Initialize system with reduced parameters
    fl_system = HierarchicalFederatedLearning()
    fl_system.global_rounds = 3
    fl_system.local_epochs = 2
    
    try:
        # Load data
        datasets = fl_system.load_data(balance_method="smote")
        
        # Run only hierarchical FL for quick test
        print("Running hierarchical federated learning...")
        hierarchical_metrics = fl_system.train_hierarchical_federated(datasets, "mlp")
        
        # Show results
        print("\n" + "=" * 50)
        print("QUICK EXPERIMENT RESULTS:")
        print("-" * 50)
        print(f"Hierarchical FL Accuracy: {hierarchical_metrics['accuracy']:.4f}")
        print(f"Hierarchical FL Precision: {hierarchical_metrics['precision']:.4f}")
        print(f"Hierarchical FL Recall: {hierarchical_metrics['recall']:.4f}")
        print(f"Hierarchical FL F1-Score: {hierarchical_metrics['f1_score']:.4f}")
        print(f"Hierarchical FL ROC-AUC: {hierarchical_metrics['roc_auc']:.4f}")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Quick experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run tests
    test_success = test_system()
    
    if test_success:
        # Run quick experiment
        experiment_success = run_quick_experiment()
        
        if experiment_success:
            print("\n" + "="*60)
            print("SYSTEM READY FOR FULL EXPERIMENTS!")
            print("="*60)
            print("To run full experiments, execute:")
            print("python hierarchical_federated_learning.py")
            print("="*60)
        else:
            print("\n[WARNING] Quick experiment had issues, but basic system works")
    else:
        print("\n[ERROR] Basic tests failed. Please check your data setup.")