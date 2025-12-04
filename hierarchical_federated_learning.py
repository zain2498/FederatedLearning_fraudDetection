"""
Hierarchical Federated Learning for Credit Card Fraud Detection with Edge Computing
================================================================================

This implementation provides a complete federated learning system with:
- Merchants (Clients) -> Edge Aggregators -> Central Server -> Global Model
- Support for SMOTE, ADASYN, and undersampling datasets
- Performance comparison between centralized, pure FL, and hierarchical FL
- Comprehensive evaluation metrics and visualization

Author: GitHub Copilot
Date: December 2025
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('federated_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HierarchicalFederatedLearning:
    """
    Hierarchical Federated Learning System for Fraud Detection
    
    Architecture:
    Merchants (5 clients) -> Edge Nodes (2-3 merchants each) -> Central Server
    """
    
    def __init__(self, project_path: str = "d:/PracticeProject/fraud_detection_project"):
        """Initialize the federated learning system"""
        self.project_path = project_path
        self.processed_data_path = os.path.join(project_path, "processed_data")
        
        # FL Configuration
        self.num_merchants = 5
        self.num_edge_nodes = 2  # Edge node 1: merchants 1,2,3 | Edge node 2: merchants 4,5
        self.global_rounds = 10
        self.local_epochs = 5
        self.learning_rate = 0.01
        
        # Model configurations
        self.model_configs = {
            'logistic': {'max_iter': 1000, 'random_state': 42},
            'mlp': {'hidden_layer_sizes': (64, 32), 'max_iter': 500, 'random_state': 42}
        }
        
        # Results storage
        self.results = {
            'centralized': {},
            'pure_fl': {},
            'hierarchical_fl': {}
        }
        
        # Edge configuration
        self.edge_config = {
            'edge_1': [1, 2, 3],  # Merchants 1, 2, 3
            'edge_2': [4, 5]      # Merchants 4, 5
        }
        
        logger.info("Hierarchical Federated Learning system initialized")
        logger.info(f"Project path: {self.project_path}")
        logger.info(f"Edge configuration: {self.edge_config}")

    def load_data(self, balance_method: str = "smote") -> Dict[str, Dict]:
        """
        Load preprocessed datasets for all merchants
        
        Args:
            balance_method: 'smote', 'adasyn', or 'undersampling'
        
        Returns:
            Dictionary containing merchant datasets
        """
        logger.info(f"Loading data with balance method: {balance_method}")
        
        datasets = {}
        
        # Find the correct dataset folder
        balance_folders = {
            'smote': 'smote',
            'adasyn': 'adasyn', 
            'undersampling': 'undersampling'
        }
        
        # Load federated client data
        federated_folder = os.path.join(self.processed_data_path, "federated_data", balance_folders[balance_method])
        
        if not os.path.exists(federated_folder):
            raise FileNotFoundError(f"Federated clients folder not found: {federated_folder}")
        
        # Load each merchant's data
        for merchant_id in range(1, self.num_merchants + 1):
            client_file = os.path.join(federated_folder, f"client_{merchant_id}_data.csv")
            
            if os.path.exists(client_file):
                # Load merchant data
                df = pd.read_csv(client_file)
                
                # Separate features and target
                if 'Class' in df.columns:
                    X = df.drop('Class', axis=1)
                    y = df['Class']
                else:
                    # Assume last column is target
                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]
                
                # Split into train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                datasets[f'merchant_{merchant_id}'] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'full_X': X,
                    'full_y': y
                }
                
                logger.info(f"Merchant {merchant_id}: {len(X_train)} train, {len(X_test)} test samples")
                logger.info(f"  Fraud rate: {(y.sum() / len(y) * 100):.2f}%")
            else:
                logger.warning(f"Client data file not found: {client_file}")
        
        if not datasets:
            raise ValueError("No merchant data loaded. Check if federated client data exists.")
        
        logger.info(f"Successfully loaded data for {len(datasets)} merchants")
        return datasets

    def create_model(self, model_type: str = "mlp", input_size: int = None) -> Any:
        """
        Create a lightweight model for fraud detection
        
        Args:
            model_type: 'logistic' or 'mlp'
            input_size: Number of input features
        
        Returns:
            Model instance
        """
        if model_type == "logistic":
            model = LogisticRegression(**self.model_configs['logistic'])
        elif model_type == "mlp":
            model = MLPClassifier(**self.model_configs['mlp'])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model

    def get_model_weights(self, model: Any) -> Dict:
        """Extract model weights/parameters"""
        if hasattr(model, 'coef_'):
            return {
                'coef_': model.coef_.copy() if model.coef_ is not None else None,
                'intercept_': model.intercept_.copy() if model.intercept_ is not None else None
            }
        else:
            # For other models, we'll use a different approach
            return {'params': model.get_params()}

    def set_model_weights(self, model: Any, weights: Dict) -> Any:
        """Set model weights/parameters"""
        if 'coef_' in weights:
            if weights['coef_'] is not None:
                model.coef_ = weights['coef_']
            if weights['intercept_'] is not None:
                model.intercept_ = weights['intercept_']
        return model

    def federated_average(self, models: List[Any], weights: List[float] = None) -> Dict:
        """
        Perform federated averaging on model weights
        
        Args:
            models: List of trained models
            weights: List of weights for averaging (e.g., based on data size)
        
        Returns:
            Averaged model weights
        """
        if weights is None:
            weights = [1.0] * len(models)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Get weights from first model to initialize
        avg_weights = self.get_model_weights(models[0])
        
        # Average coefficients
        if 'coef_' in avg_weights and avg_weights['coef_'] is not None:
            avg_coef = np.zeros_like(avg_weights['coef_'])
            avg_intercept = np.zeros_like(avg_weights['intercept_'])
            
            for i, model in enumerate(models):
                model_weights = self.get_model_weights(model)
                if model_weights['coef_'] is not None:
                    avg_coef += weights[i] * model_weights['coef_']
                    avg_intercept += weights[i] * model_weights['intercept_']
            
            avg_weights['coef_'] = avg_coef
            avg_weights['intercept_'] = avg_intercept
        
        return avg_weights

    def train_local(self, merchant_data: Dict, global_weights: Dict = None, 
                   model_type: str = "mlp") -> Tuple[Any, float]:
        """
        Train model locally at a merchant
        
        Args:
            merchant_data: Merchant's training data
            global_weights: Global model weights to initialize with
            model_type: Type of model to use
        
        Returns:
            Trained model and training loss
        """
        X_train = merchant_data['X_train']
        y_train = merchant_data['y_train']
        
        # Create and initialize model
        model = self.create_model(model_type, input_size=X_train.shape[1])
        
        # Initialize with global weights if provided
        if global_weights is not None:
            try:
                model = self.set_model_weights(model, global_weights)
            except:
                pass  # If setting weights fails, continue with random initialization
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Calculate training loss (approximation)
        try:
            train_pred = model.predict_proba(X_train)[:, 1]
            train_loss = -np.mean(y_train * np.log(train_pred + 1e-15) + 
                                (1 - y_train) * np.log(1 - train_pred + 1e-15))
        except:
            train_loss = 0.0
        
        return model, train_loss

    def aggregate_edge(self, edge_models: List[Any], edge_weights: List[float] = None) -> Dict:
        """
        Aggregate models at edge node level
        
        Args:
            edge_models: Models from merchants connected to this edge node
            edge_weights: Weights for aggregation (data sizes)
        
        Returns:
            Aggregated edge model weights
        """
        logger.info(f"Aggregating {len(edge_models)} models at edge node")
        return self.federated_average(edge_models, edge_weights)

    def aggregate_central(self, edge_weights: List[Dict], central_weights: List[float] = None) -> Dict:
        """
        Aggregate edge weights at central server
        
        Args:
            edge_weights: Weights from edge nodes
            central_weights: Weights for central aggregation
        
        Returns:
            Global model weights
        """
        logger.info(f"Aggregating {len(edge_weights)} edge models at central server")
        
        if central_weights is None:
            central_weights = [1.0] * len(edge_weights)
        
        # Normalize weights
        total_weight = sum(central_weights)
        central_weights = [w / total_weight for w in central_weights]
        
        # Average edge weights
        avg_weights = None
        
        for i, edge_weight in enumerate(edge_weights):
            if avg_weights is None:
                avg_weights = {k: v.copy() if isinstance(v, np.ndarray) else v 
                              for k, v in edge_weight.items()}
            else:
                for key in avg_weights:
                    if isinstance(avg_weights[key], np.ndarray):
                        avg_weights[key] += central_weights[i] * edge_weight[key]
        
        # Normalize by the first weight (already normalized above)
        return avg_weights

    def evaluate_model(self, model: Any, test_data: Dict, model_name: str = "") -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            test_data: Test dataset
            model_name: Name for logging
        
        Returns:
            Evaluation metrics
        """
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            pass
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary')
        }
        
        # ROC-AUC if probabilities available
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.5
        else:
            metrics['roc_auc'] = 0.5
        
        # Log results
        if model_name:
            logger.info(f"{model_name} Evaluation:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics

    def train_centralized_baseline(self, datasets: Dict, model_type: str = "mlp") -> Dict:
        """
        Train centralized baseline model
        
        Args:
            datasets: All merchant datasets
            model_type: Type of model to use
        
        Returns:
            Evaluation metrics
        """
        logger.info("Training centralized baseline model...")
        
        # Combine all training data
        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        
        for merchant_data in datasets.values():
            X_train_list.append(merchant_data['X_train'])
            y_train_list.append(merchant_data['y_train'])
            X_test_list.append(merchant_data['X_test'])
            y_test_list.append(merchant_data['y_test'])
        
        # Concatenate all data
        X_train_combined = pd.concat(X_train_list, ignore_index=True)
        y_train_combined = pd.concat(y_train_list, ignore_index=True)
        X_test_combined = pd.concat(X_test_list, ignore_index=True)
        y_test_combined = pd.concat(y_test_list, ignore_index=True)
        
        # Train model
        model = self.create_model(model_type, input_size=X_train_combined.shape[1])
        model.fit(X_train_combined, y_train_combined)
        
        # Evaluate
        test_data = {'X_test': X_test_combined, 'y_test': y_test_combined}
        metrics = self.evaluate_model(model, test_data, "Centralized Baseline")
        
        # Store results
        self.results['centralized'] = {
            'metrics': metrics,
            'model': model
        }
        
        return metrics

    def train_pure_federated(self, datasets: Dict, model_type: str = "mlp") -> Dict:
        """
        Train pure federated learning model (no edge nodes)
        
        Args:
            datasets: All merchant datasets
            model_type: Type of model to use
        
        Returns:
            Evaluation metrics and training history
        """
        logger.info("Training pure federated learning model...")
        
        # Initialize global model
        sample_data = list(datasets.values())[0]
        global_model = self.create_model(model_type, input_size=sample_data['X_train'].shape[1])
        
        # Fit once to initialize weights
        global_model.fit(sample_data['X_train'][:10], sample_data['y_train'][:10])
        
        training_history = []
        
        # Federated training rounds
        for round_num in range(self.global_rounds):
            logger.info(f"Pure FL Round {round_num + 1}/{self.global_rounds}")
            
            merchant_models = []
            merchant_weights = []
            
            # Local training at each merchant
            for merchant_name, merchant_data in datasets.items():
                global_weights = self.get_model_weights(global_model)
                
                # Train locally
                local_model, train_loss = self.train_local(
                    merchant_data, global_weights, model_type
                )
                
                merchant_models.append(local_model)
                merchant_weights.append(len(merchant_data['X_train']))
                
                logger.info(f"  {merchant_name}: {len(merchant_data['X_train'])} samples, "
                           f"loss: {train_loss:.4f}")
            
            # Federal averaging
            avg_weights = self.federated_average(merchant_models, merchant_weights)
            global_model = self.set_model_weights(global_model, avg_weights)
            
            # Evaluate on combined test set
            X_test_combined = pd.concat([d['X_test'] for d in datasets.values()], ignore_index=True)
            y_test_combined = pd.concat([d['y_test'] for d in datasets.values()], ignore_index=True)
            
            test_data = {'X_test': X_test_combined, 'y_test': y_test_combined}
            round_metrics = self.evaluate_model(global_model, test_data, f"Pure FL Round {round_num + 1}")
            
            training_history.append(round_metrics)
        
        # Final evaluation
        final_metrics = training_history[-1]
        
        # Store results
        self.results['pure_fl'] = {
            'metrics': final_metrics,
            'model': global_model,
            'history': training_history
        }
        
        return final_metrics

    def train_hierarchical_federated(self, datasets: Dict, model_type: str = "mlp") -> Dict:
        """
        Train hierarchical federated learning model
        
        Args:
            datasets: All merchant datasets
            model_type: Type of model to use
        
        Returns:
            Evaluation metrics and training history
        """
        logger.info("Training hierarchical federated learning model...")
        
        # Initialize global model
        sample_data = list(datasets.values())[0]
        global_model = self.create_model(model_type, input_size=sample_data['X_train'].shape[1])
        
        # Fit once to initialize weights
        global_model.fit(sample_data['X_train'][:10], sample_data['y_train'][:10])
        
        training_history = []
        
        # Hierarchical federated training rounds
        for round_num in range(self.global_rounds):
            logger.info(f"Hierarchical FL Round {round_num + 1}/{self.global_rounds}")
            
            edge_aggregated_weights = []
            edge_total_samples = []
            
            # Process each edge node
            for edge_name, merchant_ids in self.edge_config.items():
                logger.info(f"  Processing {edge_name} with merchants {merchant_ids}")
                
                edge_models = []
                edge_weights = []
                
                # Local training at merchants under this edge
                for merchant_id in merchant_ids:
                    merchant_name = f'merchant_{merchant_id}'
                    
                    if merchant_name in datasets:
                        merchant_data = datasets[merchant_name]
                        global_weights = self.get_model_weights(global_model)
                        
                        # Train locally
                        local_model, train_loss = self.train_local(
                            merchant_data, global_weights, model_type
                        )
                        
                        edge_models.append(local_model)
                        edge_weights.append(len(merchant_data['X_train']))
                        
                        logger.info(f"    {merchant_name}: {len(merchant_data['X_train'])} samples, "
                                   f"loss: {train_loss:.4f}")
                
                # Edge aggregation
                if edge_models:
                    edge_avg_weights = self.aggregate_edge(edge_models, edge_weights)
                    edge_aggregated_weights.append(edge_avg_weights)
                    edge_total_samples.append(sum(edge_weights))
                    
                    logger.info(f"    {edge_name} aggregated {len(edge_models)} models "
                               f"({sum(edge_weights)} total samples)")
            
            # Central aggregation
            if edge_aggregated_weights:
                global_weights = self.aggregate_central(edge_aggregated_weights, edge_total_samples)
                global_model = self.set_model_weights(global_model, global_weights)
                
                logger.info(f"  Central server aggregated {len(edge_aggregated_weights)} edge nodes")
            
            # Evaluate on combined test set
            X_test_combined = pd.concat([d['X_test'] for d in datasets.values()], ignore_index=True)
            y_test_combined = pd.concat([d['y_test'] for d in datasets.values()], ignore_index=True)
            
            test_data = {'X_test': X_test_combined, 'y_test': y_test_combined}
            round_metrics = self.evaluate_model(global_model, test_data, 
                                              f"Hierarchical FL Round {round_num + 1}")
            
            training_history.append(round_metrics)
        
        # Final evaluation
        final_metrics = training_history[-1]
        
        # Store results
        self.results['hierarchical_fl'] = {
            'metrics': final_metrics,
            'model': global_model,
            'history': training_history
        }
        
        return final_metrics

    def plot_training_curves(self):
        """Plot training curves for comparison"""
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy curves
        plt.subplot(1, 3, 1)
        if 'pure_fl' in self.results and 'history' in self.results['pure_fl']:
            rounds = range(1, len(self.results['pure_fl']['history']) + 1)
            accuracies = [h['accuracy'] for h in self.results['pure_fl']['history']]
            plt.plot(rounds, accuracies, label='Pure FL', marker='o')
        
        if 'hierarchical_fl' in self.results and 'history' in self.results['hierarchical_fl']:
            rounds = range(1, len(self.results['hierarchical_fl']['history']) + 1)
            accuracies = [h['accuracy'] for h in self.results['hierarchical_fl']['history']]
            plt.plot(rounds, accuracies, label='Hierarchical FL', marker='s')
        
        plt.xlabel('Training Round')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Comparison')
        plt.legend()
        plt.grid(True)
        
        # Plot F1-score curves
        plt.subplot(1, 3, 2)
        if 'pure_fl' in self.results and 'history' in self.results['pure_fl']:
            rounds = range(1, len(self.results['pure_fl']['history']) + 1)
            f1_scores = [h['f1_score'] for h in self.results['pure_fl']['history']]
            plt.plot(rounds, f1_scores, label='Pure FL', marker='o')
        
        if 'hierarchical_fl' in self.results and 'history' in self.results['hierarchical_fl']:
            rounds = range(1, len(self.results['hierarchical_fl']['history']) + 1)
            f1_scores = [h['f1_score'] for h in self.results['hierarchical_fl']['history']]
            plt.plot(rounds, f1_scores, label='Hierarchical FL', marker='s')
        
        plt.xlabel('Training Round')
        plt.ylabel('F1-Score')
        plt.title('Training F1-Score Comparison')
        plt.legend()
        plt.grid(True)
        
        # Plot ROC-AUC curves
        plt.subplot(1, 3, 3)
        if 'pure_fl' in self.results and 'history' in self.results['pure_fl']:
            rounds = range(1, len(self.results['pure_fl']['history']) + 1)
            roc_aucs = [h['roc_auc'] for h in self.results['pure_fl']['history']]
            plt.plot(rounds, roc_aucs, label='Pure FL', marker='o')
        
        if 'hierarchical_fl' in self.results and 'history' in self.results['hierarchical_fl']:
            rounds = range(1, len(self.results['hierarchical_fl']['history']) + 1)
            roc_aucs = [h['roc_auc'] for h in self.results['hierarchical_fl']['history']]
            plt.plot(rounds, roc_aucs, label='Hierarchical FL', marker='s')
        
        plt.xlabel('Training Round')
        plt.ylabel('ROC-AUC')
        plt.title('Training ROC-AUC Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.project_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def create_performance_comparison_table(self):
        """Create and display performance comparison table"""
        comparison_data = []
        
        for approach, data in self.results.items():
            if 'metrics' in data:
                metrics = data['metrics']
                comparison_data.append({
                    'Approach': approach.replace('_', ' ').title(),
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1_score']:.4f}",
                    'ROC-AUC': f"{metrics['roc_auc']:.4f}"
                })
        
        # Create DataFrame and display
        df_comparison = pd.DataFrame(comparison_data)
        
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON TABLE")
        print("="*80)
        print(df_comparison.to_string(index=False))
        print("="*80)
        
        # Save to CSV
        df_comparison.to_csv(os.path.join(self.project_path, 'performance_comparison.csv'), index=False)
        
        return df_comparison

    def save_models(self):
        """Save trained models"""
        models_dir = os.path.join(self.project_path, 'trained_models')
        os.makedirs(models_dir, exist_ok=True)
        
        for approach, data in self.results.items():
            if 'model' in data:
                model_path = os.path.join(models_dir, f'{approach}_model.pkl')
                
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(data['model'], f)
                    logger.info(f"Saved {approach} model to {model_path}")
                except Exception as e:
                    logger.error(f"Failed to save {approach} model: {e}")

    def run_complete_experiment(self, balance_method: str = "smote", model_type: str = "mlp"):
        """
        Run complete federated learning experiment
        
        Args:
            balance_method: 'smote', 'adasyn', or 'undersampling'
            model_type: 'logistic' or 'mlp'
        """
        logger.info(f"Starting complete FL experiment with {balance_method} data and {model_type} model")
        
        try:
            # Load data
            datasets = self.load_data(balance_method)
            
            # Train all approaches
            logger.info("Training centralized baseline...")
            self.train_centralized_baseline(datasets, model_type)
            
            logger.info("Training pure federated learning...")
            self.train_pure_federated(datasets, model_type)
            
            logger.info("Training hierarchical federated learning...")
            self.train_hierarchical_federated(datasets, model_type)
            
            # Generate results
            logger.info("Generating performance comparison...")
            self.create_performance_comparison_table()
            
            logger.info("Plotting training curves...")
            self.plot_training_curves()
            
            logger.info("Saving models...")
            self.save_models()
            
            # Save experiment results
            results_file = os.path.join(self.project_path, f'experiment_results_{balance_method}_{model_type}.json')
            
            # Prepare results for JSON (remove non-serializable objects)
            json_results = {}
            for approach, data in self.results.items():
                json_results[approach] = {
                    'metrics': data['metrics'],
                    'history': data.get('history', [])
                }
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Experiment completed successfully! Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise

def main():
    """Main execution function"""
    print("Hierarchical Federated Learning for Credit Card Fraud Detection")
    print("="*65)
    
    # Initialize system
    fl_system = HierarchicalFederatedLearning()
    
    # Run experiments with different configurations
    experiments = [
        {'balance_method': 'smote', 'model_type': 'mlp'},
        {'balance_method': 'smote', 'model_type': 'logistic'},
        {'balance_method': 'adasyn', 'model_type': 'mlp'}
    ]
    
    for i, config in enumerate(experiments, 1):
        print(f"\nExperiment {i}/{len(experiments)}: {config}")
        print("-" * 50)
        
        try:
            fl_system.run_complete_experiment(**config)
        except Exception as e:
            logger.error(f"Experiment {i} failed: {e}")
            continue
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()