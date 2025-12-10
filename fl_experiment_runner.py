"""
Federated Learning Experiment Runner
Implements Centralized, Flat FL, and Hierarchical FL (HFL) with full experiment matrix
Author: Zain Badar
"""
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import copy

# ---------------------- Synthetic Data Generation ----------------------

def generate_synthetic_data(n_clients=100, n_samples=3000, fraud_ratio=0.005, n_features=20, seed=42):
    np.random.seed(seed)
    data = []
    for client_id in range(n_clients):
        X = np.random.randn(n_samples, n_features)
        y = np.zeros(n_samples)
        # Default: uniform fraud
        fraud_indices = np.random.choice(n_samples, int(n_samples * fraud_ratio), replace=False)
        y[fraud_indices] = 1
        data.append({'client_id': client_id, 'X': X, 'y': y})
    return data

# ---------------------- Non-IID Partitioning ----------------------

def create_non_iid_partitions(data, n_edges=10, label_skew_edges=(0,1,2), skew_ratio=0.75, seed=42):
    """
    Concentrate 75% of fraud labels in Edges 1–3
    """
    np.random.seed(seed)
    edge_clients = {}
    client_counter = 0
    for edge in range(n_edges):
        n_clients = 10 if edge < n_edges-1 else 10
        edge_clients[edge] = list(range(client_counter, client_counter + n_clients))
        client_counter += n_clients
    # Skew fraud labels
    for edge in range(n_edges):
        for cid in edge_clients[edge]:
            client = data[cid]
            n = len(client['y'])
            if edge in label_skew_edges:
                n_fraud = int(n * 0.02 * skew_ratio)
            else:
                n_fraud = int(n * 0.02 * (1-skew_ratio))
            y = np.zeros(n)
            fraud_indices = np.random.choice(n, n_fraud, replace=False)
            y[fraud_indices] = 1
            client['y'] = y
    return edge_clients, data

# ---------------------- Model & Training Simulation ----------------------

def simulate_client_training(client_data, model_weights, local_epochs=1, lr=0.1, quantize=None, adversarial=False):
    """
    Simulate local training. Optionally quantize weights. Optionally flip fraud labels (adversarial).
    """
    X, y = client_data['X'], client_data['y']
    if adversarial:
        flip_mask = np.random.rand(len(y)) < 0.2
        y = np.where(flip_mask, 1-y, y)
    # Simple logistic regression update (simulate)
    grad = np.dot(X.T, (y - 1/(1+np.exp(-np.dot(X, model_weights))))) / len(y)
    new_weights = model_weights + lr * grad * local_epochs
    if quantize:
        new_weights = simulate_quantization(new_weights, quantize)
    return new_weights

def simulate_edge_aggregation(client_weights):
    return np.mean(client_weights, axis=0)

def simulate_cloud_aggregation(edge_weights):
    return np.mean(edge_weights, axis=0)

def simulate_quantization(weights, bits):
    min_w, max_w = np.min(weights), np.max(weights)
    levels = 2**bits
    scale = (max_w - min_w) / (levels - 1)
    quantized = np.round((weights - min_w) / scale) * scale + min_w
    return quantized

# ---------------------- Communication Cost Simulation ----------------------

def compute_communication_costs(weights, bits=32):
    n_params = weights.size
    bytes_sent = n_params * bits // 8
    return bytes_sent

# ---------------------- Experiment Runner ----------------------

def run_experiment_suite():
    seeds = [101, 202, 303]
    n_clients = 100
    n_edges = 10
    n_features = 20
    n_samples = 3000
    fraud_ratio = 0.005
    rounds = 50
    edge_agg_intervals = [1, 5, 20]
    quant_bits = [32, 8, 4]
    participation_rates = [0.2, 0.4]
    experiment_matrix = [
        ('A', 'Flat FL', 'IID', {'hierarchical': False, 'non_iid': False}),
        ('B', 'Hierarchical FL', 'IID', {'hierarchical': True, 'non_iid': False, 'edge_agg_interval': 1}),
        ('C', 'Flat vs HFL', 'Non-IID', {'hierarchical': False, 'non_iid': True}),
        ('C', 'Flat vs HFL', 'Non-IID', {'hierarchical': True, 'non_iid': True, 'edge_agg_interval': 1}),
        ('D', 'HFL Sparse', 'Non-IID', {'hierarchical': True, 'non_iid': True, 'edge_agg_interval': 20}),
        ('E', 'HFL Compression', 'Non-IID', {'hierarchical': True, 'non_iid': True, 'edge_agg_interval': 1, 'quantize': 8}),
        ('E', 'HFL Compression', 'Non-IID', {'hierarchical': True, 'non_iid': True, 'edge_agg_interval': 1, 'quantize': 4}),
        ('F', 'Adversarial Edge', 'Non-IID', {'hierarchical': False, 'non_iid': True, 'adversarial_edge': 0}),
        ('F', 'Adversarial Edge', 'Non-IID', {'hierarchical': True, 'non_iid': True, 'edge_agg_interval': 1, 'adversarial_edge': 0}),
    ]

    results = defaultdict(list)
    for exp_id, exp_name, exp_type, params in experiment_matrix:
        print(f"\n=== Experiment {exp_id}: {exp_name} ({exp_type}) ===")
        for seed in seeds:
            # Data generation
            data = generate_synthetic_data(n_clients=n_clients, n_samples=n_samples, fraud_ratio=fraud_ratio, n_features=n_features, seed=seed)
            if params.get('non_iid', False):
                edge_clients, data = create_non_iid_partitions(data, n_edges=n_edges, label_skew_edges=(0,1,2), skew_ratio=0.75, seed=seed)
            else:
                edge_clients = {e: list(range(e*10, (e+1)*10)) for e in range(n_edges)}
            # Initial model
            global_weights = np.zeros(n_features)
            # Metrics storage
            round_metrics = []
            comm_client_edge = 0
            comm_edge_cloud = 0
            auc_curves = []
            for rnd in range(rounds):
                # Select clients per edge
                participation_rate = random.choice(participation_rates)
                selected_clients = {}
                for edge, clients in edge_clients.items():
                    k = max(1, int(len(clients) * participation_rate))
                    selected_clients[edge] = random.sample(clients, k)
                # Local epochs per client
                local_epochs = {cid: random.choice([1,3,5]) for edge in edge_clients for cid in edge_clients[edge]}
                # Flat FL
                if not params.get('hierarchical', False):
                    client_weights = []
                    for edge, cids in selected_clients.items():
                        for cid in cids:
                            adv = params.get('adversarial_edge', None) == edge
                            cw = simulate_client_training(data[cid], global_weights, local_epochs=local_epochs[cid], quantize=params.get('quantize', None), adversarial=adv)
                            client_weights.append(cw)
                            comm_client_edge += compute_communication_costs(cw, bits=params.get('quantize', 32))
                    global_weights = simulate_cloud_aggregation(client_weights)
                    comm_edge_cloud += compute_communication_costs(global_weights, bits=params.get('quantize', 32))
                # Hierarchical FL
                else:
                    edge_weights = []
                    for edge, cids in selected_clients.items():
                        client_weights = []
                        for cid in cids:
                            adv = params.get('adversarial_edge', None) == edge
                            cw = simulate_client_training(data[cid], global_weights, local_epochs=local_epochs[cid], quantize=params.get('quantize', None), adversarial=adv)
                            client_weights.append(cw)
                            comm_client_edge += compute_communication_costs(cw, bits=params.get('quantize', 32))
                        ew = simulate_edge_aggregation(client_weights)
                        edge_weights.append(ew)
                    # Edge→Cloud aggregation interval
                    E = params.get('edge_agg_interval', 1)
                    if rnd % E == 0:
                        global_weights = simulate_cloud_aggregation(edge_weights)
                        comm_edge_cloud += compute_communication_costs(global_weights, bits=params.get('quantize', 32))
                # Metrics
                y_true = []
                y_pred = []
                auc_per_edge = []
                for edge, cids in edge_clients.items():
                    edge_y_true = []
                    edge_y_pred = []
                    for cid in cids:
                        X = data[cid]['X']
                        y = data[cid]['y']
                        preds = 1/(1+np.exp(-np.dot(X, global_weights)))
                        y_true.extend(y)
                        y_pred.extend(preds)
                        edge_y_true.extend(y)
                        edge_y_pred.extend(preds)
                    if len(set(edge_y_true)) > 1:
                        auc = roc_auc_score(edge_y_true, edge_y_pred)
                    else:
                        auc = 0.5
                    auc_per_edge.append(auc)
                # Global metrics
                y_pred_bin = [1 if p > 0.5 else 0 for p in y_pred]
                acc = accuracy_score(y_true, y_pred_bin)
                prec = precision_score(y_true, y_pred_bin, zero_division=0)
                rec = recall_score(y_true, y_pred_bin, zero_division=0)
                f1 = f1_score(y_true, y_pred_bin, zero_division=0)
                auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5
                cm = confusion_matrix(y_true, y_pred_bin)
                fnr = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
                round_metrics.append([acc, prec, rec, f1, auc, fnr, np.mean(auc_per_edge), np.std(auc_per_edge)])
                auc_curves.append(auc)
            # Store results
            results[exp_id].append({
                'metrics': round_metrics[-1],
                'auc_curve': auc_curves,
                'comm_client_edge': comm_client_edge,
                'comm_edge_cloud': comm_edge_cloud
            })
            print(f"Seed {seed}: Final Metrics: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f} FNR={fnr:.4f}")
            print(f"Client→Edge bytes: {comm_client_edge} | Edge→Cloud bytes: {comm_edge_cloud}")
        # Aggregate and print mean ± std
        all_metrics = [r['metrics'] for r in results[exp_id]]
        arr = np.array(all_metrics)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        print(f"Mean ± Std: Acc={mean[0]:.4f}±{std[0]:.4f} Prec={mean[1]:.4f}±{std[1]:.4f} Rec={mean[2]:.4f}±{std[2]:.4f} F1={mean[3]:.4f}±{std[3]:.4f} AUC={mean[4]:.4f}±{std[4]:.4f} FNR={mean[5]:.4f}±{std[5]:.4f}")
        print(f"Per-edge AUC mean={mean[6]:.4f} std={mean[7]:.4f}")
        # Plot convergence curve
        for seed, r in zip(seeds, results[exp_id]):
            plt.plot(r['auc_curve'], label=f'Seed {seed}')
        plt.xlabel('Rounds')
        plt.ylabel('AUC')
        plt.title(f'Convergence Curve: {exp_name} ({exp_type})')
        plt.legend()
        plt.show()

# ---------------------- Plotting ----------------------

def plot_convergence_curves(results, title):
    plt.figure(figsize=(8,5))
    for label, curve in results.items():
        plt.plot(curve, label=label)
    plt.xlabel('Rounds')
    plt.ylabel('AUC')
    plt.title(title)
    plt.legend()
    plt.show()

# ---------------------- Main ----------------------
if __name__ == "__main__":
    print("Federated Learning Experiment Runner - Running Experiments...")
    run_experiment_suite()
