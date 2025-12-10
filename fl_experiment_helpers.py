"""
Helper functions for Federated Learning Experiments
"""
import numpy as np
import random

def select_clients(edge_clients, participation_rate=0.3, seed=None):
    """
    Select a random subset of clients per edge for participation.
    """
    if seed is not None:
        random.seed(seed)
    selected = {}
    for edge, clients in edge_clients.items():
        n = len(clients)
        k = max(1, int(n * participation_rate))
        selected[edge] = random.sample(clients, k)
    return selected

def randomize_local_epochs(clients, min_epoch=1, max_epoch=5, seed=None):
    """
    Assign random local epochs to each client.
    """
    if seed is not None:
        np.random.seed(seed)
    return {cid: np.random.choice(range(min_epoch, max_epoch+1)) for cid in clients}

def aggregate_metrics(metrics_list):
    """
    Aggregate metrics (mean, std) across seeds/runs.
    """
    arr = np.array(metrics_list)
    return np.mean(arr, axis=0), np.std(arr, axis=0)
