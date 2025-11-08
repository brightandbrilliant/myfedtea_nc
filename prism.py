"""
PRISM-inspired Adaptive Clustering Module
-----------------------------------------
This module provides functions to automatically determine
the optimal number of clusters (K) based on the Non-IID Index (NID),
inspired by the PRISM framework's Distribution Inconsistency Detection idea.

Author: yjr
Date: [2025-11-04]
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict
from model.graphsage import GraphSAGE  # optional if used directly
from cluster import gnn_embedding_kmeans_cluster


# ===============================================================
# 1️⃣ compute_cluster_stats
# ===============================================================
def compute_cluster_stats(data, encoder, cluster_labels, device='cpu'):
    """
    Compute mean embedding and standard deviation for each cluster.

    Args:
        data: PyG Data object
        encoder: trained GNN encoder
        cluster_labels: numpy array or tensor of shape [num_nodes]
        device: computation device

    Returns:
        cluster_means (List[Tensor]): per-cluster mean embeddings
        cluster_stds (List[float]): per-cluster standard deviations
    """
    encoder.eval()
    data = data.to(device)
    cluster_labels = torch.tensor(cluster_labels, device=device)

    with torch.no_grad():
        z = encoder(data.x, data.edge_index).detach()

    num_clusters = int(cluster_labels.max().item()) + 1
    cluster_means, cluster_stds = [], []

    for c in range(num_clusters):
        mask = (cluster_labels == c)
        if mask.sum() == 0:
            cluster_means.append(torch.zeros(z.shape[1], device=device))
            cluster_stds.append(0.0)
            continue
        z_c = z[mask]
        mean = z_c.mean(dim=0)
        std = z_c.std(dim=0).mean().item()  # scalar std
        cluster_means.append(mean)
        cluster_stds.append(std)

    return cluster_means, cluster_stds


# ===============================================================
# 2️⃣ compute_cluster_nid
# ===============================================================
def compute_cluster_nid(cluster_means: List[torch.Tensor],
                        cluster_stds: List[float],
                        eps: float = 1e-8):
    """
    Compute Non-IID Index (NID) between clusters.

    Formula:
        NID = (1 / K(K-1)) * Σ_{i≠j} ||μ_i - μ_j|| / (σ_i + σ_j + ε)

    Args:
        cluster_means: list of cluster mean embeddings
        cluster_stds: list of cluster-wise std scalars
        eps: small value for numerical stability

    Returns:
        nid_value (float)
    """
    K = len(cluster_means)
    if K <= 1:
        return 0.0

    total = 0.0
    count = 0

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            diff = torch.norm(cluster_means[i] - cluster_means[j], p=2).item()
            denom = cluster_stds[i] + cluster_stds[j] + eps
            total += diff / denom
            count += 1

    return total / count


# ===============================================================
# 3️⃣ evaluate_nid_over_k
# ===============================================================
def evaluate_nid_over_k(data, encoder, k_list, device='cpu'):
    """
    Evaluate the Non-IID Index for a range of cluster numbers.

    Args:
        data: PyG Data object
        encoder: GNN encoder
        k_list: iterable of candidate K values (e.g., range(5, 21))
        device: computation device

    Returns:
        nid_results (dict): {K: NID_value}
    """
    nid_results = {}

    for k in k_list:
        labels, _ = gnn_embedding_kmeans_cluster(data, encoder,
                                                 n_clusters=k, device=device)
        means, stds = compute_cluster_stats(data, encoder, labels, device)
        nid_val = compute_cluster_nid(means, stds)
        nid_results[k] = nid_val
        print(f"[Adaptive Clustering] K={k}, NID={nid_val:.4f}")

    return nid_results


# ===============================================================
# 4️⃣ select_optimal_k
# ===============================================================
def select_optimal_k(nid_results: Dict[int, float], method="elbow"):
    """
    Select the optimal K based on NID values.

    Args:
        nid_results: dict of {K: NID_value}
        method: 'min' or 'elbow'

    Returns:
        best_K (int)
    """
    Ks = sorted(nid_results.keys())
    NIDs = [nid_results[k] for k in Ks]

    if method == "min":
        best_K = Ks[int(np.argmin(NIDs))]
        return best_K

    elif method == "elbow":
        # numerical derivative method (simple elbow detection)
        diffs = np.diff(NIDs)
        second_diffs = np.diff(diffs)
        # elbow = point where second derivative changes sign most strongly
        idx = np.argmax(np.abs(second_diffs))
        best_K = Ks[idx + 1]  # +1 because of diff shift
        return best_K

    else:
        raise ValueError("Invalid method. Choose 'min' or 'elbow'.")


# ===============================================================
# 5️⃣ adaptive_cluster_selection (main interface)
# ===============================================================
def adaptive_cluster_selection(data, encoder, k_range, device='cpu', method="elbow"):
    """
    Main interface: automatically select optimal K via NID.

    Args:
        data: PyG Data object
        encoder: trained GNN encoder
        k_range: range or list of K candidates (e.g., range(5, 21))
        device: computation device
        method: 'min' or 'elbow' for selection strategy

    Returns:
        best_K (int)
        nid_results (dict)
    """
    # print("\n[Adaptive Clustering] Evaluating Non-IID Index across K values...")
    nid_results = evaluate_nid_over_k(data, encoder, k_range, device)
    best_K = select_optimal_k(nid_results, method)
    print(f"[Adaptive Clustering] Best K determined by NID: {best_K}")
    return best_K, nid_results
