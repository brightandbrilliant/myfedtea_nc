import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
from collections import Counter
import numpy as np


# ============================================================
# 1. 基础函数：计算特征差异
# ============================================================

def compute_diff(f1, f2, mode='euclidean'):
    """计算两个嵌入/特征向量之间的差异。"""
    if mode == 'euclidean':
        # L2 距离
        return torch.norm(f1 - f2, p=2).item()
    elif mode == 'cosine':
        # 1 - Cosine 相似度
        return 1 - F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# 2. 锚点嵌入差异
# ============================================================

def compute_anchor_embedding_differences(z1, z2, anchor_pairs, device='cpu'):
    """
    使用 GNN 嵌入计算锚点对之间的差异。
    """
    results = []
    z1 = z1.to(device)
    z2 = z2.to(device)

    for pair in anchor_pairs:
        idx1, idx2 = pair[0], pair[1]
        emb1 = z1[idx1]
        emb2 = z2[idx2]
        diff = compute_diff(emb1, emb2, 'euclidean')
        results.append([idx1, idx2, diff])

    return results


# ============================================================
# 3. 聚类方法
# ============================================================

def kmeans_cluster_new(features, n_clusters=5):
    """对输入的特征/嵌入进行 KMeans 聚类。"""
    if isinstance(features, torch.Tensor):
        x = features.detach().cpu().numpy()
    else:
        x = features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(x)
    labels, inertia = kmeans.labels_, kmeans.inertia_
    return labels, inertia


def gnn_embedding_kmeans_cluster(data, encoder, n_clusters=10, device='cpu'):
    """使用 GNN 编码器生成的嵌入进行 KMeans 聚类。"""
    encoder = encoder.to(device)
    data = data.to(device)
    encoder.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index).detach()
    labels, inertia = kmeans_cluster_new(z, n_clusters=n_clusters)
    return labels, inertia


def spectral_cluster_new(features, n_clusters=10, random_state=42):
    """谱聚类（SpectralClustering）。"""
    if isinstance(features, torch.Tensor):
        x = features.detach().cpu().numpy()
    else:
        x = features

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels='kmeans',
        random_state=random_state,
        n_init=10,
        affinity='nearest_neighbors'
    )
    labels = clustering.fit_predict(x)
    inertia = 0
    return labels, inertia


def gnn_embedding_spectral_cluster(data, encoder, n_clusters=10, device='cpu'):
    """GNN 嵌入 + 谱聚类。"""
    encoder = encoder.to(device)
    data = data.to(device)
    encoder.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index).detach()
    labels, inertia = spectral_cluster_new(z, n_clusters=n_clusters)
    return labels, inertia


# ============================================================
# 4. 类别对齐：共现矩阵 + 对齐提取
# ============================================================

def build_cluster_cooccurrence_matrix(cluster_labels1, cluster_labels2, anchor_pairs,
                                      top_percent=0.75):
    """
    构建聚类共现计数矩阵（支持 K1 != K2）。

    Args:
        cluster_labels1: 图1所有节点的聚类标签（NumPy array 或 list）
        cluster_labels2: 图2所有节点的聚类标签
        anchor_pairs: 锚点对 [[i, j, diff], ...]
        top_percent: 保留相似度前 top_percent 的锚点对

    Returns:
        matrix: ndarray，形状为 [K1, K2]
        K1: 图1类数
        K2: 图2类数
    """
    labels1 = np.array(cluster_labels1)
    labels2 = np.array(cluster_labels2)

    if labels1.size == 0 or labels2.size == 0:
        raise ValueError("Empty cluster labels in build_cluster_cooccurrence_matrix.")

    K1 = int(labels1.max()) + 1
    K2 = int(labels2.max()) + 1

    if not anchor_pairs:
        return np.zeros((K1, K2), dtype=int), K1, K2

    results_sorted = sorted(anchor_pairs, key=lambda x: x[2])
    cutoff = max(1, int(len(results_sorted) * top_percent))
    filtered = results_sorted[:cutoff]

    matrix = np.zeros((K1, K2), dtype=int)

    for idx1, idx2, _ in filtered:
        try:
            idx1_i = int(idx1)
            idx2_i = int(idx2)
        except Exception:
            continue

        if idx1_i < 0 or idx1_i >= len(labels1) or idx2_i < 0 or idx2_i >= len(labels2):
            continue

        c1 = int(labels1[idx1_i])
        c2 = int(labels2[idx2_i])
        if c1 < 0 or c1 >= K1 or c2 < 0 or c2 >= K2:
            continue

        matrix[c1, c2] += 1

    return matrix, K1, K2


def extract_clear_alignments(M, min_ratio=0.3, min_count=30, mode=1):
    """
    筛选出对齐明确的类。
    Args:
        M: 共现矩阵 (numpy.ndarray)，shape = [K1, K2]
        min_ratio: 主对齐类别所占比例的阈值
        min_count: 至少多少锚点才可信
        mode: 1 (行方向, 图1->图2), 2 (列方向, 图2->图1)
    Returns:
        alignments: dict {类id: [(对齐类id, 权重), ...]}
    """
    M = np.array(M)
    alignments = {}

    if mode == 1:  # 图1 -> 图2
        for i, row in enumerate(M):
            total = np.sum(row)
            if total < min_count:
                continue
            ratios = row / (total + 1e-12)
            major_idxs = np.where(ratios >= min_ratio)[0]
            if len(major_idxs) == 0:
                continue
            selected_counts = row[major_idxs]
            weights = selected_counts / (selected_counts.sum() + 1e-12)
            alignments[i] = list(zip(major_idxs.tolist(), weights.tolist()))

    elif mode == 2:  # 图2 -> 图1
        for j in range(M.shape[1]):
            col = M[:, j]
            total = np.sum(col)
            if total < min_count:
                continue
            ratios = col / (total + 1e-12)
            major_idxs = np.where(ratios >= min_ratio)[0]
            if len(major_idxs) == 0:
                continue
            selected_counts = col[major_idxs]
            weights = selected_counts / (selected_counts.sum() + 1e-12)
            alignments[j] = list(zip(major_idxs.tolist(), weights.tolist()))

    return alignments
