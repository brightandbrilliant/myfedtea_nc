import os
import torch
# 移除 Parse_Anchors 导入
import torch.nn.functional as F
# 移除 matplotlib.pyplot, seaborn
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from collections import Counter
import numpy as np


# 移除 torch_geometric.data.Data 导入


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


def compute_anchor_embedding_differences(z1, z2, anchor_pairs, device='cpu'):
    """
    使用 GNN 嵌入计算锚点对之间的差异。
    （保留原逻辑，但只使用欧氏距离/L2范数）

    Args:
        z1 (torch.Tensor): 客户端 1 的所有节点 GNN 嵌入。
        z2 (torch.Tensor): 客户端 2 的所有节点 GNN 嵌入。
        anchor_pairs (list): 锚点对列表 [[idx1, idx2], ...]。
        device (str/torch.device): 计算设备。

    Returns:
        results (list): 包含 [idx1, idx2, diff] 的列表。
    """
    results = []
    z1 = z1.to(device)
    z2 = z2.to(device)

    for pair in anchor_pairs:
        # 注意：这里假设 anchor_pairs 已经经过 Parse_Anchors 处理，
        # 索引 idx1 对应 z1，idx2 对应 z2
        idx1, idx2 = pair[0], pair[1]

        emb1 = z1[idx1]
        emb2 = z2[idx2]

        # 欧氏距离
        diff = compute_diff(emb1, emb2, 'euclidean')
        results.append([idx1, idx2, diff])

    return results


def kmeans_cluster_new(features, n_clusters=5):
    """
    对输入的特征/嵌入进行 KMeans 聚类。
    """
    if isinstance(features, torch.Tensor):
        x = features.detach().cpu().numpy()
    else:
        x = features

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(x)
    labels, inertia = kmeans.labels_, kmeans.inertia_

    return labels, inertia


def gnn_embedding_kmeans_cluster(data, encoder, n_clusters=10, device='cpu'):
    """
    使用 GNN 编码器生成的嵌入进行 KMeans 聚类。
    """
    encoder = encoder.to(device)
    data = data.to(device)
    encoder.eval()

    # 1. 生成 GNN 嵌入
    with torch.no_grad():
        z = encoder(data.x, data.edge_index).detach()

    # 2. 执行 KMeans 聚类
    labels, inertia = kmeans_cluster_new(z, n_clusters=n_clusters)

    # 返回 NumPy 标签
    return labels, inertia


# 保留 spectral_cluster_new 和 gnn_embedding_spectral_cluster (如果需要)
# ------------------------------------------------------------------
def spectral_cluster_new(features, n_clusters=10, random_state=42):
    # ... (与原版保持一致)
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
    # ... (与原版保持一致)
    encoder = encoder.to(device)
    data = data.to(device)
    encoder.eval()

    with torch.no_grad():
        z = encoder(data.x, data.edge_index).detach()

    labels, inertia = spectral_cluster_new(z, n_clusters=n_clusters)
    return labels, inertia


# ------------------------------------------------------------------


def build_cluster_cooccurrence_matrix(cluster_labels1, cluster_labels2, anchor_pairs,
                                      num_clusters, top_percent):
    """
    构建聚类共现计数矩阵。用于确定节点类别对齐。
    （保留原逻辑，但其目的现在是节点类型对齐）

    Args:
        cluster_labels1: 图1所有节点的聚类标签（NumPy array）
        cluster_labels2: 图2所有节点的聚类标签
        anchor_pairs: 筛选后的锚点对 [[i, j, diff], ...] (diff越小越相似)
        num_clusters: 聚类类别总数 k

    Returns:
        matrix: ndarray，形状为 [k, k]，matrix[i][j] 表示图1第i类与图2第j类之间的锚点对数
    """
    # 1. 根据差异度排序（越小越相似）
    results_sorted = sorted(anchor_pairs, key=lambda x: x[2])
    cutoff = int(len(results_sorted) * top_percent)
    filtered = results_sorted[:cutoff]

    matrix = np.zeros((num_clusters, num_clusters), dtype=int)

    # 2. 统计共现次数
    for idx1, idx2, _ in filtered:
        # idx1, idx2 是本地节点索引
        c1 = cluster_labels1[idx1]
        c2 = cluster_labels2[idx2]
        matrix[c1][c2] += 1

    return matrix


def extract_clear_alignments(M, min_ratio=0.3, min_count=30, mode=1):
    """
    筛选出对齐明确的类。
    （原逻辑适用于边类型对齐，现在用于节点类别对齐，无需修改核心逻辑）

    Args:
        M: 共现矩阵 (numpy.ndarray)，shape = [n_class1, n_class2]
        min_ratio: 主对齐类别所占比例的阈值
        min_count: 至少有多少个锚点才视为可信
        mode: 1 (行/图1->图2), 2 (列/图2->图1)

    Returns:
        alignments: dict，如 {图i的类id: [(图j的类id, 权重), ...]}
    """
    M = np.array(M)
    alignments = {}

    if mode == 1:  # 图1 -> 图2
        for i, row in enumerate(M):
            total = np.sum(row)
            if total < min_count: continue
            ratios = row / total
            major_idxs = np.where(ratios >= min_ratio)[0]
            if len(major_idxs) > 0:
                selected_counts = row[major_idxs]
                weights = selected_counts / selected_counts.sum()
                alignments[i] = list(zip(major_idxs.tolist(), weights.tolist()))
    elif mode == 2:  # 图2 -> 图1
        for j in range(M.shape[1]):
            col = M[:, j]
            total = np.sum(col)
            if total < min_count: continue
            ratios = col / total
            major_idxs = np.where(ratios >= min_ratio)[0]
            if len(major_idxs) > 0:
                selected_counts = col[major_idxs]
                weights = selected_counts / selected_counts.sum()
                alignments[j] = list(zip(major_idxs.tolist(), weights.tolist()))

    return alignments

