import torch
import numpy as np


# 引入 scipy.sparse 或 sklearn.neighbors 可能更高效，此处先用 numpy/torch 实现核心逻辑

def compute_distance_matrix(Z1: torch.Tensor, Z2: torch.Tensor, metric='cosine'):
    """
    计算两个嵌入矩阵之间的距离/相似度矩阵。

    Args:
        Z1, Z2 (torch.Tensor): 客户端嵌入矩阵 (N1 x D, N2 x D)。
        metric (str): 距离度量 ('cosine' 或 'euclidean')。

    Returns:
        torch.Tensor: 距离/相似度矩阵 (N1 x N2)。
    """
    # 确保 Z1, Z2 在 CPU 上进行大规模矩阵计算，除非有专用GPU
    Z1 = Z1.cpu()
    Z2 = Z2.cpu()

    if metric == 'cosine':
        # 归一化嵌入
        Z1_norm = Z1 / Z1.norm(dim=1, keepdim=True)
        Z2_norm = Z2 / Z2.norm(dim=1, keepdim=True)
        # Cosine 相似度 (Sim) = Z1 * Z2.T
        similarity_matrix = torch.matmul(Z1_norm, Z2_norm.T)
        # 距离 (Diff) = 1 - Sim
        distance_matrix = 1.0 - similarity_matrix
        return distance_matrix

    elif metric == 'euclidean':
        # 使用广播机制计算欧氏距离（通常比循环快）
        # ||A - B||^2 = ||A||^2 + ||B||^2 - 2A·B
        # 避免直接的广播，使用高效的 Pytorch 函数
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix_np = euclidean_distances(Z1.numpy(), Z2.numpy())
        return torch.from_numpy(dist_matrix_np)

    else:
        raise ValueError("Unsupported distance metric.")


def discover_mnn_anchors(Z1: torch.Tensor, Z2: torch.Tensor, metric='cosine', TOP_PERCENTILE = 0.30):
    """
    通过最近邻距离百分位数 (30%) 挖掘锚点对。

    对于 Z1 中的每个点 i，找到 Z2 中离它最近的点 j_nn。
    如果距离 D(i, j_nn) 在所有 D(i, j_nn) 构成的集合中属于前 30% 最短距离，
    则将 (i, j_nn) 视为锚点对。

    Args:
        Z1, Z2 (torch.Tensor): 客户端嵌入。
        metric (str): 距离度量（例如 'cosine'）。

    Returns:
        list: 锚点对列表 [[idx1, idx2], ...]。
    """

    N1 = Z1.shape[0]

    if N1 == 0 or Z2.shape[0] == 0:
        print("警告: 客户端嵌入为空，未发现锚点。")
        return []

    # 1. 计算距离矩阵 (N1 x N2)
    D = compute_distance_matrix(Z1, Z2, metric)

    # 2. 找到 Z1 中每个节点到 Z2 的最近邻 (最近邻距离和索引)
    # 沿 dim=1 寻找最小值 (即每一行)，返回的索引是 Z2 的索引 j
    # min_dist_1_to_2 形状: (N1,) - Z1 中每个点的最近距离
    # indices_1_to_2 形状: (N1,) - Z2 中对应的最近邻索引 j
    min_dist_1_to_2, indices_1_to_2 = torch.min(D, dim=1)

    # 3. 计算距离阈值

    # 将所有 N1 个最近距离拉平，找到 TOP_PERCENTILE 对应的距离值
    # 即：30% 的 Z1 节点能找到比这个阈值更近的最近邻。
    threshold_value = torch.quantile(min_dist_1_to_2, q=TOP_PERCENTILE).item()

    # 4. 过滤锚点对

    # 找到所有距离小于或等于阈值的 Z1 节点索引 i
    # 形状: (N_anchors,)
    filtered_indices_i = torch.nonzero(min_dist_1_to_2 <= threshold_value, as_tuple=True)[0]

    # 提取这些被选中的 i 对应的 Z2 最近邻索引 j
    filtered_indices_j = indices_1_to_2[filtered_indices_i]

    # 5. 构造锚点对列表
    anchors = []
    if filtered_indices_i.numel() > 0:
        # 锚点对 [idx1, idx2]
        anchors_tensor = torch.stack([filtered_indices_i, filtered_indices_j], dim=1)
        anchors = anchors_tensor.tolist()

    # 6. 返回结果
    print(f"Discovered {len(anchors)} nearest-neighbor anchors (Top {int(TOP_PERCENTILE * 100)}% by distance).")
    print(f"Used distance threshold: {threshold_value:.4f}.")
    return anchors


# ------------------------------------------------------------------------
# 锚点信息结构调整：
# 由于挖掘出的锚点不包含 diff 信息，我们需要一个新函数在挖掘后立即计算 diff，
# 以便 Cluster.py 中后续的对齐函数能够使用。
# ------------------------------------------------------------------------

def compute_anchor_list_with_diff(Z1: torch.Tensor, Z2: torch.Tensor, anchors: list, device='cpu'):
    """
    为挖掘出的锚点对计算嵌入差异，用于 Cluster.py 的对齐函数。

    Args:
        Z1, Z2: 客户端嵌入。
        anchors: MNN 挖掘出的锚点列表 [[idx1, idx2], ...]

    Returns:
        results (list): 包含 [idx1, idx2, diff] 的列表。
    """
    results = []

    # 确保嵌入在正确的设备上
    Z1 = Z1.to(device)
    Z2 = Z2.to(device)

    for idx1, idx2 in anchors:
        # 使用欧氏距离 (与 Cluster.py 中的 compute_diff 保持一致)
        emb1 = Z1[idx1]
        emb2 = Z2[idx2]

        diff = torch.norm(emb1 - emb2, p=2).item()
        results.append([idx1, idx2, diff])

    return results
