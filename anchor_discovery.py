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


def discover_mnn_anchors(Z1: torch.Tensor, Z2: torch.Tensor, top_k=1, metric='cosine', threshold=None):
    """
    通过互为最近邻 (MNN) 算法挖掘锚点对。

    Args:
        Z1, Z2 (torch.Tensor): 客户端嵌入。
        top_k (int): 考虑的最近邻数量。
        metric (str): 距离度量。
        threshold (float): 可选的距离阈值。

    Returns:
        list: 锚点对列表 [[idx1, idx2], ...]。
    """
    N1 = Z1.shape[0]
    N2 = Z2.shape[0]

    # 1. 计算距离矩阵 (N1 x N2)
    D = compute_distance_matrix(Z1, Z2, metric)

    anchors = []

    # 2. 找到 Z1 中每个节点到 Z2 的最近邻
    # torch.topk 默认找最大值，因此对于距离 (Diff)，我们需要找最小值
    # 返回: [最小值，索引]
    min_dist_1_to_2, indices_1_to_2 = torch.topk(D, k=top_k, dim=1, largest=False)

    # 3. 找到 Z2 中每个节点到 Z1 的最近邻
    # 沿 dim=0 寻找 (即每一列)，返回的索引是 Z1 的索引
    min_dist_2_to_1, indices_2_to_1 = torch.topk(D, k=top_k, dim=0, largest=False)

    # 4. 寻找 MNN

    # 遍历客户端 1 的每个节点 i
    for i in range(N1):
        # i 在 Z2 中的 k 个最近邻 (索引 j)
        nn_j_from_i = indices_1_to_2[i].tolist()

        for j in nn_j_from_i:
            # 检查 j 是否在 Z1 中的 k 个最近邻包含 i

            # j 在 Z1 中的 k 个最近邻 (索引 i')
            nn_i_from_j = indices_2_to_1[:, j].tolist()

            if i in nn_i_from_j:
                # 互为最近邻 (i <-> j)

                # 检查距离阈值 (可选)
                current_dist = D[i, j].item()
                if threshold is not None and current_dist > threshold:
                    continue

                # 确保锚点只被添加一次
                anchor_pair = [i, j]
                if anchor_pair not in anchors:
                    anchors.append(anchor_pair)

    # 5. 返回锚点对
    print(f"Discovered {len(anchors)} MNN anchors.")
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
