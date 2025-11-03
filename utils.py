import matplotlib.pyplot as plt
import numpy as np
import torch
import random
# 移除 torch_geometric.transforms.RandomLinkSplit，因为它用于链接预测
from torch_geometric.utils import to_undirected


def set_seed(seed):
    """固定所有必要的随机种子以确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 强制 CUDA 使用确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_client_data(data, val_ratio=0.2, test_ratio=0.2, device='cpu'):
    """
    【修改为】节点分类任务的数据划分：生成 train_mask, val_mask, test_mask。

    Args:
        data (PyG Data): 包含 x, edge_index, y 的数据对象。
        val_ratio (float): 验证集节点比例。
        test_ratio (float): 测试集节点比例。

    Returns:
        data (PyG Data): 包含 train_mask, val_mask, test_mask 的数据对象。
    """
    data = data.to(device)

    # 确保图是无向的 (保留 GNN 预处理步骤)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    num_nodes = data.num_nodes
    # 确保总比例不超过 1.0
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("Val ratio and Test ratio must sum to less than 1.0")

    num_val = int(num_nodes * val_ratio)
    num_test = int(num_nodes * test_ratio)
    num_train = num_nodes - num_val - num_test

    # 1. 创建节点的随机排列
    # 假设所有节点都可以被用于训练/验证/测试
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    # 2. 计算划分索引
    train_indices = indices[:num_train]
    val_indices = indices[num_train: num_train + num_val]
    test_indices = indices[num_train + num_val:]

    # 3. 创建布尔掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # 4. 将掩码添加到数据对象
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # 5. 【清理】移除链接预测相关的属性（如果原数据中有）
    for attr in ['val_pos_edge_index', 'val_neg_edge_index',
                 'test_pos_edge_index', 'test_neg_edge_index',
                 'edge_label', 'edge_label_index']:
        if hasattr(data, attr):
            # 使用 try/except 或 delattr 移除属性，确保数据对象干净
            try:
                delattr(data, attr)
            except AttributeError:
                pass

    return data


def draw_loss_plot(loss_record: list):
    """绘制训练损失图（保持不变）"""
    x = []
    for i in range(1, len(loss_record) + 1):
        x.append(i)
    plt.plot(x, loss_record, marker='o', linestyle='-', color='b', label='Client')

    plt.title('Loss-Round', fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.show()
