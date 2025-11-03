import os
import torch
from collections import deque, defaultdict
import random
import numpy as np
# 确保你的项目结构中使用小写文件名 (client, model/graphsage, model/resmlp)
from client import Client
from model.graphsage import GraphSAGE
from model.resmlp import ResMLP
from cluster import (
    gnn_embedding_kmeans_cluster,
    compute_anchor_embedding_differences,  # 实际上未直接使用，但保留以防 Cluster.py 依赖
    build_cluster_cooccurrence_matrix,
    extract_clear_alignments
)
# 动态锚点挖掘导入
from anchor_discovery import discover_mnn_anchors, compute_anchor_list_with_diff
from utils import set_seed, split_client_data


# --- 辅助函数：错误聚合 ---

def aggregate_error_counts(sliding_window: deque, top_percent=1.0):
    """
    聚合滑动窗口中的错误类别计数，并筛选 Top-Percent 的错误类别。
    """
    aggregate = defaultdict(int)
    for it in sliding_window:
        for cluster_id, count in it.items():
            aggregate[cluster_id] += count

    # 筛选 top percent 的错误类别 (选择错误最多的类别进行增强)
    sorted_items = sorted(aggregate.items(), key=lambda x: x[1], reverse=True)
    cutoff = max(1, int(len(sorted_items) * top_percent))

    return dict(sorted_items[:cutoff])


# --- 辅助函数：节点知识提取 ---

def extract_augmented_node_data(target_client, source_client, error_cluster_counts,
                                cluster_labels_source, node_alignment, top_k_per_type=100):
    """
    基于目标客户端的错误类别和对齐矩阵，从 source_client 提取增强知识。

    Args:
        target_client: 知识接收方客户端对象（仅用于类型提示）。
        source_client: 知识提供方客户端对象。
        error_cluster_counts (dict): 目标客户端的错误聚类类别统计 {c_i: count}。
        cluster_labels_source (np.array): 源客户端的所有节点聚类标签。
        node_alignment (dict): 对齐矩阵 {c_i: [(c_j, weight), ...]}。
        top_k_per_type (int): 每个对齐类别要提取的节点最大数量。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 增强嵌入 Z_aug 和标签 Y_aug。
    """
    # 1. 确保源客户端模型处于评估模式，并计算嵌入
    source_client.encoder.eval()
    with torch.no_grad():
        z_j = source_client.encoder(source_client.data.x, source_client.data.edge_index).detach()
        y_j = source_client.data.y.detach()

    all_augmented_z = []
    all_augmented_y = []

    # 2. 根据错误类别和对齐矩阵抽取节点
    for c_i, error_count in error_cluster_counts.items():
        # 获取与目标客户端错误类别 c_i 对齐的源客户端类别 c_j 及其权重
        aligned_targets = node_alignment.get(c_i, [])  # [(c_j, weight), ...]

        for c_j, weight in aligned_targets:
            # 找到源客户端中属于聚类 c_j 的节点索引
            # 使用 np.where 或 (array == value).nonzero() 确保兼容性
            nodes_c_j = np.where(cluster_labels_source == c_j)[0].tolist()

            if not nodes_c_j:
                continue

            # 根据权重计算应抽取的数量
            num_to_select = int(top_k_per_type * weight)

            # 随机打乱并抽取 num_to_select 条
            random.shuffle(nodes_c_j)
            selected_indices = nodes_c_j[:max(1, num_to_select)]

            # 提取嵌入和标签
            # 确保索引是张量类型，且在 CPU/GPU 上一致
            idx_tensor = torch.tensor(selected_indices, dtype=torch.long, device=z_j.device)
            all_augmented_z.append(z_j[idx_tensor])
            all_augmented_y.append(y_j[idx_tensor])

    if not all_augmented_z:
        return None, None

    # 3. 合并所有提取的知识
    final_z = torch.cat(all_augmented_z, dim=0)
    final_y = torch.cat(all_augmented_y, dim=0)

    # 将标签数据类型转换为长整型 (CrossEntropyLoss 需要)
    return final_z, final_y.long()


# --- 通用服务器函数 (适配 NC) ---

def load_all_clients(pyg_data_paths, encoder_params, classifier_params, training_params, device, num_classes):
    """适配节点分类，加载 Classifier 和数据划分"""
    clients, raw_data_list = [], []

    for client_id, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)
        raw_data_list.append(raw_data)

        # 使用修改后的 split_client_data (生成 train/val/test mask)
        data = split_client_data(raw_data, device=device)

        encoder = GraphSAGE(**encoder_params)
        classifier = ResMLP(input_dim=encoder_params['output_dim'],
                            output_dim=num_classes, **classifier_params)

        client = Client(
            client_id=client_id,
            data=data,
            encoder=encoder,
            classifier=classifier,
            device=device,
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay']
        )
        clients.append(client)

    return clients, raw_data_list


def average_state_dicts(state_dicts):
    """聚合状态字典 (通用函数，保持不变)"""
    avg_state = {}
    for key in state_dicts[0].keys():
        # 确保堆叠的张量位于 CPU 或当前设备，并处理浮点精度
        tensors = [sd[key].float() for sd in state_dicts]
        avg_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
    return avg_state


def evaluate_all_clients(clients, use_test=False):
    """计算节点分类准确率 (Accuracy)"""
    accuracies = []
    for i, client in enumerate(clients):
        acc = client.evaluate(use_test=use_test)
        accuracies.append(acc)
        print(f"Client {client.client_id}: Acc={acc:.4f}")

    avg_acc = torch.tensor(accuracies).mean().item()
    print(f"\n===> Average: Acc={avg_acc:.4f}")

    return avg_acc


def pretrain_fedavg(clients, pretrain_rounds, training_params):
    """聚合 Encoder 和 Classifier 状态"""
    print("\n========= Phase 1: FedAvg Pre-training Start (NC) =========")

    for rnd in range(1, pretrain_rounds + 1):
        if rnd%10 == 1:
            print(f"Pretrain Processing. Round{rnd}")
        for client in clients:
            for _ in range(training_params['local_epochs']):
                client.train()

        # 聚合 Encoder 和 Classifier
        encoder_states = [client.get_encoder_state() for client in clients]
        classifier_states = [client.get_classifier_state() for client in clients]
        global_encoder_state = average_state_dicts(encoder_states)
        global_classifier_state = average_state_dicts(classifier_states)

        # 分发全局模型状态
        for client in clients:
            client.set_encoder_state(global_encoder_state)
            client.set_classifier_state(global_classifier_state)
            # 保存全局状态供 FedProx 使用
            client.last_encoder_state = {k: v.cpu().clone() for k, v in global_encoder_state.items()}
            client.last_classifier_state = {k: v.cpu().clone() for k, v in global_classifier_state.items()}

    print("========= Phase 1: FedAvg Pre-training Finished (NC) =========")
    return


def Cluster_and_Align(clients, anchor_config, nClusters, top_percent, device):
    """
    执行 N 客户端的节点聚类、动态锚点挖掘和两两类别对齐。

    Returns:
        cluster_labels (list): 所有客户端的聚类标签列表 [labels_0, labels_1, ...]
        node_alignments (dict): 嵌套字典 {(i, j): alignment_i_to_j}
    """
    num_clients = len(clients)
    cluster_labels = []
    all_z = []

    print("==================Clustering Start==================")
    # 1. 聚类并提取 GNN 嵌入
    for client in clients:
        labels, _ = gnn_embedding_kmeans_cluster(client.data, client.encoder, n_clusters=nClusters, device=device)
        cluster_labels.append(labels)

        client.encoder.eval()
        z = client.encoder(client.data.x, client.data.edge_index).detach()
        all_z.append(z)

    # 2. 动态锚点挖掘和对齐 (N * N 矩阵)
    node_alignments = {}

    print("==================Alignment Start==================")
    # 遍历所有客户端对 (i, j), i != j
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                continue

            z_i, z_j = all_z[i], all_z[j]
            labels_i, labels_j = cluster_labels[i], cluster_labels[j]

            print(f"--- Discovering Anchors and Aligning: Client {i} -> Client {j} ---")

            # 2a. 动态锚点挖掘 (服务器端)
            min_anchors = discover_mnn_anchors(z_i, z_j, metric=anchor_config['metric'])

            if not min_anchors:
                node_alignments[(i, j)] = {}
                continue

            # 2b. 计算锚点差异 [idx_i, idx_j, diff]
            results = compute_anchor_list_with_diff(z_i, z_j, min_anchors, device=device)

            # 2c. 构建共现矩阵
            co_matrix = build_cluster_cooccurrence_matrix(labels_i, labels_j, results,
                                                          nClusters, top_percent=top_percent)

            # 2d. 提取对齐 (模式 1: i -> j)
            alignment_i_to_j = extract_clear_alignments(co_matrix,
                                                        min_ratio=0.25,
                                                        min_count=anchor_config['min_count'],
                                                        mode=1)

            node_alignments[(i, j)] = alignment_i_to_j

    print("==================Alignment Finished==================")

    return cluster_labels, node_alignments


if __name__ == "__main__":
    seed_ = 826
    set_seed(seed_)

    # --- 参数设置 ---
    data_dir = "parsed_dataset/cs"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])
    NUM_CLASSES = 6  # <--- 必须根据你的数据集确定！
    NUM_CLIENTS = len(pyg_data_files)  # 动态获取客户端数量

    encoder_params = {
        'input_dim': torch.load(pyg_data_files[0]).x.shape[1],
        'hidden_dim': 128,
        'output_dim': 64,
        'num_layers': 3,
        'dropout': 0.5
    }
    classifier_params = {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3}
    training_params = {'lr': 0.001, 'weight_decay': 1e-4, 'local_epochs': 5}

    num_rounds = 700
    top_error_cluster_percent = 0.5  # 考虑错误最多的前 50% 的类别
    enhance_interval = 10
    top_k_node_per_type = 20  # 每个对齐类别抽取 20 个节点
    nClusters = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_rounds = 50
    align_update_interval = 150  # 重新聚类和对齐的间隔
    start_rnd = 1

    anchor_config = {
        'metric': 'cosine',
        'min_count': 30,
    }

    print("==================Pretraining Start==================")

    clients, raw_data_list = load_all_clients(
        pyg_data_files, encoder_params, classifier_params, training_params, device, NUM_CLASSES
    )

    pretrain_fedavg(clients, pretrain_rounds, training_params)

    # --- 最佳模型状态跟踪 (切换到 ACC) ---
    best_acc = -1
    best_encoder_state = None
    best_classifier_states = None
    best_rnd = 0

    # 初始化滑动窗口和损失记录，泛化到 N 个客户端
    sliding_error_window = [deque(maxlen=5) for _ in range(NUM_CLIENTS)]
    loss_record = [[] for _ in range(NUM_CLIENTS)]

    # 初始聚类和对齐 (使用预训练结果)
    cluster_labels, node_alignments = Cluster_and_Align(clients, anchor_config,
                                                        nClusters, top_percent=0.75, device=device)

    print("\n================ Federated Training Start (NC-Aug) ================")
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        # 1. 重新聚类和对齐 (定期更新)
        if rnd > start_rnd and rnd % align_update_interval == 0:
            cluster_labels, node_alignments = Cluster_and_Align(clients, anchor_config,
                                                                nClusters, top_percent=0.75, device=device)
            print("--- Cluster and Alignment Updated ---")

        # --- 2. 错误分析和知识注入 (服务器/Client 交互) ---
        if rnd >= start_rnd and rnd % enhance_interval == 0:

            # 2a. 客户端 i 分析错误类别
            for i in range(NUM_CLIENTS):
                client = clients[i]
                error_counts = client.analyze_prediction_errors_by_cluster(cluster_labels[i])
                sliding_error_window[i].append(error_counts)

            # 2b. 服务器聚合错误并提取知识
            # 目标客户端 i
            for i in range(NUM_CLIENTS):
                target_client = clients[i]

                # 聚合错误，确定需要增强的类别 c_i
                aggregated_errors = aggregate_error_counts(sliding_error_window[i],
                                                           top_percent=top_error_cluster_percent)

                all_Z_aug_from_j, all_Y_aug_from_j = [], []

                # 源客户端 j (j != i)
                for j in range(NUM_CLIENTS):
                    if i == j:
                        continue

                    source_client = clients[j]

                    # 获取对齐矩阵 (i -> j)
                    alignment_i_to_j = node_alignments.get((i, j), {})

                    # 提取增强知识 (Z_aug, Y_aug)
                    Z_aug_j, Y_aug_j = extract_augmented_node_data(
                        target_client,  # 知识接收方
                        source_client,  # 知识提供方
                        aggregated_errors,
                        cluster_labels[j],  # 源客户端 j 的聚类标签
                        alignment_i_to_j,
                        top_k_per_type=top_k_node_per_type
                    )

                    if Z_aug_j is not None:
                        all_Z_aug_from_j.append(Z_aug_j)
                        all_Y_aug_from_j.append(Y_aug_j)

                # 2c. 注入知识 (将所有源客户端的知识合并后注入)
                if all_Z_aug_from_j:
                    final_Z_aug = torch.cat(all_Z_aug_from_j, dim=0)
                    final_Y_aug = torch.cat(all_Y_aug_from_j, dim=0)
                    target_client.inject_augmented_node_data(final_Z_aug, final_Y_aug)

        # --- 3. 本地训练和增强训练 ---
        classifier_states_local = []
        for i in range(NUM_CLIENTS):
            client = clients[i]
            loss_avg = 0
            for _ in range(training_params['local_epochs']):
                # 本地常规训练
                loss = client.train()
                loss_avg += loss

            loss_avg /= training_params['local_epochs']
            loss_record[i].append(loss_avg)

            # 增强训练
            if client.augmented_node_data is not None:
                loss_aug = client.train_on_augmented_nodes()
                print(f"Client {i} Augmented Loss: {loss_aug:.4f}")

            classifier_states_local.append(client.get_classifier_state())

            # --- 4. 服务器聚合和分发 ---
        encoder_states = [client.get_encoder_state() for client in clients]
        global_encoder_state = average_state_dicts(encoder_states)
        global_classifier_state = average_state_dicts(classifier_states_local)

        # 分发全局 Encoder 和 Classifier 状态
        for client in clients:
            client.set_encoder_state(global_encoder_state)
            client.set_classifier_state(global_classifier_state)
            client.last_encoder_state = {k: v.cpu().clone() for k, v in global_encoder_state.items()}
            client.last_classifier_state = {k: v.cpu().clone() for k, v in global_classifier_state.items()}

        # 5. 评估和保存最佳模型
        avg_acc = evaluate_all_clients(clients, use_test=True)

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_encoder_state = {k: v.clone().detach() for k, v in global_encoder_state.items()}
            best_classifier_states = {k: v.clone().detach() for k, v in global_classifier_state.items()}
            best_rnd = rnd
            print("===> New best model saved")

    print("\n================ Federated Training Finished ================")

    # 最终评估
    for i, client in enumerate(clients):
        client.set_encoder_state(best_encoder_state)
        # 假设所有客户端共享同一个最优分类器
        client.set_classifier_state(best_classifier_states)

    print("\n================ Final Evaluation ================")
    final_avg_acc = evaluate_all_clients(clients, use_test=True)
    print(f"best rnd:{best_rnd}")
    print(f"best acc:{final_avg_acc:.4f}")
