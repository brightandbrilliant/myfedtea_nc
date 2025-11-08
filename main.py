import os
import torch
from collections import deque, defaultdict
import random
import numpy as np

from client import Client
from model.graphsage import GraphSAGE
from model.resmlp import ResMLP
from cluster import (
    gnn_embedding_kmeans_cluster,
    compute_anchor_embedding_differences,
    build_cluster_cooccurrence_matrix,
    extract_clear_alignments
)
from anchor_discovery import discover_mnn_anchors, compute_anchor_list_with_diff
from utils import set_seed, split_client_data
from prism import adaptive_cluster_selection


# ============================================================
# 辅助函数：错误聚合
# ============================================================
def aggregate_error_counts(sliding_window: deque, top_percent=1.0):
    """聚合滑动窗口中的错误类别计数，并筛选 Top-Percent 的错误类别。"""
    aggregate = defaultdict(int)
    for it in sliding_window:
        for cluster_id, count in it.items():
            aggregate[cluster_id] += count
    sorted_items = sorted(aggregate.items(), key=lambda x: x[1], reverse=True)
    cutoff = max(1, int(len(sorted_items) * top_percent))
    return dict(sorted_items[:cutoff])


# ============================================================
# 辅助函数：节点知识提取
# ============================================================
def extract_augmented_node_data(target_client, source_client, error_cluster_counts,
                                cluster_labels_source, node_alignment, top_k_per_type=100):
    """
    基于目标客户端的错误类别和对齐矩阵，从 source_client 提取增强知识。
    """
    source_client.encoder.eval()
    with torch.no_grad():
        z_j = source_client.encoder(source_client.data.x, source_client.data.edge_index).detach()
        y_j = source_client.data.y.detach()

    all_augmented_z, all_augmented_y = [], []

    for c_i, _ in error_cluster_counts.items():
        aligned_targets = node_alignment.get(c_i, [])
        for c_j, weight in aligned_targets:
            nodes_c_j = np.where(cluster_labels_source == c_j)[0].tolist()
            if not nodes_c_j:
                continue
            num_to_select = int(top_k_per_type * weight)
            random.shuffle(nodes_c_j)
            selected_indices = nodes_c_j[:max(1, num_to_select)]
            idx_tensor = torch.tensor(selected_indices, dtype=torch.long, device=z_j.device)
            all_augmented_z.append(z_j[idx_tensor])
            all_augmented_y.append(y_j[idx_tensor])

    if not all_augmented_z:
        return None, None

    final_z = torch.cat(all_augmented_z, dim=0)
    final_y = torch.cat(all_augmented_y, dim=0)
    return final_z, final_y.long()


# ============================================================
# 客户端加载与聚合
# ============================================================
def load_all_clients(pyg_data_paths, encoder_params, classifier_params, training_params, device, num_classes):
    """加载客户端（适配节点分类）"""
    clients, raw_data_list = [], []

    for client_id, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)
        raw_data_list.append(raw_data)
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
    """聚合状态字典"""
    avg_state = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key].float() for sd in state_dicts]
        avg_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
    return avg_state


def evaluate_all_clients(clients, use_test=False):
    """评估所有客户端准确率"""
    accuracies = []
    for i, client in enumerate(clients):
        acc = client.evaluate(use_test=use_test)
        accuracies.append(acc)
        print(f"Client {client.client_id}: Acc={acc:.4f}")
    avg_acc = torch.tensor(accuracies).mean().item()
    print(f"\n===> Average Acc={avg_acc:.4f}")
    return avg_acc


# ============================================================
# FedAvg预训练阶段
# ============================================================
def pretrain_fedavg(clients, pretrain_rounds, training_params):
    print("\n========= Phase 1: FedAvg Pre-training Start (NC) =========")
    for rnd in range(1, pretrain_rounds + 1):
        if rnd % 10 == 1:
            print(f"Pretrain Processing. Round {rnd}")
        for client in clients:
            for _ in range(training_params['local_epochs']):
                client.train()

        encoder_states = [client.get_encoder_state() for client in clients]
        classifier_states = [client.get_classifier_state() for client in clients]
        global_encoder_state = average_state_dicts(encoder_states)
        global_classifier_state = average_state_dicts(classifier_states)

        for client in clients:
            client.set_encoder_state(global_encoder_state)
            client.set_classifier_state(global_classifier_state)
            client.last_encoder_state = {k: v.cpu().clone() for k, v in global_encoder_state.items()}
            client.last_classifier_state = {k: v.cpu().clone() for k, v in global_classifier_state.items()}

    print("========= Phase 1: FedAvg Pre-training Finished =========")


# ============================================================
# 聚类 + 对齐
# ============================================================
def Cluster_and_Align(clients, anchor_config, top_percent, device):
    num_clients = len(clients)
    cluster_labels = []
    all_z = []
    k_list = list(range(2, 16))

    print("================== Clustering Start ==================")
    for client in clients:
        best_k, niid_idx = adaptive_cluster_selection(client.data, client.encoder, k_list, device)
        print(f"[Client {client.client_id}] Best_K={best_k}, NID={niid_idx}")
        labels, _ = gnn_embedding_kmeans_cluster(client.data, client.encoder, n_clusters=best_k, device=device)
        cluster_labels.append(labels)
        with torch.no_grad():
            z = client.encoder(client.data.x, client.data.edge_index).detach()
        all_z.append(z)

    print("================== Alignment Start ==================")
    node_alignments = {}
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                continue
            z_i, z_j = all_z[i], all_z[j]
            labels_i, labels_j = cluster_labels[i], cluster_labels[j]
            print(f"--- Aligning: Client {i} -> Client {j} ---")

            min_anchors = discover_mnn_anchors(z_i, z_j, metric=anchor_config['metric'])
            if not min_anchors:
                node_alignments[(i, j)] = {}
                print("No anchors found, skipping.")
                continue

            results = compute_anchor_list_with_diff(z_i, z_j, min_anchors, device=device)
            co_matrix, K1, K2 = build_cluster_cooccurrence_matrix(labels_i, labels_j, results,
                                                                 top_percent=top_percent)

            if np.sum(co_matrix) == 0:
                print(f"[Align] Empty matrix for ({i}->{j}), skip.")
                node_alignments[(i, j)] = {}
                continue

            alignment_i_to_j = extract_clear_alignments(
                co_matrix,
                min_ratio=0.25,
                min_count=anchor_config['min_count'],
                mode=1
            )

            node_alignments[(i, j)] = alignment_i_to_j
            print(f"[Align] ({i}->{j}) | K1={K1}, K2={K2}, nonzero={np.count_nonzero(co_matrix)}, "
                  f"aligned_pairs={len(alignment_i_to_j)}")

    print("================== Alignment Finished ==================")
    return cluster_labels, node_alignments


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    seed_ = 826
    set_seed(seed_)

    data_dir = "parsed_dataset/cs"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])
    NUM_CLASSES = 6
    NUM_CLIENTS = len(pyg_data_files)

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
    enhance_interval = 10
    align_update_interval = 150
    pretrain_rounds = 50
    top_error_cluster_percent = 0.5
    top_k_node_per_type = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    anchor_config = {
        'metric': 'cosine',
        'min_count': 30,
    }

    print("================== Pretraining Start ==================")
    clients, raw_data_list = load_all_clients(
        pyg_data_files, encoder_params, classifier_params, training_params, device, NUM_CLASSES
    )
    pretrain_fedavg(clients, pretrain_rounds, training_params)

    # 初始化
    best_acc = -1
    best_encoder_state = None
    best_classifier_state = None
    best_rnd = 0
    sliding_error_window = [deque(maxlen=5) for _ in range(NUM_CLIENTS)]

    cluster_labels, node_alignments = Cluster_and_Align(clients, anchor_config, top_percent=0.75, device=device)

    print("\n================ Federated Training Start (NC-Aug) ================")
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        # 定期更新聚类与对齐
        if rnd % align_update_interval == 0:
            cluster_labels, node_alignments = Cluster_and_Align(clients, anchor_config, top_percent=0.75, device=device)
            print("--- Cluster and Alignment Updated ---")

        # 知识增强阶段
        if rnd % enhance_interval == 0:
            for i in range(NUM_CLIENTS):
                client = clients[i]
                error_counts = client.analyze_prediction_errors_by_cluster(cluster_labels[i])
                sliding_error_window[i].append(error_counts)

            for i in range(NUM_CLIENTS):
                target_client = clients[i]
                aggregated_errors = aggregate_error_counts(sliding_error_window[i],
                                                           top_percent=top_error_cluster_percent)
                all_Z_aug_from_j, all_Y_aug_from_j = [], []

                for j in range(NUM_CLIENTS):
                    if i == j:
                        continue
                    alignment_i_to_j = node_alignments.get((i, j), {})
                    Z_aug_j, Y_aug_j = extract_augmented_node_data(
                        target_client, clients[j],
                        aggregated_errors, cluster_labels[j],
                        alignment_i_to_j, top_k_per_type=top_k_node_per_type
                    )
                    if Z_aug_j is not None:
                        all_Z_aug_from_j.append(Z_aug_j)
                        all_Y_aug_from_j.append(Y_aug_j)

                if all_Z_aug_from_j:
                    target_client.inject_augmented_node_data(
                        torch.cat(all_Z_aug_from_j, dim=0),
                        torch.cat(all_Y_aug_from_j, dim=0)
                    )

        # 本地训练与聚合
        classifier_states_local = []
        for i, client in enumerate(clients):
            loss_avg = 0
            for _ in range(training_params['local_epochs']):
                loss_avg += client.train()
            loss_avg /= training_params['local_epochs']

            if client.augmented_node_data is not None:
                loss_aug = client.train_on_augmented_nodes()
                print(f"Client {i} Augmented Loss: {loss_aug:.4f}")

            classifier_states_local.append(client.get_classifier_state())

        encoder_states = [client.get_encoder_state() for client in clients]
        global_encoder_state = average_state_dicts(encoder_states)
        global_classifier_state = average_state_dicts(classifier_states_local)

        for client in clients:
            client.set_encoder_state(global_encoder_state)
            client.set_classifier_state(global_classifier_state)
            client.last_encoder_state = {k: v.cpu().clone() for k, v in global_encoder_state.items()}
            client.last_classifier_state = {k: v.cpu().clone() for k, v in global_classifier_state.items()}

        avg_acc = evaluate_all_clients(clients, use_test=True)
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_encoder_state = {k: v.clone().detach() for k, v in global_encoder_state.items()}
            best_classifier_state = {k: v.clone().detach() for k, v in global_classifier_state.items()}
            best_rnd = rnd
            print("===> New best model saved")

    print("\n================ Federated Training Finished ================")
    for i, client in enumerate(clients):
        client.set_encoder_state(best_encoder_state)
        client.set_classifier_state(best_classifier_state)

    print("\n================ Final Evaluation ================")
    final_avg_acc = evaluate_all_clients(clients, use_test=True)
    print(f"Best Round: {best_rnd} | Best Acc: {final_avg_acc:.4f}")

