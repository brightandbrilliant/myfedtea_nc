import os
import torch
from collections import deque, defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

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
# 工具函数
# ============================================================

def average_state_dicts(state_dicts):
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def evaluate_all_clients(clients, use_test=False):
    accs = []
    for client in clients:
        accs.append(client.evaluate(use_test=use_test))
    return float(torch.tensor(accs).mean())


def plot_acc_curves(acc_records, interval, save_path=None):
    plt.figure(figsize=(7, 5))
    for aug_w, acc_list in acc_records.items():
        rounds = [interval * (i + 1) for i in range(len(acc_list))]
        plt.plot(rounds, acc_list, marker='o', label=f"aug_w={aug_w}")
    plt.xlabel("Federated Rounds")
    plt.ylabel("Accuracy")
    plt.title("Sensitivity Analysis of Augment Weight")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Client 加载
# ============================================================

def load_all_clients(pyg_data_paths, encoder_params, classifier_params,
                     training_params, device, num_classes, augment_weight):

    clients = []
    for cid, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)
        data = split_client_data(raw_data, device=device)

        encoder = GraphSAGE(**encoder_params)
        classifier = ResMLP(
            input_dim=encoder_params["output_dim"],
            output_dim=num_classes,
            **classifier_params
        )

        client = Client(
            client_id=cid,
            data=data,
            encoder=encoder,
            classifier=classifier,
            device=device,
            lr=training_params["lr"],
            weight_decay=training_params["weight_decay"],
            augment_weight=augment_weight
        )
        clients.append(client)

    return clients


# ============================================================
# 聚类与对齐（与你原始代码一致）
# ============================================================

def Cluster_and_Align(clients, anchor_config, top_percent, device):
    num_clients = len(clients)
    cluster_labels, all_z = [], []
    k_list = list(range(2, 16))

    for client in clients:
        best_k, _ = adaptive_cluster_selection(client.data, client.encoder, k_list, device)
        labels, _ = gnn_embedding_kmeans_cluster(
            client.data, client.encoder, n_clusters=best_k, device=device
        )
        cluster_labels.append(labels)
        with torch.no_grad():
            z = client.encoder(client.data.x, client.data.edge_index).detach()
        all_z.append(z)

    node_alignments = {}
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                continue

            z_i, z_j = all_z[i], all_z[j]
            labels_i, labels_j = cluster_labels[i], cluster_labels[j]

            anchors = discover_mnn_anchors(z_i, z_j, metric=anchor_config["metric"])
            if not anchors:
                node_alignments[(i, j)] = {}
                continue

            results = compute_anchor_list_with_diff(z_i, z_j, anchors, device)
            co_mat, _, _ = build_cluster_cooccurrence_matrix(
                labels_i, labels_j, results, top_percent=top_percent
            )

            if np.sum(co_mat) == 0:
                node_alignments[(i, j)] = {}
                continue

            align = extract_clear_alignments(
                co_mat,
                min_ratio=0.25,
                min_count=anchor_config["min_count"],
                mode=1
            )
            node_alignments[(i, j)] = align

    return cluster_labels, node_alignments


# ============================================================
# 主程序：敏感度分析
# ============================================================

if __name__ == "__main__":

    set_seed(62)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------- 数据 -----------------
    data_dir = "parsed_dataset/cr_10"
    pyg_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])
    NUM_CLIENTS = len(pyg_files)
    NUM_CLASSES = 7

    # ----------------- 参数 -----------------
    encoder_params = {
        "input_dim": torch.load(pyg_files[0]).x.shape[1],
        "hidden_dim": 128,
        "output_dim": 64,
        "num_layers": 3,
        "dropout": 0.5
    }
    classifier_params = {"hidden_dim": 128, "num_layers": 3, "dropout": 0.3}
    training_params = {"lr": 0.001, "weight_decay": 1e-4, "local_epochs": 5}

    pretrain_rounds = 50
    num_rounds = 700
    enhance_interval = 10
    align_update_interval = 150
    record_interval = 10

    anchor_config = {"metric": "cosine", "min_count": 25}

    augment_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    acc_records = {w: [] for w in augment_weights}

    # ========================================================
    # 敏感度实验循环
    # ========================================================

    for aug_w in augment_weights:
        print(f"\n========== Sensitivity Run: augment_weight={aug_w} ==========")

        clients = load_all_clients(
            pyg_files, encoder_params, classifier_params,
            training_params, device, NUM_CLASSES, aug_w
        )

        # ---------- 预训练 ----------
        for _ in range(pretrain_rounds):
            for c in clients:
                for _ in range(training_params["local_epochs"]):
                    c.train()

            enc_states = [c.get_encoder_state() for c in clients]
            cls_states = [c.get_classifier_state() for c in clients]
            g_enc = average_state_dicts(enc_states)
            g_cls = average_state_dicts(cls_states)

            for c in clients:
                c.set_encoder_state(g_enc)
                c.set_classifier_state(g_cls)

        # ---------- 初始化对齐 ----------
        cluster_labels, node_alignments = Cluster_and_Align(
            clients, anchor_config, top_percent=0.75, device=device
        )

        sliding_error_window = [deque(maxlen=5) for _ in range(NUM_CLIENTS)]

        # ---------- 联邦训练 ----------
        for rnd in range(1, num_rounds + 1):

            if rnd % align_update_interval == 0:
                cluster_labels, node_alignments = Cluster_and_Align(
                    clients, anchor_config, top_percent=0.75, device=device
                )

            if rnd % enhance_interval == 0:
                for i, client in enumerate(clients):
                    err = client.analyze_prediction_errors_by_cluster(cluster_labels[i])
                    sliding_error_window[i].append(err)

                for i, target in enumerate(clients):
                    agg = defaultdict(int)
                    for d in sliding_error_window[i]:
                        for k, v in d.items():
                            agg[k] += v

                    Zs, Ys = [], []
                    for j, src in enumerate(clients):
                        if i == j:
                            continue
                        align = node_alignments.get((i, j), {})
                        Zj, Yj = src.extract_augmented_node_data(
                            agg, cluster_labels[j], align
                        )
                        if Zj is not None:
                            Zs.append(Zj)
                            Ys.append(Yj)

                    if Zs:
                        target.inject_augmented_node_data(
                            torch.cat(Zs), torch.cat(Ys)
                        )

            cls_states = []
            for c in clients:
                for _ in range(training_params["local_epochs"]):
                    c.train()
                if c.augmented_node_data is not None:
                    c.train_on_augmented_nodes()
                cls_states.append(c.get_classifier_state())

            g_enc = average_state_dicts([c.get_encoder_state() for c in clients])
            g_cls = average_state_dicts(cls_states)

            for c in clients:
                c.set_encoder_state(g_enc)
                c.set_classifier_state(g_cls)

            if rnd % record_interval == 0:
                acc = evaluate_all_clients(clients, use_test=True)
                acc_records[aug_w].append(acc)
                print(f"[aug_w={aug_w}] Round {rnd} | Acc={acc:.4f}")

    # ========================================================
    # 画图
    # ========================================================
    plot_acc_curves(
        acc_records,
        interval=record_interval,
        save_path="augment_weight_sensitivity.png"
    )
