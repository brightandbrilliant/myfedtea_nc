import torch
import numpy as np
from collections import defaultdict
import random


class Client:
    def __init__(self, client_id, data, encoder, classifier, device='cpu', lr=0.005,
                 weight_decay=1e-4, max_grad_norm=30000.0,
                 # 引入增强损失权重
                 augment_weight=0.2, mu=0.01):

        self.client_id = client_id
        # 节点分类数据包含 x, edge_index, y, train/val/test_mask
        self.data = data.to(device)
        self.device = device
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)  # <--- 节点分类头

        self.augment_weight = augment_weight

        # 优化器现在包含 encoder 和 classifier 的参数
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        # 替换为节点分类损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

        self.augmented_node_data = None  # 存储增强数据 (Z_aug, Y_aug)

        self.max_grad_norm = max_grad_norm
        self.mu = mu
        self.last_encoder_state = None
        self.last_classifier_state = None  # <--- 跟踪全局 classifier 状态

    def train(self):
        """常规训练：只使用本地训练集进行节点分类"""
        self.encoder.train()
        self.classifier.train()
        self.optimizer.zero_grad()

        # 1. 前向传播和本地损失计算 (L_local)
        z = self.encoder(self.data.x, self.data.edge_index)

        # 原始训练集掩码
        raw_train_mask = self.data.train_mask

        # --- [标签过滤] 过滤掉训练集中的负标签 (-1) ---
        valid_labels_mask = (self.data.y >= 0)

        # 最终的有效训练掩码
        final_train_mask = raw_train_mask & valid_labels_mask

        if final_train_mask.sum() == 0:
            return 0.0

        pred_logits = self.classifier(z[final_train_mask])
        labels = self.data.y[final_train_mask].long()  # 确保标签为 long 类型

        # 节点分类损失
        loss_local = self.criterion(pred_logits, labels)

        # --- L_reg (Encoder 和 Classifier 的联合正则项) 计算 ---
        loss_reg = 0.0
        if self.mu > 0 and self.last_encoder_state and self.last_classifier_state:
            # Encoder 正则化
            for name, param in self.encoder.named_parameters():
                global_param = self.last_encoder_state[name].to(param.device)
                loss_reg += torch.sum(torch.pow(param.float() - global_param.float(), 2))

            # Classifier 正则化
            for name, param in self.classifier.named_parameters():
                global_param = self.last_classifier_state[name].to(param.device)
                loss_reg += torch.sum(torch.pow(param.float() - global_param.float(), 2))

            loss_reg = (self.mu / 2.0) * loss_reg
        # --- L_reg 计算结束 ---

        # 2. 计算总损失
        loss = loss_local + loss_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()

        return loss.item()

    # -------------------------------------------------------------------
    # 针对新框架新增/修改的方法
    # -------------------------------------------------------------------

    def train_on_augmented_nodes(self):
        """增强训练：仅在注入的跨图节点数据上训练 (L = L_aug)，加入标签过滤。"""
        if self.augmented_node_data is None:
            return 0.0

        # 增强训练通常只更新分类头 (Classifier)
        self.encoder.eval()
        self.classifier.train()
        self.optimizer.zero_grad()

        # 增强数据部分
        z_aug_raw, y_aug_raw = self.augmented_node_data

        # --- [标签过滤] 过滤掉增强数据中的负标签 (-1) ---
        valid_indices = (y_aug_raw >= 0).squeeze()

        if valid_indices.sum().item() == 0:
            self.clear_augmented_data()
            return 0.0

        z_aug = z_aug_raw[valid_indices]
        y_aug = y_aug_raw[valid_indices]

        # 使用当前 classifier 进行预测
        pred_logits_aug = self.classifier(z_aug)

        # 计算增强损失
        loss_aug = self.criterion(pred_logits_aug, y_aug)

        # 总损失 = 增强损失 * 权重
        loss = self.augment_weight * loss_aug

        loss.backward()
        # 只裁剪 classifier 的梯度
        torch.nn.utils.clip_grad_norm_(
            list(self.classifier.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()

        # 清理增强数据
        self.clear_augmented_data()

        return loss.item()

    def analyze_prediction_errors_by_cluster(self, cluster_labels):
        """
        分析验证集上每个聚类类别中节点的错误分类次数，跳过负标签。
        """
        self.encoder.eval()
        self.classifier.eval()

        error_cluster_counts = defaultdict(int)

        with torch.no_grad():
            z = self.encoder(self.data.x, self.data.edge_index)

            raw_mask = self.data.val_mask

            # --- [标签过滤] 过滤掉验证集中的负标签 ---
            valid_labels_mask = (self.data.y >= 0)
            final_mask = raw_mask & valid_labels_mask

            if final_mask.sum() == 0:
                return {}

            pred_logits = self.classifier(z[final_mask])
            true_labels = self.data.y[final_mask]

            predicted_labels = pred_logits.argmax(dim=1)

            # 找到被错误分类的节点索引（在 final_mask 内部的索引）
            error_mask_local = (predicted_labels != true_labels)

            # 映射回全局节点索引
            # squeeze() 确保 torch.nonzero 返回一个一维张量
            all_nodes_indices = torch.nonzero(final_mask, as_tuple=False).squeeze()

            # 处理 all_nodes_indices 为空或只包含一个元素的情况
            if all_nodes_indices.ndim == 0 and all_nodes_indices.numel() == 1:
                # 只有一个元素时，需要特殊处理索引
                error_nodes_indices = all_nodes_indices[
                    error_mask_local.item()].tolist() if error_mask_local.item() else []
            elif all_nodes_indices.numel() > 0:
                error_nodes_indices = all_nodes_indices[error_mask_local].tolist()
            else:
                error_nodes_indices = []

            # 统计错误节点所属的聚类类别
            for node_idx in error_nodes_indices:
                cluster_id = cluster_labels[node_idx]

                # 确保 cluster_id 是标准 Python int
                if isinstance(cluster_id, (np.int64, np.int32)):
                    cluster_id = int(cluster_id)
                elif isinstance(cluster_id, torch.Tensor):
                    cluster_id = cluster_id.item()

                error_cluster_counts[cluster_id] += 1

        return dict(error_cluster_counts)

    def evaluate(self, use_test=False):
        """节点分类评估：计算验证集或测试集的准确率，跳过负标签。"""
        self.encoder.eval()
        self.classifier.eval()

        with torch.no_grad():
            z = self.encoder(self.data.x, self.data.edge_index)

            raw_mask = self.data.test_mask if use_test else self.data.val_mask

            # --- [标签过滤] 过滤掉评估集中的负标签 ---
            valid_labels_mask = (self.data.y >= 0)
            final_mask = raw_mask & valid_labels_mask

            if final_mask.sum() == 0:
                return 0.0

            # 对评估集节点进行预测
            pred_logits = self.classifier(z[final_mask])
            labels = self.data.y[final_mask].long()  # 确保标签为 long 类型

            # 计算预测标签 (argmax)
            pred_labels = pred_logits.argmax(dim=1)

            # 计算准确率
            correct = (pred_labels == labels).sum().item()
            total = final_mask.sum().item()

            acc = correct / total

        return acc

    def inject_augmented_node_data(self, augmented_z: torch.Tensor, augmented_y: torch.Tensor):
        """注入增强节点嵌入和标签"""
        if augmented_z is not None and augmented_y is not None:
            # 使用 detach() 且移到设备上
            self.augmented_node_data = (augmented_z.detach().to(self.device),
                                        augmented_y.detach().to(self.device))
        else:
            self.augmented_node_data = None

    def clear_augmented_data(self):
        """清除注入的增强数据"""
        self.augmented_node_data = None

    # -------------------------------------------------------------------
    # 状态管理方法 (已修正)
    # -------------------------------------------------------------------

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_classifier_state(self):
        # 修正了参数列表，确保与 main.py 中的调用一致
        return self.classifier.state_dict()

    def set_encoder_state(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def set_classifier_state(self, state_dict):
        self.classifier.load_state_dict(state_dict)