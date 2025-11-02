import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        """
        构建一个带有残差连接的MLP分类器，用于节点分类任务。

        Args:
            input_dim (int): 输入维度，即 GCN/GraphSage 编码器输出的节点嵌入维度。
            hidden_dim (int): 隐藏层维度。
            output_dim (int): 输出维度，即类别的数量。
            num_layers (int): MLP层的总数量 (至少为2，因为需要中间层实现残差)。
            dropout (float): Dropout 比率。
        """
        super(ResMLP, self).__init__()

        if num_layers < 2:
            raise ValueError("num_layers for ResMLPClassifier must be at least 2 for meaningful residual connections.")

        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()

        # 第一层：输入维度 (GCN嵌入) -> 隐藏维度
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # 中间层：隐藏维度 -> 隐藏维度。这些层将应用残差连接。
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 最后一层：隐藏维度 -> 输出维度（类别数量）
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, node_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            node_embedding (torch.Tensor): 节点的嵌入，形状为 [N, embedding_dim]。
                                          这里的 N 是当前批次或整个图的节点数。

        Returns:
            torch.Tensor: 节点属于各个类别的 logits，形状为 [N, output_dim]。
        """
        x = node_embedding  # 输入就是编码器的节点嵌入

        # 1. 第一层 (Input_dim -> Hidden_dim)
        x = self.layers[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. 中间层 (Hidden_dim -> Hidden_dim)，应用残差连接
        # 从第二层开始到倒数第二层
        for i in range(1, self.num_layers - 1):
            identity = x  # 保存当前层的输入，作为残差连接的跳跃路径

            linear_layer = self.layers[i]
            x = linear_layer(x)

            # 残差连接：当前层输出 + 上一层输入
            x = F.relu(x + identity)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. 最后一层 (Hidden_dim -> Output_dim)
        # 最后一层不应用激活函数 (ReLU) 和 Dropout，直接输出分类 logits
        output_logits = self.layers[-1](x)

        return output_logits

# 模型的整体架构变为：
# [GCN/GraphSage] (Encoders) -> 节点嵌入 Z -> [ResMLPClassifier] (Decoder/Classifier) -> 类别 Logits
