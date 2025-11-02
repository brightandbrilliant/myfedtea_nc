# structure_encoders_v2.py
import math
from typing import List, Literal, Sequence, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import degree as pyg_degree, to_scipy_sparse_matrix, k_hop_subgraph, to_networkx
import numpy as np
from scipy import sparse as sp
import networkx as nx

# ------------------------------
# Base class
# ------------------------------
class BaseStructureEncoder(nn.Module):
    """
    所有结构信息编码器的基类
    forward(graph: Data) -> [num_nodes, feat_dim]
    """
    def __init__(self):
        super().__init__()
        self.cached_features = None

    def forward(self, graph: Data) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _infer_device(graph: Data) -> torch.device:
        if hasattr(graph, "x") and graph.x is not None:
            return graph.x.device
        return torch.device("cpu")


# ------------------------------
# 1) Random-Walk Encoder
# ------------------------------
class RWEncoder(BaseStructureEncoder):
    def __init__(self, num_steps: int, add_identity: bool = False,
                 eps: float = 1e-12, standardize: bool = False,
                 out_dim: Optional[int] = None):
        super().__init__()
        assert num_steps >= 1
        self.num_steps = num_steps
        self.add_identity = add_identity
        self.eps = eps
        self.standardize = standardize
        self.out_dim = out_dim
        self.proj: Optional[nn.Linear] = None

    @torch.no_grad()
    def forward(self, graph: Data) -> torch.Tensor:
        if self.cached_features is not None:
            X = self.cached_features.to(self._infer_device(graph))
        else:
            device = self._infer_device(graph)
            num_nodes = graph.num_nodes
            A: sp.csr_matrix = to_scipy_sparse_matrix(graph.edge_index, num_nodes=num_nodes).astype(np.float64)
            deg = np.asarray(A.sum(axis=0)).ravel()
            deg = np.maximum(deg, self.eps)
            D_inv = sp.diags(1.0 / deg)
            M = (A @ D_inv).tocsr()

            feats: List[torch.Tensor] = []
            if self.add_identity:
                feats.append(torch.ones(num_nodes, dtype=torch.float32))

            M_power = M.copy()
            for _t in range(self.num_steps):
                diag = torch.from_numpy(M_power.diagonal().astype(np.float32))
                feats.append(diag)
                M_power = (M_power @ M).tocsr()

            X = torch.stack(feats, dim=1).to(device)
            if self.standardize:
                mean = X.mean(dim=0, keepdim=True)
                std = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
                X = (X - mean) / std
            self.cached_features = X.cpu()

        if self.out_dim is not None:
            if self.proj is None:
                self.proj = nn.Linear(X.size(1), self.out_dim).to(X.device)
            X = self.proj(X)
        return X


# ------------------------------
# 2) Degree Encoder
# ------------------------------
class DegreeEncoder(BaseStructureEncoder):
    def __init__(self, max_degree: int, mode: Literal["onehot","embed","numeric"]="onehot",
                 emb_dim: Optional[int]=None, normalize_numeric: bool=True, log_transform: bool=False,
                 out_dim: Optional[int]=None):
        super().__init__()
        self.max_degree = max_degree
        self.mode = mode
        self.normalize_numeric = normalize_numeric
        self.log_transform = log_transform
        self.out_dim = out_dim
        self.proj: Optional[nn.Linear] = None

        if self.mode == "embed":
            assert emb_dim is not None
            self.embedding = nn.Embedding(num_embeddings=max_degree, embedding_dim=emb_dim)

    @torch.no_grad()
    def forward(self, graph: Data) -> torch.Tensor:
        device = self._infer_device(graph)
        N = graph.num_nodes
        deg = pyg_degree(graph.edge_index[0], num_nodes=N).to(torch.long)
        deg_clip = deg.clamp(min=1, max=self.max_degree) - 1

        if self.mode == "onehot":
            if self.cached_features is not None:
                X = self.cached_features.to(device)
            else:
                X = torch.zeros(N, self.max_degree, dtype=torch.float32)
                X[torch.arange(N), deg_clip] = 1.0
                self.cached_features = X.cpu()
        elif self.mode == "embed":
            X = self.embedding(deg_clip.to(self.embedding.weight.device))
        else:  # numeric
            if self.cached_features is not None:
                X = self.cached_features.to(device)
            else:
                d = deg.to(torch.float32)
                if self.log_transform:
                    d = torch.log1p(d)
                if self.normalize_numeric:
                    d = (d / float(self.max_degree)).clamp(0.0,1.0)
                X = d.view(-1,1)
                self.cached_features = X.cpu()

        if self.out_dim is not None:
            if self.proj is None:
                self.proj = nn.Linear(X.size(1), self.out_dim).to(X.device)
            X = self.proj(X)
        return X


# ------------------------------
# 3) Subgraph Encoder
# ------------------------------
class SubgraphEncoder(BaseStructureEncoder):
    SUPPORTED: Sequence[str] = ("num_nodes","num_edges","density","avg_deg","max_deg","min_deg","clustering","assortativity")
    def __init__(self, k_hop:int, stats: Optional[Sequence[str]]=None, standardize: bool=False,
                 safe_nan_to_num: bool=True, out_dim: Optional[int]=None):
        super().__init__()
        assert k_hop >= 1
        self.k_hop = k_hop
        self.stats = list(stats) if stats else ["num_nodes","num_edges","density","avg_deg","max_deg","min_deg","clustering"]
        for s in self.stats:
            if s not in self.SUPPORTED:
                raise ValueError(f"Unsupported stat: {s}")
        self.standardize = standardize
        self.safe_nan_to_num = safe_nan_to_num
        self.out_dim = out_dim
        self.proj: Optional[nn.Linear] = None

    @torch.no_grad()
    def forward(self, graph: Data) -> torch.Tensor:
        device = self._infer_device(graph)
        if self.cached_features is not None:
            X = self.cached_features.to(device)
        else:
            N = graph.num_nodes
            ei = graph.edge_index
            G_full = to_networkx(graph, to_undirected=True)
            feats = []
            for center in range(N):
                nodes, sub_ei, _, _ = k_hop_subgraph(center, self.k_hop, ei, relabel_nodes=True, num_nodes=N)
                G_sub = G_full.subgraph(nodes.tolist()).copy()
                n = G_sub.number_of_nodes()
                e = G_sub.number_of_edges()
                vals = []
                for s in self.stats:
                    if s=="num_nodes": vals.append(float(n))
                    elif s=="num_edges": vals.append(float(e))
                    elif s=="density": vals.append(0.0 if n<=1 else 2.0*e/(n*(n-1)))
                    elif s=="avg_deg": vals.append(0.0 if n==0 else 2.0*e/max(n,1))
                    elif s=="max_deg": vals.append(0.0 if n==0 else float(max(dict(G_sub.degree()).values())))
                    elif s=="min_deg": vals.append(0.0 if n==0 else float(min(dict(G_sub.degree()).values())))
                    elif s=="clustering": vals.append(0.0 if n<=2 else float(nx.clustering(G_sub, nodes=[center]).get(center,0.0)))
                    elif s=="assortativity":
                        try:
                            coeff = nx.degree_assortativity_coefficient(G_sub)
                            if math.isnan(coeff) and self.safe_nan_to_num: coeff=0.0
                            vals.append(float(coeff))
                        except: vals.append(0.0)
                    else: raise RuntimeError(f"Unhandled stat: {s}")
                feats.append(torch.tensor(vals, dtype=torch.float32))
            X = torch.stack(feats, dim=0)
            if self.standardize and X.numel()>0:
                mean = X.mean(dim=0, keepdim=True)
                std = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
                X = (X-mean)/std
            self.cached_features = X.cpu()
        if self.out_dim is not None:
            if self.proj is None:
                self.proj = nn.Linear(X.size(1), self.out_dim).to(X.device)
            X = self.proj(X)
        return X


# ------------------------------
# Factory
# ------------------------------
def build_structure_encoder(name: str, **kwargs) -> BaseStructureEncoder:
    name = name.lower()
    if name=="rw": return RWEncoder(**kwargs)
    elif name=="degree": return DegreeEncoder(**kwargs)
    elif name=="subgraph": return SubgraphEncoder(**kwargs)
    else: raise ValueError(f"Unknown structure encoder type: {name}")
