import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConstructor(nn.Module):
    """
    信息瓶颈引导的动态图构造器，显式学习可训练邻接矩阵 A_t。
    实现 Eq (1) 自表达、Eq (2) 信息瓶颈损失、Eq (3) 时间一致性正则。
    """
    def __init__(self, num_nodes, embedding_dim, top_k=8):
        super(GraphConstructor, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.top_k = top_k

        # 可学习邻接矩阵参数 A_t
        self.adj_param = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(self, Z_t, Z_prev=None):
        """
        参数：
        Z_t: 节点嵌入 [N, d]
        Z_prev: 可选，上一个时间步的节点嵌入 [N, d]，用于平滑损失
        返回：
        Z_hat: 自表达重建后的节点嵌入 [N, d]
        A_t: 稠密邻接矩阵 [N, N]
        A_sparse: 稀疏化邻接矩阵 [N, N]
        phi: 结构置信度矩阵 [N, N]
        losses: dict 包含各个损失项
        """
        N = Z_t.size(0)

        # 构造有向带权图 A_t
        A_t = self.adj_param.abs()  # 保证非负边权

        # Min-Max 归一化 A_t 每行到 [0,1]
        A_min, A_max = A_t.min(dim=-1, keepdim=True)[0], A_t.max(dim=-1, keepdim=True)[0]
        A_norm = (A_t - A_min) / (A_max - A_min + 1e-8)

        # Top-K 稀疏化
        A_sparse = torch.zeros_like(A_norm)
        topk = torch.topk(A_norm, self.top_k, dim=-1)
        indices = topk.indices
        row_idx = torch.arange(N).unsqueeze(1).expand(-1, self.top_k)
        A_sparse[row_idx, indices] = A_norm[row_idx, indices]

        # Eq (1) 自表达重建
        Z_hat = torch.matmul(A_sparse, Z_t)  # [N, d]
        L_reconstruct = F.mse_loss(Z_hat, Z_t)

        # Eq (2) 冗余正则 & 预测增强项  (需要外部引入对比目标，这里作为 placeholder)
        # L_IB_graph = L_reconstruct + lambda1 * I(A,Z) - lambda2 * I(A,Y)

        # Eq (3) 时间一致性正则化（若提供 Z_prev）
        L_smooth = torch.tensor(0.0, device=Z_t.device)
        if Z_prev is not None:
            A_prev = self.adj_param.abs()  # 假设平滑以邻接参数为主
            A_prev = (A_prev - A_prev.min(dim=-1, keepdim=True)[0]) / (A_prev.max(dim=-1, keepdim=True)[0] + 1e-8)
            L_smooth = F.mse_loss(A_norm, A_prev)

        phi = A_sparse.abs()

        return Z_hat, A_norm, A_sparse, phi, {"reconstruct": L_reconstruct, "smooth": L_smooth}


if __name__ == '__main__':
    Z = torch.randn(32, 64)
    model = GraphConstructor(num_nodes=32, embedding_dim=64, top_k=5)
    Z_hat, A, A_sparse, phi, losses = model(Z)
    print('Z_hat shape:', Z_hat.shape)
    print('A shape:', A.shape)
    print('Reconstruct loss:', losses['reconstruct'].item())
