import torch
import torch.nn as nn
import torch.nn.functional as F

class IRENE_Loss(nn.Module):
    """
    IRENE 总损失函数
    包括：
        - 信息瓶颈图构造损失 L_ibgraph
        - 时间平滑正则 L_smooth
        - 重建损失 L_recon
    """
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0):
        super().__init__()
        self.lambda1 = lambda1  # I(A;Z)
        self.lambda2 = lambda2  # I(A;Y)
        self.lambda3 = lambda3  # smooth
        self.lambda4 = lambda4  # recon

        self.infoNCE_score = nn.Bilinear(in1_features=64, in2_features=1, out_features=1)
        self.donsker_net = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, Z, A, Y, A_prev=None, recon=None, target=None):
        N = Z.shape[0]

        # === 自表达重建 ===
        Z_hat = torch.matmul(A, Z)
        L_rec = F.mse_loss(Z_hat, Z)

        # === 冗余项 I(A;Z) ===
        AZ_concat = torch.cat([A, Z], dim=1)
        perm = torch.randperm(N)
        AZ_neg = torch.cat([A[perm], Z], dim=1)
        T_pos = self.donsker_net(AZ_concat).mean()
        T_neg = torch.exp(self.donsker_net(AZ_neg)).mean()
        L_redundancy = T_pos - torch.log(T_neg + 1e-8)

        # === 判别项 I(A;Y) ===
        A_flat = A.mean(dim=1).unsqueeze(-1)
        pos_score = self.infoNCE_score(A_flat, Y.unsqueeze(-1)).squeeze()
        neg_Y = Y[torch.randperm(N)]
        neg_score = self.infoNCE_score(A_flat, neg_Y.unsqueeze(-1)).squeeze()
        logits = torch.stack([pos_score, neg_score], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=Z.device)
        L_info = F.cross_entropy(logits, labels)

        # === 时间平滑项 L_smooth ===
        L_smooth = torch.tensor(0.0, device=Z.device)
        if A_prev is not None:
            L_smooth = F.mse_loss(A, A_prev)

        # === 重建损失 recon ===
        L_recon = torch.tensor(0.0, device=Z.device)
        if recon is not None and target is not None:
            L_recon = F.mse_loss(recon, target)

        # === 最终总损失 ===
        L_ib = L_rec + self.lambda1 * L_redundancy - self.lambda2 * L_info
        L_total = L_ib + self.lambda3 * L_smooth + self.lambda4 * L_recon

        return L_total, {
            "ib": L_ib.item(),
            "rec": L_rec.item(),
            "redundancy": L_redundancy.item(),
            "info": L_info.item(),
            "smooth": L_smooth.item(),
            "recon": L_recon.item(),
            "total": L_total.item()
        }
