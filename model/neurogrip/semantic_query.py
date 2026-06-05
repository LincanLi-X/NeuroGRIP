import torch
import torch.nn as nn

class SemanticQuery(nn.Module):
    """
    SemanticQuery module: projects EEG node features into the semantic embedding space
    aligned with the knowledge graph.
    
    This module is designed to learn a transformation from EEG spatial-temporal features
    (e.g., output from an encoder or STGNN layer) into a semantic space where each node
    embedding can be used as a query vector for retrieving related biomedical knowledge.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        proj_dim: int,
        k_hop: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize the semantic projection network.

        Args:
            input_dim (int): Input feature dimension (EEG node feature dimension)
            hidden_dim (int): Hidden layer dimension for intermediate representation
            proj_dim (int): Output projection dimension (semantic space dimension)
            dropout (float): Dropout rate for regularization
        """
        super(SemanticQuery, self).__init__()
        self.k_hop = k_hop

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim)
        )

        self.layer_norm = nn.LayerNorm(proj_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_khop_mask(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Build k-hop reachability mask for each node.

        Args:
            adj (torch.Tensor): adjacency matrix (batch_size, num_nodes, num_nodes)

        Returns:
            torch.Tensor: reachability mask (batch_size, num_nodes, num_nodes), bool
        """
        # Binarize adjacency and ensure self is included in neighborhood.
        adj_bin = (adj > 0).to(dtype=torch.bool)
        batch_size, num_nodes, _ = adj_bin.shape
        eye = torch.eye(num_nodes, device=adj.device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)
        reach = adj_bin | eye
        frontier = reach.clone()

        for _ in range(1, max(1, self.k_hop)):
            # frontier @ adj_bin in boolean semiring (implemented via numeric matmul + threshold)
            next_reach = torch.bmm(frontier.float(), adj_bin.float()) > 0
            reach = reach | next_reach
            frontier = next_reach

        return reach

    def forward(self, node_embeddings: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for generating SemAlignQuery embeddings.

        Args:
            node_embeddings (torch.Tensor): STGNN node embeddings c_t^i (batch_size, num_nodes, input_dim)
            adj (torch.Tensor): raw graph adjacency at time t (batch_size, num_nodes, num_nodes)

        Returns:
            torch.Tensor: k-hop subgraph query vectors (batch_size, num_nodes, proj_dim)
        """
        # 1) Project node embeddings to KG semantic space: \tilde{c}_t^j
        projected = self.projector(node_embeddings)
        projected = self.layer_norm(projected)

        # 2) Build k-hop neighborhood mask for each center node v_i
        khop_mask = self._build_khop_mask(adj)  # (B, N, N)

        # 3) Subgraph-level average pooling:
        # q_i = (1 / |N_k(v_i)|) * sum_{v_j in N_k(v_i)} \tilde{c}_t^j
        neighbor_count = khop_mask.sum(dim=-1, keepdim=True).clamp_min(1)  # (B, N, 1)
        q = torch.bmm(khop_mask.float(), projected) / neighbor_count.float()
        return q


if __name__ == "__main__":
    # Example test run
    batch_size = 2
    num_nodes = 19   # EEG channels
    input_dim = 64   # Input feature dimension
    hidden_dim = 128
    proj_dim = 256

    X = torch.randn(batch_size, num_nodes, input_dim)
    A = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()

    model = SemanticQuery(input_dim, hidden_dim, proj_dim, k_hop=2)
    Q = model(X, A)

    print(f"Input shape: {X.shape}")
    print(f"Output (semantic query) shape: {Q.shape}")
