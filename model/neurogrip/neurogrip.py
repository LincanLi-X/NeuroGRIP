import torch
import torch.nn as nn
from model.neurogrip.semantic_query import SemanticQuery
from model.neurogrip.faiss_retriever import KnowledgeRetriever
from model.neurogrip.graph_refiner import GraphRefiner


class NeuroGRIP(nn.Module):
    """
    NeuroGRIP module:
    This module integrates the entire Retrieval-Augmented Graph refinement process (RAG-based refinement)
    into a single unified architecture. It performs three main steps:
    
    1. Projects EEG node features into a semantic embedding space (SemanticQuery)
    2. Retrieves related biomedical knowledge from a FAISS-based index (KnowledgeRetriever)
    3. Refines the dynamically learned adjacency matrix using knowledge-informed relations (GraphRefiner)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        proj_dim: int,
        index_path: str,
        kg_json_path: str,
        k: int = 5,
        query_k_hop: int = 2,
        refine_threshold: float = 0.6,
        refine_alpha: float = 0.7,
        save_path: str = "./results/graph_refinement",
        rag_backend: str = "auto",
        match_mode: str = "evidence",
        log_supports: bool = True,
    ):
        """
        Initialize NeuroGRIP and all its submodules.

        Args:
            input_dim (int): Dimension of EEG node input features.
            hidden_dim (int): Hidden layer size in semantic query projector.
            proj_dim (int): Dimension of semantic projection space.
            index_path (str): Path to FAISS index for knowledge retrieval.
            kg_json_path (str): Path to JSON file containing knowledge triplets.
            k (int): Number of top-K triplets to retrieve per node.
            query_k_hop (int): k-hop size for SemAlignQuery subgraph aggregation.
            refine_threshold (float): Edge confidence threshold in graph refinement.
            refine_alpha (float): Deprecated. Kept only for backward compatibility.
            save_path (str): Directory for saving triplet support logs.
            rag_backend (str): "auto", "faiss", or "numpy" retrieval backend.
            match_mode (str): edge-triplet match policy: "strict", "soft", or "evidence".
            log_supports (bool): whether to save retrieved supports for interpretation.
        """
        super(NeuroGRIP, self).__init__()

        # 1. Semantic projection network
        self.query = SemanticQuery(input_dim, hidden_dim, proj_dim, k_hop=query_k_hop)

        # 2. Knowledge retrieval module
        self.retriever = KnowledgeRetriever(
            index_path=index_path,
            kg_json_path=kg_json_path,
            k=k,
            backend=rag_backend,
        )

        # 3. Graph refinement module
        self.refiner = GraphRefiner(
            threshold=refine_threshold,
            alpha=refine_alpha,
            save_path=save_path,
            match_mode=match_mode,
            log_supports=log_supports,
        )

    def refine_graph(self, A_t: torch.Tensor, node_embeddings_t: torch.Tensor) -> torch.Tensor:
        """
        Perform NeuroGRIP refinement:
        Generate semantic queries -> Retrieve knowledge -> Refine dynamic adjacency matrix.

        Args:
            A_t (torch.Tensor): Original STGNN-learned adjacency matrix (batch_size, num_nodes, num_nodes)
            node_embeddings_t (torch.Tensor): STGNN node embeddings c_t^i (batch_size, num_nodes, input_dim)

        Returns:
            torch.Tensor: Refined adjacency matrix A_refined (batch_size, num_nodes, num_nodes)
        """
        # 1. Semantic projection to obtain query embeddings
        Q = self.query(node_embeddings_t, A_t)

        # 2. Knowledge retrieval using FAISS
        retrieved = self.retriever.retrieve(Q)

        # 3. Knowledge-based graph refinement
        A_refined = self.refiner.refine(A_t, Q, retrieved)

        return A_refined


if __name__ == "__main__":
    # Example usage demonstration
    batch_size = 1
    num_nodes = 5
    input_dim = 64
    hidden_dim = 128
    proj_dim = 256

    # Dummy EEG features and learned adjacency matrix
    X_t = torch.randn(batch_size, num_nodes, input_dim)
    A_t = torch.sigmoid(torch.randn(batch_size, num_nodes, num_nodes))

    # Example file paths (should point to real data in production)
    index_path = "knowledge/faiss.index"
    kg_json_path = "knowledge/kg_triplets.json"

    # Instantiate NeuroGRIP
    model = NeuroGRIP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        index_path=index_path,
        kg_json_path=kg_json_path,
        k=3,
        refine_threshold=0.6,
        refine_alpha=0.7,
        save_path="./results/graph_refinement"
    )

    # Perform one refinement pass
    A_refined = model.refine_graph(A_t, X_t)

    print(f"Input A_t shape: {A_t.shape}")
    print(f"Refined A_refined shape: {A_refined.shape}")
    print("NeuroGRIP refinement completed successfully.")
