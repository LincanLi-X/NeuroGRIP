import torch
import torch.nn as nn
import json
import os
import re


class GraphRefiner(nn.Module):
    """
    GraphRefiner module:
    Refines the dynamically learned adjacency matrix (A_t) from STGNN using external biomedical
    knowledge retrieved by the KnowledgeRetriever module.
    
    The refinement process computes knowledge confidence per raw edge and prunes
    edges not supported by external evidence.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        alpha: float = 0.7,
        save_path: str = "./results/graph_refinement",
        match_sim_threshold: float = 0.5,
        default_reliability: float = 0.8,
    ):
        """
        Initialize the GraphRefiner.

        Args:
            threshold (float): Confidence threshold for edge pruning.
            alpha (float): Deprecated. Kept only for backward compatibility.
            save_path (str): Directory to save knowledge-based refinement logs.
        """
        super(GraphRefiner, self).__init__()
        self.threshold = threshold
        # Deprecated: no longer used after switching to raw-edge pruning semantics.
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.save_path = save_path
        self.match_sim_threshold = match_sim_threshold
        self.default_reliability = default_reliability

        os.makedirs(save_path, exist_ok=True)
        self.support_log_path = os.path.join(save_path, "triplet_supports.json")
        self._support_records = []

    @staticmethod
    def _normalize_similarity(similarity: float) -> float:
        """
        Normalize similarity into [0, 1].
        FAISS IndexFlatIP over L2-normalized vectors yields cosine similarity in [-1, 1].
        """
        similarity = float(similarity)
        if similarity < -1.0 or similarity > 1.0:
            # If distance-like value comes in unexpectedly, clamp conservatively.
            similarity = max(min(similarity, 1.0), -1.0)
        return (similarity + 1.0) / 2.0

    def _source_reliability(self, entry: dict) -> float:
        """
        Estimate source reliability r_m in [0,1].
        Priority:
        1) explicit numeric reliability field;
        2) source/source_type heuristic;
        3) default value.
        """
        raw_rel = entry.get("reliability", None)
        if raw_rel is not None:
            try:
                return float(max(0.0, min(1.0, raw_rel)))
            except Exception:
                pass

        source_type = str(entry.get("source_type", "")).lower()
        if source_type == "guideline":
            return 0.95
        if source_type == "paper":
            return 0.85
        if source_type == "case_report":
            return 0.65

        source = str(entry.get("source", "")).lower()
        if any(k in source for k in ["guideline", "ilae", "aes", "nice", "sign"]):
            return 0.95
        if any(k in source for k in ["journal", "study", "paper", "review"]):
            return 0.85
        if any(k in source for k in ["case report", "case"]):
            return 0.65
        return self.default_reliability

    @staticmethod
    def _normalize_entity_text(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"[^a-z0-9\s\-_/]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _entity_similarity(self, a: str, b: str) -> float:
        a = self._normalize_entity_text(a)
        b = self._normalize_entity_text(b)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        if a in b or b in a:
            return 0.9
        sa = set(a.split())
        sb = set(b.split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _build_node_entity_candidates(self, evidence_triplets: list, topn: int = 10) -> list:
        """
        Build entity candidate set for a node from its retrieved triplets.
        """
        ranked = sorted(
            evidence_triplets,
            key=lambda x: float(x.get("similarity", 0.0)),
            reverse=True,
        )
        candidates = []
        for e in ranked[:topn]:
            h = str(e.get("head", "")).strip()
            t = str(e.get("tail", "")).strip()
            if h:
                candidates.append(h)
            if t:
                candidates.append(t)
        # Keep order while deduplicating
        seen = set()
        uniq = []
        for c in candidates:
            n = self._normalize_entity_text(c)
            if n and n not in seen:
                seen.add(n)
                uniq.append(c)
        return uniq

    def _entity_matches_node(self, entity: str, node_candidates: list, threshold: float = 0.5) -> bool:
        return any(self._entity_similarity(entity, c) >= threshold for c in node_candidates)

    def _match_indicator(self, entry: dict, node_i_candidates: list, node_j_candidates: list) -> float:
        """
        1_match(e_ij, tau_m): entity alignment match between edge endpoints and triplet (h,t).
        """
        head = str(entry.get("head", "")).strip()
        tail = str(entry.get("tail", "")).strip()
        relation = str(entry.get("relation", "")).strip()
        valid_triplet = bool(head) and bool(tail) and bool(relation) and bool(node_i_candidates) and bool(node_j_candidates)
        if not valid_triplet:
            return 0.0

        # allow directional or reversed alignment between (node_i,node_j) and (head,tail)
        dir_match = self._entity_matches_node(head, node_i_candidates) and self._entity_matches_node(tail, node_j_candidates)
        rev_match = self._entity_matches_node(head, node_j_candidates) and self._entity_matches_node(tail, node_i_candidates)
        return 1.0 if (dir_match or rev_match) else 0.0

    def _edge_confidence_from_evidence(self, evidence_triplets: list, node_i_candidates: list, node_j_candidates: list) -> float:
        """
        Compute edge confidence via paper-aligned aggregation:
        omega_kg(e_ij) = (1/M) * sum_m [ sim_m * r_m * 1_match_m ].
        """
        if not evidence_triplets:
            return 0.0

        scores = []
        for entry in evidence_triplets:
            sim01 = self._normalize_similarity(entry.get("similarity", 0.0))
            reliability = self._source_reliability(entry)
            match = self._match_indicator(entry, node_i_candidates, node_j_candidates)
            scores.append(sim01 * reliability * match)

        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def build_confidence_graph(self, Q: torch.Tensor, retrieved_knowledge: list) -> torch.Tensor:
        """
        Build knowledge confidence matrix omega_kg for each edge.

        For node i, use its retrieved top-M triplets as shared evidence source for
        edges incident to i. We symmetrize by averaging node-i and node-j evidence.
        """
        batch_size, num_nodes, _ = Q.shape
        omega = torch.zeros(batch_size, num_nodes, num_nodes, device=Q.device)

        # Accept both:
        # 1) batched: [B][N][K]
        # 2) flattened legacy: [B*N][K]
        if len(retrieved_knowledge) == batch_size and isinstance(retrieved_knowledge[0], list):
            # likely batched form; validate node count
            for b in range(batch_size):
                if len(retrieved_knowledge[b]) != num_nodes:
                    raise ValueError(
                        f"Batch {b} node knowledge length ({len(retrieved_knowledge[b])}) != num_nodes ({num_nodes})."
                    )
            batched_knowledge = retrieved_knowledge
        else:
            expected = batch_size * num_nodes
            if len(retrieved_knowledge) != expected:
                raise ValueError(
                    f"Retrieved knowledge length ({len(retrieved_knowledge)}) does not match B*N ({expected})."
                )
            batched_knowledge = []
            for b in range(batch_size):
                start = b * num_nodes
                batched_knowledge.append(retrieved_knowledge[start:start + num_nodes])

        for b in range(batch_size):
            node_knowledge = batched_knowledge[b]
            node_entities = [self._build_node_entity_candidates(km) for km in node_knowledge]

            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i == j:
                        continue
                    # Evidence from both endpoint neighborhoods, aggregated over top-M triplets.
                    evid_ij = node_knowledge[i] + node_knowledge[j]
                    omega[b, i, j] = self._edge_confidence_from_evidence(
                        evid_ij,
                        node_entities[i],
                        node_entities[j],
                    )
        return omega.clamp_(0.0, 1.0)

    def refine(self, A_t: torch.Tensor, Q: torch.Tensor, retrieved_knowledge: list) -> torch.Tensor:
        """
        Refine adjacency by pruning unsupported raw edges.

        Args:
            A_t (torch.Tensor): Original adjacency matrix learned by STGNN (batch_size, num_nodes, num_nodes)
            Q (torch.Tensor): EEG semantic query embeddings (batch_size, num_nodes, proj_dim)
            retrieved_knowledge (list): List of retrieved triplets with similarity values.

        Returns:
            torch.Tensor: Refined adjacency matrix A_refined (batch_size, num_nodes, num_nodes)
        """
        omega_kg = self.build_confidence_graph(Q, retrieved_knowledge)

        # Paper-aligned pruning:
        # 1) never introduce new edges;
        # 2) only keep raw edges with sufficient knowledge confidence.
        raw_edge_mask = (A_t > 0).to(dtype=A_t.dtype)
        support_mask = (omega_kg >= self.threshold).to(dtype=A_t.dtype)
        keep_mask = raw_edge_mask * support_mask
        A_refined = A_t * keep_mask

        # Save triplet supports for interpretability
        self._save_triplet_supports(retrieved_knowledge)

        return A_refined

    def _save_triplet_supports(self, retrieved_knowledge: list):
        """
        Save retrieved knowledge triplets and their associated similarity scores for interpretability.

        Args:
            retrieved_knowledge (list): Retrieved top-k knowledge triplets per node.
        """
        # Append new retrievals to internal log
        self._support_records.append(retrieved_knowledge)

        with open(self.support_log_path, "w", encoding="utf-8") as f:
            json.dump(self._support_records, f, indent=2, ensure_ascii=False)

    def reset_support_log(self):
        """Clear accumulated triplet support records."""
        self._support_records = []
        if os.path.exists(self.support_log_path):
            os.remove(self.support_log_path)


if __name__ == "__main__":
    # Example test of GraphRefiner
    batch_size = 1
    num_nodes = 5
    proj_dim = 256

    A_t = torch.rand(batch_size, num_nodes, num_nodes)
    Q = torch.rand(batch_size, num_nodes, proj_dim)

    # Example retrieved knowledge (mock data)
    mock_retrieved = [
        [
            {"head": "EEG", "relation": "diagnoses", "tail": "epilepsy", "similarity": 0.88},
            {"head": "MRI", "relation": "supports", "tail": "temporal lobe", "similarity": 0.77}
        ],
        [
            {"head": "Valproate", "relation": "treats", "tail": "seizures", "similarity": 0.92}
        ],
        [
            {"head": "GABA", "relation": "modulates", "tail": "neuronal excitability", "similarity": 0.84}
        ],
        [
            {"head": "EEG", "relation": "records", "tail": "brain activity", "similarity": 0.73}
        ],
        [
            {"head": "Trauma", "relation": "causes", "tail": "epilepsy", "similarity": 0.80}
        ]
    ]

    refiner = GraphRefiner(threshold=0.5, alpha=0.7)
    A_refined = refiner.refine(A_t, Q, mock_retrieved)

    print(f"Original A_t shape: {A_t.shape}")
    print(f"Refined A_refined shape: {A_refined.shape}")
    print("Refinement complete. Saved support log at:", refiner.support_log_path)
