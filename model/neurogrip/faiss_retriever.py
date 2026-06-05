import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - exercised only when faiss is installed
    faiss = None


def _safe_text(value) -> str:
    return "" if value is None else str(value).strip()


def _infer_source_type(source_name: str) -> str:
    source = _safe_text(source_name).lower()
    if any(k in source for k in ["guideline", "ilae", "aes", "nice", "sign"]):
        return "guideline"
    if "case" in source:
        return "case_report"
    if source:
        return "paper"
    return ""


def _source_reliability(source_type: str) -> float:
    if source_type == "guideline":
        return 0.95
    if source_type == "paper":
        return 0.85
    if source_type == "case_report":
        return 0.65
    return 0.80


class KnowledgeRetriever:
    """
    Retrieve KG triplets for SemAlignQuery vectors.

    Production mode uses a BioBERT+FAISS index over [head || tail] embeddings.
    For smoke tests and incomplete local setups, backend="auto" falls back to a
    deterministic NumPy text-hash index built from the triplet strings. The
    fallback is not meant to replace BioBERT retrieval in experiments; it keeps
    the NeuroGRIP code path executable and testable before the FAISS artifact is
    generated.
    """

    def __init__(
        self,
        index_path: Optional[str],
        kg_json_path: str,
        k: int = 5,
        normalize: bool = True,
        backend: str = "auto",
    ):
        if backend not in {"auto", "faiss", "numpy"}:
            raise ValueError("backend must be one of: auto, faiss, numpy")

        self.index_path = index_path
        self.kg_json_path = self._resolve_kg_path(kg_json_path)
        self.k = k
        self.normalize = normalize
        self.requested_backend = backend
        self.index = None
        self.backend = "numpy"
        self._numpy_index_cache: Dict[int, np.ndarray] = {}

        with open(self.kg_json_path, "r", encoding="utf-8") as f:
            raw_kg = json.load(f)
        self.kg = self._to_flat_triplets(raw_kg)

        can_use_faiss = (
            backend in {"auto", "faiss"}
            and faiss is not None
            and index_path is not None
            and os.path.exists(index_path)
        )

        if can_use_faiss:
            self.index = faiss.read_index(index_path)
            if len(self.kg) != self.index.ntotal:
                raise ValueError(
                    f"Mismatch: KG entries ({len(self.kg)}) != FAISS index vectors ({self.index.ntotal})"
                )
            self.backend = "faiss"
        elif backend == "faiss":
            if faiss is None:
                raise ImportError("faiss is required for backend='faiss' but is not installed.")
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

    @staticmethod
    def _resolve_kg_path(path: str) -> str:
        if os.path.exists(path):
            return path

        # README and args default to the flat artifact, but fresh clones may only
        # include the page-level KG. Fall back clearly and deterministically.
        dirname = os.path.dirname(path) or "."
        candidates = [
            os.path.join(dirname, "KG_triplets.json"),
            "model/neurogrip/KG_triplets.json",
            "KG_triplets.json",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"KG triplets file not found: {path}")

    @staticmethod
    def _normalize_triplet(head, relation, tail, extra: Optional[dict] = None) -> Dict:
        source = _safe_text((extra or {}).get("source", ""))
        source_type = _safe_text((extra or {}).get("source_type", "")) or _infer_source_type(source)
        reliability = (extra or {}).get("reliability", _source_reliability(source_type))
        try:
            reliability = float(reliability)
        except Exception:
            reliability = _source_reliability(source_type)

        triplet = {
            "head": _safe_text(head).lower(),
            "relation": _safe_text(relation).lower(),
            "tail": _safe_text(tail).lower(),
            "source": source,
            "source_type": source_type,
            "reliability": max(0.0, min(1.0, reliability)),
        }
        if extra:
            for key in ("source_page", "page_id", "page_no"):
                if key in extra:
                    triplet[key] = extra[key]
        return triplet

    def _to_flat_triplets(self, raw_kg) -> List[Dict]:
        """Convert flat, page-level, or list-triplet KG JSON into flat triplets."""
        if not isinstance(raw_kg, list):
            raise ValueError(f"Expected list in KG JSON, got {type(raw_kg)}")

        flat: List[Dict] = []
        for item in raw_kg:
            if isinstance(item, dict) and {"head", "relation", "tail"}.issubset(item.keys()):
                tri = self._normalize_triplet(
                    item.get("head"),
                    item.get("relation"),
                    item.get("tail"),
                    item,
                )
                if tri["head"] and tri["tail"]:
                    flat.append(tri)
                continue

            if isinstance(item, (list, tuple)) and len(item) >= 3:
                tri = self._normalize_triplet(item[0], item[1], item[2])
                if tri["head"] and tri["tail"]:
                    flat.append(tri)
                continue

            if isinstance(item, dict) and isinstance(item.get("triplets"), list):
                source = item.get("source") or item.get("source_doc") or ""
                source_type = item.get("source_type") or _infer_source_type(source)
                reliability = item.get("reliability", _source_reliability(source_type))
                page_extra = {
                    "source": source,
                    "source_type": source_type,
                    "reliability": reliability,
                    "source_page": item.get("page_id"),
                    "page_id": item.get("page_id"),
                    "page_no": item.get("page_no"),
                }
                for raw_tri in item["triplets"]:
                    if isinstance(raw_tri, dict) and {"head", "relation", "tail"}.issubset(raw_tri.keys()):
                        extra = dict(page_extra)
                        extra.update(raw_tri)
                        tri = self._normalize_triplet(
                            raw_tri.get("head"),
                            raw_tri.get("relation"),
                            raw_tri.get("tail"),
                            extra,
                        )
                    elif isinstance(raw_tri, (list, tuple)) and len(raw_tri) >= 3:
                        tri = self._normalize_triplet(raw_tri[0], raw_tri[1], raw_tri[2], page_extra)
                    else:
                        continue
                    if tri["head"] and tri["tail"]:
                        flat.append(tri)

        if not flat:
            raise ValueError("No valid triplets found after KG normalization.")
        return flat

    @staticmethod
    def _triplet_text(entry: Dict) -> str:
        return f"{entry.get('head', '')} {entry.get('relation', '')} {entry.get('tail', '')}"

    @staticmethod
    def _hash_text_embedding(text: str, dim: int) -> np.ndarray:
        vec = np.zeros(dim, dtype="float32")
        tokens = re.findall(r"[a-z0-9_/-]+", _safe_text(text).lower())
        if not tokens:
            tokens = [_safe_text(text).lower() or "empty"]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="little", signed=False)
            idx = value % dim
            sign = 1.0 if ((value >> 8) & 1) == 0 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _numpy_matrix(self, dim: int) -> np.ndarray:
        if dim not in self._numpy_index_cache:
            matrix = np.stack(
                [self._hash_text_embedding(self._triplet_text(entry), dim) for entry in self.kg],
                axis=0,
            ).astype("float32")
            self._numpy_index_cache[dim] = matrix
        return self._numpy_index_cache[dim]

    def _prepare_query(self, Q: torch.Tensor) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        original_shape = None
        if Q.dim() == 3:
            original_shape = (Q.size(0), Q.size(1))
            Q = Q.reshape(-1, Q.size(-1))
        elif Q.dim() != 2:
            raise ValueError(f"Expected Q to be 2D or 3D, got shape {tuple(Q.shape)}")

        Q_np = Q.detach().cpu().numpy().astype("float32")
        if self.normalize:
            norms = np.linalg.norm(Q_np, axis=1, keepdims=True)
            Q_np = Q_np / np.maximum(norms, 1e-12)
        return Q_np, original_shape

    def _search_faiss(self, Q_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("FAISS backend selected but index is not loaded.")
        if Q_np.shape[1] != self.index.d:
            raise ValueError(
                f"Query dim ({Q_np.shape[1]}) != FAISS index dim ({self.index.d}). "
                "Use semantic_proj_dim matching the BioBERT [head||tail] index dimension."
            )
        query = Q_np.copy()
        if self.normalize and faiss is not None:
            faiss.normalize_L2(query)
        return self.index.search(query, self.k)

    def _search_numpy(self, Q_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kg_matrix = self._numpy_matrix(Q_np.shape[1])
        sims = np.einsum("qd,kd->qk", Q_np, kg_matrix, optimize=True)
        sims = np.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
        k = min(self.k, kg_matrix.shape[0])
        top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        return top_scores.astype("float32"), top_idx.astype("int64")

    def retrieve(self, Q: torch.Tensor):
        """
        Args:
            Q: query tensor with shape [batch, nodes, dim] or [queries, dim].

        Returns:
            Batched input -> [batch][node][top-k triplet dict].
            2D input -> [query][top-k triplet dict].
        """
        Q_np, original_shape = self._prepare_query(Q)
        if self.backend == "faiss":
            scores, indices = self._search_faiss(Q_np)
        else:
            scores, indices = self._search_numpy(Q_np)

        retrieved_all = []
        for distances, idxs in zip(scores, indices):
            node_results = []
            for score, idx in zip(distances, idxs):
                if 0 <= int(idx) < len(self.kg):
                    entry = self.kg[int(idx)]
                    node_results.append({
                        "head": entry.get("head", ""),
                        "relation": entry.get("relation", ""),
                        "tail": entry.get("tail", ""),
                        "source": entry.get("source", ""),
                        "source_type": entry.get("source_type", ""),
                        "reliability": entry.get("reliability", None),
                        "source_page": entry.get("source_page", entry.get("page_id", None)),
                        "similarity": float(score),
                        "retrieval_backend": self.backend,
                    })
            retrieved_all.append(node_results)

        if original_shape is None:
            return retrieved_all

        batch_size, num_nodes = original_shape
        expected = batch_size * num_nodes
        if len(retrieved_all) != expected:
            raise ValueError(f"Retrieved entries ({len(retrieved_all)}) != B*N ({expected}).")
        return [
            retrieved_all[b * num_nodes:(b + 1) * num_nodes]
            for b in range(batch_size)
        ]
