import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - depends on optional local install
    faiss = None


def _safe_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _normalize_triplet(head: str, relation: str, tail: str) -> Dict:
    return {
        "head": _safe_text(head).lower(),
        "relation": _safe_text(relation).lower(),
        "tail": _safe_text(tail).lower(),
    }


def infer_source_type(source_name: str) -> str:
    s = _safe_text(source_name).lower()
    if any(k in s for k in ["guideline", "nice", "sign", "ilae", "aes"]):
        return "guideline"
    if "case" in s:
        return "case_report"
    return "paper"


def source_reliability(source_type: str) -> float:
    if source_type == "guideline":
        return 0.95
    if source_type == "paper":
        return 0.85
    if source_type == "case_report":
        return 0.65
    return 0.80


def flatten_triplets(kg_json_path: str) -> List[Dict]:
    """
    Supports two input schemas:
    1) Flat triplet list: [{"head": "...", "relation": "...", "tail": "..."}, ...]
    2) Page-level list: [{"page_id": 0, "triplets": [...]}, ...]
    """
    with open(kg_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat: List[Dict] = []
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {kg_json_path}, got {type(data)}")

    for item in data:
        if isinstance(item, dict) and {"head", "relation", "tail"}.issubset(item.keys()):
            tri = _normalize_triplet(item["head"], item["relation"], item["tail"])
            if tri["head"] and tri["tail"]:
                src = item.get("source", "")
                src_type = item.get("source_type", infer_source_type(src))
                flat.append(tri)
                flat[-1]["source"] = src
                flat[-1]["source_type"] = src_type
                flat[-1]["reliability"] = float(item.get("reliability", source_reliability(src_type)))
            continue

        if isinstance(item, dict) and isinstance(item.get("triplets"), list):
            source_page = item.get("page_id")
            src = item.get("source", "")
            src_type = item.get("source_type", infer_source_type(src))
            src_rel = float(item.get("reliability", source_reliability(src_type)))
            for tri_raw in item["triplets"]:
                if not isinstance(tri_raw, dict):
                    continue
                if not {"head", "relation", "tail"}.issubset(tri_raw.keys()):
                    continue
                tri = _normalize_triplet(tri_raw["head"], tri_raw["relation"], tri_raw["tail"])
                if tri["head"] and tri["tail"]:
                    tri["source_page"] = source_page
                    tri["source"] = tri_raw.get("source", src)
                    tri["source_type"] = tri_raw.get("source_type", src_type)
                    tri["reliability"] = float(tri_raw.get("reliability", src_rel))
                    flat.append(tri)

    if not flat:
        raise ValueError(f"No valid triplets found in {kg_json_path}")

    return flat


@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], device: str, batch_size: int) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        out = model(**tok).last_hidden_state  # [B, L, H]
        mask = tok["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)  # mean pooling
        vecs.append(pooled.detach().cpu().numpy().astype("float32"))
    return np.concatenate(vecs, axis=0)


def build_biobert_faiss_index(
    kg_json_path: str,
    out_triplets_path: str,
    out_index_path: str,
    biobert_model_name: str,
    batch_size: int,
    device: str,
) -> None:
    if faiss is None:
        raise ImportError("faiss-cpu is required to build the production retrieval index.")

    triplets = flatten_triplets(kg_json_path)

    tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
    model = AutoModel.from_pretrained(biobert_model_name).to(device)
    model.eval()

    heads = [t["head"] for t in triplets]
    tails = [t["tail"] for t in triplets]

    head_vec = encode_texts(model, tokenizer, heads, device=device, batch_size=batch_size)
    tail_vec = encode_texts(model, tokenizer, tails, device=device, batch_size=batch_size)

    # Paper-aligned retrieval target: [e_h || e_t]
    triplet_vec = np.concatenate([head_vec, tail_vec], axis=1).astype("float32")
    faiss.normalize_L2(triplet_vec)

    dim = triplet_vec.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(triplet_vec)

    os.makedirs(os.path.dirname(out_index_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_triplets_path) or ".", exist_ok=True)

    faiss.write_index(index, out_index_path)

    # Persist flattened, index-aligned triplets for retriever lookup.
    with open(out_triplets_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    print(
        f"Built FAISS index with {index.ntotal} vectors, dim={dim}. "
        f"Saved index to {out_index_path} and triplets to {out_triplets_path}."
    )


def parse_args():
    parser = argparse.ArgumentParser("Build BioBERT triplet embeddings and FAISS index for NeuroGRIP.")
    parser.add_argument(
        "--kg_json_path",
        type=str,
        default="model/neurogrip/KG_triplets.json",
        help="Input KG json (flat triplets or page-level with triplets field).",
    )
    parser.add_argument(
        "--out_triplets_path",
        type=str,
        default="model/neurogrip/KG_triplets_flat.json",
        help="Output flattened triplets aligned with FAISS ids.",
    )
    parser.add_argument(
        "--out_index_path",
        type=str,
        default="model/neurogrip/faiss.index",
        help="Output FAISS index path.",
    )
    parser.add_argument(
        "--biobert_model_name",
        type=str,
        default="dmis-lab/biobert-base-cased-v1.1",
        help="Hugging Face model id or local path of BioBERT encoder.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_biobert_faiss_index(
        kg_json_path=args.kg_json_path,
        out_triplets_path=args.out_triplets_path,
        out_index_path=args.out_index_path,
        biobert_model_name=args.biobert_model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
