# NeuroGRIP: Retrieval-Augmented Graph Refinement for Knowledge-Grounded EEG Seizure Diagnosis

> The official implementation of **NeuroGRIP**


## Overview

NeuroGRIP extends STGNN-style EEG graph learning with medical-knowledge-guided refinement.

Compared with plain data-driven graph learning, NeuroGRIP adds a retrieval pipeline that:
1. builds a domain knowledge graph from epilepsy guidelines/literature,
2. retrieves relevant triplets for each EEG subgraph query,
3. **prunes unsupported raw edges** in the STGNN graph.

---

## What Is Updated In This Version

This repository has been updated to better align with the paper workflow:

- `SemAlignQuery` now uses **k-hop subgraph aggregation** (not only per-node MLP projection).
- RAG input uses **STGNN node embeddings** (not raw `x` directly).
- Added BioBERT + FAISS index build script:  
  `model/neurogrip/build_biobert_faiss.py`
- KG schema handling improved:
  - supports page-level `KG_triplets.json`,
  - generates/uses flat triplets (`KG_triplets_flat.json`) for index alignment.
- `GraphRefiner` now uses paper-style edge confidence:
  - `sim * reliability * match`,
  - top-M aggregation,
  - **raw-edge pruning semantics** (no post-fusion edge creation).
- Retrieval now supports `--rag_backend auto`:
  - production uses BioBERT + FAISS when `faiss.index` is available,
  - lightweight NumPy fallback keeps debugging/tests executable before index generation.
- Batch handling for retrieval/refinement fixed.
- Training and evaluation now both re-run graph backbones with refined graphs, including DCRNN support conversion.

---

## Repository Structure (Key Parts)

```
NeuroGRIP/
├── args.py
├── main.py
├── requirements.txt
├── model/
│   ├── DCRNN.py
│   ├── EGCN.py
│   ├── EvoBrain.py
│   ├── BIOT.py
│   ├── lstm.py
│   ├── cnnlstm.py
│   └── neurogrip/
│       ├── __init__.py
│       ├── neurogrip.py
│       ├── semantic_query.py
│       ├── faiss_retriever.py
│       ├── graph_refiner.py
│       ├── knowledge_base.py
│       ├── build_biobert_faiss.py
│       ├── knowledge.json
│       └── KG_triplets.json
├── data/
│   ├── dataloader_detection.py
│   ├── dataloader_prediction.py
│   ├── dataloader_chb.py
│   └── file_markers_detection/
└── processed_data2/ (example preprocessed EEG .h5 files)
```

---

## Installation

### 1) Clone

```bash
git clone https://github.com/<your-org>/NeuroGRIP.git
cd NeuroGRIP
```

### 2) Environment

```bash
conda create -n neurogrip python=3.9 -y
conda activate neurogrip
pip install -r requirements.txt
```

`requirements.txt` includes:
- `openai`
- `pdfplumber`
- `faiss-cpu`
- `transformers`, `torch`, etc.

---

## Data Preparation

Place EEG `.h5` files under your configured input directory (default in `args.py` is `processed_data`).

For current local setup examples, many scripts use `processed_data2`.  
Set via CLI:

```bash
--input_dir processed_data2
```

Label marker files are expected under:

```text
data/file_markers_detection/
```

---

## Knowledge Pipeline

### Step 1) Build or flatten the knowledge graph

If you already have `KG_triplets.json`, generate the flat retrieval file:

```bash
python model/neurogrip/knowledge_base.py \
  --flatten_only \
  --kg_out model/neurogrip/KG_triplets.json \
  --flat_out model/neurogrip/KG_triplets_flat.json
```

To rebuild from guideline PDFs, set `OPENAI_API_KEY` and provide PDFs:

```bash
python model/neurogrip/knowledge_base.py \
  --input_pdfs path/to/ILAE.pdf path/to/NICE.pdf \
  --knowledge_out model/neurogrip/knowledge.json \
  --kg_out model/neurogrip/KG_triplets.json \
  --flat_out model/neurogrip/KG_triplets_flat.json \
  --model gpt-5.1
```

Outputs:
- `model/neurogrip/knowledge.json`
- `model/neurogrip/KG_triplets.json`
- `model/neurogrip/KG_triplets_flat.json` (flat triplets for retrieval alignment)

### Step 2) Build BioBERT embeddings + FAISS index

```bash
python model/neurogrip/build_biobert_faiss.py \
  --kg_json_path model/neurogrip/KG_triplets.json \
  --out_triplets_path model/neurogrip/KG_triplets_flat.json \
  --out_index_path model/neurogrip/faiss.index \
  --biobert_model_name dmis-lab/biobert-base-cased-v1.1
```

---

## How NeuroGRIP Works

1. **SemAlignQuery (`semantic_query.py`)**
   - STGNN node embeddings `c_t^i` are projected to KG space.
   - k-hop neighborhood aggregation builds subgraph query vectors.

2. **Knowledge Retrieval (`faiss_retriever.py`)**
   - query vectors retrieve top-K medical triplets via BioBERT+FAISS in production.
   - `--rag_backend auto` falls back to deterministic NumPy retrieval when no FAISS index is present.

3. **Graph Refinement (`graph_refiner.py`)**
   - edge confidence uses:
     - semantic similarity (`sim`),
     - source reliability (`reliability`),
     - entity alignment indicator (`1_match`).
   - top-M evidence aggregated per edge.
   - unsupported edges are pruned from raw graph:
     - keep edge only if confidence >= threshold.

---

## Training

Example (EvolveGCN + NeuroGRIP):

```bash
python main.py \
  --model_name evolvegcn \
  --task detection \
  --use_neurogrip \
  --input_dir processed_data2 \
  --kg_triplets_path model/neurogrip/KG_triplets_flat.json \
  --faiss_index_path model/neurogrip/faiss.index
```

> `model_name='neurogrip'` is not a standalone classifier.  
> Use a backbone model (`evobrain`, `evolvegcn`, `dcrnn`, etc.) with `--use_neurogrip`.

---

## Important Arguments

| Argument | Description | Default |
|---|---|---|
| `--use_neurogrip` | Enable NeuroGRIP graph refinement | `False` |
| `--kg_triplets_path` | Flat triplets aligned with FAISS ids | `model/neurogrip/KG_triplets_flat.json` |
| `--faiss_index_path` | FAISS index path | `model/neurogrip/faiss.index` |
| `--rag_backend` | `auto`, `faiss`, or `numpy` retrieval backend | `auto` |
| `--topk_retrieval` | Top-K triplets per node query | `5` |
| `--semantic_proj_dim` | Query dimension; BioBERT `[head||tail]` index uses 1536 | `1536` |
| `--semantic_k_hop` | k-hop neighborhood size for SemAlignQuery | `2` |
| `--refine_threshold` | Edge pruning threshold | `0.6` |
| `--match_mode` | Edge-triplet match policy: `evidence`, `soft`, or `strict` | `evidence` |
| `--refine_interval` | Apply refinement every N steps | `1` |
| `--num_epochs` | Training epochs | `100` |

See full options in `args.py`.

---

## Outputs

- Checkpoints and metrics: `results/...`
- RAG support logs: `results/graph_refinement/triplet_supports.json`
- Dev/test prediction artifacts (`*.npz`, ROC data, hidden features)

---

## Lightweight Validation

The core RAG modules can be tested without TUSZ/CHB-MIT, FAISS, BioBERT, or h5py:

```bash
python tests/test_neurogrip_core.py
```

This verifies SemAlignQuery shape, NumPy fallback retrieval, pruning semantics, and an end-to-end synthetic `NeuroGRIP.refine_graph` call.


---

## Citation

If you find this project useful, please considering cite our work:

```
@article{NeuroGRIP2026,
  title={NeuroGRIP: Retrieval-Augmented Graph Refinement for Knowledge-Grounded EEG Seizure Diagnosis},
  author={First Author, Second Author, Third Author, et al.},
  journal={Under Review},
  year={2026}
}
```
