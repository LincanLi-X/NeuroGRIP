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
  `model/eeg_ragnet/build_biobert_faiss.py`
- KG schema handling improved:
  - supports page-level `KG_triplets.json`,
  - generates/uses flat triplets (`KG_triplets_flat.json`) for index alignment.
- `GraphRefiner` now uses paper-style edge confidence:
  - `sim * reliability * match`,
  - top-M aggregation,
  - **raw-edge pruning semantics** (no post-fusion edge creation).
- Batch handling for retrieval/refinement fixed.
- Evaluation (`evaluate`) now also runs RAG refinement for unified train/inference behavior.

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
│   └── eeg_ragnet/
│       ├── __init__.py
│       ├── eeg_ragnet.py
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
conda create -n eeg_ragnet python=3.9 -y
conda activate eeg_ragnet
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

### Step 1) Build page-level knowledge + triplets

```bash
python model/eeg_ragnet/knowledge_base.py
```

Outputs:
- `model/eeg_ragnet/knowledge.json`
- `model/eeg_ragnet/KG_triplets.json`
- `model/eeg_ragnet/KG_triplets_flat.json` (flat triplets for retrieval alignment)

### Step 2) Build BioBERT embeddings + FAISS index

```bash
python model/eeg_ragnet/build_biobert_faiss.py \
  --kg_json_path model/eeg_ragnet/KG_triplets.json \
  --out_triplets_path model/eeg_ragnet/KG_triplets_flat.json \
  --out_index_path model/eeg_ragnet/faiss.index \
  --biobert_model_name dmis-lab/biobert-base-cased-v1.1
```

---

## How NeuroGRIP Works

1. **SemAlignQuery (`semantic_query.py`)**
   - STGNN node embeddings `c_t^i` are projected to KG space.
   - k-hop neighborhood aggregation builds subgraph query vectors.

2. **Knowledge Retrieval (`faiss_retriever.py`)**
   - query vectors retrieve top-K medical triplets via FAISS.

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
  --use_ragnet \
  --input_dir processed_data2 \
  --kg_triplets_path model/eeg_ragnet/KG_triplets_flat.json \
  --faiss_index_path model/eeg_ragnet/faiss.index
```

> `model_name='eeg_ragnet'` is not a standalone classifier.  
> Use a backbone model (`evobrain`, `evolvegcn`, `dcrnn`, etc.) with `--use_ragnet`.

---

## Important Arguments

| Argument | Description | Default |
|---|---|---|
| `--use_ragnet` | Enable RAG refinement | `False` |
| `--kg_triplets_path` | Flat triplets aligned with FAISS ids | `model/eeg_ragnet/KG_triplets_flat.json` |
| `--faiss_index_path` | FAISS index path | `model/eeg_ragnet/faiss.index` |
| `--topk_retrieval` | Top-K triplets per node query | `5` |
| `--semantic_k_hop` | k-hop neighborhood size for SemAlignQuery | `2` |
| `--refine_threshold` | Edge pruning threshold | `0.6` |
| `--refine_interval` | Apply refinement every N steps | `1` |
| `--num_epochs` | Training epochs | `100` |

See full options in `args.py`.

---

## Outputs

- Checkpoints and metrics: `results/...`
- RAG support logs: `results/graph_refinement/triplet_supports.json`
- Dev/test prediction artifacts (`*.npz`, ROC data, hidden features)


---

## Citation

If you find this project useful, please considering cite our work:

```
@article{EEGRAGNet2026,
  title={NeuroGRIP: Retrieval-Augmented Graph Refinement for Knowledge-Grounded EEG Seizure Diagnosis},
  author={First Author, Second Author, Third Author, et al.},
  journal={Under Review},
  year={2026}
}
```
