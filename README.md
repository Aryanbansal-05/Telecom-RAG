# Telecom RAG: Future-Ready Telecom RAN Assistant

A Retrieval-Augmented Generation (RAG) application built for telecom Radio Access Network (RAN) tasks — 3GPP spec Q&A, root cause analysis, and anomaly detection — combining domain-specific knowledge retrieval with LLM-based reasoning.

## Problem Statement

Telecom RANs are growing in complexity, making root cause analysis, anomaly detection, and optimization increasingly hard to manage manually. This project automates and accelerates those decisions using a RAG pipeline grounded in real telecom data — 3GPP standards (via TeleQnA), O-RAN network telemetry, and 5G core network fault data — while staying explainable, efficient, and faithful to its sources.

## Use Cases

- **3GPP Spec Q&A** — ask a telecom standards question, get an answer grounded in TeleQnA's question/explanation pairs, with sources cited.
- **Root Cause Analysis** — given an alarm or fault description, retrieve relevant context and explain likely causes.
- **Anomaly Detection** — flag abnormal network behavior using O-RAN performance telemetry.

## Architecture

```
TeleQnA.txt ──▶ ingest.py ──▶ chunker.py ──▶ embeddings.py ──▶ FAISS index
                                                                     │
                                                                     ▼
                          user question ──▶ retriever.py ──▶ generator.py ──▶ answer + sources
                                                                     ▲
                                                              llm_setup.py
```

O-RAN telemetry (`slice_mixed`, `slice_traffic`) and the Zenodo fault/network-element data are loaded via `ingest.py` for direct pandas-based analysis (the anomaly-detection / RCA modules) rather than embedded into the vector store — that data is structured telemetry, not retrievable text.

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python |
| Embedding model | `BAAI/bge-large-en` (sentence-transformers) |
| Vector store | FAISS |
| LLM | Mistral-7B-Instruct |
| Backend (planned) | FastAPI |
| Demo UI (planned) | Streamlit |
| Evaluation (planned) | RAGAS |

## Datasets

| Dataset | Purpose | Source |
|---|---|---|
| TeleQnA | 3GPP Q&A knowledge base | [github.com/netop-team/TeleQnA](https://github.com/netop-team/TeleQnA) (zip password: `teleqnadataset`) |
| O-RAN COMMAG | Network performance telemetry for anomaly detection | [github.com/wineslab/colosseum-oran-commag-dataset](https://github.com/wineslab/colosseum-oran-commag-dataset) |
| Zenodo fault data | Network element + performance metrics for RCA | [zenodo.org/records/7003755](https://zenodo.org/records/7003755) |

> Note: TeleQnA's full dataset has 10,000 questions across 5 categories. A 1,827-question subset (Standards overview + Standards specifications only) matches a narrower "3GPP standards" scope if you want to filter to just that.

## Project Structure

```
Telecom RAG/
├── data/
│   ├── raw/            # downloaded datasets -- NOT tracked in git, see setup below
│   └── processed/      # FAISS index + metadata -- tracked, built by embeddings.py
├── RAG-pipeline/
│   ├── ingest.py        # loads TeleQnA, O-RAN CSVs, Zenodo Excel files
│   ├── chunker.py       # splits oversized text (no-op for TeleQnA's short entries)
│   ├── embeddings.py    # builds the FAISS index, checkpointed/resumable
│   ├── retriever.py     # searches the FAISS index for a given query
│   ├── llm_setup.py     # loads the LLM (in progress)
│   └── generator.py     # combines retrieval + LLM into a full answer (in progress)
├── venv/                # NOT tracked in git
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd "Telecom RAG"
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then download the datasets above into `data/raw/`:
- `data/raw/TeleQnA.txt`
- `data/raw/slice_mixed/`, `data/raw/slice_traffic/`
- `data/raw/7003755/` (the two Zenodo Excel files)

## Running the Pipeline

```bash
python RAG-pipeline/ingest.py        # confirms all three data sources load correctly
python RAG-pipeline/chunker.py       # verifies chunk sizes (mostly a no-op here)
python RAG-pipeline/embeddings.py    # builds the FAISS index (resumable if interrupted)
python RAG-pipeline/retriever.py     # test retrieval against the index
```

## Target KPIs

| Metric | Target |
|---|---|
| Mean Reciprocal Rank (MRR) | Above 75% |
| Top-k Accuracy | Above 85% |
| Accuracy | Above 80% |
| Recall | Above 85% |
| Faithfulness | Above 90% |

## Status

- [x] Data ingestion (TeleQnA, O-RAN, Zenodo)
- [x] Chunking
- [x] Embeddings + FAISS index
- [x] Retriever
- [ ] LLM setup
- [ ] Full generation pipeline (retrieval + LLM)
- [ ] FastAPI backend
- [ ] Streamlit demo UI
- [ ] RAGAS evaluation against KPI targets

## Contributors

- Nancy
- Aryan Bansal

