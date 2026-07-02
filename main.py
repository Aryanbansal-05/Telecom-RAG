import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# RAG-pipeline/ holds all the actual logic -- add it to the path so its
# modules import as plain top-level names regardless of the hyphen in the
# folder name (we're not importing the folder itself, just its .py files).
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAG-pipeline"))

from generator import answer_question          # noqa: E402
from rca import explain_anomaly                 # noqa: E402
from llm_setup import load_llm, get_llm_info   # noqa: E402
from retriever import retrieve_chunks, get_index_stats  # noqa: E402
from anomaly_detector import (                  # noqa: E402
    load_sample_csvs,
    detect_anomalies,
    describe_anomaly,
    anomaly_to_query,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load both models once at startup, so the first real request from
    # a user/demo isn't the one that eats the load time.
    global _start_time
    _start_time = time.time()
    print("Pre-loading models at startup...")
    retrieve_chunks("warmup", k=1)  # triggers embedding model + FAISS index load
    load_llm()
    print("Models loaded. Server ready.\n")
    yield


app = FastAPI(title="Telecom RAN RAG Assistant", version="1.0.0", lifespan=lifespan)

# Track startup time for uptime calculation
_start_time: float = time.time()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request/response models ----------

class QuestionRequest(BaseModel):
    question: str
    k: int = 5


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]


class RootCauseRequest(BaseModel):
    description: str
    k: int = 5


class RootCauseResponse(BaseModel):
    explanation: str
    sources: List[str]


class AnomalyItem(BaseModel):
    metric: str
    value: float
    z_score: float
    source_file: str
    time: Optional[float] = None
    description: str


# ---------- Endpoints ----------

# ── Data directory (for folder listing) ──────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

APP_VERSION = "1.0.0"


@app.get("/")
def root():
    llm  = get_llm_info()
    idx  = get_index_stats()
    models_loaded = llm["llm_loaded"] and idx["embedding_loaded"] and idx["index_loaded"]
    return {
        "status": "ok",
        "version": APP_VERSION,
        "models_loaded": models_loaded,
        "message": "Telecom RAN RAG Assistant API -- see /docs",
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """3GPP spec Q&A use case."""
    try:
        result = answer_question(request.question, k=request.k)
        return AnswerResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Live system statistics: models, knowledge base, uptime."""
    llm = get_llm_info()
    idx = get_index_stats()
    uptime_s = int(time.time() - _start_time)
    return {
        "version": APP_VERSION,
        "uptime_seconds": uptime_s,
        "uptime_human": _fmt_uptime(uptime_s),
        "models": {
            "llm_name":        llm["model_name"],
            "llm_file":        llm["model_file"],
            "llm_quantization":llm["quantization"],
            "llm_n_ctx":       llm["n_ctx"],
            "llm_loaded":      llm["llm_loaded"],
            "embedding_model": idx["embedding_model"],
            "embedding_loaded":idx["embedding_loaded"],
            "index_loaded":    idx["index_loaded"],
        },
        "knowledge_base": {
            "name":          "TeleQnA",
            "faiss_vectors": idx["vectors"],
            "documents":     idx["documents"],
            "index_type":    "FAISS / L2",
        },
    }


def _fmt_uptime(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


@app.get("/folders")
def list_folders():
    """List available telemetry data folders for anomaly scanning."""
    try:
        entries = [
            d for d in os.listdir(_DATA_DIR)
            if os.path.isdir(os.path.join(_DATA_DIR, d)) and not d.startswith(".")
        ]
        return {"folders": sorted(entries)}
    except Exception:
        return {"folders": ["slice_mixed", "slice_traffic"]}  # safe fallback


@app.get("/anomalies", response_model=List[AnomalyItem])
def get_anomalies(
    folder: str = "slice_mixed",
    max_files: int = 30,
    z_threshold: float = 3.0,
    limit: int = 10,
):
    """Anomaly detection use case -- scans a sample of O-RAN telemetry CSVs."""
    try:
        df = load_sample_csvs(folder, max_files=max_files)
        anomalies_df = detect_anomalies(df, z_threshold=z_threshold)

        items = []
        for _, row in anomalies_df.head(limit).iterrows():
            metric = row["anomaly_metric"]
            time_val = row.get("time")
            if pd.isna(time_val):
                time_val = None

            items.append(AnomalyItem(
                metric=metric,
                value=float(row[metric]),
                z_score=float(row["z_score"]),
                source_file=row["source_file"],
                time=time_val,
                description=describe_anomaly(row),
            ))
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rootcause", response_model=RootCauseResponse)
def rootcause(request: RootCauseRequest):
    """Root cause analysis use case -- explains a given anomaly/alarm description."""
    try:
        result = explain_anomaly(request.description, k=request.k)
        return RootCauseResponse(explanation=result["explanation"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rootcause/auto", response_model=RootCauseResponse)
def rootcause_auto(
    folder: str = "slice_mixed",
    max_files: int = 30,
    z_threshold: float = 3.0,
):
    """
    Convenience endpoint that chains the other two use cases together:
    detects a real anomaly from telemetry, then explains it -- exactly
    what rca.py's test script does, exposed as one API call.
    """
    try:
        df = load_sample_csvs(folder, max_files=max_files)
        anomalies_df = detect_anomalies(df, z_threshold=z_threshold)

        if len(anomalies_df) == 0:
            return RootCauseResponse(explanation="No anomalies detected in this sample.", sources=[])

        row = anomalies_df.iloc[0]
        description = describe_anomaly(row)
        query = anomaly_to_query(row)

        result = explain_anomaly(description, retrieval_query=query)
        return RootCauseResponse(explanation=result["explanation"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))