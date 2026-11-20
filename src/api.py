import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import RagPipeline

app = FastAPI(title="rag-claim-audit")
_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    # built on first request so importing this module (and /health) stays fast
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(lightweight=os.environ.get("RAG_LIGHTWEIGHT") == "1")
    return _pipeline


class RetrieveRequest(BaseModel):
    question: str
    top_k: int = 5


class AuditRequest(BaseModel):
    question: str
    draft_answer: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    start = time.perf_counter()
    out = get_pipeline().query(req.question, top_k=req.top_k)
    out["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return out


@app.post("/audit")
def audit(req: AuditRequest) -> dict:
    return get_pipeline().audit(req.question, draft_answer=req.draft_answer)
