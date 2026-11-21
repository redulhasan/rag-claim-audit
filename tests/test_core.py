import os
from pathlib import Path

os.environ.setdefault("RAG_LIGHTWEIGHT", "1")

import pytest

from src.chunking import Chunk, fixed_chunks, header_aware_chunks, load_corpus
from src.grounding import ClaimAuditor, NLIVerifier, extract_numbers, label_index_map, should_refuse, softmax, split_claims
from src.retrievers import rrf_fuse

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def docs():
    return load_corpus(ROOT / "corpus")


def test_corpus_has_32_notes(docs):
    assert len(docs) == 32
    assert all(name.endswith(".md") for name, _ in docs)


def test_header_aware_chunking_keeps_metadata(docs):
    chunks = header_aware_chunks(docs)
    assert len(chunks) > 10
    for chunk in chunks:
        assert chunk.source_file.endswith(".md")
        assert chunk.section not in ("", "fixed")
        assert chunk.chunk_id


def test_fixed_chunking_runs_and_keeps_source(docs):
    chunks = fixed_chunks(docs)
    assert chunks
    assert all(c.source_file.endswith(".md") for c in chunks)


def test_rrf_rewards_documents_ranked_by_both():
    a = {"chunk_id": "a", "text": "", "source_file": "x.md", "section": "s", "score": 9.0}
    b = {"chunk_id": "b", "text": "", "source_file": "x.md", "section": "s", "score": 1.0}
    fused = rrf_fuse([[a, b], [b, a]], top_k=2)
    # b is rank 2 in both lists: 2/(60+2) beats a rank-1 plus rank-2 split... both
    # appear in both lists here, so check the fused scores are ordered and bounded
    assert fused[0]["score"] >= fused[1]["score"]
    solo = {"chunk_id": "solo", "text": "", "source_file": "x.md", "section": "s", "score": 5.0}
    fused2 = rrf_fuse([[solo], [b, solo]], top_k=2)
    assert fused2[0]["chunk_id"] == "solo"


def test_extract_numbers_normalizes():
    assert extract_numbers("about 0.5 MB and 32 layers, 100% sure") == {"0.5", "32", "100"}


def test_numeric_drift_is_flagged():
    chunks = [
        {
            "chunk_id": "c1",
            "text": "The KV cache holds 0.5 MB per token for a 32-layer model with d_model 4096 in fp16.",
            "source_file": "transformers_attention.md",
            "section": "KV cache",
            "score": 1.0,
        }
    ]
    auditor = ClaimAuditor()
    good = auditor.audit("The KV cache holds 0.5 MB per token for a 32-layer model.", chunks)
    assert good["claims"][0]["label"] == "supported"
    drifted = auditor.audit("The KV cache holds 5 MB per token for a 32-layer model.", chunks)
    assert drifted["claims"][0]["label"] == "unsupported"
    assert drifted["claims"][0]["numeric_mismatch"] is True


def test_unsupported_claim_is_flagged():
    chunks = [
        {
            "chunk_id": "c1",
            "text": "BM25 uses term frequency saturation with k1 = 1.5 and b = 0.75 by default.",
            "source_file": "rag_retrieval_notes.md",
            "section": "BM25",
            "score": 1.0,
        }
    ]
    report = ClaimAuditor().audit("The moon is made of green cheese and orbits every 9 days.", chunks)
    assert report["claims"][0]["label"] == "unsupported"
    assert report["grounding_ratio"] == 0.0


def test_refusal_gate():
    assert should_refuse(0.05, 0.30) is True
    assert should_refuse(0.80, 0.30) is False


def test_softmax_produces_valid_probabilities():
    import numpy as np

    probs = softmax([2.0, 1.0, 0.1])
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert abs(float(probs.sum()) - 1.0) < 1e-9
    # large negative logits must not underflow the whole vector to zero
    shifted = softmax([-1000.0, -1001.0, -999.0])
    assert abs(float(shifted.sum()) - 1.0) < 1e-9
    assert float(shifted.max()) > 0.6
    rows = softmax([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
    assert np.allclose(rows.sum(axis=-1), 1.0)


def test_label_index_map_uses_config_when_present():
    assert label_index_map(None) == {"contradiction": 0, "entailment": 1, "neutral": 2}
    custom = label_index_map({0: "ENTAILMENT", 1: "Neutral", 2: "contradiction"})
    assert custom == {"contradiction": 2, "entailment": 0, "neutral": 1}


def test_lightweight_nli_probs_are_valid():
    verifier = NLIVerifier(lightweight=True)
    probs = verifier.probs(
        "LoRA freezes the pretrained weight W and learns a low-rank update instead.",
        "LoRA freezes the pretrained weight W.",
    )
    assert set(probs) == {"contradiction", "entailment", "neutral"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["entailment"] >= 0.60


def test_negation_contradiction_is_flagged():
    chunks = [
        {
            "chunk_id": "c1",
            "text": "The reranker improves ranking quality on this corpus by reordering the candidates with a cross-encoder.",
            "source_file": "rag_retrieval_notes.md",
            "section": "Cross-encoder reranking",
            "score": 1.0,
        }
    ]
    report = ClaimAuditor().audit("The reranker does not improve ranking quality on this corpus.", chunks)
    claim = report["claims"][0]
    assert claim["label"] == "unsupported"
    assert claim["contradiction"] is True
    assert claim["reason"] == "contradiction"


def test_nli_rules_applied_in_order():
    chunks = [
        {
            "chunk_id": "c1",
            "text": "The KV cache holds 0.5 MB per token for a 32-layer model.",
            "source_file": "transformers_attention.md",
            "section": "KV cache",
            "score": 1.0,
        }
    ]

    def fake_nli(premise, hypothesis):
        return {"contradiction": 0.8, "entailment": 0.1, "neutral": 0.1}

    auditor = ClaimAuditor(nli=fake_nli)
    contradicted = auditor.audit("The KV cache holds 0.5 MB per token for a 32-layer model.", chunks)
    assert contradicted["claims"][0]["label"] == "unsupported"
    assert contradicted["claims"][0]["reason"] == "contradiction"
    # numeric check runs before the NLI verdict
    drifted = auditor.audit("The KV cache holds 7 MB per token for a 32-layer model.", chunks)
    assert drifted["claims"][0]["reason"] == "numeric_mismatch"


@pytest.fixture(scope="module")
def pipe():
    from src.pipeline import RagPipeline

    return RagPipeline(lightweight=True)


def test_false_premise_query_is_refused(pipe):
    out = pipe.query("According to the notes, does LoRA increase the rank of the pretrained weight matrix W itself during training?")
    assert out["refused"] is True
    assert out["refusal_reason"] == "false_premise"


def test_answerable_query_is_not_refused(pipe):
    out = pipe.query("How much memory does the KV cache use per token for a 32-layer model with d_model 4096 in fp16?")
    assert out["refused"] is False
    assert out["refusal_reason"] is None


def test_reranker_candidate_pool_is_capped(pipe):
    seen = {}
    original = pipe.reranker.score_pairs

    def spy(query, texts):
        seen["n"] = len(texts)
        return original(query, texts)

    pipe.reranker.score_pairs = spy
    try:
        pipe.retrieve("What is LoRA and how does it work?")
    finally:
        pipe.reranker.score_pairs = original
    assert seen["n"] == 8  # top_k_candidates default, not the full pool of 20


def test_split_claims():
    claims = split_claims("First claim is here. Second claim follows it! Is this a third one?")
    assert len(claims) == 3


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api import app

    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_retrieve_refuses_gibberish(client):
    resp = client.post("/retrieve", json={"question": "zxqw blorp fnord quuxly"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "out_of_scope"


def test_audit_flags_ungrounded_draft(client):
    resp = client.post(
        "/audit",
        json={
            "question": "How much memory does the KV cache use per token for a 32-layer model with d_model 4096 in fp16?",
            "draft_answer": "The KV cache holds 9 MB per token for a 32-layer model.",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert body["ungrounded"] is True
    assert body["claims"][0]["reason"] == "numeric_mismatch"


def test_audit_grounded_draft_is_not_flagged(client):
    resp = client.post(
        "/audit",
        json={
            "question": "How much memory does the KV cache use per token for a 32-layer model with d_model 4096 in fp16?",
            "draft_answer": "The KV cache holds 0.5 MB per token for a 32-layer model with d_model 4096 in fp16.",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert body["ungrounded"] is False
    assert body["claims"][0]["label"] == "supported"


def test_audit_endpoint_catches_drift(client):
    resp = client.post(
        "/audit",
        json={
            "question": "How much memory does the KV cache use per token for a 32-layer model with d_model 4096 in fp16?",
            "draft_answer": "The KV cache holds 0.5 MB per token for a 32-layer model with d_model 4096 in fp16. The KV cache holds 9 MB per token for a 32-layer model.",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    labels = [c["label"] for c in body["claims"]]
    assert "supported" in labels
    assert "unsupported" in labels
