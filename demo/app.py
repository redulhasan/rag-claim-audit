"""Standalone demo for the rag-claim-audit pipeline.

Run from the repository root:
    pip install -r demo/requirements.txt
    python demo/app.py

Set RAG_LIGHTWEIGHT=1 to run without downloading model weights (TF-IDF
retrieval and heuristic verification instead of the neural models).
"""

import html
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from src.pipeline import RagPipeline

_pipeline: RagPipeline | None = None

_BADGE_COLORS = {
    "supported": "#1a7f37",
    "weakly_supported": "#9a6700",
    "unsupported": "#cf222e",
}


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(lightweight=os.environ.get("RAG_LIGHTWEIGHT") == "1")
    return _pipeline


def badge(claim: dict) -> str:
    color = _BADGE_COLORS.get(claim["label"], "#57606a")
    citation = claim.get("citation") or {}
    source = f"{citation.get('source_file', '?')} / {citation.get('section', '?')}"
    return (
        f'<div style="margin:6px 0;padding:8px;border:1px solid {color};border-radius:6px">'
        f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">'
        f"{html.escape(claim['label'])}</span> "
        f"<b>{html.escape(claim.get('reason', ''))}</b> "
        f"<span style='color:#57606a'>score {claim.get('score', 0)} - {html.escape(source)}</span><br>"
        f"{html.escape(claim['claim'])}</div>"
    )


def run(question: str, draft_answer: str):
    if not question.strip():
        return "Enter a question first.", "", ""
    pipe = get_pipeline()
    out = pipe.audit(question, draft_answer=draft_answer.strip() or None)
    latency = f"Latency: {out.get('latency_ms', 0)} ms"
    if out["refused"]:
        reason = out.get("refusal_reason") or "out_of_scope"
        return f"Refused ({reason}): {out.get('message', '')}", "", latency
    status = "Answer returned. Ungrounded draft flagged." if out.get("ungrounded") else "Answer returned."
    badges = "".join(badge(c) for c in out["claims"]) or "No claims to audit."
    header = f"<p><b>Answer:</b> {html.escape(out.get('answer', ''))}</p>"
    header += f"<p>Grounding ratio: {out.get('grounding_ratio', 0)} ({out.get('n_supported', 0)}/{out.get('n_claims', 0)} claims supported)</p>"
    return status, header + badges, latency


with gr.Blocks(title="rag-claim-audit demo") as demo:
    gr.Markdown("# rag-claim-audit demo")
    gr.Markdown("Ask a technical question against the indexed notes, or paste a draft answer to audit it claim by claim.")
    question = gr.Textbox(label="Technical query", placeholder="How much memory does the KV cache use per token?")
    draft = gr.Textbox(label="Draft answer to audit (optional)", lines=4)
    button = gr.Button("Run")
    status = gr.Textbox(label="Status", interactive=False)
    results = gr.HTML(label="Claim audit")
    latency = gr.Textbox(label="Latency", interactive=False)
    button.click(run, inputs=[question, draft], outputs=[status, results, latency])

if __name__ == "__main__":
    demo.launch()
