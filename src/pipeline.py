import argparse
import os
import re
import time
from pathlib import Path

from src.chunking import fixed_chunks, header_aware_chunks, load_corpus
from src.grounding import PREMISE_REFUSAL_TEXT, REFUSAL_TEXT, ClaimAuditor, NLIVerifier, should_refuse
from src.retrievers import BM25Retriever, CrossEncoderReranker, DenseFAISSRetriever, HybridRRFRetriever

ROOT = Path(__file__).resolve().parents[1]


class RagPipeline:
    def __init__(
        self,
        corpus_dir: str | Path = ROOT / "corpus",
        lightweight: bool = False,
        chunking: str = "header",
        refusal_threshold: float = 0.30,
        top_k_candidates: int = 8,
        premise_threshold: float = 0.55,
    ):
        docs = load_corpus(corpus_dir)
        self.chunks = header_aware_chunks(docs) if chunking == "header" else fixed_chunks(docs)
        self.bm25 = BM25Retriever(self.chunks)
        self.dense = DenseFAISSRetriever(self.chunks, lightweight=lightweight)
        self.hybrid = HybridRRFRetriever(self.bm25, self.dense)
        self.reranker = CrossEncoderReranker(lightweight=lightweight)
        self.refusal_threshold = refusal_threshold
        self.top_k_candidates = top_k_candidates
        self.premise_threshold = premise_threshold
        self.lightweight = lightweight
        self.nli = NLIVerifier(lightweight=lightweight)

        if lightweight:
            self.auditor = ClaimAuditor(nli=self.nli.probs)
        else:
            # relevance scorer picks the best chunk per claim; the NLI verifier
            # then decides supported / weakly_supported / unsupported on it
            def ce_scorer(claim: str, chunk_text: str) -> float:
                return self.reranker.score_pairs(claim, [chunk_text])[0]

            self.auditor = ClaimAuditor(scorer=ce_scorer, nli=self.nli.probs)

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        # only the top fused candidates go to the cross-encoder; reranking the
        # whole pool was the main CPU latency cost in the v1 ablation
        candidates = self.hybrid.search(question, top_k=20, pool=20)
        return self.reranker.rerank(question, candidates[: self.top_k_candidates], top_k=top_k)

    def query(self, question: str, top_k: int = 5) -> dict:
        start = time.perf_counter()
        results = self.retrieve(question, top_k=top_k)
        top_score = results[0]["score"] if results else 0.0
        # check 1 (out of scope): the reranker's top score is a sigmoid-squashed
        # logit in full mode; the lightweight coverage score plays the same role
        if should_refuse(top_score, self.refusal_threshold):
            return {
                "question": question,
                "refused": True,
                "refusal_reason": "out_of_scope",
                "message": REFUSAL_TEXT,
                "results": results,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        # check 2 (false premise): run NLI with the top chunk as premise and the
        # question itself as hypothesis. checking the extracted answer sentences
        # instead would be circular - they come from that same chunk.
        if results:
            probs = self.nli.probs(results[0]["text"], question)
            if probs.get("contradiction", 0.0) >= self.premise_threshold:
                return {
                    "question": question,
                    "refused": True,
                    "refusal_reason": "false_premise",
                    "message": PREMISE_REFUSAL_TEXT,
                    "results": results,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                }
        return {
            "question": question,
            "refused": False,
            "refusal_reason": None,
            "answer": extractive_answer(results),
            "results": results,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    def audit(self, question: str, draft_answer: str | None = None) -> dict:
        start = time.perf_counter()
        found = self.query(question)
        if found["refused"]:
            return {
                "question": question,
                "refused": True,
                "refusal_reason": found.get("refusal_reason"),
                "message": found.get("message", REFUSAL_TEXT),
                "latency_ms": found["latency_ms"],
            }
        answer = draft_answer or found["answer"]
        report = self.auditor.audit(answer, found["results"])
        # check 3 (audit mode): the draft as a whole is ungrounded if most of it
        # is unsupported or any single claim flatly contradicts its source
        ungrounded = report["grounding_ratio"] < 0.50 or any(c["contradiction"] for c in report["claims"])
        return {
            "question": question,
            "refused": False,
            "refusal_reason": None,
            "answer": answer,
            **report,
            "ungrounded": ungrounded,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }


def extractive_answer(results: list[dict]) -> str:
    # no generation model involved: first sentence of the top two chunks
    sentences = []
    for item in results[:2]:
        parts = re.split(r"(?<=[.!?])\s+", item["text"].replace("\n", " ").strip())
        if parts:
            sentences.append(parts[0])
    return " ".join(sentences)


def synthetic_drift(answer: str) -> str:
    # double the first number in the answer; a working auditor should flag it
    def _double(match: re.Match) -> str:
        value = float(match.group(0)) * 2
        return str(int(value)) if value == int(value) else str(value)

    return re.sub(r"\d+(?:\.\d+)?", _double, answer, count=1)


def generate_live(question: str, context: str) -> str:
    # optional path, only used when an api key is configured in the environment
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    resp = client.chat.completions.create(
        model=os.environ.get("GEN_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Answer only from the provided notes. Say you do not know if the notes do not cover it."},
            {"role": "user", "content": f"Notes:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--audit", action="store_true", help="audit the extractive answer claim by claim")
    parser.add_argument("--drift", action="store_true", help="audit a number-drifted answer instead")
    parser.add_argument("--lightweight", action="store_true", help="TF-IDF dense mode, no model downloads")
    args = parser.parse_args()

    pipe = RagPipeline(lightweight=args.lightweight)
    if args.audit or args.drift:
        found = pipe.query(args.question)
        answer = found.get("answer", "")
        if args.drift and answer:
            answer = synthetic_drift(answer)
        print(pipe.audit(args.question, draft_answer=answer or None))
    else:
        print(pipe.query(args.question))


if __name__ == "__main__":
    main()
