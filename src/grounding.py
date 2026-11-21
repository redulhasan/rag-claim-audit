import re

import numpy as np

REFUSAL_TEXT = "Not enough evidence in the indexed notes to verify an answer."
PREMISE_REFUSAL_TEXT = (
    "The premise of this question contradicts the indexed notes, so no answer is returned. "
    "The notes say the opposite of what the question assumes."
)

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
_FALLBACK_LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}

# lightweight contradiction signals, used when no NLI model is loaded (tests, CI).
# negation only counts in the claim: notes are full of incidental negations
# ("keys and values do not change"), so a negated chunk alone proves nothing.
_NEGATION_RE = re.compile(r"\b(not|never|cannot|no longer|does not|do not|did not|is not|are not|was not|were not|without)\b|n't\b")
_ANTONYM_PAIRS = [
    ("increas", "decreas"),
    ("increas", "freez"),
    ("increas", "does not change"),
    ("increas", "unchanged"),
    ("rais", "lower"),
    ("faster", "slower"),
    ("faster", "unchanged"),
    ("higher", "lower"),
    ("larger", "smaller"),
    ("doubl", "halv"),
    ("improv", "worsen"),
    ("expand", "shrink"),
]

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?|\d+/\d+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_STOP = set("the a an is are was were of to in on for and or it its this that with as at by from be has have not".split())


def split_claims(answer: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(answer.strip()) if len(s.strip()) > 10]


def extract_numbers(text: str) -> set[str]:
    out = set()
    for raw in _NUMBER_RE.findall(text):
        raw = raw.rstrip("%")
        if "/" in raw:
            out.add(raw)
            continue
        try:
            value = float(raw)
            out.add(str(int(value)) if value == int(value) else str(value))
        except ValueError:
            out.add(raw)
    return out


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def lexical_score(claim: str, chunk_text: str) -> float:
    words = _content_words(claim)
    if not words:
        return 0.0
    return len(words & _content_words(chunk_text)) / len(words)


def should_refuse(top_score: float, threshold: float) -> bool:
    return top_score < threshold


def softmax(logits) -> np.ndarray:
    # explicit, numerically stable softmax over the last axis. do not rely on a
    # library flag for this: cross-encoder predict() behaviour for multi-label
    # NLI heads differs across sentence-transformers versions.
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def label_index_map(id2label: dict | None) -> dict[str, int]:
    # map label names to logit indices from the model config when available
    # (case-insensitive); fall back to the standard deberta NLI order otherwise
    if not id2label:
        return dict(_FALLBACK_LABELS)
    found = {}
    for idx, name in id2label.items():
        key = str(name).lower()
        if key in _FALLBACK_LABELS:
            found[key] = int(idx)
    return {**_FALLBACK_LABELS, **found}


def heuristic_contradiction(claim: str, chunk_text: str) -> bool:
    # deterministic stand-in for the NLI contradiction check in lightweight mode:
    # antonym flips (increases vs decreases/frozen) or a negated claim whose
    # content otherwise matches the chunk closely
    low_claim = claim.lower()
    low_chunk = chunk_text.lower()
    for a, b in _ANTONYM_PAIRS:
        if (a in low_claim and b in low_chunk) or (b in low_claim and a in low_chunk):
            return True
    if _NEGATION_RE.search(low_claim) and not _NEGATION_RE.search(low_chunk):
        return lexical_score(claim, chunk_text) >= 0.35
    return False


def heuristic_nli_probs(premise: str, hypothesis: str) -> dict[str, float]:
    # deterministic probabilities with the same shape as the NLI model output,
    # so the gate thresholds mean the same thing in lightweight and full mode
    hyp_numbers = extract_numbers(hypothesis)
    prem_numbers = extract_numbers(premise)
    if hyp_numbers and prem_numbers and not hyp_numbers.issubset(prem_numbers):
        return {"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05}
    if heuristic_contradiction(hypothesis, premise):
        return {"contradiction": 0.85, "entailment": 0.05, "neutral": 0.10}
    overlap = lexical_score(hypothesis, premise)
    if overlap >= 0.35:
        return {"contradiction": 0.05, "entailment": 0.80, "neutral": 0.15}
    if overlap >= 0.15:
        return {"contradiction": 0.10, "entailment": 0.30, "neutral": 0.60}
    return {"contradiction": 0.10, "entailment": 0.05, "neutral": 0.85}


class NLIVerifier:
    # wraps cross-encoder/nli-deberta-v3-small in full mode. pairs are always
    # (premise, hypothesis) = (chunk_text, claim_or_query), in that order.
    def __init__(self, lightweight: bool = False, model_name: str = NLI_MODEL):
        self.lightweight = lightweight
        self.model = None
        self._label_map = label_index_map(None)
        if not lightweight:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
            config = getattr(self.model.model, "config", None)
            self._label_map = label_index_map(getattr(config, "id2label", None))

    def probs(self, premise: str, hypothesis: str) -> dict[str, float]:
        if self.lightweight or self.model is None:
            return heuristic_nli_probs(premise, hypothesis)
        raw = self.model.predict([(premise, hypothesis)])
        logits = np.atleast_2d(np.asarray(raw, dtype=np.float64))[0]
        probs = softmax(logits)
        return {label: float(probs[idx]) for label, idx in self._label_map.items()}


class ClaimAuditor:
    # scorer(claim, chunk_text) -> 0..1 relevance, used to pick the best chunk.
    # nli(premise, hypothesis) -> {"contradiction", "entailment", "neutral"}
    # probabilities, used for the verdict when provided. without nli, the
    # lightweight heuristics in this module stand in for it (tests and CI).
    # rule order per claim: numeric check, contradiction, entailment, fallback.
    def __init__(self, scorer=None, supported_threshold: float = 0.45, weak_threshold: float = 0.22, nli=None):
        self.scorer = scorer or lexical_score
        self.supported_threshold = supported_threshold
        self.weak_threshold = weak_threshold
        self.nli = nli

    def audit(self, answer: str, chunks: list[dict]) -> dict:
        claims = []
        for claim in split_claims(answer):
            if not chunks:
                claims.append(
                    {
                        "claim": claim,
                        "label": "unsupported",
                        "score": 0.0,
                        "numeric_mismatch": False,
                        "contradiction": False,
                        "reason": "no_evidence",
                        "citation": None,
                    }
                )
                continue
            scored = [(self.scorer(claim, c["text"]), c) for c in chunks]
            best_score, best = max(scored, key=lambda pair: pair[0])
            claim_numbers = extract_numbers(claim)
            chunk_numbers = extract_numbers(best["text"])
            # rule A: a relevant chunk with the wrong number in it is how altered
            # figures slip through; treat any unmatched number as unsupported
            numeric_mismatch = bool(claim_numbers) and not claim_numbers.issubset(chunk_numbers)
            contradiction = False
            if numeric_mismatch:
                label, reason = "unsupported", "numeric_mismatch"
            elif self.nli is not None:
                probs = self.nli(best["text"], claim)
                if probs.get("contradiction", 0.0) >= 0.50:  # rule B
                    label, reason, contradiction = "unsupported", "contradiction", True
                elif probs.get("entailment", 0.0) >= 0.60:  # rule C
                    label, reason = "supported", "entailed"
                else:  # rule D
                    label, reason = "weakly_supported", "insufficient_entailment"
            elif heuristic_contradiction(claim, best["text"]):
                label, reason, contradiction = "unsupported", "contradiction", True
            elif best_score >= self.supported_threshold:
                label, reason = "supported", "lexical_support"
            elif best_score >= self.weak_threshold:
                label, reason = "weakly_supported", "weak_evidence"
            else:
                label, reason = "unsupported", "low_relevance"
            claims.append(
                {
                    "claim": claim,
                    "label": label,
                    "score": round(float(best_score), 3),
                    "numeric_mismatch": numeric_mismatch,
                    "contradiction": contradiction,
                    "reason": reason,
                    "citation": {"source_file": best["source_file"], "section": best["section"]},
                }
            )
        supported = sum(1 for c in claims if c["label"] == "supported")
        return {
            "claims": claims,
            "grounding_ratio": round(supported / len(claims), 3) if claims else 0.0,
            "n_claims": len(claims),
            "n_supported": supported,
            "n_unsupported": sum(1 for c in claims if c["label"] == "unsupported"),
        }
