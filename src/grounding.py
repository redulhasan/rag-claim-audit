import re

REFUSAL_TEXT = "Not enough evidence in the indexed notes to verify an answer."

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


class ClaimAuditor:
    # scorer(claim, chunk_text) -> 0..1. default is lexical overlap; pass a
    # cross-encoder based scorer for the real thing. relevance alone is not
    # entailment - see the limitations section of the readme.
    def __init__(self, scorer=None, supported_threshold: float = 0.45, weak_threshold: float = 0.22):
        self.scorer = scorer or lexical_score
        self.supported_threshold = supported_threshold
        self.weak_threshold = weak_threshold

    def audit(self, answer: str, chunks: list[dict]) -> dict:
        claims = []
        for claim in split_claims(answer):
            if not chunks:
                claims.append({"claim": claim, "label": "unsupported", "score": 0.0, "citation": None})
                continue
            scored = [(self.scorer(claim, c["text"]), c) for c in chunks]
            best_score, best = max(scored, key=lambda pair: pair[0])
            claim_numbers = extract_numbers(claim)
            chunk_numbers = extract_numbers(best["text"])
            # a relevant chunk with the wrong number in it is how altered
            # figures slip through; treat any unmatched number as unsupported
            numeric_mismatch = bool(claim_numbers) and not claim_numbers.issubset(chunk_numbers)
            if numeric_mismatch:
                label = "unsupported"
            elif best_score >= self.supported_threshold:
                label = "supported"
            elif best_score >= self.weak_threshold:
                label = "weakly_supported"
            else:
                label = "unsupported"
            claims.append(
                {
                    "claim": claim,
                    "label": label,
                    "score": round(float(best_score), 3),
                    "numeric_mismatch": numeric_mismatch,
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
