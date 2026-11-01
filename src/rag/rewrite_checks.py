"""Conservative checks for obvious rewrite failures, not semantic verification."""
import re

_EN_QUESTION = re.compile(r"^\s*(?:who|what|when|where|why|how|which|whose|whom|is|are|was|were|do|does|did|can|could|will|would|has|have|had|should)\b", re.I)
_NUMBERS = re.compile(r"(?<![\w])\d+(?:[.,]\d+)*(?![\w])")
_NEGATION = re.compile(r"\b(?:not|never|without|cannot|can't|didn't|doesn't|isn't|wasn't|weren't|won't|couldn't|shouldn't|haven't|hasn't|hadn't|no)\b", re.I)


def check_rewrite(question: str, rewrite: str) -> dict:
    reasons = []
    if not rewrite.strip():
        reasons.append("empty")
    if '\n' in rewrite or '\r' in rewrite or len(rewrite) > max(512, 4 * len(question)):
        reasons.append("invalid_format")
    # Apply only to explicit English questions; Chinese and imperative search
    # queries are intentionally not judged by this rule.
    if _EN_QUESTION.match(question) and '?' in question and not (_EN_QUESTION.match(rewrite) or '?' in rewrite):
        reasons.append("question_to_nonquestion")
    original_numbers = set(_NUMBERS.findall(question))
    if original_numbers - set(_NUMBERS.findall(rewrite)):
        reasons.append("number_omitted")
    if _NEGATION.search(question) and not _NEGATION.search(rewrite):
        reasons.append("negation_omitted")
    return {"passed": not reasons, "reasons": reasons, "semantic_fidelity": "not_verified"}
