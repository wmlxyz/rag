"""Syntactic citation checks; they do not verify evidence entailment."""
import re


def inspect_citations(text: str, valid_ids: set[int]) -> dict:
    # Recognize single and comma-separated numeric references; flag ranges as
    # unsupported rather than pretending their endpoints are the whole range.
    ids, malformed = set(), []
    for match in re.finditer(r"\[([^\[\]]+)\]", text):
        inner = match.group(1).strip()
        if re.fullmatch(r"\d+(?:\s*[,，]\s*\d+)*", inner):
            ids.update(int(part) for part in re.split(r"\s*[,，]\s*", inner))
        elif re.match(r"^\d", inner):
            malformed.append(match.group(0))
    return {"cited_ids": sorted(ids), "valid_ids": sorted(ids & valid_ids),
            "invalid_ids": sorted(ids - valid_ids), "malformed": malformed,
            "has_valid": bool(ids & valid_ids),
            "passed": bool(ids & valid_ids) and not (ids - valid_ids) and not malformed}
