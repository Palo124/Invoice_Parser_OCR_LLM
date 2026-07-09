from dataclasses import dataclass, field

from app.services.validation.types import ValidationError


@dataclass
class FieldMergeResult:
    merged: dict
    disagreements: list[ValidationError] = field(default_factory=list)


def normalize_value(value):
    if isinstance(value, (list, dict)):
        import json

        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(value)
    return value


def denormalize_value(value):
    if isinstance(value, str):
        stripped = value.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            import json

            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
    return value


def merge_extractions_with_flags(dict1: dict, dict2: dict) -> FieldMergeResult:
    """Merge two LLM extractions and record fields where models disagree."""
    merged: dict = {}
    disagreements: list[ValidationError] = []
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in sorted(all_keys):
        left = dict1.get(key)
        right = dict2.get(key)
        if normalize_value(left) == normalize_value(right):
            merged[key] = left if key in dict1 else right
            continue

        merged[key] = left
        disagreements.append(
            ValidationError(
                field=key,
                code="tmr_disagreement",
                message=f"Models disagree on '{key}': {left!r} vs {right!r}.",
                severity="warning",
            )
        )

    return FieldMergeResult(merged=merged, disagreements=disagreements)
