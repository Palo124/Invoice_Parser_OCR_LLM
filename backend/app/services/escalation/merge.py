from app.services.validation.types import ValidationError


def _get_nested(data: dict, path: str):
    current = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_nested(data: dict, path: str, value) -> None:
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def fields_to_override(
    validation_errors: list[ValidationError],
    disagreement_fields: list[str],
) -> list[str]:
    fields = {error.field for error in validation_errors if error.severity == "error"}
    fields.update(disagreement_fields)
    return sorted(fields)


def merge_escalation_overrides(
    base_data: dict,
    escalation_data: dict,
    override_fields: list[str],
) -> tuple[dict, list[str]]:
    merged = dict(base_data)
    applied: list[str] = []

    for field_path in override_fields:
        value = _get_nested(escalation_data, field_path)
        if value is None and field_path not in escalation_data:
            top_level = escalation_data.get(field_path)
            if top_level is not None:
                value = top_level

        if value is None:
            continue

        if "." in field_path:
            _set_nested(merged, field_path, value)
        else:
            merged[field_path] = value
        applied.append(field_path)

    return merged, applied
