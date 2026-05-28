from __future__ import annotations

from typing import Dict, Sequence
import json


def _try_load_dict(candidate: str) -> Dict[str, object] | None:
    try:
        data = json.loads(candidate)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def parse_json_object(payload: str) -> Dict[str, object]:
    candidates = []
    stripped = payload.strip()
    if stripped:
        candidates.append(stripped)

    fence_stripped = _strip_code_fence(payload)
    if fence_stripped and fence_stripped not in candidates:
        candidates.append(fence_stripped)

    for candidate in list(candidates):
        extracted = _extract_first_json_object(candidate)
        if extracted and extracted not in candidates:
            candidates.append(extracted)

    for candidate in candidates:
        data = _try_load_dict(candidate)
        if data is not None:
            return data
    raise ValueError("No JSON object found in LLM output.")


def empty_value_from_schema(schema: Dict[str, object]) -> object:
    schema_type = schema.get("type")
    if schema_type == "object":
        return {
            key: empty_value_from_schema(value)
            for key, value in dict(schema.get("properties", {})).items()
            if isinstance(value, dict)
        }
    if schema_type == "array":
        return []
    if schema_type == "string":
        return ""
    if schema_type == "boolean":
        return False
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    return None


def normalize_value_with_schema(value: object, schema: Dict[str, object]) -> object:
    schema_type = schema.get("type")
    if schema_type == "object":
        source = value if isinstance(value, dict) else {}
        normalized: Dict[str, object] = {}
        for key, child_schema in dict(schema.get("properties", {})).items():
            if isinstance(child_schema, dict):
                normalized[key] = normalize_value_with_schema(source.get(key), child_schema)
        return normalized
    if schema_type == "array":
        return value if isinstance(value, list) else []
    if schema_type == "string":
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)
    if schema_type == "boolean":
        return value if isinstance(value, bool) else False
    if schema_type == "integer":
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return 0
    if schema_type == "number":
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return 0.0
    return value


def normalize_with_response_format(data: object, response_format: Dict[str, object]) -> Dict[str, object]:
    schema = dict(response_format.get("json_schema", {})).get("schema", {})
    if not isinstance(schema, dict):
        return data if isinstance(data, dict) else {}
    normalized = normalize_value_with_schema(data if isinstance(data, dict) else {}, schema)
    if isinstance(normalized, dict):
        return normalized
    empty = empty_value_from_schema(schema)
    return empty if isinstance(empty, dict) else {}


def coerce_structured_output(payload: str, response_format: Dict[str, object]) -> Dict[str, object]:
    try:
        data: object = parse_json_object(payload)
    except Exception:
        data = {}
    return normalize_with_response_format(data, response_format)


def has_meaningful_value(data: Dict[str, object], path: Sequence[str]) -> bool:
    current: object = data
    for key in path:
        if not isinstance(current, dict):
            return False
        current = current.get(key)

    if isinstance(current, str):
        return bool(current.strip())
    if current is None:
        return False
    if isinstance(current, list):
        return len(current) > 0
    if isinstance(current, dict):
        return len(current) > 0
    return True
