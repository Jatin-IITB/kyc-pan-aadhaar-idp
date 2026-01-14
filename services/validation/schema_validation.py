"/services/validation/schema_validation.py"
from pathlib import Path
import json
import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "config" / "schemas"


def _load_schema(doc_type: str) -> dict:
    schema_path = SCHEMA_DIR / f"{doc_type}.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def get_required_fields(doc_type: str):
    try:
        schema = _load_schema(doc_type)
    except Exception:
        return []
    req = schema.get("required", [])
    return req if isinstance(req, list) else []


def validate_with_schema(data: dict, doc_type: str):
    try:
        schema = _load_schema(doc_type)
    except Exception as e:
        return False, str(e)
    
    validator = jsonschema.Draft7Validator(schema)
    errors = list(validator.iter_errors(data))

    if not errors:
        return True, "Valid"
    error_messages = [f"Field '{err.json_path.split('.')[-1]}' {err.message}" for err in errors]
    return False, " | ".join(error_messages)
