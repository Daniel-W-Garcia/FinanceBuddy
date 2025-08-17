# tools/task_callbacks.py
import os
import json
import tempfile
import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def _atomic_write_json(path: str, data: Any) -> None:
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _friendly_errors(exc: ValidationError) -> list[str]:
    return [".".join(str(x) for x in e.get("loc", [])) + ": " + e.get("msg", "") for e in exc.errors()]


def render_synthesis_to_md(synth: Dict[str, Any], out_md_path: str) -> None:
    """Simple renderer of synthesis JSON -> Markdown (human readable)."""
    lines = []
    lines.append(f"# Executive Brief (as of {synth.get('raw_input', {}).get('as_of') or synth.get('as_of')})\n")
    lines.append("## TL;DR\n")
    for b in synth.get("tldr", []):
        lines.append(f"- {b}")
    lines.append("\n## Positives\n")
    for p in synth.get("positives", []):
        lines.append(f"- {p}")
    lines.append("\n## Risks\n")
    for r in synth.get("risks", []):
        lines.append(f"- {r}")
    lines.append("\n## Notable Filings\n")
    for f in synth.get("notable_filings", []):
        lines.append(f"- {f}")
    lines.append("\n## Fundamentals Takeaways\n")
    for f in synth.get("fundamentals_takeaways", []):
        lines.append(f"- {f}")
    lines.append("\n## Recent Catalysts\n")
    for c in synth.get("recent_catalysts", []):
        lines.append(f"- {c}")
    lines.append("\n## Next Steps\n")
    for n in synth.get("next_steps", []):
        lines.append(f"- {n}")
    lines.append("\n---\n")
    lines.append(synth.get("disclaimer", ""))
    # write atomically
    tmp = out_md_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")
    os.replace(tmp, out_md_path)


def task_complete_callback(task_result: Any, task_obj: Any = None, task_name: Optional[str] = None) -> None:
    """
    Generic callback to run after a task finishes.

    Typical callback signatures:
      - callback(result, task)
      - callback(result) where result is a CrewOutput with metadata about task
    This function is defensive and attempts to handle common variants.
    """

    # Extract a dict payload from task_result
    try:
        if hasattr(task_result, "to_dict"):
            payload = task_result.to_dict()
        elif hasattr(task_result, "model_dump"):
            payload = task_result.model_dump()
        elif isinstance(task_result, dict):
            payload = task_result
        else:
            # Try attribute access .pydantic / .pydantic.model_dump()
            payload = getattr(task_result, "pydantic", getattr(task_result, "data", None))
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            if payload is None:
                payload = {}
    except Exception as e:
        logger.exception("Failed to extract payload from task_result: %s", e)
        payload = {}

    # Determine the task object and name if not provided
    # The caller should pass the task object if available; otherwise attempt best-effort extraction
    t_obj = task_obj
    t_name = task_name or (getattr(task_obj, "config", {}).get("id") if task_obj else None)

    # Determine Pydantic model bound to the Task (support both output_pydantic and output_json keys)
    model_cls = None
    output_file = None
    try:
        if t_obj is not None:
            model_cls = getattr(t_obj, "output_pydantic", None) or getattr(t_obj, "output_json", None)
            # Many Task objects expose output_file property; fallback to config mapping
            output_file = getattr(t_obj, "output_file", None) or getattr(t_obj, "config", {}).get("output_file")
    except Exception:
        pass

    # If output_file not found, attempt to build from task_name
    if not output_file and t_name:
        output_file = f"output/{t_name}.json"

    # Validate with pydantic model if present
    validated_payload = None
    diagnostics = None
    if isinstance(model_cls, type) and issubclass(model_cls, BaseModel):
        try:
            m = model_cls.model_validate(payload)
            validated_payload = m.model_dump()
        except ValidationError as exc:
            warnings = _friendly_errors(exc)
            logger.warning("Validation failed for task %s: %s", t_name or "<unknown>", warnings)
            # Salvage partial payload (simple per-field attempt)
            partial = {}
            raw_dict = payload if isinstance(payload, dict) else {"value": payload}
            for fld in model_cls.model_fields.keys():
                if fld in raw_dict:
                    try:
                        tmp = model_cls.model_validate({fld: raw_dict[fld]})
                        dumped = tmp.model_dump()
                        if fld in dumped:
                            partial[fld] = dumped[fld]
                    except Exception:
                        continue
            partial["raw_input"] = raw_dict
            partial["warnings"] = warnings
            validated_payload = partial
            diagnostics = {"task": t_name, "warnings": warnings, "raw_input_preview": json.dumps(raw_dict)[:500]}
    else:
        # No model: use the raw payload but ensure warnings/raw_input keys exist
        validated_payload = payload if isinstance(payload, dict) else {"raw_input": payload}
        validated_payload.setdefault("warnings", [])
        validated_payload.setdefault("raw_input", payload)

    # Write the output JSON atomically (pretty-printed)
    if output_file:
        try:
            # backup original if exists
            if os.path.exists(output_file):
                bak = output_file + ".orig"
                if not os.path.exists(bak):
                    shutil.copy2(output_file, bak)
            _atomic_write_json(output_file, validated_payload)
            logger.info("Wrote validated output for task %s -> %s", t_name, output_file)
        except Exception as e:
            logger.exception("Failed to write validated output for task %s: %s", t_name, e)

    # Write diagnostics if present
    if diagnostics:
        diag_dir = os.path.join("output", "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        diag_path = os.path.join(diag_dir, (t_name or "unknown") + ".diagnostic.json")
        _atomic_write_json(diag_path, diagnostics)

    # Special-case: if this is the synthesis task, also render Markdown
    try:
        if t_name and "synthesis" in t_name.lower():
            md_path = os.path.splitext(output_file)[0] + ".md"
            render_synthesis_to_md(validated_payload, md_path)
            logger.info("Rendered synthesis markdown -> %s", md_path)
    except Exception:
        logger.exception("Failed to render synthesis markdown for task %s", t_name)