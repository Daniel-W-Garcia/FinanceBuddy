# tools/validate_task_outputs.py
"""
Validate JSON task outputs listed in tasks.yaml against Pydantic v2 models.

Usage:
  from financebuddy.tools.validate_task_outputs import validate_all_task_outputs
  validate_all_task_outputs(tasks_yaml_path="tasks.yaml", base_dir=".", diagnostics_dir="output/diagnostics")
"""

from __future__ import annotations

import os
import json
import shutil
import logging
import importlib
import tempfile
from typing import Any, Dict, List, Optional, Type

import yaml
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def _resolve_model(model_path: str) -> Type[BaseModel]:
    """
    Resolve a string like 'crew.EdgarBasicsSchema' into the actual Pydantic model class.
    Raises ImportError/AttributeError on failure.
    """
    module_name, _, class_name = model_path.rpartition(".")
    if not module_name:
        raise ImportError(f"Invalid model_path '{model_path}': must be 'module.ClassName'")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    if not issubclass(cls, BaseModel):
        raise TypeError(f"Resolved object {model_path} is not a pydantic BaseModel subclass")
    return cls


def _friendly_from_error(err: Dict[str, Any]) -> str:
    loc = ".".join(str(x) for x in err.get("loc", []))
    msg = err.get("msg", "")
    return f"{loc}: {msg}" if loc else msg


def _atomic_write_json(path: str, data: Any) -> None:
    """
    Write JSON atomically: write to temp file then rename.
    """
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _validate_payload_against_model(model_cls: Type[BaseModel], raw: Any) -> Dict[str, Any]:
    """
    Validate raw payload against model_cls.
    Returns a dict suitable for JSON serialization:
      - on success: model_dump()
      - on ValidationError: partial dict with 'warnings' and 'raw_input'
    """
    try:
        m = model_cls.model_validate(raw)
        out = m.model_dump()
        # ensure diagnostics keys exist
        if isinstance(out, dict):
            out.setdefault("warnings", [])
            out.setdefault("raw_input", None)
        return out
    except ValidationError as exc:
        errs = exc.errors()
        warnings = [_friendly_from_error(e) for e in errs]
        logger.warning("Validation errors: %s", warnings)

        partial: Dict[str, Any] = {}
        raw_dict = raw if isinstance(raw, dict) else {"value": raw}

        # Salvage strategy:
        # If model has an 'items' field (we use wrapper models that use items), try to validate items
        if "items" in model_cls.model_fields:
            items_valid: List[Any] = []
            if isinstance(raw, list):
                for item in raw:
                    try:
                        # Try validating one-item list via wrapper model if possible
                        tmp = model_cls.model_validate({"items": [item]})
                        dumped = tmp.model_dump()
                        items = dumped.get("items") or []
                        if items:
                            items_valid.extend(items)
                        else:
                            items_valid.append(item)
                    except Exception:
                        # skip invalid item
                        continue
            partial["items"] = items_valid
        else:
            # per-field salvage for top-level fields
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
        return partial


def validate_all_task_outputs(tasks_yaml_path: str = "tasks.yaml", base_dir: str = ".", diagnostics_dir: str = "output/diagnostics") -> None:
    """
    Validate all task output files declared in tasks.yaml.

    For each task that declares 'output_model' and 'output_file':
      - If file exists: validate and rewrite canonical output (or partial with warnings).
      - If not exists: logs debug and continues.

    Creates diagnostics files in diagnostics_dir with full ValidationError.errors() when validation fails.
    """
    # Load tasks yaml
    with open(tasks_yaml_path, "r", encoding="utf-8") as fh:
        tasks_cfg = yaml.safe_load(fh)

    tasks = {k: v for k, v in tasks_cfg.items() if k != "meta"}

    os.makedirs(diagnostics_dir, exist_ok=True)

    for task_name, task_def in tasks.items():
        output_model_path = task_def.get("output_model")
        output_file = task_def.get("output_file")
        if not output_model_path or not output_file:
            logger.debug("Skipping task '%s' (no output_model or output_file)", task_name)
            continue

        # Resolve absolute path of output file
        out_path = output_file if os.path.isabs(output_file) else os.path.join(base_dir, output_file)

        if not os.path.exists(out_path):
            logger.info("Task output file not found for '%s': %s", task_name, out_path)
            continue

        try:
            model_cls = _resolve_model(output_model_path)
        except Exception as e:
            logger.exception("Failed to resolve model '%s' for task '%s': %s", output_model_path, task_name, e)
            continue

        # Load existing JSON
        try:
            with open(out_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as e:
            logger.exception("Failed to load JSON for task '%s' from %s: %s", task_name, out_path, e)
            continue

        # Backup original file
        backup_path = out_path + ".orig.json"
        try:
            if not os.path.exists(backup_path):
                shutil.copy2(out_path, backup_path)
        except Exception:
            logger.exception("Failed to back up file %s -> %s", out_path, backup_path)

        # Validate
        validated = _validate_payload_against_model(model_cls, raw)

        # If the result contains warnings and they differ from empty, write diagnostics
        if isinstance(validated, dict) and validated.get("warnings"):
            diag = {
                "task": task_name,
                "model": output_model_path,
                "file": out_path,
                "warnings": validated.get("warnings"),
                "raw_input_preview": (json.dumps(raw)[:200] + "...") if raw else None,
            }
            diag_path = os.path.join(diagnostics_dir, f"{task_name}.diagnostic.json")
            try:
                _atomic_write_json(diag_path, diag)
            except Exception:
                logger.exception("Failed to write diagnostic for task %s", task_name)

        # Write validated/partial payload back to file atomically
        try:
            _atomic_write_json(out_path, validated)
            logger.info("Validated and updated output for task '%s' -> %s", task_name, out_path)
        except Exception:
            logger.exception("Failed to write validated output for task '%s' to %s", task_name, out_path)
