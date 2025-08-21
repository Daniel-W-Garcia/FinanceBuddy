#!/usr/bin/env python3
import argparse
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, get_args, get_origin, get_type_hints, Union

# Make project root importable (same strategy as your original script)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Module path for your tools file
TOOLS_MODULE_PATH = "tools.extraction_tools"
OUTPUT_FILENAME = "project_tool_specs.json"

# Try to import Pydantic BaseModel for richer schema capture
try:
    from pydantic import BaseModel
except Exception:
    BaseModel = None  # type: ignore

# Try to import CrewAI BaseTool to detect subclassed tools (optional)
try:
    import crewai  # type: ignore

    try:
        BaseToolClass = getattr(crewai.tools, "BaseTool", None)
    except Exception:
        BaseToolClass = None
except Exception:
    crewai = None  # type: ignore
    BaseToolClass = None


def python_type_to_json_type(tp: Any) -> Dict[str, Any]:
    """
    Map Python typing objects to simple JSON-like schema descriptors.
    Returns a dict describing type (and nested items when applicable).
    """
    if tp is None:
        return {"type": "null"}

    origin = get_origin(tp)
    args = get_args(tp)

    # direct builtin types
    if tp is str:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is list or origin is list or origin is List:
        item = args[0] if args else Any
        return {"type": "array", "items": python_type_to_json_type(item)}
    if tp is dict or origin is dict:
        val = args[1] if len(args) > 1 else Any
        return {"type": "object", "additionalProperties": python_type_to_json_type(val)}

    # Optional[T] -> Union[T, NoneType]
    if origin is Union:
        # filter NoneType
        non_none = [a for a in args if a is not type(None)]
        subs = [python_type_to_json_type(a) for a in non_none]
        if len(subs) == 1:
            subs[0]["nullable"] = True
            return subs[0]
        return {"anyOf": subs}

    # Pydantic BaseModel
    try:
        if BaseModel and inspect.isclass(tp) and issubclass(tp, BaseModel):
            # Use pydantic's schema() for a richer description
            try:
                return {"type": "object", "pydantic_schema": tp.schema()}
            except Exception:
                return {"type": "object", "pydantic_model": tp.__name__}
    except Exception:
        pass

    # Fallback: represent as string
    return {"type": "string"}


def extract_param_info(param: inspect.Parameter, type_hints: Dict[str, Any]) -> Dict[str, Any]:
    hint = type_hints.get(param.name, None)
    schema = python_type_to_json_type(hint) if hint is not None else {"type": "string"}

    info: Dict[str, Any] = {
        "name": param.name,
        "kind": str(param.kind),
        "required": param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ),
        "schema": schema,
    }
    if param.default is not inspect.Parameter.empty:
        try:
            json.dumps(param.default)
            info["default"] = param.default
        except Exception:
            info["default"] = repr(param.default)
    return info


def detect_crewai_tool(obj, module) -> Dict[str, Any]:
    """
    Return detection info dict for the object in the tools module.
    Keys:
    - is_tool (bool)
    - tool_type: "function", "crewai_decorated", "crewai_basetool_class", "class"
    - display_name (optional)
    """
    name = getattr(obj, "name", repr(obj))
    info = {"is_tool": False, "tool_type": None, "display_name": name}

    # If crewai is installed, check for subclass of BaseTool
    try:
        if inspect.isclass(obj) and BaseToolClass and issubclass(obj, BaseToolClass):
            info.update({
                "is_tool": True,
                "tool_type": "crewai_basetool_class",
                "display_name": getattr(obj, "name", name)
            })
            return info
    except Exception:
        pass

    # If function, check for known decorator artifacts:
    if inspect.isfunction(obj):
        # check typical attributes added by tool decorators
        if hasattr(obj, "_tool") or hasattr(obj, "name") and isinstance(getattr(obj, "name"), str):
            info.update({
                "is_tool": True,
                "tool_type": "crewai_decorated",
                "display_name": getattr(obj, "name", name)
            })
            return info

        # many decorators wrap functions. inspect __wrapped__ chain
        wrapped = getattr(obj, "__wrapped__", None)
        if wrapped:
            # check wrapped attributes
            if hasattr(wrapped, "_tool") or hasattr(wrapped, "name"):
                info.update({
                    "is_tool": True,
                    "tool_type": "crewai_decorated_wrapped",
                    "display_name": getattr(wrapped, "name", name)
                })
                return info

    # fallback: treat public functions/classes defined in the module as plain tools (if they appear relevant)
    if getattr(obj, "__module__", None) == module.__name__:
        # mark as generic function/class (not explicitly crewai)
        info.update({
            "is_tool": True,
            "tool_type": "function_or_class",
            "display_name": name
        })
    return info


def extract_tools_from_module(module_name: str) -> List[Dict[str, Any]]:
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"ERROR: Could not import module {module_name}: {e}")
        return []

    members = inspect.getmembers(module, predicate=lambda o: (inspect.isfunction(o) or inspect.isclass(o)))
    results: List[Dict[str, Any]] = []

    for name, member in members:
        # Skip private/internal names
        if name.startswith("_"):
            continue
        # Only consider objects defined in that module
        if getattr(member, "__module__", None) != module.__name__:
            continue

        detection = detect_crewai_tool(member, module)

        # Resolve type hints for the member (safe)
        try:
            type_hints = get_type_hints(member, globalns=module.__dict__, localns=module.__dict__)
        except Exception:
            try:
                # fallback without resolving forward refs
                type_hints = get_type_hints(member, globalns={}, localns={})
            except Exception:
                type_hints = {}

        sig = None
        params = {}
        return_schema = {"type": "unknown"}

        try:
            sig = inspect.signature(member)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                params[param_name] = extract_param_info(param, type_hints)
        except Exception:
            params = {}

        # return type
        try:
            rhint = type_hints.get("return", None)
            return_schema = python_type_to_json_type(rhint) if rhint is not None else {"type": "any"}
        except Exception:
            return_schema = {"type": "any"}

        tool_info = {
            "name": name,
            "display_name": detection.get("display_name", name),
            "description": inspect.getdoc(member) or "",
            "detected_as": detection["tool_type"],
            "is_crewai_tool": detection["is_tool"],
            "parameters": params,
            "return_type": return_schema,
            "source_file": getattr(member, "__module__", module_name),
        }
        results.append(tool_info)

    return results


def save_spec(spec: Dict[str, Any], output_path: Path) -> None:
    payload = {
        "project": output_path.parent.name if output_path.parent else "FinanceBuddy",
        "extraction_tools": spec,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved project tool specs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate tool specs for CrewAI project tools")
    parser.add_argument("--module", "-m", default=TOOLS_MODULE_PATH,
                        help="Python module path to scan for tools (default: tools.extraction_tools)")
    parser.add_argument("--output", "-o", default=str(Path(__file__).parent / OUTPUT_FILENAME))
    args = parser.parse_args()

    specs = extract_tools_from_module(args.module)
    save_spec(specs, Path(args.output))

    print(f"\n✅ Extracted {len(specs)} tool definitions:")
    for s in specs:
        print(f" - {s['name']} ({s['detected_as']}): {s['description'][:80]}")


if __name__ == "__main__":
    main()