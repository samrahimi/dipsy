"""Load a conversational agent definition from a YAML file.

YAML format::

    signature: "question -> answer"
    model: "openai/gpt-4o-mini"
    tools:
      - tools/search_tool.py
      - tools/calculator_tool.py
    subagents:
      - name: math_expert
        description: "Delegate math questions to this specialist"
        agent: ./math_expert.yaml
    max_iters: 20
    compression: concise
    max_context_turns: 20

Tool and agent paths are resolved relative to the YAML file's directory.
Subagents are recursively loaded and wrapped as tool functions.
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable


def _import_yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        print(
            "Error: PyYAML is required for YAML agent definitions.\n"
            "Install it with: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)


def load_tools_from_file(file_path: str) -> list[Callable]:
    """Load all public callable functions from a Python file.

    A function is included if its name does not start with ``_`` and
    it is a regular function (not a class or built-in).
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Tool file not found: {file_path}")

    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)

    # Temporarily add tool dir to sys.path so the tool's own imports resolve
    tool_dir = os.path.dirname(file_path)
    inserted = False
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)
        inserted = True
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f"Failed to load tool file {file_path}: {exc}") from exc
    finally:
        if inserted:
            sys.path.remove(tool_dir)

    tools: list[Callable] = []
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if inspect.isfunction(obj):
            tools.append(obj)

    if not tools:
        raise ValueError(
            f"No public functions found in tool file: {file_path}\n"
            f"Tool files must contain at least one public function (no leading underscore)."
        )
    return tools


def _build_subagent_tool(entry: dict, yaml_dir: str, _visited: set[str]) -> Callable:
    """Build a tool function that delegates to a child agent.

    The child agent YAML is loaded recursively, constructed as a full
    ``ReAct + ConversationalAgent``, and wrapped in a function that the
    parent agent can call like any other tool.
    """
    import dspy
    from dspy.predict.conversational import ConversationalAgent

    # Validate entry
    missing = {"name", "description", "agent"} - set(entry.keys())
    if missing:
        raise ValueError(f"Subagent entry missing required keys: {missing}. Got: {entry}")

    name = entry["name"]
    desc = entry["description"]
    agent_path = entry["agent"]

    if not os.path.isabs(agent_path):
        agent_path = os.path.join(yaml_dir, agent_path)
    agent_path = os.path.normpath(agent_path)

    # Recursively load subagent config (with cycle detection)
    config = load_agent_config(agent_path, _visited=_visited)

    # Build the subagent
    inner = dspy.ReAct(
        config["signature"],
        tools=config["tools"],
        max_iters=config["max_iters"],
    )
    agent = ConversationalAgent(
        inner,
        compression_level=config["compression"],
        max_context_turns=config["max_context_turns"],
    )

    # Determine input/output field names from the subagent's signature
    sig = dspy.Signature(config["signature"])
    input_field = list(sig.input_fields.keys())[0]
    output_field = list(sig.output_fields.keys())[0]

    # Create a wrapper function that looks like a regular tool
    def subagent_tool(request: str) -> str:
        result = agent(**{input_field: request})
        return str(getattr(result, output_field, ""))

    subagent_tool.__name__ = name
    subagent_tool.__qualname__ = name
    subagent_tool.__doc__ = desc

    return subagent_tool


_REQUIRED_FIELDS = {"signature"}
_OPTIONAL_FIELDS = {"model", "tools", "subagents", "max_iters", "compression", "max_context_turns"}
_ALL_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


def load_agent_config(yaml_path: str, _visited: set[str] | None = None) -> dict[str, Any]:
    """Parse and validate a YAML agent definition.

    Returns a dict with keys: ``signature``, ``model``, ``tools``,
    ``max_iters``, ``compression``, ``max_context_turns``.

    Subagent YAML references are recursively loaded and built into tool
    functions that appear in the ``tools`` list alongside regular tools.
    """
    yaml = _import_yaml()
    yaml_path = os.path.abspath(yaml_path)

    # Circular reference detection
    if _visited is None:
        _visited = set()
    if yaml_path in _visited:
        raise ValueError(f"Circular agent reference detected: {yaml_path}")
    _visited.add(yaml_path)

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Agent YAML file not found: {yaml_path}")

    with open(yaml_path, "r") as fh:
        try:
            config = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {yaml_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Agent YAML must be a mapping, got {type(config).__name__}")

    unknown = set(config.keys()) - _ALL_FIELDS
    if unknown:
        raise ValueError(f"Unknown fields in agent YAML: {unknown}")

    missing = _REQUIRED_FIELDS - set(config.keys())
    if missing:
        raise ValueError(f"Missing required fields in agent YAML: {missing}")

    # Resolve tool paths relative to the YAML file's directory
    yaml_dir = os.path.dirname(yaml_path)
    raw_tool_paths = config.get("tools", [])
    if not isinstance(raw_tool_paths, list):
        raise ValueError(f"'tools' must be a list of file paths, got {type(raw_tool_paths).__name__}")

    all_tools: list[Callable] = []
    for tool_path in raw_tool_paths:
        if not os.path.isabs(tool_path):
            tool_path = os.path.join(yaml_dir, tool_path)
        tool_path = os.path.normpath(tool_path)
        all_tools.extend(load_tools_from_file(tool_path))

    # Build subagent tools
    raw_subagents = config.get("subagents", [])
    if not isinstance(raw_subagents, list):
        raise ValueError(f"'subagents' must be a list, got {type(raw_subagents).__name__}")

    for entry in raw_subagents:
        if not isinstance(entry, dict):
            raise ValueError(f"Each subagent entry must be a mapping, got {type(entry).__name__}")
        all_tools.append(_build_subagent_tool(entry, yaml_dir, _visited))

    compression = config.get("compression", "concise")
    if compression not in ("verbose", "concise"):
        raise ValueError(f"compression must be 'verbose' or 'concise', got '{compression}'")

    return {
        "signature": config["signature"],
        "model": config.get("model"),
        "tools": all_tools,
        "max_iters": config.get("max_iters", 20),
        "compression": compression,
        "max_context_turns": config.get("max_context_turns", 20),
    }
