# YAML Agent Definitions

dipsy extends DSPy with declarative agent definitions using YAML. This allows you to define complex multi-agent systems without writing Python classes.

## Quick Example

**Define an agent** (`my_agent.yaml`):

```yaml
signature: "question -> answer"
model: "openai/gpt-4o-mini"

tools:
  - tools/search_tool.py
  - tools/calculator_tool.py

max_iters: 20
compression: concise
max_context_turns: 20
```

**Use it in Python:**

```python
import dspy
from dspy.utils import create_agent_from_yaml

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

agent = create_agent_from_yaml("my_agent.yaml")
result = agent(question="What is 2 + 2?")
print(result.answer)
```

## YAML Schema

### Required Fields

- **`signature`** (string): DSPy signature defining input and output fields
  ```yaml
  signature: "question -> answer"
  ```

### Optional Fields

- **`model`** (string): LiteLLM model identifier (can be overridden via CLI `--model`)
  ```yaml
  model: "openai/gpt-4o-mini"
  ```

- **`tools`** (list): Paths to Python files containing tool functions
  ```yaml
  tools:
    - ../tools/search_tool.py
    - ../tools/calculator_tool.py
  ```

  Tool paths are resolved relative to the YAML file's directory.

- **`subagents`** (list): Child agents for hierarchical delegation (see [Multi-Agent Systems](multi-agent-systems.md))
  ```yaml
  subagents:
    - name: researcher
      description: "Delegate research questions"
      agent: ./researcher.yaml
  ```

- **`max_iters`** (integer): Maximum ReAct reasoning iterations (default: 20)
  ```yaml
  max_iters: 10
  ```

- **`compression`** (string): Context compression level: `"concise"` or `"verbose"` (default: `"concise"`)
  ```yaml
  compression: verbose
  ```

- **`max_context_turns`** (integer): Maximum conversation turns to retain (default: 20)
  ```yaml
  max_context_turns: 30
  ```

## Writing Tools

Tools are Python files containing type-annotated functions. Each function becomes a tool the agent can call.

### Requirements

1. **Type annotations** on all parameters and return type
2. **Docstring** describing what the tool does (agent reads this!)
3. **Public function** (name doesn't start with `_`)

### Example Tool

```python
# tools/search_tool.py
"""Search tool for finding information."""

def search(query: str) -> str:
    """Search for information on a topic. Returns relevant snippets."""
    # Implementation here
    return f"Search results for '{query}': ..."
```

### Multiple Tools per File

A single Python file can contain multiple tools:

```python
# tools/math_tools.py
"""Mathematical operations."""

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b
```

Both `add` and `multiply` become available tools.

## API Reference

### `create_agent_from_yaml(yaml_path: str)`

Convenience function to load and build a `ConversationalAgent` from YAML.

**Parameters:**
- `yaml_path` (str): Path to the YAML agent definition file

**Returns:**
- `ConversationalAgent`: Fully initialized agent ready to use

**Example:**
```python
agent = create_agent_from_yaml("agents/my_agent.yaml")
result = agent(question="Hello!")
```

### `load_agent_config(yaml_path: str)`

Lower-level function that parses YAML and returns a configuration dict.

**Parameters:**
- `yaml_path` (str): Path to the YAML agent definition file

**Returns:**
- `dict`: Configuration dictionary with keys:
  - `signature`: DSPy signature string
  - `model`: Model identifier or None
  - `tools`: List of callable tool functions
  - `max_iters`: Maximum iterations
  - `compression`: Compression level
  - `max_context_turns`: Context retention limit

**Example:**
```python
from dspy.utils import load_agent_config
from dspy.predict.conversational import ConversationalAgent
import dspy

config = load_agent_config("agents/my_agent.yaml")

# Build agent with custom settings
inner = dspy.ReAct(
    config["signature"],
    tools=config["tools"],
    max_iters=config["max_iters"]
)

agent = ConversationalAgent(
    inner,
    compression_level="verbose",  # Override YAML setting
    max_context_turns=50          # Override YAML setting
)
```

### `load_tools_from_file(file_path: str)`

Utility to load all public functions from a Python file as tools.

**Parameters:**
- `file_path` (str): Path to Python file containing tool functions

**Returns:**
- `list[Callable]`: List of tool functions

**Example:**
```python
from dspy.utils import load_tools_from_file

tools = load_tools_from_file("tools/search_tool.py")
agent = dspy.ReAct("question -> answer", tools=tools)
```

## CLI Usage

The example CLI supports YAML agents:

```bash
# Run a YAML-defined agent
python examples/conversational_agent_demo.py --agent examples/agents/demo_agent.yaml

# Override model from CLI
python examples/conversational_agent_demo.py \
    --agent examples/agents/demo_agent.yaml \
    --model openai/gpt-4o
```

See the [examples/README.md](https://github.com/samrahimi/dipsy/blob/main/examples/README.md) for complete CLI documentation.

## Next Steps

- Learn about [Conversational Agents](conversational-agents.md) for multi-turn memory
- Build [Hierarchical Multi-Agent Systems](multi-agent-systems.md) with subagents
- Explore [examples/](https://github.com/samrahimi/dipsy/tree/main/examples) in the repository
