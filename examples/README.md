# Conversational Agent CLI

A multi-turn conversational agent powered by DSPy, with real-time LLM streaming, YAML-based agent definitions, and nested agentic hierarchies via subagents.

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run a YAML-defined agent with tools
python examples/conversational_agent_demo.py --agent examples/agents/demo_agent.yaml

# Run the orchestrator (delegates to subagents)
python examples/conversational_agent_demo.py --agent examples/agents/orchestrator.yaml
```

## CLI Reference

```
python examples/conversational_agent_demo.py [OPTIONS]
```

| Flag | Description |
|---|---|
| `--agent YAML_FILE` | Path to a YAML agent definition (recommended) |
| `--model MODEL` | LM model string, e.g. `openai/gpt-4o-mini` |
| `--api-key KEY` | API key (defaults to env var, e.g. `OPENAI_API_KEY`) |
| `--base-url URL` | Base URL for the LM API (proxies, custom endpoints) |
| `--react` | Use ReAct agent with built-in demo tools (mutually exclusive with `--agent`) |
| `--no-stream` | Disable streaming; use non-streaming observer mode |
| `--quiet` | Disable all observability output |
| `--debug` | Show compressed conversation context at each turn |
| `--verbose` | Use verbose compression (preserves more detail) |
| `--max-turns N` | Max conversation turns to retain (default: 20) |

### Precedence

When using `--agent`, settings are resolved as **CLI flag > YAML config > defaults**. For example, `--model` on the command line overrides the `model` field in the YAML file.

### In-Session Commands

Once the agent is running, you can type these commands at the prompt:

| Command | Description |
|---|---|
| `quit` | Exit the session |
| `clear` | Reset conversation history |
| `history` | View compressed turn summaries |
| `save <name>` | Save a named checkpoint |
| `restore <name>` | Restore a named checkpoint |


## Output Modes

The CLI supports three output modes:

**Streaming (default)** — LLM tokens are streamed to the terminal in real time, with ANSI-styled status messages for module lifecycle, tool calls, and LM invocations.

```bash
python examples/conversational_agent_demo.py --agent examples/agents/demo_agent.yaml
```

**Observer mode (`--no-stream`)** — Full non-streaming observability. Shows the LM prompt fields, response fields, tool calls with arguments and results, and timing information.

```bash
python examples/conversational_agent_demo.py --agent examples/agents/demo_agent.yaml --no-stream
```

**Quiet mode (`--quiet`)** — No observability output. Only the final assistant response is shown.

```bash
python examples/conversational_agent_demo.py --agent examples/agents/demo_agent.yaml --quiet
```


## YAML Agent Definitions

Agents are defined declaratively in YAML files. Each agent is built as a `dspy.ReAct` module wrapped in a `ConversationalAgent` for multi-turn memory.

**For programmatic usage**, use the `create_agent_from_yaml()` function from `dspy.utils` (see [Programmatic Usage](#programmatic-usage) section).

### Schema

```yaml
# Required
signature: "question -> answer"

# Optional
model: "openai/gpt-4o-mini"
tools:
  - path/to/tool.py
subagents:
  - name: agent_name
    description: "When to delegate to this agent"
    agent: ./child_agent.yaml
max_iters: 20           # Max ReAct iterations (default: 20)
compression: concise     # "concise" or "verbose" (default: "concise")
max_context_turns: 20    # Max conversation turns retained (default: 20)
```

| Field | Required | Description |
|---|---|---|
| `signature` | Yes | DSPy signature string, e.g. `"question -> answer"` |
| `model` | No | LiteLLM model identifier. Can be overridden with `--model` on the CLI. |
| `tools` | No | List of paths to Python files containing tool functions. |
| `subagents` | No | List of child agent definitions to expose as tools. |
| `max_iters` | No | Maximum ReAct reasoning iterations. Default: `20`. |
| `compression` | No | Turn compression level: `"concise"` or `"verbose"`. Default: `"concise"`. |
| `max_context_turns` | No | Maximum number of compressed turn summaries to retain. Default: `20`. |

All file paths (`tools`, `subagents[].agent`) are resolved relative to the YAML file's directory.

### Example: Flat Agent

```yaml
# examples/agents/demo_agent.yaml
signature: "question -> answer"
model: "openai/gpt-4o-mini"

tools:
  - ../tools/search_tool.py
  - ../tools/calculator_tool.py
```

```bash
python examples/conversational_agent_demo.py --agent examples/agents/demo_agent.yaml
```


## Writing Tools

A tool is a Python file containing one or more public functions. Each function becomes a tool that the ReAct agent can call. DSPy auto-extracts the tool name from `__name__`, the description from the docstring, and the argument schema from type hints.

### Requirements

- The file must contain at least one public function (no leading underscore).
- Each function must have **type-annotated parameters** and a **return type**.
- Each function must have a **docstring** describing what it does (the agent reads this to decide when to call the tool).

### Example Tool

```python
# examples/tools/search_tool.py
"""Search tool for the demo agent."""


def search(query: str) -> str:
    """Search for information on a topic. Returns a short snippet of relevant information."""
    return f"Search result for '{query}': ..."
```

```python
# examples/tools/calculator_tool.py
"""Calculator tool for the demo agent."""


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, and parentheses."""
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters"
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

If a Python file contains multiple public functions, all of them are loaded as separate tools.


## Nested Agentic Hierarchies (Subagents)

An agent can delegate to child agents by listing them in the `subagents` section. Each subagent is a full `ReAct + ConversationalAgent` (with its own tools and conversation memory), wrapped as a callable tool that the parent agent can invoke like any other tool.

### How It Works

1. The parent agent's YAML references child agent YAML files in `subagents`.
2. Each child YAML is recursively loaded and built into a full agent.
3. The child agent is wrapped in a function with the given `name` and `description`.
4. The parent's ReAct loop sees the child as a tool: `def researcher(request: str) -> str`.
5. When the parent calls the tool, the child agent processes the request independently.

### Subagent Entry Schema

Subagents can be defined in two ways:

**Option 1: External File (original)**

```yaml
subagents:
  - name: researcher          # Tool function name (used by ReAct)
    description: "..."        # When to delegate (read by the agent)
    agent: ./researcher.yaml  # Path to child agent YAML
```

**Option 2: Inline Definition (new)**

```yaml
subagents:
  - name: researcher
    description: "..."
    agent:                    # Inline agent definition
      signature: "question -> answer"
      tools:
        - ../tools/search_tool.py
      max_iters: 10
```

| Field | Required | Description |
|---|---|---|
| `name` | Yes | The tool function name exposed to the parent agent. |
| `description` | Yes | Description the parent agent reads to decide when to delegate. |
| `agent` | Yes | Either a path to a child agent YAML file (string) or an inline agent definition (dict). |

### Example: Orchestrator with Subagents (Separate Files)

**Parent — `orchestrator.yaml`:**

```yaml
# examples/agents/orchestrator.yaml
signature: "question -> answer"
model: "openai/gpt-4o-mini"

subagents:
  - name: researcher
    description: "Delegate research and information-lookup questions to this agent. It can search for facts and general knowledge."
    agent: ./researcher.yaml

  - name: math_expert
    description: "Delegate math, arithmetic, and calculation questions to this agent. It can evaluate mathematical expressions."
    agent: ./math_expert.yaml
```

**Child — `researcher.yaml`:**

```yaml
# examples/agents/researcher.yaml
signature: "question -> answer"

tools:
  - ../tools/search_tool.py
```

**Child — `math_expert.yaml`:**

```yaml
# examples/agents/math_expert.yaml
signature: "question -> answer"

tools:
  - ../tools/calculator_tool.py
```

```bash
python examples/conversational_agent_demo.py --agent examples/agents/orchestrator.yaml
```

The orchestrator has no tools of its own. It routes questions to the appropriate subagent, each of which is a full ReAct agent with its own tools and conversation memory.

### Example: Orchestrator with Inline Subagents

For simpler workflows, you can define subagents directly in the parent file:

**All-in-one — `orchestrator_inline.yaml`:**

```yaml
# examples/agents/orchestrator_inline.yaml
signature: "question -> answer"
model: "openai/gpt-4o-mini"

subagents:
  # Inline subagent definition - no separate file needed!
  - name: researcher
    description: "Delegate research and information-lookup questions to this agent."
    agent:
      signature: "question -> answer"
      tools:
        - ../tools/search_tool.py

  # Another inline subagent
  - name: math_expert
    description: "Delegate math, arithmetic, and calculation questions to this agent."
    agent:
      signature: "question -> answer"
      tools:
        - ../tools/calculator_tool.py
      max_iters: 10  # Can customize settings per subagent
```

```bash
python examples/conversational_agent_demo.py --agent examples/agents/orchestrator_inline.yaml
```

This achieves the same result as separate files, but keeps everything in one place for easier management.

### Mixing Tools and Subagents

A single agent can have both `tools` and `subagents`. All of them appear as callable tools in the parent's ReAct loop:

```yaml
signature: "question -> answer"
model: "openai/gpt-4o-mini"

tools:
  - ../tools/search_tool.py

subagents:
  - name: math_expert
    description: "Delegate math questions"
    agent: ./math_expert.yaml
```

### Mixing File-Based and Inline Subagents

You can also mix both approaches in the same parent agent:

```yaml
signature: "question -> answer"
model: "openai/gpt-4o-mini"

subagents:
  # External file for complex subagent
  - name: researcher
    description: "Research specialist"
    agent: ./researcher.yaml

  # Inline definition for simple subagent
  - name: calculator
    description: "Math specialist"
    agent:
      signature: "question -> answer"
      tools:
        - ../tools/calculator_tool.py
```

This gives you flexibility to:
- Use **inline definitions** for simple, single-purpose subagents
- Use **separate files** for complex subagents that you want to reuse or test independently

### Circular Reference Detection

The loader detects circular agent references (A references B, B references A) and raises a `ValueError` before entering an infinite loop.


## ConversationalAgent Module

The `ConversationalAgent` module wraps any DSPy module to add multi-turn conversational memory. It is the core building block used by the CLI.

### Programmatic Usage

#### Basic ConversationalAgent

```python
import dspy
from dspy.predict.conversational import ConversationalAgent

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

inner = dspy.ChainOfThought("question -> answer")
agent = ConversationalAgent(inner)

# First turn
r1 = agent(question="What is the capital of France?")
print(r1.answer)

# Second turn (agent remembers the first)
r2 = agent(question="What is its population?")
print(r2.answer)

# Manage history
agent.save_checkpoint("after_two")
agent.clear_history()
agent.restore_checkpoint("after_two")
print(agent.get_history())
```

#### Loading from YAML

The `create_agent_from_yaml` convenience function makes it easy to load agents programmatically:

```python
import dspy
from dspy.utils import create_agent_from_yaml

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Load agent from YAML (including all tools and subagents)
agent = create_agent_from_yaml("examples/agents/orchestrator.yaml")

# Use it immediately
result = agent(question="What is 25 * 17?")
print(result.answer)

# Multi-turn conversation works automatically
result = agent(question="What did I just ask you about?")
print(result.answer)
```

For more control over the loading process, use `load_agent_config`:

```python
import dspy
from dspy.utils import load_agent_config
from dspy.predict.conversational import ConversationalAgent

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Load config dict with all tools and subagents resolved
config = load_agent_config("examples/agents/orchestrator.yaml")

# Build the agent with custom settings
inner = dspy.ReAct(
    config["signature"],
    tools=config["tools"],
    max_iters=config["max_iters"],
)

# Create agent with custom compression settings
agent = ConversationalAgent(
    inner,
    compression_level="verbose",  # Override YAML setting
    max_context_turns=50,         # Override YAML setting
)

result = agent(question="Calculate 123 + 456")
print(result.answer)
```

### With ReAct and Tools

```python
def search(query: str) -> str:
    """Search for information."""
    return f"Result for '{query}': ..."

inner = dspy.ReAct("question -> answer", tools=[search])
agent = ConversationalAgent(inner, compression_level="verbose", max_context_turns=10)

result = agent(question="Look up the weather in Tokyo")
print(result.answer)
```

### How Context Compression Works

After each turn, the agent:

1. Extracts the turn's input, output, and any tool usage (ReAct trajectories are captured automatically).
2. Compresses this into a summary using a `ChainOfThought(TurnSummary)` module.
3. Stores the summary and injects all prior summaries as `conversation_context` on the next turn.

The `compression_level` controls summary detail:

- `"concise"` — keeps only the essentials (recommended for most use cases).
- `"verbose"` — preserves more detail from each turn.

Older turns beyond `max_context_turns` are dropped to keep the context window manageable.


## File Layout

```
dspy/
└── utils/
    └── agent_loader.py                # YAML parser and agent builder (core framework)

examples/
├── conversational_agent_demo.py       # CLI entry point
├── agents/
│   ├── demo_agent.yaml                # Flat agent (search + calculator)
│   ├── orchestrator.yaml              # Parent agent with file-based subagents
│   ├── orchestrator_inline.yaml       # Parent agent with inline subagents
│   ├── researcher.yaml                # Child agent (search)
│   └── math_expert.yaml               # Child agent (calculator)
└── tools/
    ├── search_tool.py                 # Search tool function
    └── calculator_tool.py             # Calculator tool function
```

The agent loader is now part of the core DSPy framework and can be imported from `dspy.utils`.
