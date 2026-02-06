# Conversational Agents

The `ConversationalAgent` module wraps any DSPy module to add multi-turn conversational memory with intelligent context compression.

## Overview

Conversational agents maintain conversation history across turns and automatically compress older context to manage the LM's context window. This allows building chatbots, interactive assistants, and stateful agents that remember previous interactions.

## Quick Example

```python
import dspy
from dspy.predict.conversational import ConversationalAgent

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Wrap any DSPy module
inner = dspy.ChainOfThought("question -> answer")
agent = ConversationalAgent(inner)

# First turn
r1 = agent(question="What is the capital of France?")
print(r1.answer)  # "Paris"

# Second turn - agent remembers the first question
r2 = agent(question="What is its population?")
print(r2.answer)  # Agent knows "its" refers to Paris
```

## How It Works

After each turn, the `ConversationalAgent`:

1. Extracts the turn's input, output, and any tool usage
2. Compresses this into a summary using `ChainOfThought(TurnSummary)`
3. Stores the summary
4. Injects all prior summaries as `conversation_context` on the next turn

Older turns beyond `max_context_turns` are automatically dropped.

## Constructor Parameters

```python
ConversationalAgent(
    inner_module,
    compression_level="concise",
    max_context_turns=20
)
```

**Parameters:**

- **`inner_module`** (dspy.Module): The DSPy module to wrap (e.g., `ChainOfThought`, `ReAct`, `Predict`)
- **`compression_level`** (str): Summary detail level
  - `"concise"` - Keep only essentials (recommended)
  - `"verbose"` - Preserve more detail from each turn
- **`max_context_turns`** (int): Maximum conversation turns to retain (default: 20)

## Usage with ReAct and Tools

```python
def search(query: str) -> str:
    """Search for information."""
    return f"Result for '{query}': ..."

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Create ReAct agent with tools
inner = dspy.ReAct("question -> answer", tools=[search, calculator])

# Wrap with conversational memory
agent = ConversationalAgent(inner, compression_level="verbose", max_context_turns=10)

# Use the agent
result = agent(question="What is 25 * 17?")
print(result.answer)

# Agent remembers context
result = agent(question="Can you search for information about that number?")
print(result.answer)  # Knows "that number" is 425
```

## History Management

### View History

```python
summaries = agent.get_history()
for i, summary in enumerate(summaries, 1):
    print(f"Turn {i}: {summary}")
```

### Clear History

```python
agent.clear_history()
```

### Save/Restore Checkpoints

```python
# Save current state
agent.save_checkpoint("after_greeting")

# Do more work
agent(question="Tell me about quantum computing")
agent(question="What are its applications?")

# Restore previous state
agent.restore_checkpoint("after_greeting")

# History is back to the saved point
print(agent.get_history())
```

## Context Compression

The agent uses a `TurnSummary` signature to compress each turn:

```python
class TurnSummary(dspy.Signature):
    """Compress a conversation turn into a concise summary."""

    conversation_context: str = dspy.InputField(
        desc="Previous conversation summaries"
    )
    user_input: str = dspy.InputField(
        desc="What the user said this turn"
    )
    assistant_output: str = dspy.InputField(
        desc="What the assistant responded"
    )
    tool_usage: str = dspy.InputField(
        desc="Tools used and their results"
    )

    summary: str = dspy.OutputField(
        desc="Concise summary of this turn"
    )
```

### Compression Levels

**Concise (default):**
- Focuses on key facts and decisions
- Drops minor details
- Best for long conversations

**Verbose:**
- Preserves more context per turn
- Better for complex reasoning chains
- Uses more context window

## With YAML Agents

Specify compression settings in YAML:

```yaml
signature: "question -> answer"
model: "openai/gpt-4o-mini"

tools:
  - tools/search_tool.py

compression: verbose
max_context_turns: 30
```

Load with `create_agent_from_yaml()`:

```python
from dspy.utils import create_agent_from_yaml

agent = create_agent_from_yaml("agent.yaml")
# Compression settings are automatically applied
```

## Streaming Support

Conversational agents work with streaming:

```python
import dspy.streaming

# Set up streaming
streamed_agent = dspy.streamify(
    agent,
    stream_listeners=[...],
    status_message_provider=status_provider
)

# Stream tokens in real-time
for chunk in streamed_agent(question="Explain quantum computing"):
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk.chunk, end="", flush=True)
```

See `examples/conversational_agent_demo.py` for a complete streaming example.

## Debugging Context

To see what context is being passed to the LM:

```python
# Access the internal context string
context = agent._build_context_string()
print("Context being passed to LM:")
print(context)
```

Or use the CLI with `--debug`:

```bash
python examples/conversational_agent_demo.py \
    --agent examples/agents/demo_agent.yaml \
    --debug
```

## API Reference

### `ConversationalAgent` Class

```python
class ConversationalAgent(dspy.Module):
    def __init__(
        self,
        inner_module: dspy.Module,
        compression_level: str = "concise",
        max_context_turns: int = 20
    )
```

**Methods:**

- **`forward(**kwargs)`**: Process a turn and return the result
- **`get_history()`**: Return list of turn summaries
- **`clear_history()`**: Reset conversation history
- **`save_checkpoint(name: str)`**: Save current history state
- **`restore_checkpoint(name: str)`**: Restore saved history state

## Examples

See:
- `examples/conversational_agent_demo.py` - Interactive CLI
- `tests/predict/test_conversational.py` - Unit tests
- [YAML Agents Guide](yaml-agents.md) - Using with YAML definitions

## Next Steps

- Build [Multi-Agent Systems](multi-agent-systems.md) with hierarchical subagents
- Learn about [YAML Agent Definitions](yaml-agents.md)
- Explore [Streaming](../../tutorials/streaming/index.md) for real-time responses
