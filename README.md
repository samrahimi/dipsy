<p align="center">
  <img align="center" src="docs/docs/static/img/dspy_logo.png" width="460px" />
</p>

# dipsy: Declarative Multi-Agent Systems on DSPy

**Documentation:** [dipsy Docs](https://samrahimi.github.io/dipsy/)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dspy?period=monthly)](https://pepy.tech/projects/dspy)

----

**dipsy** is an opinionated fork of [DSPy](https://github.com/stanfordnlp/dspy) that extends the framework with declarative agent definitions and native support for hierarchical multi-agent systems. While DSPy focuses on programming language models through compositional Python code, dipsy adds a configuration-first approach that makes building complex agentic systems more accessible and maintainable.

## What is dipsy?

dipsy enhances DSPy with:

- 🎯 **YAML-based agent definitions** - Define agents declaratively without writing Python classes
- 🪆 **Inline subagent support** - Nest agents within agents for true hierarchical architectures
- 🔧 **Convenience APIs** - `create_agent_from_yaml()` and other helpers for rapid development
- 💬 **ConversationalAgent module** - Built-in multi-turn memory with intelligent context compression
- 🎨 **Streaming support** - Real-time LLM token streaming with rich observability
- 📦 **Agent composability** - Mix declarative YAML definitions with programmatic Python

## Philosophy: dipsy vs DSPy

### DSPy's Philosophy
DSPy pioneered "programming over prompting" - treating LM pipelines as code rather than brittle prompt strings. It provides powerful modules like `ReAct`, `ChainOfThought`, and optimizers like `MIPROv2` for teaching LMs to deliver high-quality outputs. DSPy is intentionally code-first: you write Python classes, compose modules, and use teleprompters to optimize your programs.

### dipsy's Philosophy
dipsy embraces "configuration over code" for agent systems while preserving DSPy's programmatic power. We believe:

1. **Agents should be declarative** - Complex multi-agent hierarchies are easier to reason about in YAML than Python
2. **Composition should be flexible** - Mix declarative definitions with programmatic control as needed
3. **Developer experience matters** - Getting started should be simple; power should be available when needed
4. **Agents are first-class primitives** - Just as DSPy made modules composable, dipsy makes agents composable

dipsy is for developers who want to:
- Rapidly prototype multi-agent systems without boilerplate
- Define agent hierarchies declaratively and iterate quickly
- Build conversational agents with managed memory and context
- Maintain complex agent configurations separate from application logic

**Note:** dipsy is a community fork and is not affiliated with Stanford NLP. For the original DSPy framework and research, see [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy).

## Quick Start

### Installation

```bash
pip install dspy  # Install base DSPy
```

To install the latest dipsy from source:

```bash
pip install git+https://github.com/samrahimi/dipsy.git
```

### Example: YAML Agent with Subagents

**Define an agent in YAML** (`orchestrator.yaml`):

```yaml
signature: "question -> answer"
model: "openai/gpt-4o-mini"

subagents:
  # Inline subagent definition
  - name: researcher
    description: "Delegate research and fact-finding questions"
    agent:
      signature: "question -> answer"
      tools:
        - tools/search_tool.py

  # Another inline subagent
  - name: math_expert
    description: "Delegate math and calculation questions"
    agent:
      signature: "question -> answer"
      tools:
        - tools/calculator_tool.py
```

**Use it in Python:**

```python
import dspy
from dspy.utils import create_agent_from_yaml

# Configure your LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Load agent from YAML (includes all subagents)
agent = create_agent_from_yaml("orchestrator.yaml")

# Multi-turn conversation with automatic memory management
response = agent(question="What is 25 * 17?")
print(response.answer)  # Delegates to math_expert subagent

response = agent(question="What did I just ask you?")
print(response.answer)  # Agent remembers previous context
```

### Example: Programmatic Agent (Pure DSPy)

dipsy is fully backward compatible with DSPy's programmatic approach:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Define tools
def search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}..."

# Build agent programmatically
agent = dspy.ReAct("question -> answer", tools=[search])
result = agent(question="What is quantum computing?")
```

## Key Features

### 1. YAML Agent Definitions

Define agents declaratively with tools and subagents:

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

### 2. Hierarchical Multi-Agent Systems

Nest agents within agents for sophisticated delegation:

```yaml
# Parent orchestrator
subagents:
  - name: analyst
    agent: ./analyst.yaml      # External file

  - name: writer
    agent:                      # Inline definition
      signature: "task -> response"
      tools: [./tools/writer.py]
```

Each subagent is a full `ReAct + ConversationalAgent` with:
- Its own tools and conversation memory
- Independent reasoning loop
- Automatic context compression

### 3. ConversationalAgent Module

Built-in multi-turn memory with intelligent compression:

```python
from dspy.predict.conversational import ConversationalAgent

inner = dspy.ChainOfThought("question -> answer")
agent = ConversationalAgent(inner, max_context_turns=20)

# First turn
agent(question="What is the capital of France?")

# Second turn - agent remembers previous context
agent(question="What is its population?")

# History management
agent.save_checkpoint("checkpoint1")
agent.clear_history()
agent.restore_checkpoint("checkpoint1")
```

### 4. Streaming Support

Real-time token streaming with observability:

```python
import dspy.streaming

# Wrap your agent for streaming
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

## Documentation

**Full documentation:** [samrahimi.github.io/dipsy](https://samrahimi.github.io/dipsy/)

Key sections:
- [YAML Agent Guide](https://samrahimi.github.io/dipsy/) - Complete guide to declarative agents
- [ConversationalAgent API](https://samrahimi.github.io/dipsy/api/modules/ConversationalAgent/) - Multi-turn memory module
- [Examples](https://github.com/samrahimi/dipsy/tree/main/examples) - Working examples in the repo

## Examples

The `examples/` directory includes:

- **`conversational_agent_demo.py`** - Interactive CLI with streaming
- **`agents/orchestrator.yaml`** - Multi-agent orchestrator (file-based subagents)
- **`agents/orchestrator_inline.yaml`** - Multi-agent orchestrator (inline subagents)
- **`agents/demo_agent.yaml`** - Simple agent with tools
- **`tools/`** - Example tool definitions

Run an example:

```bash
python examples/conversational_agent_demo.py --agent examples/agents/orchestrator.yaml
```

## Compatibility

dipsy maintains **100% backward compatibility** with DSPy:
- All DSPy modules work unchanged (`Predict`, `ChainOfThought`, `ReAct`, etc.)
- All optimizers work unchanged (`MIPROv2`, `BootstrapFewShot`, `COPRO`, etc.)
- You can mix YAML agents with programmatic DSPy modules
- Existing DSPy programs run without modification

## When to Use dipsy vs DSPy

**Use dipsy when:**
- Building multi-agent systems with delegation hierarchies
- You want declarative configuration for agents
- Managing conversational agents with memory
- Rapid prototyping and iteration are priorities

**Use vanilla DSPy when:**
- Building single-agent pipelines or classifiers
- You prefer pure Python and compositional code
- Using DSPy optimizers (MIPROv2, GEPA) is central to your workflow
- You want the latest research features from Stanford NLP

**Use both when:**
- You want declarative agents with custom optimization
- Building complex systems that benefit from both approaches

## Contributing

dipsy is a community-driven fork. Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Citation & Reading More

dipsy builds on DSPy's research. If you use dipsy, please cite the original DSPy work:

**[Oct'23] [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)**

Additional DSPy papers:
- **[Jul'25] [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)**
- **[Jun'24] [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695)**
- [Jul'24] [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930)
- [Dec'23] [DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines](https://arxiv.org/abs/2312.13382)
- [Dec'22] [Demonstrate-Search-Predict: Composing Retrieval & Language Models for Knowledge-Intensive NLP](https://arxiv.org/abs/2212.14024.pdf)

For DSPy updates, follow [@DSPyOSS](https://twitter.com/DSPyOSS) on Twitter.

## Credits

dipsy is a fork of [DSPy](https://github.com/stanfordnlp/dspy) by Stanford NLP. We're grateful to the DSPy team for their groundbreaking work on programming language models.

This fork adds agent-centric features while preserving DSPy's core philosophy and research contributions.

## License

Apache 2.0 (same as DSPy)
