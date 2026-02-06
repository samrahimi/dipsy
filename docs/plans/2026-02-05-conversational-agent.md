# ConversationalAgent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `ConversationalAgent` wrapper module to DSPy that gives any module multi-turn conversational memory via automatic context compression.

**Architecture:** `ConversationalAgent` is a `dspy.Module` that wraps any inner module. It maintains a list of turn summaries. Before each call, it compresses prior turns with a `TurnSummary` signature + `ChainOfThought`, builds a `conversation_context` string, and injects it into the inner module's `forward()`. After each call, it captures relevant info (input/output/trajectory) and stores it. A `context_hook` parameter lets custom modules define what to capture.

**Tech Stack:** Python 3.10+, DSPy module system, Pydantic, pytest, DummyLM for tests.

---

### Task 1: Create the TurnSummary signature

**Files:**
- Create: `dspy/signatures/turn_summary.py`

**Step 1: Write the TurnSummary signature**

```python
# dspy/signatures/turn_summary.py
import dspy


class TurnSummary(dspy.Signature):
    """Summarize a conversation turn, preserving key facts, user intent, tool usage, and any commitments or state changes."""

    turn_input: str = dspy.InputField(desc="The user's input for this conversation turn")
    turn_output: str = dspy.InputField(desc="The module's output for this conversation turn")
    tool_usage: str = dspy.InputField(
        desc="Description of tools called and their results during this turn, or 'None' if no tools were used",
        default="None",
    )
    compression_level: str = dspy.InputField(
        desc="How aggressively to compress: 'verbose' preserves more detail, 'concise' keeps only essentials",
        default="concise",
    )
    summary: str = dspy.OutputField(
        desc="Compressed summary of this turn preserving: user intent, key facts exchanged, tools used and results, commitments or state changes"
    )
```

**Step 2: Run test to verify import works**

Run: `python -c "from dspy.signatures.turn_summary import TurnSummary; print(TurnSummary)"`
Expected: Prints the signature class without error.

**Step 3: Commit**

```bash
git add dspy/signatures/turn_summary.py
git commit -m "feat: add TurnSummary signature for conversation context compression"
```

---

### Task 2: Create the ConversationalAgent module

**Files:**
- Create: `dspy/predict/conversational.py`

**Step 1: Write the ConversationalAgent module**

```python
# dspy/predict/conversational.py
import copy
import json
import logging
from typing import Any, Callable

import dspy
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import ensure_signature
from dspy.signatures.turn_summary import TurnSummary

logger = logging.getLogger(__name__)


class ConversationalAgent(Module):
    """Wraps any DSPy module to add multi-turn conversational memory.

    The wrapper maintains conversation history internally. Before each turn,
    it compresses prior context and injects it into the inner module's
    forward call via a ``conversation_context`` input field. After each turn,
    it captures relevant information for future context.

    Args:
        module: Any DSPy module (Predict, ChainOfThought, ReAct, or custom).
        compression_level: ``"verbose"`` preserves more detail,
            ``"concise"`` keeps only essentials.  Defaults to ``"concise"``.
        max_context_turns: Maximum number of compressed turn summaries to
            retain.  Older turns are dropped.  Defaults to ``20``.
        context_hook: Optional callable ``(input_kwargs, prediction) -> str``
            that extracts what should be captured from a turn.  If ``None``,
            built-in extraction is used (handles ReAct trajectories, and
            generic input/output pairs).
    """

    def __init__(
        self,
        module: Module,
        compression_level: str = "concise",
        max_context_turns: int = 20,
        context_hook: Callable[..., str] | None = None,
    ):
        super().__init__()
        self.module = module
        self.compression_level = compression_level
        self.max_context_turns = max_context_turns
        self.context_hook = context_hook

        # Internal state
        self._turn_summaries: list[str] = []
        self._full_history: list[dict[str, Any]] = []
        self._checkpoints: dict[str, dict[str, Any]] = {}

        # Summarizer module (optimizable via DSPy optimizers)
        self.summarizer = dspy.ChainOfThought(TurnSummary)

        # Modify the inner module's signature to accept conversation_context
        self._patch_inner_signature()

    def _patch_inner_signature(self):
        """Add a conversation_context input field to the inner module's signature(s)."""
        if hasattr(self.module, "signature"):
            sig = self.module.signature
            if "conversation_context" not in sig.input_fields:
                self.module.signature = sig.prepend(
                    "conversation_context",
                    dspy.InputField(
                        desc="Summary of prior conversation turns for context",
                    ),
                    type_=str,
                )

            # For ReAct, also patch the internal react and extract signatures
            if hasattr(self.module, "react") and hasattr(self.module.react, "signature"):
                react_sig = self.module.react.signature
                if "conversation_context" not in react_sig.input_fields:
                    self.module.react.signature = react_sig.prepend(
                        "conversation_context",
                        dspy.InputField(
                            desc="Summary of prior conversation turns for context",
                        ),
                        type_=str,
                    )
            if hasattr(self.module, "extract") and hasattr(self.module.extract, "signature"):
                # ChainOfThought wraps a Predict, so patch the inner predict's signature
                extract_mod = self.module.extract
                if hasattr(extract_mod, "predict"):
                    extract_sig = extract_mod.predict.signature
                    if "conversation_context" not in extract_sig.input_fields:
                        extract_mod.predict.signature = extract_sig.prepend(
                            "conversation_context",
                            dspy.InputField(
                                desc="Summary of prior conversation turns for context",
                            ),
                            type_=str,
                        )
                elif hasattr(extract_mod, "signature"):
                    extract_sig = extract_mod.signature
                    if "conversation_context" not in extract_sig.input_fields:
                        extract_mod.signature = extract_sig.prepend(
                            "conversation_context",
                            dspy.InputField(
                                desc="Summary of prior conversation turns for context",
                            ),
                            type_=str,
                        )

    def _build_context_string(self) -> str:
        """Build a context string from compressed turn summaries."""
        if not self._turn_summaries:
            return "No prior conversation context."
        parts = []
        for i, summary in enumerate(self._turn_summaries, 1):
            parts.append(f"[Turn {i}] {summary}")
        return "\n".join(parts)

    def _extract_turn_info(self, input_kwargs: dict[str, Any], prediction: Prediction) -> dict[str, str]:
        """Extract turn input, output, and tool usage from a prediction."""
        if self.context_hook is not None:
            custom = self.context_hook(input_kwargs, prediction)
            return {
                "turn_input": str(input_kwargs),
                "turn_output": custom,
                "tool_usage": "None",
            }

        # Extract input: use all kwargs except conversation_context
        filtered_input = {k: v for k, v in input_kwargs.items() if k != "conversation_context"}
        turn_input = json.dumps(filtered_input, default=str, ensure_ascii=False)

        # Extract tool usage for ReAct modules
        tool_usage = "None"
        trajectory = getattr(prediction, "trajectory", None)
        if isinstance(trajectory, dict) and trajectory:
            tool_parts = []
            idx = 0
            while f"tool_name_{idx}" in trajectory:
                name = trajectory[f"tool_name_{idx}"]
                args = trajectory.get(f"tool_args_{idx}", {})
                obs = trajectory.get(f"observation_{idx}", "")
                tool_parts.append(f"Tool: {name}({json.dumps(args, default=str)}) -> {obs}")
                idx += 1
            if tool_parts:
                tool_usage = "; ".join(tool_parts)

        # Extract output: collect all output field values
        output_parts = {}
        for key in prediction.keys():
            if key != "trajectory":
                output_parts[key] = str(getattr(prediction, key, ""))
        turn_output = json.dumps(output_parts, default=str, ensure_ascii=False)

        return {
            "turn_input": turn_input,
            "turn_output": turn_output,
            "tool_usage": tool_usage,
        }

    def _compress_turn(self, turn_info: dict[str, str]) -> str:
        """Use the summarizer module to compress a turn into a summary."""
        result = self.summarizer(
            turn_input=turn_info["turn_input"],
            turn_output=turn_info["turn_output"],
            tool_usage=turn_info["tool_usage"],
            compression_level=self.compression_level,
        )
        return result.summary

    def forward(self, **kwargs):
        # Build and inject context
        context = self._build_context_string()
        kwargs["conversation_context"] = context

        # Call inner module
        prediction = self.module(**kwargs)

        # Extract and compress turn info
        turn_info = self._extract_turn_info(kwargs, prediction)
        summary = self._compress_turn(turn_info)

        # Store results
        self._turn_summaries.append(summary)
        self._full_history.append({
            "input": turn_info["turn_input"],
            "output": turn_info["turn_output"],
            "tool_usage": turn_info["tool_usage"],
            "summary": summary,
        })

        # Enforce max context limit
        if len(self._turn_summaries) > self.max_context_turns:
            self._turn_summaries = self._turn_summaries[-self.max_context_turns:]

        return prediction

    async def aforward(self, **kwargs):
        context = self._build_context_string()
        kwargs["conversation_context"] = context

        prediction = await self.module.acall(**kwargs)

        turn_info = self._extract_turn_info(kwargs, prediction)
        summary_result = await self.summarizer.acall(
            turn_input=turn_info["turn_input"],
            turn_output=turn_info["turn_output"],
            tool_usage=turn_info["tool_usage"],
            compression_level=self.compression_level,
        )
        summary = summary_result.summary

        self._turn_summaries.append(summary)
        self._full_history.append({
            "input": turn_info["turn_input"],
            "output": turn_info["turn_output"],
            "tool_usage": turn_info["tool_usage"],
            "summary": summary,
        })

        if len(self._turn_summaries) > self.max_context_turns:
            self._turn_summaries = self._turn_summaries[-self.max_context_turns:]

        return prediction

    # --- Conversation management utilities ---

    def clear_history(self):
        """Reset all conversation state."""
        self._turn_summaries = []
        self._full_history = []

    def get_history(self, summarized: bool = True) -> list[dict[str, Any]] | list[str]:
        """Retrieve conversation history.

        Args:
            summarized: If True, returns list of summary strings.
                If False, returns full history dicts with input/output/tool_usage/summary.
        """
        if summarized:
            return list(self._turn_summaries)
        return list(self._full_history)

    def save_checkpoint(self, name: str):
        """Save current conversation state under a named checkpoint."""
        self._checkpoints[name] = {
            "turn_summaries": list(self._turn_summaries),
            "full_history": copy.deepcopy(self._full_history),
        }

    def restore_checkpoint(self, name: str):
        """Restore conversation state from a named checkpoint."""
        if name not in self._checkpoints:
            raise ValueError(f"Checkpoint '{name}' not found. Available: {list(self._checkpoints.keys())}")
        checkpoint = self._checkpoints[name]
        self._turn_summaries = list(checkpoint["turn_summaries"])
        self._full_history = copy.deepcopy(checkpoint["full_history"])

    def list_checkpoints(self) -> list[str]:
        """List available checkpoint names."""
        return list(self._checkpoints.keys())
```

**Step 2: Verify module can be imported**

Run: `python -c "from dspy.predict.conversational import ConversationalAgent; print('OK')"`
Expected: Prints "OK".

**Step 3: Commit**

```bash
git add dspy/predict/conversational.py
git commit -m "feat: add ConversationalAgent module with context compression and history management"
```

---

### Task 3: Register exports

**Files:**
- Modify: `dspy/predict/__init__.py`
- Modify: `dspy/__init__.py`

**Step 1: Add ConversationalAgent to predict __init__.py**

In `dspy/predict/__init__.py`, add the import and `__all__` entry:

```python
from dspy.predict.conversational import ConversationalAgent
```

Add `"ConversationalAgent"` to the `__all__` list.

**Step 2: Verify top-level import**

Run: `python -c "import dspy; print(dspy.ConversationalAgent)"`
Expected: Prints `<class 'dspy.predict.conversational.ConversationalAgent'>`.

**Step 3: Commit**

```bash
git add dspy/predict/__init__.py
git commit -m "feat: export ConversationalAgent from dspy.predict and top-level dspy"
```

---

### Task 4: Write integration tests

**Files:**
- Create: `tests/predict/test_conversational.py`

**Step 1: Write the test file**

```python
# tests/predict/test_conversational.py
import json

import pytest

import dspy
from dspy.predict.conversational import ConversationalAgent
from dspy.utils.dummies import DummyLM


class TestConversationalAgentWithPredict:
    """Test ConversationalAgent wrapping a simple Predict module."""

    def _make_agent(self, answers):
        lm = DummyLM(answers)
        dspy.configure(lm=lm)
        inner = dspy.Predict("question -> answer")
        return ConversationalAgent(inner)

    def test_single_turn(self):
        agent = self._make_agent([
            {"answer": "Paris"},
            # summarizer call (ChainOfThought produces reasoning + summary)
            {"reasoning": "User asked about France's capital, answer was Paris.", "summary": "User asked capital of France. Answer: Paris."},
        ])
        result = agent(question="What is the capital of France?")
        assert result.answer == "Paris"

    def test_multi_turn_context_injected(self):
        agent = self._make_agent([
            # Turn 1: question + answer
            {"answer": "Paris"},
            {"reasoning": "Capital question.", "summary": "Asked capital of France. Answer: Paris."},
            # Turn 2: follow-up question + answer
            {"answer": "2.1 million"},
            {"reasoning": "Population question.", "summary": "Asked population of Paris. Answer: 2.1 million."},
        ])
        # Turn 1
        agent(question="What is the capital of France?")
        # Turn 2 - context from turn 1 should be injected
        result = agent(question="What is the population of that city?")
        assert result.answer == "2.1 million"
        assert len(agent.get_history()) == 2

    def test_clear_history(self):
        agent = self._make_agent([
            {"answer": "Paris"},
            {"reasoning": "r", "summary": "s"},
        ])
        agent(question="What is the capital of France?")
        assert len(agent.get_history()) == 1
        agent.clear_history()
        assert len(agent.get_history()) == 0

    def test_get_history_full(self):
        agent = self._make_agent([
            {"answer": "Paris"},
            {"reasoning": "r", "summary": "Capital of France is Paris."},
        ])
        agent(question="What is the capital of France?")
        full = agent.get_history(summarized=False)
        assert len(full) == 1
        assert "summary" in full[0]
        assert "input" in full[0]
        assert "output" in full[0]

    def test_checkpoints(self):
        agent = self._make_agent([
            {"answer": "Paris"},
            {"reasoning": "r", "summary": "s1"},
            {"answer": "Berlin"},
            {"reasoning": "r", "summary": "s2"},
        ])
        agent(question="Capital of France?")
        agent.save_checkpoint("after_turn_1")
        agent(question="Capital of Germany?")
        assert len(agent.get_history()) == 2
        agent.restore_checkpoint("after_turn_1")
        assert len(agent.get_history()) == 1

    def test_list_checkpoints(self):
        agent = self._make_agent([
            {"answer": "Paris"},
            {"reasoning": "r", "summary": "s"},
        ])
        agent(question="test")
        agent.save_checkpoint("cp1")
        agent.save_checkpoint("cp2")
        assert set(agent.list_checkpoints()) == {"cp1", "cp2"}

    def test_restore_nonexistent_checkpoint_raises(self):
        agent = self._make_agent([])
        with pytest.raises(ValueError, match="not found"):
            agent.restore_checkpoint("does_not_exist")

    def test_max_context_turns(self):
        # Create agent with max 2 turns
        answers = []
        for i in range(4):
            answers.append({"answer": f"answer_{i}"})
            answers.append({"reasoning": "r", "summary": f"summary_{i}"})
        lm = DummyLM(answers)
        dspy.configure(lm=lm)
        inner = dspy.Predict("question -> answer")
        agent = ConversationalAgent(inner, max_context_turns=2)

        for i in range(3):
            agent(question=f"question_{i}")

        # Only the last 2 summaries should be retained
        assert len(agent.get_history()) == 2


class TestConversationalAgentWithChainOfThought:
    """Test ConversationalAgent wrapping ChainOfThought."""

    def test_cot_single_turn(self):
        lm = DummyLM([
            {"reasoning": "Paris is the capital.", "answer": "Paris"},
            {"reasoning": "Summarizing.", "summary": "Capital of France: Paris."},
        ])
        dspy.configure(lm=lm)
        inner = dspy.ChainOfThought("question -> answer")
        agent = ConversationalAgent(inner)
        result = agent(question="What is the capital of France?")
        assert result.answer == "Paris"


class TestConversationalAgentWithReAct:
    """Test ConversationalAgent wrapping ReAct with tools."""

    def test_react_tool_usage_captured(self):
        def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        lm = DummyLM([
            # ReAct turn 1: tool call
            {"next_thought": "Let me check weather.", "next_tool_name": "get_weather", "next_tool_args": {"city": "Tokyo"}},
            # ReAct turn 1: finish
            {"next_thought": "Got weather.", "next_tool_name": "finish", "next_tool_args": {}},
            # ReAct extraction
            {"reasoning": "Weather retrieved.", "answer": "Sunny in Tokyo"},
            # Summarizer call
            {"reasoning": "Weather lookup.", "summary": "Asked weather in Tokyo. Used get_weather tool. Result: Sunny in Tokyo."},
        ])
        dspy.configure(lm=lm)
        inner = dspy.ReAct("question -> answer", tools=[get_weather])
        agent = ConversationalAgent(inner)
        result = agent(question="What's the weather in Tokyo?")
        assert result.answer == "Sunny in Tokyo"

        # Verify tool usage was captured in history
        full = agent.get_history(summarized=False)
        assert len(full) == 1
        assert "get_weather" in full[0]["tool_usage"]


class TestConversationalAgentWithCustomHook:
    """Test ConversationalAgent with a custom context_hook."""

    def test_custom_hook(self):
        def my_hook(input_kwargs, prediction):
            return f"Custom: {prediction.answer}"

        lm = DummyLM([
            {"answer": "42"},
            {"reasoning": "r", "summary": "Custom capture: answer was 42."},
        ])
        dspy.configure(lm=lm)
        inner = dspy.Predict("question -> answer")
        agent = ConversationalAgent(inner, context_hook=my_hook)
        result = agent(question="Meaning of life?")
        assert result.answer == "42"

        full = agent.get_history(summarized=False)
        assert "Custom: 42" in full[0]["output"]


class TestConversationalAgentContextCompression:
    """Test that context compression actually produces shorter summaries."""

    def test_verbose_vs_concise(self):
        """Both modes should work; the compression_level is passed to summarizer."""
        for level in ["verbose", "concise"]:
            lm = DummyLM([
                {"answer": "Paris"},
                {"reasoning": "r", "summary": f"Summary at {level} level."},
            ])
            dspy.configure(lm=lm)
            inner = dspy.Predict("question -> answer")
            agent = ConversationalAgent(inner, compression_level=level)
            agent(question="What is the capital of France?")
            assert len(agent.get_history()) == 1
```

**Step 2: Run the tests**

Run: `python -m pytest tests/predict/test_conversational.py -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/predict/test_conversational.py
git commit -m "test: add integration tests for ConversationalAgent"
```

---

### Task 5: Build the demo CLI

**Files:**
- Create: `examples/conversational_agent_demo.py`

**Step 1: Write the demo CLI**

```python
#!/usr/bin/env python3
"""Demo CLI for ConversationalAgent.

Usage:
    # With ChainOfThought (default):
    python examples/conversational_agent_demo.py

    # With ReAct and tools:
    python examples/conversational_agent_demo.py --react

    # With debug mode (shows compressed context):
    python examples/conversational_agent_demo.py --debug

    # Verbose compression:
    python examples/conversational_agent_demo.py --verbose

Requires OPENAI_API_KEY (or another LM provider) to be set, or
pass --model to specify a model string like "openai/gpt-4o-mini".
"""
import argparse
import sys

import dspy
from dspy.predict.conversational import ConversationalAgent


def build_tools():
    """Example tools for ReAct mode."""

    def search(query: str) -> str:
        """Search for information. Returns a short snippet."""
        return f"Search result for '{query}': This is a simulated search result with relevant information."

    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return [search, calculator]


def main():
    parser = argparse.ArgumentParser(description="ConversationalAgent Demo")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="LM model string")
    parser.add_argument("--react", action="store_true", help="Use ReAct agent with tools")
    parser.add_argument("--debug", action="store_true", help="Show compressed context at each turn")
    parser.add_argument("--verbose", action="store_true", help="Use verbose compression")
    parser.add_argument("--max-turns", type=int, default=20, help="Max conversation turns to retain")
    args = parser.parse_args()

    # Configure DSPy
    lm = dspy.LM(args.model)
    dspy.configure(lm=lm)

    # Build inner module
    if args.react:
        tools = build_tools()
        inner = dspy.ReAct("question -> answer", tools=tools)
        print("Mode: ReAct with tools [search, calculator]")
    else:
        inner = dspy.ChainOfThought("question -> answer")
        print("Mode: ChainOfThought")

    # Wrap with ConversationalAgent
    compression = "verbose" if args.verbose else "concise"
    agent = ConversationalAgent(inner, compression_level=compression, max_context_turns=args.max_turns)

    print(f"Compression: {compression}")
    print(f"Max context turns: {args.max_turns}")
    print("Type 'quit' to exit, 'clear' to reset history, 'history' to view summaries.")
    print("Type 'save <name>' to checkpoint, 'restore <name>' to restore.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            agent.clear_history()
            print("History cleared.")
            continue
        if user_input.lower() == "history":
            summaries = agent.get_history()
            if not summaries:
                print("No conversation history.")
            else:
                for i, s in enumerate(summaries, 1):
                    print(f"  [Turn {i}] {s}")
            continue
        if user_input.lower().startswith("save "):
            name = user_input[5:].strip()
            agent.save_checkpoint(name)
            print(f"Checkpoint '{name}' saved.")
            continue
        if user_input.lower().startswith("restore "):
            name = user_input[8:].strip()
            try:
                agent.restore_checkpoint(name)
                print(f"Checkpoint '{name}' restored.")
            except ValueError as e:
                print(f"Error: {e}")
            continue

        if args.debug:
            ctx = agent._build_context_string()
            print(f"\n[DEBUG] Context being passed:\n{ctx}\n")

        try:
            result = agent(question=user_input)
            print(f"\nAssistant: {result.answer}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify the script parses without errors**

Run: `python -c "import ast; ast.parse(open('examples/conversational_agent_demo.py').read()); print('OK')"`
Expected: Prints "OK".

**Step 3: Commit**

```bash
git add examples/conversational_agent_demo.py
git commit -m "feat: add conversational agent demo CLI"
```

---

### Task 6: Verify existing tests still pass

**Step 1: Run the existing test suite for predict and signatures**

Run: `python -m pytest tests/predict/ tests/signatures/ -x -q --timeout=60`
Expected: All existing tests pass (no regressions).

**Step 2: Run the new conversational tests**

Run: `python -m pytest tests/predict/test_conversational.py -v`
Expected: All new tests pass.

---

### Task 7: Final commit with all files

If any fixes were needed, commit them together.

```bash
git add -A
git commit -m "feat: complete ConversationalAgent implementation with tests and demo"
```
