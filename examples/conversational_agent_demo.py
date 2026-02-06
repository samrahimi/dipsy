#!/usr/bin/env python3
"""Demo CLI for ConversationalAgent.

Usage:
    # With a YAML agent definition (recommended):
    python examples/conversational_agent_demo.py --agent examples/agents/demo_agent.yaml

    # Override YAML model from CLI:
    python examples/conversational_agent_demo.py \\
        --agent examples/agents/demo_agent.yaml --model openai/gpt-4o

    # Basic with streaming (uses OPENAI_API_KEY from env):
    python examples/conversational_agent_demo.py --model openai/gpt-4o-mini

    # With explicit API key and base URL:
    python examples/conversational_agent_demo.py \\
        --model openai/gpt-4o-mini \\
        --api-key sk-... \\
        --base-url https://my-proxy.example.com/v1

    # With ReAct and tools:
    python examples/conversational_agent_demo.py --model openai/gpt-4o-mini --react

    # Disable streaming (non-streaming observer mode):
    python examples/conversational_agent_demo.py --model openai/gpt-4o-mini --no-stream

    # With debug mode (shows compressed context each turn):
    python examples/conversational_agent_demo.py --model openai/gpt-4o-mini --debug

    # Verbose compression:
    python examples/conversational_agent_demo.py --model openai/gpt-4o-mini --verbose
"""
import argparse
import os
import re
import sys
import textwrap
import time

import dspy
import dspy.streaming
from dspy.predict.conversational import ConversationalAgent
from dspy.streaming.messages import StatusMessage, StatusMessageProvider
from dspy.utils.callback import BaseCallback


# -- ANSI helpers -------------------------------------------------------------

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

FIELD_HEADER = re.compile(r"\[\[ ## (\w+) ## \]\]")


def _styled(text: str, *codes: str) -> str:
    prefix = "".join(codes)
    return f"{prefix}{text}{RESET}"


# -- Live observability callback (non-streaming mode) -------------------------

class TerminalObserver(BaseCallback):
    """Prints live, detailed output of every DSPy operation to the terminal.

    Shows which modules run, the raw LLM input/output, tool calls and results,
    and timing information.  Used in --no-stream mode.
    """

    def __init__(self):
        self._t0 = None
        self._module_depth = 0

    def reset(self):
        self._t0 = time.monotonic()
        self._module_depth = 0

    def _elapsed(self) -> str:
        if self._t0 is None:
            return ""
        return f"{time.monotonic() - self._t0:.1f}s"

    def _prefix(self) -> str:
        return _styled("  " + "│ " * self._module_depth, DIM)

    # -- Module events --------------------------------------------------------

    def on_module_start(self, call_id, instance, inputs):
        name = type(instance).__name__
        p = self._prefix()

        if name == "ConversationalAgent":
            return

        # Identify the module
        sig_str = ""
        sig = None
        if hasattr(instance, "predict") and hasattr(instance.predict, "signature"):
            sig = instance.predict.signature
        elif hasattr(instance, "signature"):
            sig = instance.signature
        if sig:
            ins = ", ".join(sig.input_fields.keys())
            outs = ", ".join(sig.output_fields.keys())
            sig_str = f" ({ins} -> {outs})"

        print(f"{p}{_styled('┌ ' + name + sig_str, BOLD, CYAN)}", flush=True)
        self._module_depth += 1

    def on_module_end(self, call_id, outputs, exception=None):
        if self._module_depth > 0:
            self._module_depth -= 1
        p = self._prefix()
        if exception:
            print(f"{p}{_styled('└ error: ' + str(exception), BOLD, YELLOW)}", flush=True)
        else:
            # Show the prediction fields
            if hasattr(outputs, '_store'):
                for k, v in outputs._store.items():
                    val = str(v)
                    if len(val) > 200:
                        val = val[:200] + "..."
                    print(f"{p}{_styled('│ ', DIM)}{_styled(k, BOLD)}: {val}", flush=True)
            print(f"{p}{_styled('└ done', DIM)} {_styled('(' + self._elapsed() + ')', DIM)}", flush=True)

    # -- LM events ------------------------------------------------------------

    def on_lm_start(self, call_id, instance, inputs):
        p = self._prefix()
        model = getattr(instance, "model", "?")
        print(f"{p}{_styled('▶ LM call', YELLOW)} {_styled(model, DIM)}", flush=True)

        # Show the last user message (the actual prompt)
        messages = inputs.get("messages") or []
        if messages:
            last_msg = messages[-1]
            content = last_msg.get("content", "")
            if isinstance(content, str):
                # Extract field headers and their values for a compact view
                fields = FIELD_HEADER.split(content)
                if len(fields) > 1:
                    # fields = ['preamble', 'field1', 'value1', 'field2', 'value2', ...]
                    for i in range(1, len(fields), 2):
                        field_name = fields[i]
                        field_val = fields[i + 1].strip() if i + 1 < len(fields) else ""
                        if len(field_val) > 300:
                            field_val = field_val[:300] + "..."
                        print(f"{p}  {_styled(field_name, DIM, BOLD)}: {_styled(field_val, DIM)}", flush=True)
                else:
                    preview = content[:300] + ("..." if len(content) > 300 else "")
                    print(f"{p}  {_styled(preview, DIM)}", flush=True)

    def on_lm_end(self, call_id, outputs, exception=None):
        p = self._prefix()
        if exception:
            print(f"{p}{_styled('✗ LM error: ' + str(exception), YELLOW)}", flush=True)
            return

        # outputs is a list of strings or dicts
        if not outputs:
            return

        for output in outputs:
            if isinstance(output, str):
                # Parse the field-header format
                fields = FIELD_HEADER.split(output)
                if len(fields) > 1:
                    for i in range(1, len(fields), 2):
                        field_name = fields[i]
                        field_val = fields[i + 1].strip() if i + 1 < len(fields) else ""
                        if len(field_val) > 500:
                            field_val = field_val[:500] + "..."
                        print(f"{p}{_styled('◀ ', GREEN)}{_styled(field_name, BOLD, GREEN)}: {field_val}", flush=True)
                else:
                    preview = output[:500] + ("..." if len(output) > 500 else "")
                    print(f"{p}{_styled('◀ ', GREEN)}{preview}", flush=True)
            elif isinstance(output, dict):
                text = output.get("text", "")
                reasoning = output.get("reasoning_content", "")
                if reasoning:
                    preview = reasoning[:300] + ("..." if len(reasoning) > 300 else "")
                    print(f"{p}{_styled('◀ reasoning', MAGENTA)}: {preview}", flush=True)
                if text:
                    fields = FIELD_HEADER.split(text)
                    if len(fields) > 1:
                        for i in range(1, len(fields), 2):
                            field_name = fields[i]
                            field_val = fields[i + 1].strip() if i + 1 < len(fields) else ""
                            if len(field_val) > 500:
                                field_val = field_val[:500] + "..."
                            print(f"{p}{_styled('◀ ', GREEN)}{_styled(field_name, BOLD, GREEN)}: {field_val}", flush=True)
                    else:
                        preview = text[:500] + ("..." if len(text) > 500 else "")
                        print(f"{p}{_styled('◀ ', GREEN)}{preview}", flush=True)

    # -- Tool events ----------------------------------------------------------

    def on_tool_start(self, call_id, instance, inputs):
        p = self._prefix()
        name = getattr(instance, "name", "?")
        kwargs = inputs.get("kwargs", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        print(f"{p}{_styled('⚙ tool', MAGENTA)} {_styled(name, BOLD)}({args_str})", flush=True)

    def on_tool_end(self, call_id, outputs, exception=None):
        p = self._prefix()
        if exception:
            print(f"{p}{_styled('⚙ tool error: ' + str(exception), YELLOW)}", flush=True)
        else:
            result = str(outputs) if outputs is not None else ""
            if len(result) > 300:
                result = result[:300] + "..."
            print(f"{p}{_styled('⚙ → ', MAGENTA)}{result}", flush=True)


# -- Streaming status provider ------------------------------------------------

class TerminalStatusProvider(StatusMessageProvider):
    """Provides rich ANSI-styled status messages for streaming mode.

    Module hierarchy, tool calls, and LM calls are reported as StatusMessages
    through the streamify pipeline, keeping all output serialized through a
    single thread (the sync generator consumer).
    """

    def __init__(self):
        self._t0 = None
        self._module_stack = []  # None for skipped modules (ConversationalAgent)
        self.stream_prefix = ""  # current prefix for streaming chunks

    def reset(self):
        self._t0 = time.monotonic()
        self._module_stack = []
        self.stream_prefix = ""

    def _elapsed(self) -> str:
        if self._t0 is None:
            return ""
        return f"{time.monotonic() - self._t0:.1f}s"

    def _prefix(self) -> str:
        depth = sum(1 for x in self._module_stack if x is not None)
        return "  " + "│ " * depth

    def module_start_status_message(self, instance, inputs):
        name = type(instance).__name__
        if name == "ConversationalAgent":
            self._module_stack.append(None)
            return None

        p = self._prefix()

        sig_str = ""
        sig = None
        if hasattr(instance, "predict") and hasattr(instance.predict, "signature"):
            sig = instance.predict.signature
        elif hasattr(instance, "signature"):
            sig = instance.signature
        if sig:
            ins = ", ".join(sig.input_fields.keys())
            outs = ", ".join(sig.output_fields.keys())
            sig_str = f" ({ins} -> {outs})"

        self._module_stack.append(name)
        self.stream_prefix = self._prefix()
        return f"{p}{_styled('┌ ' + name + sig_str, BOLD, CYAN)}"

    def module_end_status_message(self, outputs):
        name = self._module_stack.pop() if self._module_stack else None
        if name is None:
            return None
        p = self._prefix()
        self.stream_prefix = p
        return f"{p}{_styled('└ done (' + self._elapsed() + ')', DIM)}"

    def lm_start_status_message(self, instance, inputs):
        model = getattr(instance, "model", "?")
        p = self._prefix()
        self.stream_prefix = p
        return f"{p}{_styled('▶ LM ' + model, YELLOW)}"

    def lm_end_status_message(self, outputs):
        return None  # streamed fields replace LM end output

    def tool_start_status_message(self, instance, inputs):
        name = getattr(instance, "name", "?")
        kwargs = inputs.get("kwargs", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        p = self._prefix()
        return f"{p}{_styled('⚙ tool ', MAGENTA)}{_styled(name, BOLD, MAGENTA)}({args_str})"

    def tool_end_status_message(self, outputs):
        if outputs == "Completed.":
            return None
        result = str(outputs) if outputs is not None else ""
        if len(result) > 300:
            result = result[:300] + "..."
        p = self._prefix()
        return f"{p}{_styled('⚙ → ', MAGENTA)}{result}"


# -- Helpers ------------------------------------------------------------------

def _find_stream_targets(inner_module):
    """Return (predict_instance, output_field_names) for the inner module."""
    if isinstance(inner_module, dspy.ChainOfThought):
        return inner_module.predict, ["reasoning", "answer"]
    elif isinstance(inner_module, dspy.ReAct):
        if hasattr(inner_module, "extract"):
            extract = inner_module.extract
            if hasattr(extract, "predict"):
                return extract.predict, ["reasoning", "answer"]
            return extract, ["answer"]
        return None, []
    elif isinstance(inner_module, dspy.Predict):
        return inner_module, ["answer"]
    else:
        return None, []


# -- Tools for ReAct mode ----------------------------------------------------

def build_tools():
    """Example tools for ReAct mode."""

    def search(query: str) -> str:
        """Search for information. Returns a short snippet."""
        return f"Search result for '{query}': This is a simulated search result with relevant information."

    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return "Error: expression contains disallowed characters"
        try:
            result = eval(expression, {"__builtins__": {}})  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return [search, calculator]


# -- Main --------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_MAX_TURNS = 20


def main():
    parser = argparse.ArgumentParser(description="ConversationalAgent Demo")
    parser.add_argument("--agent", default=None, metavar="YAML_FILE",
                        help="Path to a YAML agent definition file (ReAct with tools from YAML)")
    parser.add_argument("--model", default=None,
                        help=f"LM model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", default=None,
                        help="API key for the LM provider (defaults to env var, e.g. OPENAI_API_KEY)")
    parser.add_argument("--base-url", default=None,
                        help="Base URL for the LM API (for proxies or custom endpoints)")
    parser.add_argument("--react", action="store_true", help="Use ReAct agent with built-in demo tools")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming (use non-streaming observer mode)")
    parser.add_argument("--debug", action="store_true", help="Show compressed context at each turn")
    parser.add_argument("--verbose", action="store_true", help="Use verbose compression")
    parser.add_argument("--max-turns", type=int, default=None,
                        help=f"Max conversation turns to retain (default: {DEFAULT_MAX_TURNS})")
    parser.add_argument("--quiet", action="store_true", help="Disable all observability output")
    args = parser.parse_args()

    if args.agent and args.react:
        parser.error("--agent and --react are mutually exclusive")

    # Load agent config from YAML if provided
    agent_config = None
    if args.agent:
        from dspy.utils import load_agent_config

        try:
            agent_config = load_agent_config(args.agent)
        except (FileNotFoundError, ValueError, ImportError) as exc:
            print(f"Error loading agent definition: {exc}", file=sys.stderr)
            sys.exit(1)

    # Resolve settings: CLI > YAML > defaults
    if agent_config:
        model_id = args.model or agent_config["model"] or DEFAULT_MODEL
        max_turns = args.max_turns if args.max_turns is not None else agent_config["max_context_turns"]
        compression = "verbose" if args.verbose else agent_config["compression"]
    else:
        model_id = args.model or DEFAULT_MODEL
        max_turns = args.max_turns if args.max_turns is not None else DEFAULT_MAX_TURNS
        compression = "verbose" if args.verbose else "concise"

    # Configure DSPy
    lm_kwargs = {}
    if args.api_key:
        lm_kwargs["api_key"] = args.api_key
    if args.base_url:
        lm_kwargs["api_base"] = args.base_url
    lm = dspy.LM(model_id, **lm_kwargs)

    # Set up callbacks based on output mode
    use_streaming = not args.quiet and not args.no_stream
    observer = None
    if args.no_stream and not args.quiet:
        observer = TerminalObserver()
        dspy.configure(lm=lm, callbacks=[observer])
    else:
        dspy.configure(lm=lm)

    print(f"Model: {model_id}")
    if args.base_url:
        print(f"Base URL: {args.base_url}")

    # Build inner module
    if agent_config:
        tools = agent_config["tools"]
        inner = dspy.ReAct(
            agent_config["signature"],
            tools=tools,
            max_iters=agent_config["max_iters"],
        )
        tool_names = [getattr(t, "__name__", "?") for t in tools]
        print(f"Agent: {args.agent}")
        print(f"Mode: ReAct with tools {tool_names}")
    elif args.react:
        tools = build_tools()
        inner = dspy.ReAct("question -> answer", tools=tools)
        print("Mode: ReAct with tools [search, calculator]")
    else:
        inner = dspy.ChainOfThought("question -> answer")
        print("Mode: ChainOfThought")

    # Wrap with ConversationalAgent
    agent = ConversationalAgent(inner, compression_level=compression, max_context_turns=max_turns)

    # Set up streaming
    streamed_agent = None
    status_provider = None
    if use_streaming:
        target_predict, stream_fields = _find_stream_targets(inner)
        stream_listeners = []
        if target_predict:
            stream_listeners = [
                dspy.streaming.StreamListener(
                    signature_field_name=field,
                    predict=target_predict,
                    allow_reuse=True,
                )
                for field in stream_fields
            ]
        status_provider = TerminalStatusProvider()
        streamed_agent = dspy.streamify(
            agent,
            status_message_provider=status_provider,
            stream_listeners=stream_listeners,
            async_streaming=False,
        )

    print(f"Compression: {compression}")
    print(f"Streaming: {'enabled' if use_streaming else 'disabled'}")
    print(f"Max context turns: {max_turns}")
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

        if use_streaming:
            _run_streaming_turn(streamed_agent, user_input, status_provider)
        elif observer:
            _run_observer_turn(agent, user_input, observer)
        else:
            _run_quiet_turn(agent, user_input)


def _run_streaming_turn(streamed_agent, user_input, status_provider):
    """Execute one turn with streaming output."""
    if status_provider:
        status_provider.reset()
    print()

    try:
        result = None
        current_field = None

        for value in streamed_agent(question=user_input):
            if isinstance(value, dspy.streaming.StreamResponse):
                if value.signature_field_name != current_field:
                    if current_field is not None:
                        print(RESET)  # reset color and newline
                    current_field = value.signature_field_name
                    p = status_provider.stream_prefix if status_provider else ""
                    if value.signature_field_name == "reasoning":
                        print(f"{p}{DIM}{MAGENTA}reasoning: ", end="", flush=True)
                    else:
                        print(f"{p}{GREEN}{value.signature_field_name}: ", end="", flush=True)
                print(value.chunk, end="", flush=True)
                if value.is_last_chunk:
                    print(RESET)
                    current_field = None
            elif isinstance(value, StatusMessage):
                if current_field is not None:
                    print(RESET)
                    current_field = None
                print(value.message, flush=True)
            elif isinstance(value, dspy.Prediction):
                if current_field is not None:
                    print(RESET)
                    current_field = None
                result = value

        if result:
            print(f"\n{_styled('Assistant:', BOLD)} {result.answer}")
    except Exception as e:
        if current_field is not None:
            print(RESET)
        print(f"\nError: {e}")


def _run_observer_turn(agent, user_input, observer):
    """Execute one turn with non-streaming TerminalObserver output."""
    observer.reset()
    print()
    try:
        result = agent(question=user_input)
        print(f"\n{_styled('Assistant:', BOLD)} {result.answer}")
    except Exception as e:
        print(f"\nError: {e}")


def _run_quiet_turn(agent, user_input):
    """Execute one turn with no observability output."""
    try:
        result = agent(question=user_input)
        print(f"\n{_styled('Assistant:', BOLD)} {result.answer}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
