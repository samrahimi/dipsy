import copy
import json
import logging
from typing import Any, Callable

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.field import InputField
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
        if compression_level not in ("verbose", "concise"):
            raise ValueError(f"compression_level must be 'verbose' or 'concise', got '{compression_level}'")
        self.module = module
        self.compression_level = compression_level
        self.max_context_turns = max_context_turns
        self.context_hook = context_hook

        # Internal state
        self._turn_summaries: list[str] = []
        self._full_history: list[dict[str, Any]] = []
        self._checkpoints: dict[str, dict[str, Any]] = {}

        # Summarizer module (optimizable via DSPy optimizers)
        self.summarizer = ChainOfThought(TurnSummary)

        # Modify the inner module's signature to accept conversation_context
        self._patch_inner_signature()

    @staticmethod
    def _prepend_context_field(sig, context_field):
        """Prepend conversation_context to a signature if not already present."""
        if "conversation_context" not in sig.input_fields:
            return sig.prepend("conversation_context", context_field, type_=str)
        return sig

    def _patch_inner_signature(self):
        """Add a conversation_context input field to the inner module's signature(s)."""
        context_field = InputField(desc="Summary of prior conversation turns for context")
        patched = False

        # Case 1: Module has a direct .signature (Predict, ReAct)
        if hasattr(self.module, "signature"):
            self.module.signature = self._prepend_context_field(
                self.module.signature, context_field,
            )
            patched = True

        # Case 2: Module wraps a Predict via .predict (ChainOfThought)
        if hasattr(self.module, "predict") and hasattr(self.module.predict, "signature"):
            self.module.predict.signature = self._prepend_context_field(
                self.module.predict.signature, context_field,
            )
            patched = True

        # Case 3: ReAct internals — patch .react and .extract sub-modules
        if hasattr(self.module, "react") and hasattr(self.module.react, "signature"):
            self.module.react.signature = self._prepend_context_field(
                self.module.react.signature, context_field,
            )
        if hasattr(self.module, "extract"):
            extract_mod = self.module.extract
            if hasattr(extract_mod, "predict") and hasattr(extract_mod.predict, "signature"):
                extract_mod.predict.signature = self._prepend_context_field(
                    extract_mod.predict.signature, context_field,
                )
            elif hasattr(extract_mod, "signature"):
                extract_mod.signature = self._prepend_context_field(
                    extract_mod.signature, context_field,
                )

        if not patched:
            logger.warning(
                "Inner module %s has no 'signature' or 'predict.signature' attribute. "
                "conversation_context will be passed as a keyword argument "
                "but the module may not use it.",
                type(self.module).__name__,
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
        # Always filter out the synthetic conversation_context field
        filtered_input = {k: v for k, v in input_kwargs.items() if k != "conversation_context"}

        if self.context_hook is not None:
            custom = self.context_hook(filtered_input, prediction)
            return {
                "turn_input": json.dumps(filtered_input, default=str, ensure_ascii=False),
                "turn_output": custom,
                "tool_usage": "None",
            }
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
        self._truncate_history()

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

        # Enforce max context limit
        self._truncate_history()

        return prediction

    def _truncate_history(self):
        """Truncate both turn summaries and full history to max_context_turns."""
        if len(self._turn_summaries) > self.max_context_turns:
            self._turn_summaries = self._turn_summaries[-self.max_context_turns:]
            self._full_history = self._full_history[-self.max_context_turns:]

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
