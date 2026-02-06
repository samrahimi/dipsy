from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature


class TurnSummary(Signature):
    """Summarize a conversation turn, preserving key facts, user intent, tool usage, and any commitments or state changes."""

    turn_input: str = InputField(desc="The user's input for this conversation turn")
    turn_output: str = InputField(desc="The module's output for this conversation turn")
    tool_usage: str = InputField(
        desc="Description of tools called and their results during this turn, or 'None' if no tools were used",
        default="None",
    )
    compression_level: str = InputField(
        desc="How aggressively to compress: 'verbose' preserves more detail, 'concise' keeps only essentials",
        default="concise",
    )
    summary: str = OutputField(
        desc="Compressed summary of this turn preserving: user intent, key facts exchanged, tools used and results, commitments or state changes"
    )
