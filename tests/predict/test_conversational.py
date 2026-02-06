import pytest

import dspy
from dspy.predict.conversational import ConversationalAgent
from dspy.utils.dummies import DummyLM


class TestConversationalAgentWithPredict:
    """Test ConversationalAgent wrapping a simple Predict module."""

    def _make_agent(self, answers, **agent_kwargs):
        lm = DummyLM(answers)
        dspy.configure(lm=lm)
        inner = dspy.Predict("question -> answer")
        return ConversationalAgent(inner, **agent_kwargs)

    def test_single_turn(self):
        agent = self._make_agent([
            {"answer": "Paris"},
            # summarizer call (ChainOfThought produces reasoning + summary)
            {"reasoning": "User asked about France's capital, answer was Paris.",
             "summary": "User asked capital of France. Answer: Paris."},
        ])
        result = agent(question="What is the capital of France?")
        assert result.answer == "Paris"

    def test_multi_turn_context_injected(self):
        agent = self._make_agent([
            # Turn 1: question + answer
            {"answer": "Paris"},
            {"reasoning": "Capital question.",
             "summary": "Asked capital of France. Answer: Paris."},
            # Turn 2: follow-up question + answer
            {"answer": "2.1 million"},
            {"reasoning": "Population question.",
             "summary": "Asked population of Paris. Answer: 2.1 million."},
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
        answers = []
        for i in range(4):
            answers.append({"answer": f"answer_{i}"})
            answers.append({"reasoning": "r", "summary": f"summary_{i}"})
        agent = self._make_agent(answers, max_context_turns=2)

        for i in range(3):
            agent(question=f"question_{i}")

        # Only the last 2 summaries should be retained
        assert len(agent.get_history()) == 2
        # Full history should also be truncated to match
        assert len(agent.get_history(summarized=False)) == 2

    def test_clear_history_also_clears_full_history(self):
        agent = self._make_agent([
            {"answer": "Paris"},
            {"reasoning": "r", "summary": "s"},
        ])
        agent(question="test")
        assert len(agent.get_history(summarized=False)) == 1
        agent.clear_history()
        assert len(agent.get_history(summarized=False)) == 0

    def test_invalid_compression_level_raises(self):
        lm = DummyLM([])
        dspy.configure(lm=lm)
        inner = dspy.Predict("question -> answer")
        with pytest.raises(ValueError, match="compression_level must be"):
            ConversationalAgent(inner, compression_level="invalid")

    def test_checkpoint_independence(self):
        """Modifying state after saving should not corrupt the saved checkpoint."""
        agent = self._make_agent([
            {"answer": "a1"},
            {"reasoning": "r", "summary": "s1"},
            {"answer": "a2"},
            {"reasoning": "r", "summary": "s2"},
        ])
        agent(question="q1")
        agent.save_checkpoint("cp")
        agent(question="q2")
        # After another turn, checkpoint should still have only 1 entry
        agent.restore_checkpoint("cp")
        assert len(agent.get_history()) == 1
        assert len(agent.get_history(summarized=False)) == 1


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
            {"next_thought": "Let me check weather.",
             "next_tool_name": "get_weather",
             "next_tool_args": {"city": "Tokyo"}},
            # ReAct turn 1: finish
            {"next_thought": "Got weather.",
             "next_tool_name": "finish",
             "next_tool_args": {}},
            # ReAct extraction
            {"reasoning": "Weather retrieved.", "answer": "Sunny in Tokyo"},
            # Summarizer call
            {"reasoning": "Weather lookup.",
             "summary": "Asked weather in Tokyo. Used get_weather tool. Result: Sunny in Tokyo."},
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

    def test_custom_hook_does_not_receive_conversation_context(self):
        """The context_hook should receive filtered kwargs (no conversation_context)."""
        received_keys = []

        def my_hook(input_kwargs, prediction):
            received_keys.extend(input_kwargs.keys())
            return f"Custom: {prediction.answer}"

        lm = DummyLM([
            {"answer": "yes"},
            {"reasoning": "r", "summary": "s"},
        ])
        dspy.configure(lm=lm)
        inner = dspy.Predict("question -> answer")
        agent = ConversationalAgent(inner, context_hook=my_hook)
        agent(question="test")
        assert "conversation_context" not in received_keys
        assert "question" in received_keys


class TestConversationalAgentContextCompression:
    """Test that context compression produces summaries and passes compression_level."""

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


class TestConversationalAgentAsync:
    """Test async path (aforward)."""

    @pytest.mark.asyncio
    async def test_async_single_turn(self):
        lm = DummyLM([
            {"answer": "Paris"},
            {"reasoning": "r", "summary": "Capital of France: Paris."},
        ])
        dspy.configure(lm=lm)
        inner = dspy.Predict("question -> answer")
        agent = ConversationalAgent(inner)
        result = await agent.acall(question="What is the capital of France?")
        assert result.answer == "Paris"
        assert len(agent.get_history()) == 1

    @pytest.mark.asyncio
    async def test_async_multi_turn(self):
        lm = DummyLM([
            {"answer": "Paris"},
            {"reasoning": "r", "summary": "Asked capital of France. Answer: Paris."},
            {"answer": "2.1 million"},
            {"reasoning": "r", "summary": "Asked population. Answer: 2.1 million."},
        ])
        inner = dspy.Predict("question -> answer")
        agent = ConversationalAgent(inner)
        with dspy.context(lm=lm):
            await agent.acall(question="What is the capital of France?")
            result = await agent.acall(question="What is its population?")
        assert result.answer == "2.1 million"
        assert len(agent.get_history()) == 2
