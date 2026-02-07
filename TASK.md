# dipsy v0.1: Rebuild as Evolutionary Agent Framework

## Current State
This repo is a DSPy fork with YAML agent definitions, conversational memory, and hierarchical multi-agent systems. The ideas are solid but DSPy adds massive overhead and is unusably slow for interactive agents.

## Mission
Rip out ALL DSPy dependencies. Rebuild dipsy as a lightweight framework (direct OpenRouter API calls) while preserving the best concepts.

## Core Vision
Build agents that can:
- **Make tools** - generate new Python functions with LLM assistance, save to `~/.dipsy/tools/`
- **Share skills** - markdown docs with knowledge, stored in `~/.dipsy/skills/`
- **Spawn offspring** - create child agents with inherited knowledge and tools
- **Long-term coherence** - persistent tool/skill libraries across sessions

Think: evolutionary intelligence, not chatbots.

## Required Builtins (Every Agent Has These)

```python
make_tool(name: str, functionality: str, save_to_path="~/.dipsy/tools") -> str
show_available_tools() -> list[dict]
load_tool(name: str, load_from_path="~/.dipsy/tools") -> str

make_skill(name: str, domain: str, knowledge: str, save_to_path="~/.dipsy/skills") -> str
show_available_skills() -> list[dict]
load_skill(name: str, load_from_path="~/.dipsy/skills") -> str

SPAWN(parents: list[str], agent_name: str, agent_purpose: str,
      foundational_knowledge: str, default_tools: list[str]) -> dict
```

## Technical Requirements
- **Streaming:** All LLM output + tool calls stream to terminal in real-time
- **Client:** OpenRouter via OpenAI SDK (compatible API, just change base_url)
- **Memory:** Conversational history with compression when context gets long
- **Tools:** Simple Python functions with type hints + docstrings (no DSPy signatures)
- **Config:** Keep YAML agent definitions - they're good
- **No restrictions:** Generated tools can do any valid Python

## Keep From Current Codebase
- YAML agent definition concept (`examples/README.md` has good docs)
- Conversational memory architecture
- Tool loading from Python files
- The vision - it came from a frontier Claude model doing creative work

## Discard Entirely
- All DSPy imports and dependencies
- The entire `dspy/` directory
- Signatures, modules, optimizers, adapters
- Anything that says "dspy" in the code

## Success Criteria
Working demo where an agent:
1. Has a streaming conversation
2. Uses `make_tool()` to generate a new Python tool
3. Uses `load_tool()` to equip itself with that tool
4. Uses the tool it just created
5. Uses `SPAWN()` to create a child agent
6. Can be defined via YAML config

## Implementation Notes
- Build clean, minimal code (~500 LOC core)
- Direct API calls, no heavy abstractions
- Start with streaming conversations, then add builtin tools
- Test as you build
- Update README.md with new vision

## Philosophy
Agents should evolve like humans did:
- Monkeys use tools → Humans make tools
- Individual learning dies → Collective knowledge persists
- Static entities → Self-improving systems that spawn offspring

You're building the substrate for autonomous, evolutionary intelligence.

---

**Time:** ~5 hours
**Deliverable:** Working dipsy v0.1 with above features demonstrated

Figure out the implementation. Build something remarkable.
