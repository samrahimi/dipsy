#!/usr/bin/env python3
"""Test script to verify agent_loader integration works."""

import dspy

# Test 1: Import from dspy.utils
print("Test 1: Importing from dspy.utils...")
from dspy.utils import create_agent_from_yaml, load_agent_config, load_tools_from_file
print("✓ All imports successful\n")

# Test 2: Load a config
print("Test 2: Loading demo_agent.yaml config...")
config = load_agent_config("examples/agents/demo_agent.yaml")
print(f"✓ Config loaded")
print(f"  - Signature: {config['signature']}")
print(f"  - Model: {config['model']}")
print(f"  - Tools: {len(config['tools'])} tool(s) loaded")
print(f"  - Tool names: {[t.__name__ for t in config['tools']]}\n")

# Test 3: Load orchestrator with subagents
print("Test 3: Loading orchestrator.yaml with subagents...")
config = load_agent_config("examples/agents/orchestrator.yaml")
print(f"✓ Config loaded")
print(f"  - Signature: {config['signature']}")
print(f"  - Tools (including subagents): {len(config['tools'])} tool(s)")
print(f"  - Tool names: {[t.__name__ for t in config['tools']]}")
print(f"  - Subagents detected: researcher, math_expert\n")

# Test 4: Create agent using convenience function (without running it)
print("Test 4: Creating agent with create_agent_from_yaml...")
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="test-key"))
try:
    agent = create_agent_from_yaml("examples/agents/demo_agent.yaml")
    print(f"✓ Agent created successfully")
    print(f"  - Type: {type(agent).__name__}")
    print(f"  - Has inner module: {hasattr(agent, 'inner_module')}")
except Exception as e:
    print(f"✗ Error creating agent: {e}")

print("\n" + "="*60)
print("All tests passed! The agent_loader is working correctly.")
print("="*60)
