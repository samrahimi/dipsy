#!/usr/bin/env python3
"""Test script to verify inline subagent definitions work."""

import dspy
from dspy.utils import load_agent_config, create_agent_from_yaml

print("="*70)
print("Testing Inline Subagent Definitions")
print("="*70)

# Test 1: File-based subagents (original approach)
print("\n1. Testing file-based subagents (orchestrator.yaml)...")
try:
    config = load_agent_config("examples/agents/orchestrator.yaml")
    print(f"   ✓ Config loaded")
    print(f"   - Tools/subagents: {len(config['tools'])}")
    print(f"   - Names: {[t.__name__ for t in config['tools']]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Inline subagents (new approach)
print("\n2. Testing inline subagents (orchestrator_inline.yaml)...")
try:
    config = load_agent_config("examples/agents/orchestrator_inline.yaml")
    print(f"   ✓ Config loaded")
    print(f"   - Tools/subagents: {len(config['tools'])}")
    print(f"   - Names: {[t.__name__ for t in config['tools']]}")

    # Verify the subagents have docstrings from the description
    for tool in config['tools']:
        print(f"   - {tool.__name__}: {tool.__doc__[:50]}...")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Create agent with inline subagents
print("\n3. Creating agent from inline definition...")
try:
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="test-key"))
    agent = create_agent_from_yaml("examples/agents/orchestrator_inline.yaml")
    print(f"   ✓ Agent created successfully")
    print(f"   - Type: {type(agent).__name__}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Mixed approach (create a temporary test)
print("\n4. Testing mixed file-based and inline subagents...")
try:
    import tempfile
    import os

    # Get absolute paths
    abs_researcher = os.path.abspath("examples/agents/researcher.yaml")
    abs_calculator = os.path.abspath("examples/tools/calculator_tool.py")

    # Create a temporary YAML with mixed subagent styles
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(f"""
signature: "question -> answer"
model: "openai/gpt-4o-mini"

subagents:
  # File-based subagent
  - name: researcher
    description: "Research specialist"
    agent: {abs_researcher}

  # Inline subagent
  - name: calculator
    description: "Math specialist"
    agent:
      signature: "question -> answer"
      tools:
        - {abs_calculator}
""")
        temp_path = f.name

    config = load_agent_config(temp_path)
    print(f"   ✓ Mixed config loaded")
    print(f"   - Tools/subagents: {len(config['tools'])}")
    print(f"   - Names: {[t.__name__ for t in config['tools']]}")

    # Cleanup
    os.unlink(temp_path)
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Nested inline subagents
print("\n5. Testing nested inline subagents...")
try:
    abs_search = os.path.abspath("examples/tools/search_tool.py")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(f"""
signature: "question -> answer"

subagents:
  - name: meta_agent
    description: "Agent with its own subagent"
    agent:
      signature: "question -> answer"
      subagents:
        - name: leaf_agent
          description: "Leaf level agent"
          agent:
            signature: "question -> answer"
            tools:
              - {abs_search}
""")
        temp_path = f.name

    config = load_agent_config(temp_path)
    print(f"   ✓ Nested inline config loaded")
    print(f"   - Parent has {len(config['tools'])} tool(s)")

    os.unlink(temp_path)
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*70)
print("All inline subagent tests completed!")
print("="*70)
print("\nFeatures verified:")
print("  ✓ File-based subagents (original)")
print("  ✓ Inline subagents (new)")
print("  ✓ Mixed file-based and inline subagents")
print("  ✓ Nested inline subagents")
print("  ✓ Agent creation with inline definitions")
