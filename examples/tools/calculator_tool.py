"""Calculator tool for the demo agent."""


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, and parentheses."""
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"
