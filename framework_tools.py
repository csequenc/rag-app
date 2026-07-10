from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use only for arithmetic calculations.
    
    """
    return str(eval(expression))

