from langchain_core.tools import tool

@tool
def reverse_text(text: str) -> str:
    """
    Reverses the given text.
    """

    return text[::-1]
