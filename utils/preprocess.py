import re


def clean_text(text: str) -> str:
    """
    Clean and normalize text for analysis.

    Steps:
    - convert to lowercase
    - remove extra spaces
    - remove most special characters
    - keep useful symbols like +, #, . and -

    Args:
        text: raw input text

    Returns:
        cleaned text string
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#.\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()