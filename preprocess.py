"""Shared text-cleaning function used by training and the app."""
import re


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # URLs
    text = re.sub(r"@\w+", " ", text)                    # mentions
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep letters only
    text = re.sub(r"\s+", " ", text).strip()              # extra whitespace
    return text
