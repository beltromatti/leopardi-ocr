from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_headings(text: str) -> str:
    normalized_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            hashes, _, rest = stripped.partition(" ")
            normalized_lines.append(f"{hashes} {rest.strip()}".rstrip())
        else:
            normalized_lines.append(line.rstrip())
    return "\n".join(normalized_lines)


def normalize_bullets(text: str) -> str:
    return re.sub(r"^[\*\+]\s+", "- ", text, flags=re.MULTILINE)


def normalize_math_delimiters(text: str) -> str:
    text = text.replace("\\(", "$").replace("\\)", "$")
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    return text


def canonicalize_tables(text: str) -> str:
    lines = text.splitlines()
    normalized: list[str] = []
    for line in lines:
        if "|" in line:
            parts = [part.strip() for part in line.split("|")]
            normalized.append("|".join(parts))
        else:
            normalized.append(line.rstrip())
    return "\n".join(normalized)


def normalize_markdown(text: str) -> str:
    text = normalize_whitespace(text)
    text = normalize_headings(text)
    text = normalize_bullets(text)
    text = normalize_math_delimiters(text)
    text = canonicalize_tables(text)
    return normalize_whitespace(text)


def normalize_latex(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(r"\,", " ")
    return text.strip()
