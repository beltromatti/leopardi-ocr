from __future__ import annotations

import re
from typing import Iterable
from xml.etree import ElementTree as ET


_SECTION_COMMANDS = {
    "part": "#",
    "chapter": "#",
    "section": "##",
    "subsection": "###",
    "subsubsection": "####",
    "paragraph": "#####",
}

_TEXT_UNWRAP_COMMANDS = (
    "textbf",
    "textit",
    "textrm",
    "textsf",
    "texttt",
    "emph",
    "underline",
    "mbox",
)


def normalize_target_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_tex_comments(tex: str) -> str:
    stripped_lines: list[str] = []
    for line in tex.splitlines():
        escaped = False
        current: list[str] = []
        for char in line:
            if char == "%" and not escaped:
                break
            current.append(char)
            escaped = char == "\\"
        stripped_lines.append("".join(current))
    return "\n".join(stripped_lines)


def _replace_tex_command(content: str, command: str, replacement: str) -> str:
    pattern = re.compile(rf"\\{command}\*?\{{([^{{}}]+)\}}", re.DOTALL)
    return pattern.sub(lambda match: f"{replacement} {match.group(1).strip()}\n\n", content)


def _unwrap_tex_command(content: str, command: str) -> str:
    pattern = re.compile(rf"\\{command}\{{([^{{}}]+)\}}", re.DOTALL)
    return pattern.sub(lambda match: match.group(1).strip(), content)


def tex_to_markdown(tex: str) -> str:
    content = _strip_tex_comments(tex)
    content = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^{}]+\}", "", content)
    content = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^{}]+\}", "", content)
    content = re.sub(r"\\bibliographystyle\{[^{}]+\}", "", content)
    content = re.sub(r"\\bibliography\{[^{}]+\}", "", content)
    content = re.sub(r"\\label\{[^{}]+\}", "", content)
    content = re.sub(r"\\cite[t|p]?\{[^{}]+\}", "[CITE]", content)
    content = re.sub(r"\\ref\{[^{}]+\}", "[REF]", content)
    content = re.sub(r"\\maketitle", "", content)
    content = re.sub(r"\\tableofcontents", "", content)

    content = re.sub(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        lambda match: f"## Abstract\n\n{match.group(1).strip()}\n\n",
        content,
        flags=re.DOTALL,
    )
    for command, heading in _SECTION_COMMANDS.items():
        content = _replace_tex_command(content, command, heading)
    for command in _TEXT_UNWRAP_COMMANDS:
        content = _unwrap_tex_command(content, command)

    content = re.sub(
        r"\\item\s+",
        "\n- ",
        content,
    )
    content = re.sub(r"\\begin\{(?:itemize|enumerate)\}", "\n", content)
    content = re.sub(r"\\end\{(?:itemize|enumerate)\}", "\n", content)

    def _block_math(match: re.Match[str]) -> str:
        payload = match.group(1).strip()
        return f"\n$$\n{payload}\n$$\n"

    for environment in ("equation", "equation*", "align", "align*", "gather", "gather*"):
        content = re.sub(
            rf"\\begin\{{{re.escape(environment)}\}}(.*?)\\end\{{{re.escape(environment)}\}}",
            _block_math,
            content,
            flags=re.DOTALL,
        )

    content = re.sub(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", "", content, flags=re.DOTALL)
    content = re.sub(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", "", content, flags=re.DOTALL)
    content = re.sub(r"\\begin\{document\}", "", content)
    content = re.sub(r"\\end\{document\}", "", content)
    content = re.sub(r"\\[a-zA-Z@]+(\[[^\]]*\])?", "", content)
    content = content.replace("{", "").replace("}", "")
    return normalize_target_text(content)


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _iter_jats_sections(root: ET.Element) -> Iterable[str]:
    article_title = _node_text(root.find(".//article-title"))
    if article_title:
        yield f"# {article_title}"

    abstract = root.find(".//abstract")
    abstract_text = _node_text(abstract)
    if abstract_text:
        yield "## Abstract"
        yield abstract_text

    for sec in root.findall(".//body//sec"):
        title = _node_text(sec.find("./title"))
        if title:
            yield f"## {title}"
        for paragraph in sec.findall("./p"):
            text = _node_text(paragraph)
            if text:
                yield text
        for formula in sec.findall(".//disp-formula"):
            formula_text = _node_text(formula)
            if formula_text:
                yield "$$"
                yield formula_text
                yield "$$"
        for table_wrap in sec.findall(".//table-wrap"):
            table_title = _node_text(table_wrap.find("./caption/title")) or _node_text(
                table_wrap.find("./label")
            )
            if table_title:
                yield f"### {table_title}"
            caption_text = _node_text(table_wrap.find("./caption/p"))
            if caption_text:
                yield caption_text


def jats_to_markdown(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    parts = [part for part in _iter_jats_sections(root) if part]
    return normalize_target_text("\n\n".join(parts))


def normalize_alignment_text(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for index, char in enumerate(text):
        if char.isalnum():
            normalized_chars.append(char.lower())
            index_map.append(index)
        elif char.isspace():
            if normalized_chars and normalized_chars[-1] != " ":
                normalized_chars.append(" ")
                index_map.append(index)
    normalized = "".join(normalized_chars).strip()
    if not normalized:
        return "", []
    leading = len("".join(normalized_chars)) - len(normalized.lstrip())
    trailing = len("".join(normalized_chars)) - len(normalized.rstrip())
    return normalized, index_map[leading : len(index_map) - trailing if trailing else None]


def project_markdown_to_pages(markdown: str, page_texts: list[str]) -> list[str]:
    if not page_texts:
        return [markdown]
    normalized_markdown, markdown_map = normalize_alignment_text(markdown)
    if not normalized_markdown or not markdown_map:
        return [normalize_target_text(markdown)] * len(page_texts)

    page_norms = [normalize_alignment_text(page_text)[0] for page_text in page_texts]
    starts: list[int] = [0]
    cursor = 0
    for page_norm in page_norms[1:]:
        anchor = page_norm[:120].strip()
        if not anchor:
            starts.append(cursor)
            continue
        match = normalized_markdown.find(anchor, cursor)
        if match == -1:
            proportional = int(len(normalized_markdown) * (len(starts) / max(len(page_texts), 1)))
            starts.append(max(cursor, proportional))
            cursor = starts[-1]
            continue
        starts.append(match)
        cursor = match
    starts.append(len(normalized_markdown))

    projected: list[str] = []
    for index in range(len(page_texts)):
        start_norm = starts[index]
        end_norm = starts[index + 1]
        if start_norm >= len(markdown_map):
            projected.append("")
            continue
        start_original = markdown_map[start_norm]
        end_original = markdown_map[min(max(end_norm - 1, start_norm), len(markdown_map) - 1)] + 1
        projected.append(normalize_target_text(markdown[start_original:end_original]))
    return projected
