from __future__ import annotations

import json
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

_FRONT_MATTER_COMMANDS = (
    "TITLE",
    "title",
    "RUNTITLE",
    "runtitle",
    "ABSTRACT",
    "abstract",
    "ARTICLEAUTHORS",
    "author",
)

_INLINE_MATH_PATTERN = r"\$(?:[^\s$\n]|[^\s$\n][^$\n]*?[^\s$\n])\$"


def normalize_target_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    normalized_lines: list[str] = []
    in_fence = False
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            normalized_lines.append(stripped)
            in_fence = not in_fence
            continue
        if in_fence:
            normalized_lines.append(line)
            continue
        if not stripped:
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")
            continue
        leading = re.match(r"^\s*", line).group(0).replace("\t", "  ")
        normalized_lines.append(leading + re.sub(r"[ \t]+", " ", stripped))
    while normalized_lines and normalized_lines[0] == "":
        normalized_lines.pop(0)
    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()
    return "\n".join(normalized_lines)


def _normalize_markdown_math_boundaries(text: str) -> str:
    lines = text.split("\n")
    normalized_lines: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            normalized_lines.append(line)
            in_fence = not in_fence
            continue
        if in_fence:
            normalized_lines.append(line)
            continue
        updated = line
        updated = re.sub(rf"([A-Za-z0-9)])({_INLINE_MATH_PATTERN})", r"\1 \2", updated)
        updated = re.sub(rf"({_INLINE_MATH_PATTERN})([A-Za-z0-9(])", r"\1 \2", updated)
        updated = re.sub(r"(?<=\S)(\$\$)", r" \1", updated)
        updated = re.sub(r"(\$\$)(?=\S)", r"\1 ", updated)
        normalized_lines.append(updated)
    return "\n".join(normalized_lines)


def _normalize_inline_math_content(text: str) -> str:
    value = text.strip()
    value = re.sub(r"\(\s+", "(", value)
    value = re.sub(r"\[\s+", "[", value)
    value = re.sub(r"\s+([,;:.!?])", r"\1", value)
    value = re.sub(r"\s+([)\]])", r"\1", value)
    return value


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


def _extract_tex_braced_payload(content: str, command: str) -> str:
    match = re.search(rf"\\{re.escape(command)}\*?\s*\{{", content)
    if match is None:
        return ""
    index = match.end() - 1
    depth = 0
    start = index + 1
    for cursor in range(index, len(content)):
        char = content[cursor]
        if char == "{":
            depth += 1
            if depth == 1:
                start = cursor + 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start:cursor]
    return ""


def _extract_braced_from_index(content: str, brace_index: int) -> tuple[str, int]:
    if brace_index < 0 or brace_index >= len(content) or content[brace_index] != "{":
        return "", brace_index
    depth = 0
    start = brace_index + 1
    for cursor in range(brace_index, len(content)):
        char = content[cursor]
        if char == "{":
            depth += 1
            if depth == 1:
                start = cursor + 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start:cursor], cursor + 1
    return "", brace_index


def _extract_tex_environment(content: str, environment: str) -> str:
    match = re.search(
        rf"\\begin\{{{re.escape(environment)}\}}(.*?)\\end\{{{re.escape(environment)}\}}",
        content,
        flags=re.DOTALL,
    )
    if match is None:
        return ""
    return match.group(1).strip()


def _extract_simple_tex_macros(content: str) -> dict[str, str]:
    macros: dict[str, str] = {}
    command_pattern = re.compile(
        r"\\(?:newcommand|renewcommand|providecommand)\*?\s*\{\\([A-Za-z@]+)\}\s*(?:\[(\d+)\])?\s*\{",
        flags=re.DOTALL,
    )
    for match in command_pattern.finditer(content):
        if match.group(2) not in {None, "0"}:
            continue
        name = match.group(1)
        if len(name) < 3:
            continue
        value, _ = _extract_braced_from_index(content, match.end() - 1)
        cleaned = _strip_tex_wrappers(value).strip()
        if cleaned and cleaned != "\\":
            macros[name] = cleaned
    def_pattern = re.compile(r"\\def\\([A-Za-z@]+)\s*\{", flags=re.DOTALL)
    for match in def_pattern.finditer(content):
        name = match.group(1)
        if len(name) < 3:
            continue
        value, _ = _extract_braced_from_index(content, match.end() - 1)
        cleaned = _strip_tex_wrappers(value).strip()
        if cleaned and cleaned != "\\":
            macros[name] = cleaned
    return macros


def _apply_simple_tex_macros(content: str, macros: dict[str, str]) -> str:
    for name, value in sorted(macros.items(), key=lambda item: -len(item[0])):
        content = re.sub(rf"\\{re.escape(name)}\b", lambda _match, replacement=value: replacement, content)
    return content


def _extract_tex_front_matter(content: str) -> list[str]:
    blocks: list[str] = []
    title = ""
    for command in ("TITLE", "title", "RUNTITLE", "runtitle"):
        title = _strip_tex_wrappers(_extract_tex_braced_payload(content, command))
        if title:
            blocks.append(f"# {title}")
            break

    article_authors = _extract_tex_braced_payload(content, "ARTICLEAUTHORS")
    if article_authors:
        authors = [
            _strip_tex_wrappers(match.group(1))
            for match in re.finditer(r"\\AUTHOR\{([^{}]+)\}", article_authors, flags=re.DOTALL)
        ]
        affiliations = [
            _strip_tex_wrappers(match.group(1))
            for match in re.finditer(r"\\AFF\{([^{}]+)\}", article_authors, flags=re.DOTALL)
        ]
        if authors:
            blocks.append("\n".join(authors))
        for affiliation in affiliations:
            if affiliation:
                blocks.append(affiliation)
    else:
        authors = _strip_tex_wrappers(_extract_tex_braced_payload(content, "author"))
        if authors:
            blocks.append(authors)

    abstract = _clean_tex_text_preserve_math(_extract_tex_braced_payload(content, "ABSTRACT"))
    if not abstract:
        abstract = _clean_tex_text_preserve_math(_extract_tex_environment(content, "abstract"))
    if abstract:
        blocks.append("## Abstract")
        blocks.append(abstract)
    return [block for block in blocks if normalize_target_text(block)]


def _strip_tex_named_commands(content: str, commands: tuple[str, ...]) -> str:
    output = content
    for command in commands:
        cursor = 0
        chunks: list[str] = []
        while True:
            match = re.search(rf"\\{re.escape(command)}\*?\s*\{{", output[cursor:])
            if match is None:
                chunks.append(output[cursor:])
                break
            absolute_start = cursor + match.start()
            absolute_brace = cursor + match.end() - 1
            chunks.append(output[cursor:absolute_start])
            _, next_index = _extract_braced_from_index(output, absolute_brace)
            cursor = next_index if next_index > absolute_brace else absolute_brace + 1
        output = "".join(chunks)
    return output


def _extract_tex_document_body(content: str) -> str:
    match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", content, flags=re.DOTALL)
    if match is not None:
        return match.group(1)
    return content


def _replace_tex_command(content: str, command: str, replacement: str) -> str:
    pattern = re.compile(rf"\\{command}\*?\{{([^{{}}]+)\}}", re.DOTALL)
    return pattern.sub(lambda match: f"{replacement} {match.group(1).strip()}\n\n", content)


def _unwrap_tex_command(content: str, command: str) -> str:
    pattern = re.compile(rf"\\{command}\{{([^{{}}]+)\}}", re.DOTALL)
    return pattern.sub(lambda match: match.group(1).strip(), content)


def _strip_tex_wrappers(text: str) -> str:
    value = text.strip()
    for command in _TEXT_UNWRAP_COMMANDS:
        value = _unwrap_tex_command(value, command)
    value = re.sub(r"\\label\{[^{}]+\}", "", value)
    value = re.sub(r"\\cite[t|p]?\{[^{}]+\}", "[CITE]", value)
    value = re.sub(r"\\ref\{[^{}]+\}", "[REF]", value)
    value = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", "", value)
    value = value.replace("{", "").replace("}", "")
    return normalize_target_text(value)


def _clean_tex_text_preserve_math(text: str) -> str:
    value = text.strip()
    for command in _TEXT_UNWRAP_COMMANDS:
        value = _unwrap_tex_command(value, command)
    value = re.sub(r"\\label\{[^{}]+\}", "", value)
    value = re.sub(r"\\cite[t|p]?\{[^{}]+\}", "[CITE]", value)
    value = re.sub(r"\\ref\{[^{}]+\}", "[REF]", value)
    value = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", "", value)
    return normalize_target_text(value)


def _markdown_escape_cell(text: str) -> str:
    return normalize_target_text(text).replace("|", r"\|")


def _canonical_table_block(caption: str, cells: list[tuple[int, int, int, int, str]], columns: int) -> str:
    lines = ["```table"]
    if caption:
        lines.append(f"caption: {normalize_target_text(caption)}")
    lines.append(f"columns: {columns}")
    lines.append("cells:")
    for row_start, col_start, row_end, col_end, text in cells:
        payload = json.dumps(normalize_target_text(text), ensure_ascii=False)
        lines.append(f"  - [{row_start}, {col_start}, {row_end}, {col_end}, {payload}]")
    lines.append("```")
    return "\n".join(lines)


def _simple_markdown_table(caption: str, rows: list[list[str]]) -> str:
    header = rows[0]
    body = rows[1:] or [["" for _ in header]]
    lines: list[str] = []
    if caption:
        lines.append(f"Table: {normalize_target_text(caption)}")
        lines.append("")
    lines.append("| " + " | ".join(_markdown_escape_cell(cell) for cell in header) + " |")
    lines.append("| " + " | ".join(":---" for _ in header) + " |")
    for row in body:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(_markdown_escape_cell(cell) for cell in padded[: len(header)]) + " |")
    return "\n".join(lines)


def _parse_tex_table_cell(cell: str) -> tuple[int, int, str]:
    value = cell.strip()
    rowspan = 1
    colspan = 1
    multirow = re.match(r"\\multirow\{(\d+)\}\{[^{}]*\}\{(.*)\}", value, flags=re.DOTALL)
    if multirow is not None:
        rowspan = max(1, int(multirow.group(1)))
        value = multirow.group(2).strip()
    multicol = re.match(r"\\multicolumn\{(\d+)\}\{[^{}]*\}\{(.*)\}", value, flags=re.DOTALL)
    if multicol is not None:
        colspan = max(1, int(multicol.group(1)))
        value = multicol.group(2).strip()
    return rowspan, colspan, _strip_tex_wrappers(value)


def _split_tex_rows(tabular: str) -> list[str]:
    text = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|cline\{[^{}]+\})", "", tabular)
    rows = re.split(r"(?<!\\)\\\\", text)
    return [row.strip() for row in rows if row.strip()]


def _tex_table_to_markdown(tabular: str, caption: str) -> str:
    rows_raw = _split_tex_rows(tabular)
    active_rowspans: dict[int, int] = {}
    cells: list[tuple[int, int, int, int, str]] = []
    simple_rows: list[list[str]] = []
    max_columns = 0
    complex_table = False
    for row_index, row in enumerate(rows_raw):
        row_cells = [cell for cell in re.split(r"(?<!\\)&", row) if cell.strip()]
        rendered_row: list[str] = []
        col_index = 0
        while active_rowspans.get(col_index, 0) > 0:
            active_rowspans[col_index] -= 1
            col_index += 1
        for raw_cell in row_cells:
            while active_rowspans.get(col_index, 0) > 0:
                active_rowspans[col_index] -= 1
                col_index += 1
            rowspan, colspan, text = _parse_tex_table_cell(raw_cell)
            cells.append((row_index, col_index, row_index + rowspan - 1, col_index + colspan - 1, text))
            if rowspan > 1 or colspan > 1:
                complex_table = True
            if rowspan > 1:
                for offset in range(colspan):
                    active_rowspans[col_index + offset] = rowspan - 1
            rendered_row.extend([text] + [""] * (colspan - 1))
            col_index += colspan
        max_columns = max(max_columns, col_index)
        simple_rows.append(rendered_row)
    rectangular = all(len(row) == max_columns for row in simple_rows if row)
    if not complex_table and rectangular and simple_rows and max_columns > 1:
        return _simple_markdown_table(caption, simple_rows)
    return _canonical_table_block(caption, cells, max_columns or 1)


def _extract_tex_tables(content: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        body = match.group(1)
        caption_match = re.search(r"\\caption\{(.*?)\}", body, flags=re.DOTALL)
        caption = _strip_tex_wrappers(caption_match.group(1)) if caption_match else ""
        tabular_match = re.search(
            r"\\begin\{tabular\*?\}(?:\{[^{}]*\})?\{[^{}]*\}(.*?)\\end\{tabular\*?\}",
            body,
            flags=re.DOTALL,
        )
        if tabular_match is None:
            return ""
        return "\n" + _tex_table_to_markdown(tabular_match.group(1), caption) + "\n"

    return re.sub(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", _replace, content, flags=re.DOTALL)


def _extract_tex_figures(content: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        body = match.group(1)
        caption_match = re.search(r"\\caption\{(.*?)\}", body, flags=re.DOTALL)
        if caption_match is None:
            return ""
        caption = _strip_tex_wrappers(caption_match.group(1))
        if not caption:
            return ""
        return f"\nFigure: {caption}\n"

    return re.sub(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", _replace, content, flags=re.DOTALL)


def tex_to_markdown(tex: str) -> str:
    content = _strip_tex_comments(tex)
    macros = _extract_simple_tex_macros(content)
    content = _apply_simple_tex_macros(content, macros)
    front_matter = _extract_tex_front_matter(content)
    content = _extract_tex_document_body(content)
    content = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^{}]+\}", "", content)
    content = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^{}]+\}", "", content)
    content = re.sub(r"\\bibliographystyle\{[^{}]+\}", "", content)
    content = re.sub(r"\\bibliography\{[^{}]+\}", "", content)
    content = re.sub(r"\\(?:newcommand|renewcommand|providecommand)\*?\s*\{\\[A-Za-z@]+\}\s*(?:\[[0-9]+\])?\s*\{[^{}]*\}", "", content)
    content = re.sub(r"\\def\\[A-Za-z@]+\s*\{[^{}]*\}", "", content)
    content = re.sub(r"\\(?:definecolor|hypersetup)\*?(?:\[[^\]]*\])?\s*\{[^{}]*\}", "", content)
    content = re.sub(r"\\(?:TheoremsNumberedThrough|TheoremsNumberedByChapter|ECRepeatTheorems|EquationsNumberedThrough|EquationsNumberedBySection)\b", "", content)
    content = re.sub(r"\\label\{[^{}]+\}", "", content)
    content = re.sub(r"\\cite[t|p]?\{[^{}]+\}", "[CITE]", content)
    content = re.sub(r"\\ref\{[^{}]+\}", "[REF]", content)
    content = re.sub(r"\\maketitle", "", content)
    content = re.sub(r"\\tableofcontents", "", content)
    content = _strip_tex_named_commands(content, _FRONT_MATTER_COMMANDS)

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

    content = _extract_tex_figures(content)
    content = _extract_tex_tables(content)
    content = re.sub(r"\\begin\{document\}", "", content)
    content = re.sub(r"\\end\{document\}", "", content)
    content = re.sub(r"\\[a-zA-Z@]+(\[[^\]]*\])?", "", content)
    payload = normalize_target_text(content)
    if front_matter:
        payload = normalize_target_text("\n\n".join((*front_matter, payload)))
    payload = re.sub(r"^\{([^{}\n]+)\}$", r"\1", payload, flags=re.MULTILINE)
    payload = re.sub(
        r"\$\s*([^$\n]+?)\s*\$",
        lambda match: f"${_normalize_inline_math_content(match.group(1))}$",
        payload,
    )
    payload = _normalize_markdown_math_boundaries(payload)
    return payload


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _jats_tag(node: ET.Element) -> str:
    return node.tag.split("}")[-1]


def _jats_formula_text(node: ET.Element) -> str:
    tex = node.find(".//tex-math")
    if tex is not None and _node_text(tex):
        return normalize_target_text(_node_text(tex))
    return normalize_target_text(_node_text(node))


def _collect_jats_inline(node: ET.Element | None) -> str:
    if node is None:
        return ""
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in list(node):
        tag = _jats_tag(child)
        if tag == "inline-formula":
            formula = _jats_formula_text(child)
            if formula:
                parts.append(f"${formula}$")
        elif tag == "disp-formula":
            formula = _jats_formula_text(child)
            if formula:
                parts.append(f"\n$$\n{formula}\n$$\n")
        else:
            parts.append(_collect_jats_inline(child))
        if child.tail:
            parts.append(child.tail)
    return normalize_target_text("".join(parts))


def _jats_table_to_markdown(table_wrap: ET.Element) -> str:
    caption_bits = [
        _collect_jats_inline(table_wrap.find("./caption/title")),
        _collect_jats_inline(table_wrap.find("./caption/p")),
    ]
    caption = normalize_target_text(" ".join(bit for bit in caption_bits if bit))
    rows = table_wrap.findall(".//tr")
    cells: list[tuple[int, int, int, int, str]] = []
    simple_rows: list[list[str]] = []
    active_rowspans: dict[int, int] = {}
    max_columns = 0
    complex_table = False
    for row_index, row in enumerate(rows):
        rendered_row: list[str] = []
        col_index = 0
        while active_rowspans.get(col_index, 0) > 0:
            active_rowspans[col_index] -= 1
            col_index += 1
        for cell in row:
            tag = _jats_tag(cell)
            if tag not in {"th", "td"}:
                continue
            while active_rowspans.get(col_index, 0) > 0:
                active_rowspans[col_index] -= 1
                col_index += 1
            rowspan = max(1, int(cell.attrib.get("rowspan", "1")))
            colspan = max(1, int(cell.attrib.get("colspan", "1")))
            text = _collect_jats_inline(cell)
            cells.append((row_index, col_index, row_index + rowspan - 1, col_index + colspan - 1, text))
            if rowspan > 1 or colspan > 1:
                complex_table = True
            if rowspan > 1:
                for offset in range(colspan):
                    active_rowspans[col_index + offset] = rowspan - 1
            rendered_row.extend([text] + [""] * (colspan - 1))
            col_index += colspan
        max_columns = max(max_columns, col_index)
        if rendered_row:
            simple_rows.append(rendered_row)
    rectangular = all(len(row) == max_columns for row in simple_rows if row)
    if not cells:
        return ""
    if not complex_table and rectangular and simple_rows and max_columns > 1:
        return _simple_markdown_table(caption, simple_rows)
    return _canonical_table_block(caption, cells, max_columns or 1)


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
            text = _collect_jats_inline(paragraph)
            if text:
                yield text
        for formula in sec.findall(".//disp-formula"):
            formula_text = _jats_formula_text(formula)
            if formula_text:
                yield "$$"
                yield formula_text
                yield "$$"
        for table_wrap in sec.findall(".//table-wrap"):
            table_markdown = _jats_table_to_markdown(table_wrap)
            if table_markdown:
                yield table_markdown
        for figure in sec.findall(".//fig"):
            caption = normalize_target_text(
                " ".join(
                    part
                    for part in (
                        _collect_jats_inline(figure.find("./caption/title")),
                        _collect_jats_inline(figure.find("./caption/p")),
                    )
                    if part
                )
            )
            if caption:
                yield f"Figure: {caption}"


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
