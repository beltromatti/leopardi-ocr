from __future__ import annotations

from dataclasses import dataclass

from leopardi.inference.config import AssemblyConfig


@dataclass(slots=True)
class DocumentPage:
    page_number: int
    markdown: str


def _first_nonempty_line(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.strip():
            return line.strip()
    return None


def _last_nonempty_line(markdown: str) -> str | None:
    for line in reversed(markdown.splitlines()):
        if line.strip():
            return line.strip()
    return None


def _repetition_candidates(lines: list[str | None], min_repeat_pages: int) -> set[str]:
    counts: dict[str, int] = {}
    for line in lines:
        if line is None:
            continue
        counts[line] = counts.get(line, 0) + 1
    return {line for line, count in counts.items() if count >= min_repeat_pages}


def assemble_document(pages: list[DocumentPage], config: AssemblyConfig) -> str:
    if not pages:
        return ""
    header_candidates = _repetition_candidates(
        [_first_nonempty_line(page.markdown) for page in pages],
        config.min_repeat_pages,
    )
    footer_candidates = _repetition_candidates(
        [_last_nonempty_line(page.markdown) for page in pages],
        config.min_repeat_pages,
    )

    assembled_pages: list[str] = []
    for page in sorted(pages, key=lambda item: item.page_number):
        lines = page.markdown.splitlines()
        if config.suppress_repeated_headers and lines:
            first = _first_nonempty_line(page.markdown)
            if first in header_candidates:
                dropped = False
                kept: list[str] = []
                for line in lines:
                    if not dropped and line.strip() == first:
                        dropped = True
                        continue
                    kept.append(line)
                lines = kept
        if config.suppress_repeated_footers and lines:
            last = _last_nonempty_line("\n".join(lines))
            if last in footer_candidates:
                for idx in range(len(lines) - 1, -1, -1):
                    if lines[idx].strip() == last:
                        del lines[idx]
                        break
        page_text = "\n".join(lines).strip()
        if page_text:
            assembled_pages.append(page_text)
    separator = f"\n\n{config.page_break_marker}\n\n" if config.emit_page_break_markers else "\n\n"
    return separator.join(assembled_pages)
