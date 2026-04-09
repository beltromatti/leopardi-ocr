from __future__ import annotations

import io
import random
import re
import textwrap
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from leopardi.data_pipeline.canonicalize import normalize_target_text


EUROPEAN_WIKIPEDIA_CONFIGS: dict[str, str] = {
    "de": "20231101.de",
    "fr": "20231101.fr",
    "es": "20231101.es",
    "it": "20231101.it",
    "pt": "20231101.pt",
}

PAGE_W = 595
PAGE_H = 842
RENDER_SCALE = 2
MARGIN_BASE = 45
BODY_FONT_SIZE = 11
HEADING_FONT_SIZE = 14
SUBHEADING_FONT_SIZE = 12
LINE_HEIGHT_BASE = 15
HEADING_GAP = 8
PARA_GAP = 6
CHARS_PER_LINE = 75
MIN_ARTICLE_LEN = 1500

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

_BOLD_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


@dataclass(slots=True)
class SynthDoGSample:
    sample_id: str
    doc_id: str
    language: str
    title: str
    canonical_target: str
    image_png: bytes


def _resolve_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = _BOLD_FONT_CANDIDATES if bold else _FONT_CANDIDATES
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, AttributeError):
            continue
    return ImageFont.load_default()


def _is_heading(line: str, next_line: str | None) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80 or len(stripped) < 2:
        return False
    if stripped.endswith((".", ",", ";", ":", "!", "?")):
        return False
    if stripped.startswith(("-", "*", "•", "(", "[", "{")):
        return False
    if next_line and len(next_line.strip()) > len(stripped) * 2:
        return True
    if re.match(r"^[A-ZÀ-ÖÙ-Ýa-zà-öù-ý\s\-–—:()]+$", stripped) and len(stripped) < 60:
        return True
    return False


def _is_list_item(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^[-*•]\s+", stripped) or re.match(r"^\d+\.\s+", stripped))


def wiki_text_to_markdown(title: str, text: str) -> str:
    lines = text.split("\n")
    md_blocks: list[str] = [f"# {title.strip()}"]
    current_para: list[str] = []
    i = 0

    def flush_para() -> None:
        if current_para:
            md_blocks.append(" ".join(current_para))
            current_para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else None

        if not stripped:
            flush_para()
            i += 1
            continue

        if _is_heading(stripped, next_line):
            flush_para()
            depth = 2 if len(stripped) < 30 else 3
            md_blocks.append(f"{'#' * depth} {stripped}")
            i += 1
            continue

        if _is_list_item(stripped):
            flush_para()
            clean = re.sub(r"^[-*•]\s+", "- ", stripped)
            clean = re.sub(r"^\d+\.\s+", "- ", clean)
            md_blocks.append(clean)
            i += 1
            continue

        if stripped.startswith(("Kategorie:", "Catégorie:", "Categoría:", "Categoria:")):
            i += 1
            continue
        if "|" in stripped and stripped.count("|") > 2:
            i += 1
            continue

        current_para.append(stripped)
        i += 1

    flush_para()
    return normalize_target_text("\n\n".join(md_blocks))


def render_markdown_page(markdown: str, *, rng: random.Random, scale: int = RENDER_SCALE) -> tuple[bytes, str]:
    w = PAGE_W * scale
    h = PAGE_H * scale
    margin = MARGIN_BASE * scale
    line_h = LINE_HEIGHT_BASE * scale
    heading_gap = HEADING_GAP * scale
    para_gap = PARA_GAP * scale

    bg = rng.randint(248, 255)
    img = Image.new("RGB", (w, h), (bg, bg, max(0, bg - rng.randint(0, 2))))
    draw = ImageDraw.Draw(img)

    body_font = _resolve_font(BODY_FONT_SIZE * scale)
    h1_font = _resolve_font(HEADING_FONT_SIZE * scale, bold=True)
    h2_font = _resolve_font(SUBHEADING_FONT_SIZE * scale, bold=True)

    y = margin
    max_y = h - margin
    rendered_blocks: list[str] = []

    for block in markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("# ") and not block.startswith("## "):
            font = h1_font
            text = block[2:]
            prefix = "# "
            pre_gap = heading_gap
        elif block.startswith("## "):
            font = h2_font
            text = block.lstrip("#").strip()
            prefix = "## " if block.startswith("## ") and not block.startswith("### ") else "### "
            pre_gap = heading_gap
        elif block.startswith("### "):
            font = h2_font
            text = block.lstrip("#").strip()
            prefix = "### "
            pre_gap = heading_gap
        elif block.startswith("- "):
            font = body_font
            text = block
            prefix = ""
            pre_gap = 2 * scale
        else:
            font = body_font
            text = block
            prefix = ""
            pre_gap = para_gap

        wrapped = textwrap.fill(text, width=CHARS_PER_LINE)
        lines = wrapped.split("\n")
        needed = pre_gap + len(lines) * line_h
        if y + needed > max_y:
            break

        y += pre_gap
        ink = rng.randint(5, 25)
        for line in lines:
            draw.text((margin, y), line, fill=(ink, ink, ink), font=font)
            y += line_h
        rendered_blocks.append(f"{prefix}{text}" if prefix else text)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), normalize_target_text("\n\n".join(rendered_blocks))


def iter_synthdog_european_samples(*, total_limit: int, seed: int = 1337) -> list[SynthDoGSample]:
    from datasets import load_dataset

    rng = random.Random(seed)
    per_language = max(1, (total_limit + len(EUROPEAN_WIKIPEDIA_CONFIGS) - 1) // len(EUROPEAN_WIKIPEDIA_CONFIGS))
    samples: list[SynthDoGSample] = []

    for lang, hf_config in EUROPEAN_WIKIPEDIA_CONFIGS.items():
        if len(samples) >= total_limit:
            break
        ds = load_dataset("wikimedia/wikipedia", hf_config, split="train", streaming=True)
        generated = 0
        for article in ds:
            if len(samples) >= total_limit or generated >= per_language:
                break
            title = str(article.get("title") or "").strip()
            text = str(article.get("text") or "")
            if not title or len(text) < MIN_ARTICLE_LEN:
                continue
            markdown = wiki_text_to_markdown(title, text)
            if not markdown or markdown.count("\n\n") < 2:
                continue
            png_bytes, canonical = render_markdown_page(markdown, rng=rng)
            if not canonical or len(canonical) < 100:
                continue
            sample_id = f"synthdog_european-{lang}-{generated:06d}"
            samples.append(
                SynthDoGSample(
                    sample_id=sample_id,
                    doc_id=f"{lang}-wiki-{generated:06d}",
                    language=lang,
                    title=title,
                    canonical_target=canonical,
                    image_png=png_bytes,
                )
            )
            generated += 1
    return samples
