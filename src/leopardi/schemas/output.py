from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    heading = "heading"
    paragraph = "paragraph"
    table = "table"
    figure = "figure"
    list_item = "list_item"
    equation = "equation"


class Box(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class Block(BaseModel):
    type: BlockType
    markdown: str = Field(description="Markdown fragment for this block.")
    bbox: Box
    confidence: float = Field(ge=0.0, le=1.0)


class ParsedPage(BaseModel):
    page_id: str
    markdown: str
    blocks: list[Block]
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def example(cls) -> "ParsedPage":
        return cls(
            page_id="sample-page-0001",
            markdown="# Leopardi\n\nEuler identity: $e^{i\\pi} + 1 = 0$",
            blocks=[
                Block(
                    type=BlockType.heading,
                    markdown="# Leopardi",
                    bbox=Box(x0=0.05, y0=0.04, x1=0.45, y1=0.12),
                    confidence=0.99,
                ),
                Block(
                    type=BlockType.paragraph,
                    markdown="Euler identity: $e^{i\\pi} + 1 = 0$",
                    bbox=Box(x0=0.05, y0=0.15, x1=0.70, y1=0.24),
                    confidence=0.97,
                ),
            ],
        )

