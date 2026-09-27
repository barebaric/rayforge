"""Explicit nodes used by the limited Markdown parser."""

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class Text:
    value: str


@dataclass(frozen=True)
class Emphasis:
    children: tuple["Inline", ...]


@dataclass(frozen=True)
class Strong:
    children: tuple["Inline", ...]


@dataclass(frozen=True)
class InlineCode:
    value: str


@dataclass(frozen=True)
class Link:
    label: tuple["Inline", ...]
    destination: str


@dataclass(frozen=True)
class LineBreak:
    pass


Inline: TypeAlias = Text | Emphasis | Strong | InlineCode | Link | LineBreak


@dataclass(frozen=True)
class Heading:
    level: int
    children: tuple[Inline, ...]


@dataclass(frozen=True)
class Paragraph:
    children: tuple[Inline, ...]


@dataclass(frozen=True)
class ListItem:
    blocks: tuple["Block", ...]


@dataclass(frozen=True)
class ListBlock:
    ordered: bool
    items: tuple[ListItem, ...]


@dataclass(frozen=True)
class BlockQuote:
    blocks: tuple["Block", ...]


@dataclass(frozen=True)
class CodeBlock:
    value: str
    language: str | None = None


@dataclass(frozen=True)
class Details:
    title: str
    blocks: tuple["Block", ...]


Block: TypeAlias = (
    Heading | Paragraph | ListBlock | BlockQuote | CodeBlock | Details
)


@dataclass(frozen=True)
class Document:
    blocks: tuple[Block, ...]
