"""Parser for Rayforge's deliberately limited Markdown subset."""

import re

from .model import (
    Block,
    BlockQuote,
    CodeBlock,
    Details,
    Document,
    Emphasis,
    Heading,
    Inline,
    InlineCode,
    LineBreak,
    Link,
    ListBlock,
    ListItem,
    Paragraph,
    Strong,
    Text,
)

_DETAILS_START = re.compile(r"^:::details(?:[ \t]+(.+))?$")
_DETAILS_END = ":::enddetails"
_FENCE = re.compile(r"^```[ \t]*([^`]*)$")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*)$")
_LIST = re.compile(r"^([*+-])[ \t]+(.*)$|^(\d+)\.[ \t]+(.*)$")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+[^)]*)?\)")

# Longest markers first so bold italic is not mistaken for bold.
_EMPHASIS_MARKERS = (
    ("***", (Strong, Emphasis)),
    ("___", (Strong, Emphasis)),
    ("**", (Strong,)),
    ("__", (Strong,)),
    ("*", (Emphasis,)),
    ("_", (Emphasis,)),
)

# Quotes, lists, and details sections nest by recursion. The limit keeps
# pathological input (such as a line of hundreds of ">") from exhausting
# the interpreter stack.
_MAX_NESTING_DEPTH = 16


class MarkdownParser:
    """Parse Markdown into an immutable :class:`Document`."""

    def parse(self, source: str) -> Document:
        return Document(tuple(self._parse_blocks(source.splitlines())))

    def _parse_blocks(self, lines: list[str], depth: int = 0) -> list[Block]:
        blocks: list[Block] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue

            fence = _FENCE.match(line)
            if fence:
                index += 1
                content: list[str] = []
                while index < len(lines) and not lines[index].startswith(
                    "```"
                ):
                    content.append(lines[index])
                    index += 1
                if index < len(lines):
                    index += 1
                blocks.append(
                    CodeBlock(
                        "\n".join(content), fence.group(1).strip() or None
                    )
                )
                continue

            details = _DETAILS_START.match(line)
            if details:
                end = self._details_end(lines, index + 1)
                if end is not None:
                    blocks.append(
                        Details(
                            details.group(1) or "",
                            tuple(
                                self._parse_nested(
                                    lines[index + 1 : end], depth
                                )
                            ),
                        )
                    )
                    index = end + 1
                    continue

            heading = _HEADING.match(line)
            if heading:
                blocks.append(
                    Heading(
                        len(heading.group(1)),
                        tuple(parse_inlines(heading.group(2))),
                    )
                )
                index += 1
                continue

            if line.startswith(">"):
                quote: list[str] = []
                while index < len(lines) and lines[index].startswith(">"):
                    quote.append(lines[index][1:].lstrip())
                    index += 1
                blocks.append(
                    BlockQuote(tuple(self._parse_nested(quote, depth)))
                )
                continue

            list_match = _LIST.match(line)
            if list_match:
                block, index = self._parse_list(
                    lines, index, list_match, depth
                )
                blocks.append(block)
                continue

            paragraph: list[str] = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                if (
                    _HEADING.match(lines[index])
                    or _FENCE.match(lines[index])
                    or _DETAILS_START.match(lines[index])
                    or lines[index].startswith(">")
                    or _LIST.match(lines[index])
                ):
                    break
                paragraph.append(lines[index])
                index += 1
            blocks.append(
                Paragraph(tuple(parse_inlines("\n".join(paragraph))))
            )
        return blocks

    def _parse_nested(self, lines: list[str], depth: int) -> list[Block]:
        """Parse nested lines, keeping them literal past the depth limit.

        Beyond the limit the lines are kept as plain paragraphs, so deeply
        nested input renders as the text the user typed instead of
        recursing until the stack runs out.
        """
        if depth >= _MAX_NESTING_DEPTH:
            return [
                Paragraph(tuple(parse_inlines(line)))
                for line in lines
                if line.strip()
            ]
        return self._parse_blocks(lines, depth + 1)

    @staticmethod
    def _details_end(lines: list[str], start: int) -> int | None:
        for index in range(start, len(lines)):
            if lines[index] == _DETAILS_END:
                return index
        return None

    def _parse_list(
        self,
        lines: list[str],
        start: int,
        first_match: re.Match[str],
        depth: int = 0,
    ) -> tuple[ListBlock, int]:
        ordered = first_match.group(3) is not None
        items: list[ListItem] = []
        index = start
        while index < len(lines):
            match = _LIST.match(lines[index])
            if not match or (match.group(3) is not None) != ordered:
                break
            item_lines = [match.group(2) or match.group(4) or ""]
            index += 1
            while (
                index < len(lines)
                and lines[index].startswith(("  ", "\t"))
                and lines[index].strip()
            ):
                item_lines.append(lines[index].lstrip())
                index += 1
            items.append(
                ListItem(tuple(self._parse_nested(item_lines, depth)))
            )
        return ListBlock(ordered, tuple(items)), index


def parse(source: str) -> Document:
    """Parse *source* using the standard limited Markdown parser."""

    return MarkdownParser().parse(source)


def parse_inlines(source: str) -> list[Inline]:
    """Parse inline syntax without interpreting raw HTML."""

    result: list[Inline] = []
    index = 0
    while index < len(source):
        if source[index] == "`":
            end = source.find("`", index + 1)
            if end != -1:
                result.append(InlineCode(source[index + 1 : end]))
                index = end + 1
                continue
        link = _LINK.match(source, index)
        if link:
            result.append(
                Link(tuple(parse_inlines(link.group(1))), link.group(2))
            )
            index = link.end()
            continue
        matched = False
        for marker, node_types in _EMPHASIS_MARKERS:
            if source.startswith(marker, index):
                end = source.find(marker, index + len(marker))
                if end > index + len(marker):
                    inner = parse_inlines(source[index + len(marker) : end])
                    node: tuple[Inline, ...] = tuple(inner)
                    for node_type in reversed(node_types):
                        node = (node_type(node),)
                    result.append(node[0])
                    index = end + len(marker)
                    matched = True
                    break
        if matched:
            continue
        if source[index] == "\n":
            result.append(LineBreak())
            index += 1
            continue
        start = index
        while index < len(source) and source[index] not in "`[*_\n":
            index += 1
        if start == index:
            index += 1
            result.append(Text(source[start:index]))
        else:
            result.append(Text(source[start:index]))
    return result
