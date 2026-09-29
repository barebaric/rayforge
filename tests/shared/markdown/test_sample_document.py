from pathlib import Path

import pytest

from rayforge.shared.markdown import (
    Block,
    BlockQuote,
    CodeBlock,
    Details,
    Emphasis,
    Heading,
    Inline,
    InlineCode,
    LineBreak,
    Link,
    ListBlock,
    Strong,
    parse,
)

SAMPLE = Path(__file__).parents[2] / "assets" / "markdown_sample.md"


def _nodes(node) -> list:
    """Flatten a parsed document into every nested node it contains."""
    found = [node]
    for field in ("blocks", "children", "items"):
        for child in getattr(node, field, ()) or ():
            found.extend(_nodes(child))
    return found


@pytest.fixture
def sample_nodes() -> list:
    document = parse(SAMPLE.read_text(encoding="utf-8"))
    return [node for block in document.blocks for node in _nodes(block)]


@pytest.mark.parametrize(
    "node_type",
    (
        BlockQuote,
        CodeBlock,
        Details,
        Emphasis,
        Heading,
        InlineCode,
        LineBreak,
        Link,
        ListBlock,
        Strong,
    ),
)
def test_sample_document_covers_every_supported_node(
    sample_nodes, node_type: type[Block] | type[Inline]
):
    assert any(isinstance(node, node_type) for node in sample_nodes)


def test_sample_document_covers_every_heading_level(sample_nodes):
    levels = {node.level for node in sample_nodes if isinstance(node, Heading)}
    assert levels == set(range(1, 7))


def test_sample_document_covers_ordered_and_unordered_lists(sample_nodes):
    ordered = {
        node.ordered for node in sample_nodes if isinstance(node, ListBlock)
    }
    assert ordered == {True, False}


def test_sample_document_covers_nested_bold_italic(sample_nodes):
    assert any(
        isinstance(node, Strong)
        and any(isinstance(child, Emphasis) for child in node.children)
        for node in sample_nodes
    )


def test_sample_document_covers_nested_italic_bold(sample_nodes):
    assert any(
        isinstance(node, Emphasis)
        and any(isinstance(child, Strong) for child in node.children)
        for node in sample_nodes
    )


def test_sample_document_repeats_a_details_title(sample_nodes):
    titles = [node.title for node in sample_nodes if isinstance(node, Details)]
    assert len(titles) != len(set(titles))


def test_sample_document_keeps_unsafe_content_literal(sample_nodes):
    destinations = [
        node.destination for node in sample_nodes if isinstance(node, Link)
    ]
    assert any(
        destination.startswith("javascript:") for destination in destinations
    )
