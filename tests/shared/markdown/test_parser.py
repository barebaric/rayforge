from rayforge.shared.markdown import (
    BlockQuote,
    CodeBlock,
    Details,
    Emphasis,
    Heading,
    InlineCode,
    LineBreak,
    Link,
    ListBlock,
    ListItem,
    MarkdownParser,
    Paragraph,
    Strong,
    Text,
    parse,
)


def test_parses_headings_paragraphs_and_inline_markup():
    document = parse(
        "# Title\n\nA **bold** and *italic* `value` with "
        "[a link](https://example.com).\nnext"
    )

    assert document.blocks == (
        Heading(1, (Text("Title"),)),
        Paragraph(
            (
                Text("A "),
                Strong((Text("bold"),)),
                Text(" and "),
                Emphasis((Text("italic"),)),
                Text(" "),
                InlineCode("value"),
                Text(" with "),
                Link((Text("a link"),), "https://example.com"),
                Text("."),
                LineBreak(),
                Text("next"),
            )
        ),
    )


def test_parses_bold_italic_with_triple_markers():
    document = parse("A ***strong point*** and ___another one___.")

    assert document.blocks == (
        Paragraph(
            (
                Text("A "),
                Strong((Emphasis((Text("strong point"),)),)),
                Text(" and "),
                Strong((Emphasis((Text("another one"),)),)),
                Text("."),
            )
        ),
    )


def test_parses_nested_markers_of_different_lengths():
    document = parse("*foo **bar** baz* and __bold _in_ italic__")

    assert document.blocks == (
        Paragraph(
            (
                Emphasis(
                    (
                        Text("foo "),
                        Strong((Text("bar"),)),
                        Text(" baz"),
                    )
                ),
                Text(" and "),
                Strong(
                    (
                        Text("bold "),
                        Emphasis((Text("in"),)),
                        Text(" italic"),
                    )
                ),
            )
        ),
    )


def test_extra_star_after_bold_stays_literal():
    document = parse("**a***")

    assert document.blocks == (Paragraph((Strong((Text("a"),)), Text("*"))),)


def test_parses_lists_quotes_and_fenced_code():
    document = parse(
        "- one\n- **two**\n\n1. first\n2. second\n\n"
        "> quoted\n> text\n\n```python\nprint('x')\n```"
    )

    assert isinstance(document.blocks[0], ListBlock)
    assert document.blocks[0].ordered is False
    assert isinstance(document.blocks[1], ListBlock)
    assert document.blocks[1].ordered is True
    assert document.blocks[2] == BlockQuote(
        (Paragraph((Text("quoted"), LineBreak(), Text("text"))),)
    )
    assert document.blocks[3] == CodeBlock("print('x')", "python")


def test_fence_with_info_string_does_not_close_block():
    document = parse("```\ncode\n```python\nmore code\n```")

    assert document.blocks == (CodeBlock("code\n```python\nmore code", None),)


def test_parses_details_only_for_matching_directives():
    document = parse(":::details Linux\nUse `ip -br address`.\n:::enddetails")

    assert document.blocks == (
        Details(
            "Linux",
            (
                Paragraph(
                    (Text("Use "), InlineCode("ip -br address"), Text("."))
                ),
            ),
        ),
    )


def test_details_directive_inside_fence_does_not_close_details():
    document = parse(
        ":::details Syntax\n"
        "```\n"
        ":::details Example\ncontent\n:::enddetails\n"
        "```\n"
        ":::enddetails"
    )

    details = document.blocks[0]
    assert isinstance(details, Details)
    assert details.blocks == (
        CodeBlock(":::details Example\ncontent\n:::enddetails", None),
    )


def test_raw_html_and_malformed_details_are_literal():
    document = parse("<b>literal</b>\n\n:::details Missing\ncontent")

    assert document.blocks == (
        Paragraph((Text("<b>literal</b>"),)),
        Paragraph(
            (
                Text(":::details Missing"),
                LineBreak(),
                Text("content"),
            )
        ),
    )


def test_excessive_nesting_stays_literal_instead_of_recursing():
    document = parse(">" * 500 + " deep")

    text = ""
    blocks = list(document.blocks)
    while blocks:
        block = blocks.pop()
        blocks.extend(getattr(block, "blocks", ()))
        for inline in getattr(block, "children", ()):
            text += getattr(inline, "value", "")
    assert "deep" in text


def test_block_handlers_report_where_parsing_continues():
    parser = MarkdownParser()
    lines = ["# Title", "", "text"]

    assert parser._parse_heading(lines, 0, 0) == (
        Heading(1, (Text("Title"),)),
        1,
    )
    assert parser._parse_paragraph(lines, 2) == (
        Paragraph((Text("text"),)),
        3,
    )


def test_code_block_handler_tolerates_a_missing_closing_fence():
    parser = MarkdownParser()

    assert parser._parse_code_block(["```python", "code"], 0, 0) == (
        CodeBlock("code", "python"),
        2,
    )
    assert parser._parse_code_block(["```", "a", "```", "b"], 0, 0) == (
        CodeBlock("a", None),
        3,
    )


def test_details_handler_declines_unmatched_and_unclosed_directives():
    parser = MarkdownParser()

    assert parser._parse_details([":::note Not ours"], 0, 0) is None
    assert parser._parse_details([":::details Unclosed"], 0, 0) is None
    assert parser._parse_details(
        [":::details T", "content", ":::enddetails"], 0, 0
    ) == (
        Details("T", (Paragraph((Text("content"),)),)),
        3,
    )


def test_list_handler_stops_at_the_other_list_kind():
    parser = MarkdownParser()

    assert parser._parse_list(["plain"], 0, 0) is None
    assert parser._parse_list(["- one", "1. two", "- three"], 0, 0) == (
        ListBlock(False, (ListItem((Paragraph((Text("one"),)),)),)),
        1,
    )
    assert parser._parse_list(["1. one", "2. two"], 0, 0) == (
        ListBlock(
            True,
            (
                ListItem((Paragraph((Text("one"),)),)),
                ListItem((Paragraph((Text("two"),)),)),
            ),
        ),
        2,
    )


def test_starts_block_covers_every_construct():
    parser = MarkdownParser()
    starting = ("# h", "```", ":::details T", "> quote", "- item", "1. item")
    continuing = ("plain text", "  indented", ":::note")

    assert all(parser._starts_block(line) for line in starting)
    assert not any(parser._starts_block(line) for line in continuing)
