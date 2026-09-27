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
