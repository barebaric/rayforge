"""The built-in help page describing the supported Markdown subset."""

from gettext import gettext as _


def build_help_markdown() -> str:
    """Describe the supported syntax using the supported syntax itself.

    Only the prose is translated; every example stays a literal so a
    translation can never break the syntax it demonstrates.
    """
    topics = (
        (
            _("Bold and italic"),
            _("Wrap words in two stars for bold, or one star for italic."),
            "**bold text**\n*italic text*",
        ),
        (
            _("Headings"),
            _("Start a line with number signs. More signs, smaller heading."),
            "# Main title\n## Section\n### Subsection",
        ),
        (
            _("Lists"),
            _(
                "Start lines with a dash for bullets, or with a number and a "
                "dot for a numbered list."
            ),
            "- First item\n- Second item\n\n1. First step\n2. Second step",
        ),
        (
            _("Links"),
            _(
                "Put the link text in square brackets and the address in "
                "round brackets."
            ),
            "[Rayforge website](https://rayforge.org)",
        ),
        (
            _("Inline code"),
            _(
                "Wrap short commands in single backticks to keep them "
                "readable."
            ),
            "Send `G0 X0 Y0` to move to the origin.",
        ),
        (
            _("Quotes"),
            _("Start a line with a greater-than sign to call out a remark."),
            "> Always wear laser safety glasses.",
        ),
        (
            _("Expandable sections"),
            _("Hide long details behind a title that readers can unfold."),
            (
                ":::details Advanced tuning\n"
                "Only needed for thick material.\n"
                ":::enddetails"
            ),
        ),
    )
    parts = [
        f"# {_('Formatting help')}",
        _("Type the text shown below to get the matching result."),
    ]
    for name, description, example in topics:
        parts.append(f"## {name}")
        parts.append(description)
        parts.append(f"```\n{example}\n```")
    parts.append(f"## {_('Code blocks')}")
    parts.append(
        _(
            "Put a line with three backticks before and after a block of "
            "commands to show it unchanged."
        )
    )
    return "\n\n".join(parts)
