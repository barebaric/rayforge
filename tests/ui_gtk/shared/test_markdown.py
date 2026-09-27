# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Adw, GLib, Gtk

from rayforge.ui_gtk.shared.expander import Expander
from rayforge.ui_gtk.shared.markdown import (
    MarkdownEditor,
    MarkdownEditorDialog,
    MarkdownPreviewEditor,
    MarkdownToolbar,
    MarkdownView,
    build_help_markdown,
)

pytestmark = pytest.mark.ui


def _widgets(widget):
    yield widget
    child = widget.get_first_child()
    while child:
        yield from _widgets(child)
        child = child.get_next_sibling()


def _labels(widget):
    return [
        child.get_text()
        for child in _widgets(widget)
        if isinstance(child, Gtk.Label)
    ]


def test_markdown_view_renders_supported_nodes_and_safe_text(
    ui_context_initializer,
):
    view = MarkdownView(
        "# Heading\n\n**bold** and *italic* `<b>not html</b>` and `sample` "
        "with [a link](https://example.com)\n\n"
        "- one\n- two\n\n:::details More\nDetails\n:::enddetails"
    )

    labels = _labels(view)
    assert "Heading" in "".join(labels)
    assert "<b>not html</b>" in "".join(labels)
    paragraph = next(
        label
        for label in _widgets(view)
        if isinstance(label, Gtk.Label) and "bold" in label.get_text()
    )
    assert paragraph.get_use_markup()
    assert "<b>bold</b>" in paragraph.get_label()
    assert "<i>italic</i>" in paragraph.get_label()
    assert "<tt>sample</tt>" in paragraph.get_label()
    assert 'background="#e8edf2"' in paragraph.get_label() or (
        'background="#42464c"' in paragraph.get_label()
    )
    assert '<a href="https://example.com">a link</a>' in (
        paragraph.get_label()
    )
    assert any(isinstance(child, Expander) for child in _widgets(view))


def test_markdown_links_only_activate_safe_schemes(ui_context_initializer):
    view = MarkdownView(
        "[web](https://example.com) "
        "[email](mailto:help@example.com) "
        "[unsafe](javascript:alert(1))"
    )
    label = next(
        child
        for child in _widgets(view)
        if isinstance(child, Gtk.Label) and "web" in child.get_text()
    )

    assert label.get_use_markup()
    assert '<a href="https://example.com">web</a>' in label.get_label()
    assert '<a href="mailto:help@example.com">email</a>' in (label.get_label())
    assert "javascript:" not in label.get_label()
    assert "unsafe" in label.get_text()


def test_markdown_list_items_keep_inline_content_together(
    ui_context_initializer,
):
    view = MarkdownView(
        "- **First item** with a [link](https://example.com)\n"
        "- Second item with `inline code`"
    )
    labels = [
        child.get_text()
        for child in _widgets(view)
        if isinstance(child, Gtk.Label)
    ]

    assert any("First item with a link" in label for label in labels)
    assert any("Second item with inline code" in label for label in labels)


def test_code_blocks_have_distinct_styles_inside_details(
    ui_context_initializer,
):
    view = MarkdownView(
        "```\nnormal code\n```\n\n"
        ":::details More\n\n```\ndetails code\n```\n:::enddetails"
    )
    code_blocks = [
        child
        for child in _widgets(view)
        if isinstance(child, Gtk.Label)
        and (
            child.has_css_class("markdown-code")
            or child.has_css_class("markdown-code-in-details")
        )
    ]

    assert len(code_blocks) == 2
    assert any(child.has_css_class("markdown-code") for child in code_blocks)
    assert any(
        child.has_css_class("markdown-code-in-details")
        for child in code_blocks
    )


def test_details_keep_expanded_state_across_rerenders(ui_context_initializer):
    source = (
        ":::details Alpha\nA\n:::enddetails\n\n"
        ":::details Alpha\nB\n:::enddetails"
    )
    view = MarkdownView(source)
    expanders = [
        child for child in _widgets(view) if isinstance(child, Expander)
    ]
    assert len(expanders) == 2
    assert not expanders[1].revealer.get_reveal_child()

    expanders[1].set_expanded(True)
    view.set_text(source + "\n\nTrailing paragraph")

    expanders = [
        child for child in _widgets(view) if isinstance(child, Expander)
    ]
    assert not expanders[0].revealer.get_reveal_child()
    assert expanders[1].revealer.get_reveal_child()


def test_details_state_is_forgotten_when_sections_disappear(
    ui_context_initializer,
):
    view = MarkdownView(":::details Alpha\nA\n:::enddetails")
    expanders = [
        child for child in _widgets(view) if isinstance(child, Expander)
    ]
    expanders[0].set_expanded(True)
    assert view._details_expanded

    view.set_text("Just a paragraph")
    assert view._details_expanded == {}


def test_preview_split_handle_is_wide_and_styled(ui_context_initializer):
    editor = MarkdownPreviewEditor("text")
    assert editor._paned.get_wide_handle()
    assert editor._paned.has_css_class("even-split")


def test_toolbar_wraps_selection_with_markers(ui_context_initializer):
    editor = MarkdownEditor("focus height")
    start, end = editor.buffer.get_bounds()
    editor.buffer.select_range(start, end)
    toolbar = MarkdownToolbar(editor)

    toolbar.buttons["bold"].emit("clicked")

    assert editor.get_text() == "**focus height**"


def test_toolbar_inserts_example_without_a_selection(ui_context_initializer):
    editor = MarkdownEditor("")
    toolbar = MarkdownToolbar(editor)

    toolbar.buttons["link"].emit("clicked")

    text = editor.get_text()
    assert text.startswith("[")
    assert text.endswith("](https://example.com)")


def test_toolbar_numbers_every_selected_list_line(ui_context_initializer):
    editor = MarkdownEditor("first\nsecond\nthird")
    start, end = editor.buffer.get_bounds()
    editor.buffer.select_range(start, end)
    toolbar = MarkdownToolbar(editor)

    toolbar.buttons["numbered-list"].emit("clicked")

    assert editor.get_text() == "1. first\n2. second\n3. third"


def test_toolbar_block_snippets_stay_parseable(ui_context_initializer):
    editor = MarkdownEditor("")
    toolbar = MarkdownToolbar(editor)

    toolbar.buttons["details"].emit("clicked")

    view = MarkdownView(editor.get_text())
    assert any(isinstance(child, Expander) for child in _widgets(view))


def test_help_markdown_only_uses_supported_syntax(ui_context_initializer):
    view = MarkdownView(build_help_markdown())
    labels = [
        child.get_text()
        for child in _widgets(view)
        if isinstance(child, Gtk.Label)
    ]

    assert any(":::details" in label for label in labels)
    assert not any(":::enddetails" == label.strip() for label in labels[:1])
    assert any(
        child.has_css_class("markdown-code")
        for child in _widgets(view)
        if isinstance(child, Gtk.Label)
    )


def test_help_pane_replaces_preview_and_closes_on_edit(ui_context_initializer):
    editor = MarkdownPreviewEditor("start")

    editor.toolbar.help_button.set_active(True)
    assert editor.preview_stack.get_visible_child_name() == "help"

    editor.editor.set_text("changed")
    assert editor.preview_stack.get_visible_child_name() == "preview"
    assert not editor.toolbar.help_button.get_active()


def test_markdown_renders_bold_italic_as_nested_markup(ui_context_initializer):
    view = MarkdownView("A ***strong point*** here.")
    labels = [
        child.get_label()
        for child in _widgets(view)
        if isinstance(child, Gtk.Label)
    ]

    assert any("<b><i>strong point</i></b>" in label for label in labels)


def test_toolbar_sits_above_the_split(ui_context_initializer):
    editor = MarkdownPreviewEditor("text")

    assert editor.toolbar.get_parent() is editor.action_bar
    assert editor.layout_dropdown.get_parent() is editor.action_bar
    assert editor.toolbar not in list(_widgets(editor._paned))


def test_markdown_editor_round_trips_raw_source(ui_context_initializer):
    editor = MarkdownEditor("# title\n\n:::unknown\nraw\n")
    assert editor.get_text() == "# title\n\n:::unknown\nraw\n"
    editor.set_text("<script>alert(1)</script>")
    assert editor.get_text() == "<script>alert(1)</script>"


def test_preview_editor_debounces_until_refresh(ui_context_initializer):
    preview = MarkdownPreviewEditor("before")
    preview.editor.set_text("after")
    assert preview.preview.get_text() == "before"
    preview.refresh_preview()
    assert preview.preview.get_text() == "after"


def test_preview_editor_layout_can_be_selected(ui_context_initializer):
    preview = MarkdownPreviewEditor()
    preview.layout_dropdown.set_selected(0)
    assert preview.get_layout() == MarkdownPreviewEditor.SIDE_BY_SIDE
    preview._on_width_changed()
    assert preview.get_layout() == MarkdownPreviewEditor.SIDE_BY_SIDE
    preview.layout_dropdown.set_selected(1)
    assert preview.get_layout() == MarkdownPreviewEditor.STACKED


def test_preview_editor_splits_panes_evenly_until_dragged(
    ui_context_initializer,
):
    def settle():
        for _ in range(25):
            while GLib.MainContext.default().iteration(False):
                pass

    preview = MarkdownPreviewEditor()
    window = Gtk.Window()
    window.set_default_size(900, 650)
    window.set_child(preview)
    window.present()
    settle()

    paned = preview._paned
    assert paned.get_resize_start_child()
    assert paned.get_resize_end_child()
    assert preview.get_layout() == MarkdownPreviewEditor.SIDE_BY_SIDE
    assert paned.get_position() == paned.get_width() // 2

    paned.set_position(200)
    settle()
    assert paned.get_position() == 200

    preview.layout_dropdown.set_selected(1)
    settle()
    assert preview.get_layout() == MarkdownPreviewEditor.STACKED
    assert paned.get_position() == paned.get_height() // 2


def test_preview_editor_stays_even_after_resizing(ui_context_initializer):
    def settle():
        for _ in range(25):
            while GLib.MainContext.default().iteration(False):
                pass

    preview = MarkdownPreviewEditor()
    window = Gtk.Window()
    window.set_default_size(1150, 750)
    window.set_child(preview)
    window.present()
    settle()

    paned = preview._paned
    assert preview.get_layout() == MarkdownPreviewEditor.SIDE_BY_SIDE
    assert paned.get_position() == paned.get_width() // 2

    window.set_default_size(820, 600)
    settle()
    assert paned.get_position() == paned.get_width() // 2
    window.destroy()


def test_markdown_editor_dialog_preserves_description_and_source(
    ui_context_initializer,
):
    dialog = MarkdownEditorDialog(
        title="Edit notes",
        description="Machine-specific guidance",
        initial_text="raw **source**",
    )

    assert dialog.get_title() == "Edit notes"
    assert "Machine-specific guidance" in _labels(dialog)
    assert dialog.get_text() == "raw **source**"
    dialog._on_save()
    assert dialog.get_result() == "raw **source**"


def test_markdown_editor_dialog_uses_draggable_header_bar(
    ui_context_initializer,
):
    dialog = MarkdownEditorDialog(title="Edit notes")
    header_bars = [
        child for child in _widgets(dialog) if isinstance(child, Adw.HeaderBar)
    ]

    assert len(header_bars) == 1
    assert dialog.cancel_button.get_ancestor(Adw.HeaderBar) is header_bars[0]
    assert dialog.save_button.get_ancestor(Adw.HeaderBar) is header_bars[0]


def test_markdown_editor_dialog_allows_omitted_description(
    ui_context_initializer,
):
    dialog = MarkdownEditorDialog(title="Edit notes", initial_text="notes")

    assert dialog.get_title() == "Edit notes"
    assert dialog.get_text() == "notes"
    assert "Machine-specific guidance" not in _labels(dialog)
