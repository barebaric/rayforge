import logging
from typing import Optional
from blinker import Signal
from gi.repository import Gdk, Gtk, Pango, GLib
from ...core.expression import (
    ExpressionContext,
    ExpressionTokenizer,
    ExpressionValidator,
    Token,
    TokenType,
)
from .gtk import apply_css


logger = logging.getLogger(__name__)

# Self-contained CSS for the widget
css = """
/* Styles for ExpressionEntry Widget */

/* Add a border and background to the frame to mimic a Gtk.Entry */
.expression-entry-frame {
  background-color: @theme_bg_color;
  border: 1px solid @borders;
}

/* Add a red border when the expression is invalid */
.expression-entry-frame.error {
  border-color: @error_color;
}

/* Set a transparent background for the TextView inside the frame */
.expression-entry-frame > GtkTextView {
  padding: 8px;
  background-color: transparent;
}

/* Label for displaying validation errors below the entry */
.expression-error-label {
    color: @error_color;
    margin: 10px;
}

.autocomplete-selector > contents {
  padding: 1px;
}
"""


class AutoCompleteSelector(Gtk.Popover):
    """
    A specialized Gtk.Popover for displaying autocompletion results.
    Its contents are explicitly made non-focusable to prevent them from
    stealing keyboard events from the text entry.
    """

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.add_css_class("autocomplete-selector")
        self.set_can_focus(False)

        self.list_box = Gtk.ListBox()
        self.list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.list_box.set_focusable(False)

        scroller = Gtk.ScrolledWindow(
            child=self.list_box,
            min_content_height=150,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
        )
        scroller.set_size_request(200, -1)

        self.set_child(scroller)
        self.set_autohide(False)
        self.set_has_arrow(False)
        self.set_position(Gtk.PositionType.BOTTOM)


class ExpressionEntry(Gtk.Box):
    """
    A GTK widget for entering mathematical expressions with live validation,
    syntax highlighting, and autocompletion.

    It uses a Gtk.Popover to display a completion list without grabbing
    input or disrupting the application layout.

    Signals:
        activated (blinker.Signal): Emitted when the user presses Enter. The
                                    sender is the ExpressionEntry instance.
        validated (blinker.Signal): Emitted after the text changes and is
                                    validated. The sender is the instance,
                                    and a keyword argument `is_valid` (bool)
                                    is provided.
    """

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        apply_css(css)

        # Signals
        self.activated = Signal()
        self.validated = Signal()

        self._context: Optional[ExpressionContext] = None
        self._validator = ExpressionValidator()
        self._tokenizer = ExpressionTokenizer()

        # Highlighting Tags
        self.tag_table = Gtk.TextTagTable()
        self._tags = {
            "variable": self._create_tag(
                self.tag_table, "variable", foreground="#809bbd"
            ),
            "function": self._create_tag(
                self.tag_table, "function", foreground="#f5c211"
            ),
            "number": self._create_tag(
                self.tag_table, "number", foreground="#3d84c7da"
            ),
            "string": self._create_tag(
                self.tag_table, "string", foreground="#A39464"
            ),
            "operator": self._create_tag(
                self.tag_table, "operator", weight=Pango.Weight.BOLD
            ),
            "paren": self._create_tag(self.tag_table, "paren"),
            "error": self._create_tag(
                self.tag_table, "error", underline=Pango.Underline.ERROR
            ),
        }

        # Build the main UI components
        self._buffer = Gtk.TextBuffer(tag_table=self.tag_table)
        self.textview = Gtk.TextView(
            buffer=self._buffer,
            wrap_mode=Gtk.WrapMode.NONE,
            accepts_tab=False,
            monospace=True,
            left_margin=12,
            right_margin=12,
            top_margin=10,
            bottom_margin=10,
        )
        self.textview.set_size_request(-1, 40)
        self.textview.set_vexpand(True)

        self.frame = Gtk.Frame(child=self.textview)
        self.frame.add_css_class("expression-entry-frame")
        self.append(self.frame)

        # Use our dedicated AutoCompleteMenu
        self._completion_menu = AutoCompleteSelector()
        # The popover's position is relative to its parent. Setting the parent
        # to the textview allows us to position it relative to the text.
        self._completion_menu.set_parent(self.textview)

        # Build and add the error label
        self._error_label = Gtk.Label(
            wrap=True, wrap_mode=Pango.WrapMode.WORD_CHAR, xalign=0
        )
        self._error_label.add_css_class("expression-error-label")
        self._error_label.set_visible(False)
        self.append(self._error_label)

        # Connect Controllers and Signals
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.textview.add_controller(key_controller)

        self._buffer_changed_handler_id = self._buffer.connect(
            "changed", self._on_buffer_changed
        )
        self._completion_menu.list_box.connect(
            "row-activated", self._on_completion_activated
        )

    def _create_tag(self, table: Gtk.TextTagTable, name: str, **properties):
        """Creates a Gtk.TextTag and applies properties directly."""
        tag = Gtk.TextTag(name=name)
        for key, value in properties.items():
            tag.set_property(key, value)
        table.add(tag)
        return tag

    def set_context(self, context: ExpressionContext):
        """
        Sets the expression context, which defines the available variables
        and functions for validation and autocompletion.
        """
        self._context = context
        self._populate_completion_model()
        self._validate_and_highlight()

    def get_text(self) -> str:
        """Returns the text content of the entry."""
        start, end = self._buffer.get_bounds()
        return self._buffer.get_text(start, end, True)

    def set_text(self, text: str):
        """Sets the text content of the entry."""
        self._buffer.set_text(text)

    def _on_buffer_changed(self, buffer: Gtk.TextBuffer):
        self._validate_and_highlight()
        self._update_completion_popup()

    def _validate_and_highlight(self):
        """
        Runs the full validation and syntax highlighting pipeline.
        This is the core update logic of the widget.
        """
        if not self._context:
            return

        text = self.get_text()
        # Validation
        result = self._validator.validate(text, self._context)

        if result.is_valid:
            self.frame.remove_css_class("error")
            self._error_label.set_text("")
            self._error_label.set_visible(False)
        else:
            self.frame.add_css_class("error")
            if result.error_info:
                message = result.error_info.get_message()
                self._error_label.set_text(message)
                self._error_label.set_visible(True)
            else:
                # Fallback for invalid state with no specific message
                self._error_label.set_text("")
                self._error_label.set_visible(False)

        self.validated.send(self, is_valid=result.is_valid)

        # Highlighting
        self._buffer.remove_all_tags(*self._buffer.get_bounds())
        tokens = self._tokenizer.tokenize(text)
        for token in tokens:
            self._apply_highlighting_for_token(token)

    def _apply_highlighting_for_token(self, token: Token):
        """Applies the correct Gtk.TextTag for a given Token."""
        if not self._context:
            return
        start = self._buffer.get_iter_at_offset(token.start)
        end = self._buffer.get_iter_at_offset(token.end)

        tag_name: Optional[str] = None
        if token.type == TokenType.NUMBER:
            tag_name = "number"
        elif token.type == TokenType.STRING:
            tag_name = "string"
        elif token.type == TokenType.OPERATOR:
            tag_name = "operator"
        elif token.type == TokenType.PARENTHESIS:
            tag_name = "paren"
        elif token.type == TokenType.NAME:
            if self._context.is_variable(token.value):
                tag_name = "variable"
            elif self._context.is_function(token.value):
                tag_name = "function"
            else:
                tag_name = "error"

        if tag_name:
            self._buffer.apply_tag_by_name(tag_name, start, end)

    def _on_key_pressed(self, controller, keyval, keycode, state):
        is_completion_visible = self._completion_menu.is_visible()

        if is_completion_visible:
            if keyval == Gdk.KEY_Up:
                self._navigate_completion(-1)
                return True
            if keyval == Gdk.KEY_Down:
                # Select first item if nothing is selected
                if not self._completion_menu.list_box.get_selected_row():
                    self._select_first_visible_completion()
                else:
                    self._navigate_completion(1)
                return True
            if keyval in (Gdk.KEY_Tab, Gdk.KEY_ISO_Left_Tab):
                completion_row = (
                    self._completion_menu.list_box.get_selected_row()
                )
                # If nothing is selected, find the first visible one
                if not completion_row:
                    completion_row = self._find_first_visible_row()

                if completion_row:
                    self._on_completion_activated(
                        self._completion_menu.list_box, completion_row
                    )
                return True
            if keyval == Gdk.KEY_Escape:
                self._completion_menu.popdown()
                return True

        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            selected_row = self._completion_menu.list_box.get_selected_row()
            if is_completion_visible and selected_row:
                self._on_completion_activated(
                    self._completion_menu.list_box, selected_row
                )
            else:
                self.activated.send(self)
            return True

        return False

    def _do_apply_completion(self, completion_text: str) -> bool:
        """Safely modifies the buffer. Runs deferred via GLib.idle_add."""
        self._buffer.handler_block(self._buffer_changed_handler_id)
        self._buffer.begin_user_action()

        cursor_iter = self._buffer.get_iter_at_mark(self._buffer.get_insert())
        word_start_iter = cursor_iter.copy()
        if word_start_iter.backward_word_start():
            self._buffer.delete(word_start_iter, cursor_iter)
        self._buffer.insert_at_cursor(completion_text)

        self._buffer.end_user_action()
        self._buffer.handler_unblock(self._buffer_changed_handler_id)

        self._completion_menu.popdown()
        self._validate_and_highlight()

        return GLib.SOURCE_REMOVE

    # Autocompletion Logic
    def _populate_completion_model(self):
        """Fills the completion listbox with items from the context."""
        list_box = self._completion_menu.list_box
        # Clear existing children efficiently
        while child := list_box.get_first_child():
            list_box.remove(child)

        if not self._context:
            return

        symbols = sorted(
            list(self._context.variables.keys())
            + list(self._context.functions.keys())
        )
        for symbol in symbols:
            label = Gtk.Label(label=symbol, xalign=0)
            list_box.append(label)
            # Make the row containing the label non-focusable
            row = label.get_parent()
            if isinstance(row, Gtk.ListBoxRow):
                row.set_focusable(False)

    def _update_completion_popup(self):
        """Shows or hides the completion popover based on the current text."""
        cursor_iter = self._buffer.get_iter_at_mark(self._buffer.get_insert())
        word_start_iter = cursor_iter.copy()

        # Find the start of the current word
        if not word_start_iter.backward_word_start():
            if self._completion_menu.is_visible():
                self._completion_menu.popdown()
            return

        partial_word = self._buffer.get_text(
            word_start_iter, cursor_iter, True
        )

        # Filter the list and find the first visible match
        has_matches = False
        list_box = self._completion_menu.list_box
        child = list_box.get_first_child()
        while child:
            if isinstance(child, Gtk.ListBoxRow):
                label = child.get_child()
                if isinstance(label, Gtk.Label):
                    is_match = label.get_label().startswith(partial_word)
                    child.set_visible(is_match)
                    if is_match:
                        has_matches = True
            child = child.get_next_sibling()

        if has_matches and partial_word:
            # Position the popover under the word being typed.
            # Get the Gdk.Rectangle for the character at the start of the word.
            # Coordinates are relative to the buffer's contents.
            start_location = self.textview.get_iter_location(word_start_iter)

            # Convert buffer coordinates to coordinates relative to the
            # TextView widget.
            start_x, start_y = self.textview.buffer_to_window_coords(
                Gtk.TextWindowType.WIDGET, start_location.x, start_location.y
            )

            # Create the rectangle for the popover to point to.
            # Gtk.Popover centers its arrow on the provided rectangle.
            # To left-align the popover with the start of the word, we
            # give it a rectangle that is only 1 pixel wide at the word's
            # starting x-coordinate.
            pointing_rect = Gdk.Rectangle()
            pointing_rect.x = start_x
            pointing_rect.y = start_y
            pointing_rect.width = 1  # This is the key change
            pointing_rect.height = start_location.height

            # Tell the popover where to point, relative to its parent
            # (the TextView).
            self._completion_menu.set_pointing_to(pointing_rect)

            if not self._completion_menu.is_visible():
                self._completion_menu.popup()
            self._select_first_visible_completion()
        else:
            if self._completion_menu.is_visible():
                self._completion_menu.popdown()

    def _find_first_visible_row(self) -> Optional[Gtk.ListBoxRow]:
        """Finds the first visible row without selecting it."""
        child = self._completion_menu.list_box.get_first_child()
        while child:
            if isinstance(child, Gtk.ListBoxRow) and child.is_visible():
                return child
            child = child.get_next_sibling()
        return None

    def _select_first_visible_completion(self):
        """Selects the first visible row, safe for arrow keys."""
        first_row = self._find_first_visible_row()
        if first_row:
            self._completion_menu.list_box.select_row(first_row)

    def _navigate_completion(self, step: int):
        """Moves the selection in the completion list up or down by 'step'."""
        list_box = self._completion_menu.list_box
        selected = list_box.get_selected_row()
        if not selected:
            return

        next_row: Optional[Gtk.Widget] = None
        if step > 0:
            sibling = selected.get_next_sibling()
            while sibling:
                if sibling.is_visible():
                    next_row = sibling
                    break
                sibling = sibling.get_next_sibling()
        else:
            sibling = selected.get_prev_sibling()
            while sibling:
                if sibling.is_visible():
                    next_row = sibling
                    break
                sibling = sibling.get_prev_sibling()

        if isinstance(next_row, Gtk.ListBoxRow):
            list_box.select_row(next_row)

    def _on_completion_activated(self, listbox, row: Optional[Gtk.ListBoxRow]):
        """Inserts the selected completion into the text buffer."""
        if not row:
            return
        label = row.get_child()
        if not isinstance(label, Gtk.Label):
            return

        completion_text = label.get_label()
        self._do_apply_completion(completion_text)
