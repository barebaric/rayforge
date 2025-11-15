from .var import Var


class TextAreaVar(Var[str]):
    """
    A Var subclass for multi-line string values that hints to the UI
    that it should be represented by a text area (Gtk.TextView) rather than
    a single-line entry.
    """

    pass
