from gettext import gettext as _

from .choicevar import ChoiceVar


class LabeledChoiceVar(ChoiceVar):
    """A :class:`ChoiceVar` that shows human-readable labels while
    storing machine-readable values.

    The ``choices`` are given as ``(label, value)`` pairs. The UI
    dropdown shows the labels; the stored value is the corresponding
    value. This is used for enum-backed recipe settings (e.g.
    ``CutSide``) so the editor displays "Centerline" rather than
    "CENTERLINE".
    """

    display_name = _("Choice (Labeled)")

    def __init__(
        self,
        key: str,
        label: str,
        choices: list[tuple[str, str]],
        description: str | None = None,
        default: str | None = None,
        value: str | None = None,
        allow_none: bool = True,
    ):
        self._label_to_value = {lbl: val for lbl, val in choices}
        self._value_to_label = {val: lbl for lbl, val in choices}
        display_choices = [lbl for lbl, _ in choices]
        super().__init__(
            key=key,
            label=label,
            choices=display_choices,
            description=description,
            default=default,
            value=value,
            allow_none=allow_none,
        )
        valid_values = list(self._value_to_label)

        def _labeled_validator(val: str | None):
            if val is not None and val not in valid_values:
                raise ValueError(
                    f"Value '{val}' is not a valid choice for '{self.key}'"
                )

        self.validator = _labeled_validator

    def get_display_for_value(self, value: str | None) -> str | None:
        if value is None:
            return None
        return self._value_to_label.get(value, value)

    def get_value_for_display(self, display: str | None) -> str | None:
        if display is None:
            return None
        return self._label_to_value.get(display, display)
