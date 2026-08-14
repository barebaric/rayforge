from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from .var import ValidationError, Var


class IntVar(Var[int]):
    """A Var subclass for integer values with optional bounds."""

    display_name = _("Integer")

    def __init__(
        self,
        key: str,
        label: str,
        description: str | None = None,
        default: int | None = None,
        value: int | None = None,
        min_val: int | None = None,
        max_val: int | None = None,
        validator: Callable[[int | None], None] | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
        sensitive_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        self.min_val = min_val
        self.max_val = max_val

        def thevalidator(v: int | None):
            if self.min_val is not None and v is not None and v < self.min_val:
                raise ValidationError(
                    _("Value must be at least {min_val}.").format(
                        min_val=self.min_val
                    )
                )
            if self.max_val is not None and v is not None and v > self.max_val:
                raise ValidationError(
                    _("Value must be at most {max_val}.").format(
                        max_val=self.max_val
                    )
                )
            if validator:
                validator(v)

        super().__init__(
            key=key,
            label=label,
            var_type=int,
            description=description,
            default=default,
            value=value,
            validator=thevalidator,
            visible_when=visible_when,
            sensitive_when=sensitive_when,
        )

    def to_dict(self, include_value: bool = False) -> dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data.update({"min_val": self.min_val, "max_val": self.max_val})
        return data


class SliderIntVar(IntVar):
    """
    An IntVar subclass that hints to the UI that it should be
    represented by a slider rather than a spinbox.
    """

    display_name = _("Slider (Integer)")

    def __init__(
        self,
        key: str,
        label: str,
        description: str | None = None,
        default: int | None = None,
        value: int | None = None,
        min_val: int | None = None,
        max_val: int | None = None,
        validator: Callable[[int | None], None] | None = None,
        show_value: bool = True,
        format_suffix: str | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
        sensitive_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        self.show_value = show_value
        self.format_suffix = format_suffix
        super().__init__(
            key=key,
            label=label,
            description=description,
            default=default,
            value=value,
            min_val=min_val,
            max_val=max_val,
            validator=validator,
            visible_when=visible_when,
            sensitive_when=sensitive_when,
        )

    def to_dict(self, include_value: bool = False) -> dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data.update(
            {
                "show_value": self.show_value,
                "format_suffix": self.format_suffix,
            }
        )
        return data
