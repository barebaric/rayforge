from __future__ import annotations
from typing import Optional, List, Dict, Any
from .var import Var


class ChoiceVar(Var[str]):
    """
    A variable that represents a choice from a predefined list of strings.
    """

    def __init__(
        self,
        key: str,
        label: str,
        choices: List[str],
        description: Optional[str] = None,
        default: Optional[str] = None,
        value: Optional[str] = None,
    ):
        """
        Initializes a new ChoiceVar instance.

        Args:
            key: The unique machine-readable identifier.
            label: The human-readable name for the UI.
            choices: A list of string options for the user to choose from.
            description: A longer, human-readable description.
            default: The default value. Must be one of the choices.
            value: The initial value. If provided, it overrides the default.
        """
        super().__init__(
            key=key,
            label=label,
            var_type=str,
            description=description,
            default=default,
            value=value,
        )
        self.choices = choices

        # Validator to ensure the value is always one of the allowed choices.
        def _choice_validator(val: Optional[str]):
            if val is not None and val not in self.choices:
                raise ValueError(
                    f"Value '{val}' is not a valid choice for '{self.key}'"
                )

        self.validator = _choice_validator

    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data.update({"choices": self.choices})
        return data

    def get_display_for_value(self, value: Optional[str]) -> Optional[str]:
        """
        For simple ChoiceVar, the display value is the same as the stored
        value. Subclasses can override this for mapping.
        """
        return value

    def get_value_for_display(self, display: Optional[str]) -> Optional[str]:
        """
        For simple ChoiceVar, the stored value is the same as the display
        value. Subclasses can override this for mapping.
        """
        return display
