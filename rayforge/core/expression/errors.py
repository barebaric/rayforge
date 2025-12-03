from __future__ import annotations
import enum
from typing import Optional, Type


class ValidationStatus(enum.Enum):
    """Indicates the result of an expression validation."""

    OK = "ok"
    ERROR = "error"


class ErrorInfo:
    """Base class for detailed error information."""

    def get_message(self) -> str:
        """Returns a user-friendly, translatable error message."""
        raise NotImplementedError


class SyntaxErrorInfo(ErrorInfo):
    """Details for a syntax error."""

    def __init__(self, message: str, offset: int):
        self.message = message
        self.offset = offset

    def get_message(self) -> str:
        return _("Syntax Error: {message}").format(
            message=self.message.capitalize()
        )


class UnknownVariableInfo(ErrorInfo):
    """Details for an undefined variable or function."""

    def __init__(self, name: str):
        self.name = name

    def get_message(self) -> str:
        return _("Unknown variable or function: '{name}'").format(
            name=self.name
        )


class TypeMismatchInfo(ErrorInfo):
    """Details for an operation between incompatible types."""

    def __init__(
        self,
        operator: str,
        left_type: Type,
        right_type: Type,
    ):
        self.operator = operator
        self.left_type = left_type.__name__
        self.right_type = right_type.__name__

    def get_message(self) -> str:
        return _(
            "Cannot use operator '{op}' between types '{left}' and '{right}'"
        ).format(op=self.operator, left=self.left_type, right=self.right_type)


class ValidationResult:
    """A container for the complete result of a validation check."""

    def __init__(
        self,
        status: ValidationStatus,
        error_info: Optional[ErrorInfo] = None,
    ):
        self.status = status
        self.error_info = error_info

    @property
    def is_valid(self) -> bool:
        """Returns True if the validation status is OK."""
        return self.status == ValidationStatus.OK

    @classmethod
    def success(cls) -> ValidationResult:
        """Factory method for a successful result."""
        return cls(ValidationStatus.OK)

    @classmethod
    def failure(cls, error_info: ErrorInfo) -> ValidationResult:
        """Factory method for a failed result."""
        return cls(ValidationStatus.ERROR, error_info)
