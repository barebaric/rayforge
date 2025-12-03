import pytest
from rayforge.core.expression import errors


def test_error_info_base_class():
    """Ensures the base ErrorInfo class abstract method raises an error."""
    with pytest.raises(NotImplementedError):
        errors.ErrorInfo().get_message()


def test_syntax_error_info():
    """Tests the SyntaxErrorInfo class."""
    error_info = errors.SyntaxErrorInfo(message="invalid syntax", offset=5)
    assert error_info.message == "invalid syntax"
    assert error_info.offset == 5
    assert error_info.get_message() == "Syntax Error: Invalid syntax"


def test_unknown_variable_info():
    """Tests the UnknownVariableInfo class."""
    error_info = errors.UnknownVariableInfo(name="width")
    assert error_info.name == "width"
    assert error_info.get_message() == "Unknown variable or function: 'width'"


def test_type_mismatch_info():
    """Tests the TypeMismatchInfo class."""
    error_info = errors.TypeMismatchInfo(
        operator="+", left_type=str, right_type=int
    )
    assert error_info.operator == "+"
    assert error_info.left_type == "str"
    assert error_info.right_type == "int"
    expected_msg = "Cannot use operator '+' between types 'str' and 'int'"
    assert error_info.get_message() == expected_msg


def test_validation_result_success():
    """Tests the factory method for a successful ValidationResult."""
    result = errors.ValidationResult.success()
    assert result.status == errors.ValidationStatus.OK
    assert result.error_info is None
    assert result.is_valid is True


def test_validation_result_failure():
    """Tests the factory method for a failed ValidationResult."""
    error_info = errors.UnknownVariableInfo(name="foo")
    result = errors.ValidationResult.failure(error_info)
    assert result.status == errors.ValidationStatus.ERROR
    assert result.error_info is error_info
    assert result.is_valid is False
