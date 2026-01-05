def get_spinrow_int(spinrow):
    # Workaround: Adw.SpinRow seems to have a bug that the value is not
    # always updated if it was edited using the keyboard in the edit
    # field. I.e. get_value() still returns the previous value.
    # So I convert it manually from text if possible.
    try:
        value = int(spinrow.get_text())
    except ValueError:
        value = int(spinrow.get_value())
    lower = spinrow.get_adjustment().get_lower()
    upper = spinrow.get_adjustment().get_upper()
    return max(lower, min(value, upper))


def get_spinrow_float(spinrow):
    # Workaround: Adw.SpinRow seems to have a bug that the value is not
    # always updated if it was edited using the keyboard in the edit
    # field. I.e. get_value() still returns the previous value.
    # So I convert it manually from text if possible.
    try:
        value = float(spinrow.get_text())
    except ValueError:
        value = float(spinrow.get_value())
    lower = spinrow.get_adjustment().get_lower()
    upper = spinrow.get_adjustment().get_upper()
    return max(lower, min(value, upper))
