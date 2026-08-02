"""Laser PWM row widgets."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class _PwmRow(SpinRow):
    """Base for rows shown only when the selected head has PWM."""

    def __init__(
        self,
        editor: Any,
        step: Any,
        attr: str,
        title: str,
        subtitle: str,
    ):
        super().__init__(
            editor,
            step,
            attr,
            title,
            subtitle,
            1,
            100000,
            1,
            0,
            is_int=True,
        )

    def _sync_dependencies(self):
        machine = self.get_machine()
        head = self.get_selected_head()
        if machine is None or head is None:
            self.set_visible(False)
            return
        self.set_visible(machine.get_pwm_params(head) is not None)


class FrequencyRow(_PwmRow):
    """A spin row bound to the ``LaserStep.frequency`` attribute."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "frequency",
            _("Frequency"),
            _("Laser PWM frequency in Hz"),
        )


class PulseWidthRow(_PwmRow):
    """A spin row bound to the ``LaserStep.pulse_width`` attribute."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "pulse_width",
            _("Pulse Width"),
            _("Laser PWM pulse width in ns"),
        )
