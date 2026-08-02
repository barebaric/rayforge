"""Laser-domain capabilities interpreting driver-reported features."""

from gettext import gettext as _
from typing import Optional, Tuple

from rayforge.core.capability import Capability
from rayforge.core.varset import IntVar, VarSet
from rayforge.machine.driver.driver import DriverFeatures, PWMParams


class PWMCapability(Capability):
    """Laser-addon capability interpreting ``DriverFeatures.pwm``."""

    def __init__(self, params: PWMParams):
        self._params = params

    @property
    def name(self) -> str:
        return "PWM"

    @property
    def label(self) -> str:
        return _("PWM")

    @property
    def varset(self) -> VarSet:
        return VarSet(
            vars=[
                IntVar(
                    key="frequency",
                    label=_("Frequency"),
                    description=_("PWM frequency in Hz"),
                    default=self._params.frequency,
                    min_val=1,
                    max_val=self._params.max_frequency,
                ),
                IntVar(
                    key="pulse_width",
                    label=_("Pulse Width"),
                    description=_("Pulse width in microseconds"),
                    default=self._params.pulse_width,
                    min_val=self._params.min_pulse_width,
                    max_val=self._params.max_pulse_width,
                ),
            ]
        )


def pwm_capability_from_features(
    features: DriverFeatures,
) -> Optional[PWMCapability]:
    """Interpret driver-reported PWM features into a ``PWMCapability``."""
    if features.pwm:
        return PWMCapability(features.pwm)
    return None


def resolve_pwm_capability(
    features: DriverFeatures,
) -> Tuple[Capability, ...]:
    """Resolver: turn driver-reported PWM features into capabilities."""
    cap = pwm_capability_from_features(features)
    return (cap,) if cap else ()
