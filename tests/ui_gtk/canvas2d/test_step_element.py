"""Tests for the canvas StepElement visibility rules."""

from unittest.mock import MagicMock

import pytest

from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.ui_gtk.canvas2d.elements.step import StepElement


@pytest.mark.ui
def test_disabled_workflow_hides_its_steps(ui_context_initializer):
    layer = Layer("Layer")
    workflow = layer.workflows[0]
    step = Step(typelabel="Contour", name="Cut")
    workflow.add_step(step)
    element = StepElement(step=step, pipeline=MagicMock())

    assert element.visible is True

    workflow.set_enabled(False)
    assert element.visible is False

    workflow.set_enabled(True)
    assert element.visible is True
