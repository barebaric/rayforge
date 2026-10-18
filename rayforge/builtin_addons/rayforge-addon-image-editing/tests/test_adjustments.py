from image_editing.adjustments import ImageAdjustments

from rayforge.core.image_edit import ImageEditStep


def test_defaults_need_no_edits():
    assert ImageAdjustments().to_edits() == {"sharpen": None, "tone": None}


def test_tone_values_become_a_tone_edit():
    edits = ImageAdjustments(brightness=10, gamma=1.5).to_edits()
    assert edits["sharpen"] is None
    assert edits["tone"] == ImageEditStep(
        "tone", {"brightness": 10, "contrast": 0.0, "gamma": 1.5}
    )


def test_sharpen_needs_amount_and_radius():
    assert ImageAdjustments(sharpen_amount=50).to_edits()["sharpen"] == (
        ImageEditStep("sharpen", {"amount": 50, "radius_mm": 0.2})
    )
    no_radius = ImageAdjustments(sharpen_amount=50, sharpen_radius_mm=0)
    assert no_radius.to_edits()["sharpen"] is None


def test_sharpen_comes_before_tone():
    edits = ImageAdjustments(contrast=5, sharpen_amount=50).to_edits()
    assert list(edits) == ["sharpen", "tone"]


def test_round_trip_through_the_edit_stack():
    values = ImageAdjustments(
        brightness=-5,
        contrast=20,
        gamma=0.8,
        sharpen_amount=120,
        sharpen_radius_mm=0.3,
    )
    edits = [e for e in values.to_edits().values() if e is not None]
    stack = [ImageEditStep("invert"), *edits]
    assert ImageAdjustments.from_edits(stack) == values


def test_empty_stack_reads_as_defaults():
    assert ImageAdjustments.from_edits([]) == ImageAdjustments()
