from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from rayforge.doceditor.layout.auto import (
    PixelPerfectLayoutStrategy,
    WorkpieceVariant,
)

if TYPE_CHECKING:
    from rayforge.core.item import DocItem
    from rayforge.shared.tasker.context import ExecutionContext


class _Item:
    def __init__(self, uid: str):
        self.uid = uid
        self.name = uid
        self.parent = None


class _Context:
    def __init__(self):
        self.progress: list[float] = []
        self.messages: list[str] = []

    def set_progress(self, value: float) -> None:
        self.progress.append(value)

    def set_message(self, message: str) -> None:
        self.messages.append(message)


def _variant(uid: str, mask: np.ndarray, angle: int = 0) -> WorkpieceVariant:
    return WorkpieceVariant(
        item=cast("DocItem", _Item(uid)),
        mask=np.asarray(mask, dtype=bool),
        local_bbox=(0, 0, 1, 1),
        angle_offset=angle,
        unrotated_size_mm=(1.0, 1.0),
    )


def _strategy(**kwargs) -> PixelPerfectLayoutStrategy:
    seed = cast("DocItem", _Item("seed"))
    return PixelPerfectLayoutStrategy([seed], **kwargs)


def _positions(placements) -> list[tuple[str, int, tuple[int, int]]]:
    return [
        (item.variant.item.uid, item.variant.angle_offset, item.position_px)
        for item in placements
    ]


def _dilate_reference(binary: np.ndarray, iterations: int) -> np.ndarray:
    result = binary != 0
    for _ in range(iterations):
        padded = np.pad(result, 1, constant_values=False)
        result = (
            result
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return result


def test_packs_in_a_row_with_the_smallest_bounding_box():
    l_shape = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
    groups = [
        [_variant("a", l_shape, 0), _variant("a", np.rot90(l_shape), 90)],
        [_variant("b", np.ones((2, 2)))],
        [_variant("c", np.ones((1, 4)))],
        [_variant("d", np.ones((3, 1)))],
    ]

    placements, unplaced = _strategy()._pack_items(
        groups, np.zeros((12, 16), dtype=bool)
    )

    assert _positions(placements) == [
        ("a", 0, (0, 0)),
        ("b", 0, (0, 2)),
        ("c", 0, (0, 4)),
        ("d", 0, (0, 8)),
    ]
    assert unplaced == []


def test_reports_items_that_do_not_fit():
    groups = [[_variant("big", np.ones((5, 5)))]]

    placements, unplaced = _strategy()._pack_items(
        groups, np.zeros((3, 3), dtype=bool)
    )

    assert placements == []
    assert [item.uid for item in unplaced] == ["big"]


def test_chooses_the_variant_with_the_smallest_score():
    groups = [
        [
            _variant("a", np.ones((2, 2)), 0),
            _variant("a", np.ones((1, 1)), 90),
        ]
    ]

    placements, _ = _strategy()._pack_items(
        groups, np.zeros((10, 10), dtype=bool)
    )

    assert len(placements) == 1
    assert placements[0].variant.mask.shape == (1, 1)
    assert placements[0].position_px == (0, 0)


def test_reports_packing_progress():
    groups = [
        [_variant("a", np.ones((1, 1)))],
        [_variant("b", np.ones((1, 1)))],
    ]
    context = _Context()

    _strategy()._pack_items(
        groups,
        np.zeros((5, 5), dtype=bool),
        cast("ExecutionContext", context),
    )

    assert context.progress == [0.5, 0.9]
    assert context.messages == [
        "Packing item 1 of 2...",
        "Packing item 2 of 2...",
    ]


def test_prepare_variants_dilates_the_margin(monkeypatch):
    strategy = _strategy(
        margin_mm=0.5,
        resolution_px_per_mm=2.0,
        allow_rotation=False,
    )
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    monkeypatch.setattr(
        strategy,
        "_render_and_mask",
        lambda item, angle: (mask, (0, 0, 4, 4), (4.0, 4.0)),
    )

    groups, total_area = strategy._prepare_variants()

    margin_px = int(0.5 * 2.0)
    padded = np.pad(mask, margin_px, constant_values=False)
    expected = _dilate_reference(padded, margin_px)

    assert len(groups) == 1
    assert len(groups[0]) == 1
    np.testing.assert_array_equal(groups[0][0].mask, expected)
    assert total_area == int(expected.sum())


def test_prepare_variants_without_margin_keeps_the_mask(monkeypatch):
    strategy = _strategy(
        margin_mm=0.0,
        resolution_px_per_mm=2.0,
        allow_rotation=False,
    )
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    monkeypatch.setattr(
        strategy,
        "_render_and_mask",
        lambda item, angle: (mask, (0, 0, 4, 4), (4.0, 4.0)),
    )

    groups, _ = strategy._prepare_variants()

    np.testing.assert_array_equal(groups[0][0].mask, mask)
