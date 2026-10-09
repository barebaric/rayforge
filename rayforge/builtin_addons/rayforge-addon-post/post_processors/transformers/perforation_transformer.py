from __future__ import annotations

from collections.abc import Sequence
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.geo import Matrix
from raygeo.ops.transform.tabs import TabsSpec

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import OpsTransformer

if TYPE_CHECKING:
    from raygeo.geo import Geometry


MIN_LENGTH = 0.01
EPSILON = 1e-4


class PerforationTransformer(OpsTransformer):
    """
    Turns cut paths into a dashed cut: the laser fires for
    ``cut_length`` and is off for ``skip_length``, repeated along
    every contour of the workpiece.

    The gaps are placed in millimetres along each contour of the
    workpiece's boundaries, starting with a full cut at the start of
    every contour, and handed to the tabs gap machinery, which cuts
    them out of the nearest toolpath.
    """

    SPEC_NAME = "tabs"
    DEFAULT_CUT_LENGTH = 2.0
    DEFAULT_SKIP_LENGTH = 1.0

    def __init__(
        self,
        enabled: bool = True,
        cut_length: float = DEFAULT_CUT_LENGTH,
        skip_length: float = DEFAULT_SKIP_LENGTH,
    ):
        super().__init__(enabled=enabled)
        self._cut_length = max(MIN_LENGTH, cut_length)
        self._skip_length = max(MIN_LENGTH, skip_length)

    @property
    def cut_length(self) -> float:
        return self._cut_length

    @cut_length.setter
    def cut_length(self, value: float) -> None:
        self._cut_length = max(MIN_LENGTH, value)
        self.changed.send(self)

    @property
    def skip_length(self) -> float:
        return self._skip_length

    @skip_length.setter
    def skip_length(self, value: float) -> None:
        self._skip_length = max(MIN_LENGTH, value)
        self.changed.send(self)

    @property
    def label(self) -> str:
        return _("Perforation")

    @property
    def description(self) -> str:
        return _(
            "Cuts in dashes: the laser fires for the cut length and "
            "skips the skip length"
        )

    def _gap_centres(self, length: float) -> list[float]:
        period = self._cut_length + self._skip_length
        half_gap = self._skip_length / 2.0
        centres: list[float] = []
        centre = self._cut_length + half_gap
        while centre + half_gap <= length + EPSILON:
            centres.append(centre)
            centre += period
        return centres

    def _clips_for_contour(
        self, contour: Geometry
    ) -> list[tuple[float, float, float]]:
        centres = self._gap_centres(contour.distance())
        if not centres:
            return []
        positions = contour.get_positions_at_distances(centres)
        return [
            (point[0], point[1], self._skip_length)
            for _seg, _t, point in positions
        ]

    def _generate_clips(
        self, workpiece: WorkPiece
    ) -> list[tuple[float, float, float]]:
        boundaries = workpiece.boundaries
        if boundaries is None or boundaries.is_empty():
            return []
        width, height = workpiece.size
        scaled = boundaries.copy()
        if width > 0 and height > 0:
            scaled.transform(Matrix.scale(width, height))
        clips: list[tuple[float, float, float]] = []
        for contour in scaled.split_into_contours():
            clips.extend(self._clips_for_contour(contour))
        return clips

    def to_spec(
        self,
        workpiece: WorkPiece | None,
        stock_geometries: Sequence[Geometry] | None,
        settings: dict[str, Any] | None,
    ) -> TabsSpec:
        clips = self._generate_clips(workpiece) if workpiece else []
        return TabsSpec(tab_power=0.0, original_power=1.0, clips=clips)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["cut_length"] = self._cut_length
        data["skip_length"] = self._skip_length
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerforationTransformer:
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(
            enabled=data.get("enabled", True),
            cut_length=data.get("cut_length", cls.DEFAULT_CUT_LENGTH),
            skip_length=data.get("skip_length", cls.DEFAULT_SKIP_LENGTH),
        )
