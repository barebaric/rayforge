from __future__ import annotations

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

import numpy as np
from raygeo.ops.transform.mesh_correction import MeshCorrectionSpec

from rayforge.pipeline.transformer.base import OpsTransformer

if TYPE_CHECKING:
    from raygeo.geo import Geometry

    from rayforge.core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


class MeshCorrectionTransformer(OpsTransformer):
    """
    Warps a toolpath's Z onto the machine's probed bed height map so
    the focal point (laser) or cut depth (CNC) follows the real
    surface. The height map is probed and stored on the machine (see
    the Bed Mesh page in the machine settings); this transformer adds
    the bilinearly interpolated map height to every move's Z.

    The transformer itself is geometry-free: it queries the machine's
    probed bed mesh in ``to_spec`` and raises when none has been
    probed, which the pipeline handles by skipping it.

    The mesh heights are machine Z positions where the probe touched
    the bed. Every move's Z becomes its own Z plus the bed height under
    it, plus the focus offset, minus the active work offset's Z, so the
    result is in the work coordinates the job is emitted in.
    """

    SPEC_NAME = "mesh_correction"

    #: Applies to a layer's merged toolpath in machine space.
    LAYER_APPLICABLE = True

    def __init__(self, enabled: bool = True, z_offset: float = 0.0):
        super().__init__(enabled=enabled)
        self._z_offset: float = 0.0
        self.z_offset = z_offset

    @property
    def z_offset(self) -> float:
        """The focus offset above the probed surface, in mm."""
        return self._z_offset

    @z_offset.setter
    def z_offset(self, value: float) -> None:
        self._z_offset = float(value)

    @property
    def label(self) -> str:
        return _("Bed Mesh Correction")

    @property
    def description(self) -> str:
        return _(
            "Compensates an uneven work surface using the machine's "
            "probed bed height map."
        )

    def to_spec(
        self,
        workpiece: WorkPiece | None,
        stock_geometries: list[Geometry] | None,
        machine=None,
        settings=None,
    ) -> MeshCorrectionSpec:
        if machine is None or machine.bed_mesh is None:
            raise ValueError(
                "machine has no probed bed mesh; probe the bed in the "
                "machine settings first"
            )
        mesh = machine.bed_mesh
        heights = np.asarray(mesh.heights, dtype=np.float64).reshape(
            mesh.ny, mesh.nx
        )
        wcs_z = float(machine.get_active_wcs_offset()[2])
        return MeshCorrectionSpec(
            mesh.x0,
            mesh.y0,
            mesh.dx,
            mesh.dy,
            heights,
            z_offset=self.z_offset - wcs_z,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {**super().to_dict()}
        if self.z_offset:
            data["z_offset"] = self.z_offset
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeshCorrectionTransformer:
        return cls(
            enabled=data.get("enabled", True),
            z_offset=data.get("z_offset", 0.0),
        )
