"""Bed height-map model: a grid of probed Z values for the bed."""

import statistics
import uuid
from datetime import UTC, datetime
from gettext import gettext as _
from typing import Any

from blinker import Signal


class BedMesh:
    """A probed bed height map in machine coordinates.

    The grid stores the raw Z values reported by the machine's probe
    at ``(x0 + i * dx, y0 + j * dy)``, in row-major order (``ny``
    rows of ``nx`` values). The values are machine-space; converting
    them into the coordinate frame a job is emitted in happens at
    pipeline-build time, not here, so the mesh stays valid when WCS
    or axis-direction settings change.

    A mesh is replaced wholesale by a new probe run; the revision
    counter lets consumers (e.g. cache tokens) distinguish probe
    runs without comparing height arrays.
    """

    def __init__(self):
        self.uid: str = str(uuid.uuid4())
        self.x0: float = 0.0
        self.y0: float = 0.0
        self.dx: float = 1.0
        self.dy: float = 1.0
        self.nx: int = 1
        self.ny: int = 1
        self.heights: list[float] = [0.0]
        self.probed_at: str = ""
        self.probe_params: dict[str, float] = {}
        self.changed = Signal()
        self.extra: dict[str, Any] = {}

    @property
    def mean_z(self) -> float:
        return statistics.fmean(self.heights)

    @property
    def z_range(self) -> tuple[float, float]:
        return min(self.heights), max(self.heights)

    def sample(self, x: float, y: float) -> float:
        """Bilinearly interpolated height at ``(x, y)``.

        Coordinates outside the grid are clamped to the nearest edge
        sample.
        """
        fx = (x - self.x0) / self.dx if self.nx > 1 and self.dx > 0 else 0.0
        fy = (y - self.y0) / self.dy if self.ny > 1 and self.dy > 0 else 0.0
        fx = min(max(fx, 0.0), float(self.nx - 1))
        fy = min(max(fy, 0.0), float(self.ny - 1))
        i = min(int(fx), self.nx - 1)
        j = min(int(fy), self.ny - 1)
        i1 = min(i + 1, self.nx - 1)
        j1 = min(j + 1, self.ny - 1)
        tx = fx - i
        ty = fy - j
        lo = (
            self.heights[j * self.nx + i] * (1.0 - tx)
            + self.heights[j * self.nx + i1] * tx
        )
        hi = (
            self.heights[j1 * self.nx + i] * (1.0 - tx)
            + self.heights[j1 * self.nx + i1] * tx
        )
        return lo * (1.0 - ty) + hi * ty

    def set_params(
        self,
        *,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        nx: int,
        ny: int,
        heights: list[float],
        probe_params: dict[str, float] | None = None,
    ) -> None:
        """Replace the grid with a new probe run's results."""
        if nx < 1 or ny < 1:
            raise ValueError(_("mesh grid must be at least 1x1"))
        if len(heights) != nx * ny:
            raise ValueError(_("heights do not match the grid size"))
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.heights = list(heights)
        self.probed_at = datetime.now(UTC).isoformat()
        if probe_params is not None:
            self.probe_params = dict(probe_params)
        self.changed.send(self)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "uid": self.uid,
            "x0": self.x0,
            "y0": self.y0,
            "dx": self.dx,
            "dy": self.dy,
            "nx": self.nx,
            "ny": self.ny,
            "heights": list(self.heights),
            "probed_at": self.probed_at,
            "probe_params": dict(self.probe_params),
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BedMesh":
        known_keys = {
            "uid",
            "x0",
            "y0",
            "dx",
            "dy",
            "nx",
            "ny",
            "heights",
            "probed_at",
            "probe_params",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        mesh = cls()
        mesh.uid = data.get("uid", mesh.uid)
        mesh.x0 = float(data.get("x0", 0.0))
        mesh.y0 = float(data.get("y0", 0.0))
        mesh.dx = float(data.get("dx", 1.0))
        mesh.dy = float(data.get("dy", 1.0))
        mesh.nx = int(data.get("nx", 1))
        mesh.ny = int(data.get("ny", 1))
        mesh.heights = [float(z) for z in data.get("heights", [0.0])]
        mesh.probed_at = data.get("probed_at", "")
        mesh.probe_params = {
            k: float(v) for k, v in data.get("probe_params", {}).items()
        }
        if len(mesh.heights) != mesh.nx * mesh.ny:
            # Malformed data: fall back to a flat 1x1 mesh rather than
            # crashing profile loading.
            mesh.nx = 1
            mesh.ny = 1
            mesh.heights = [mesh.mean_z if mesh.heights else 0.0]
        mesh.extra = extra
        return mesh

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("changed", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
