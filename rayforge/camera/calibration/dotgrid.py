"""Calibration target backed by a printed grid of black dots."""

import logging
from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np

from .target import (
    CalibrationTarget,
    CalibrationTargetType,
    ConfigField,
    SummaryRow,
    TargetConfig,
    normalize_detection,
    to_gray,
)

logger = logging.getLogger(__name__)


@dataclass
class DotGridConfig(TargetConfig):
    dots_x: int = 6
    dots_y: int = 8
    spacing_mm: float = 20.0
    dot_diameter_mm: float = 10.0
    # Vertical distance between rows. 0 means "same as spacing_mm",
    # which is the plain rectangular grid.
    row_spacing_mm: float = 0.0
    # How far every second row is shifted along x. A half-pitch shift
    # gives a hexagonal lattice; 0 gives a rectangular one.
    row_offset_mm: float = 0.0

    FIELDS: ClassVar[tuple[ConfigField, ...]] = (
        ConfigField(key="dots_x", lower=2, upper=30, integer=True),
        ConfigField(key="dots_y", lower=2, upper=30, integer=True),
        ConfigField(
            key="spacing_mm",
            lower=1.0,
            upper=300.0,
            step=0.5,
        ),
        ConfigField(
            key="row_spacing_mm",
            lower=0.0,
            upper=300.0,
            step=0.5,
        ),
        ConfigField(
            key="row_offset_mm",
            lower=0.0,
            upper=300.0,
            step=0.5,
        ),
        ConfigField(
            key="dot_diameter_mm",
            lower=0.5,
            upper=300.0,
            step=0.5,
        ),
    )


class DotGridTarget(CalibrationTarget):
    """A sheet of solid black dots on a white background.

    This is the cheapest pattern to produce at home: a plain inkjet or
    laser printer on ordinary paper is enough, because a filled dot has
    no fine structure to blur. Detection thresholds the sheet, keeps the
    blobs that are round enough to be dots, and orders the survivors into
    a lattice by their principal directions.

    Rows may be staggered: :attr:`DotGridConfig.row_spacing_mm` sets the
    distance between rows and :attr:`DotGridConfig.row_offset_mm` shifts
    every second row sideways, which covers both the plain rectangular
    grid and the hexagonal arrangement that many printed dot sheets
    use. A staggered sheet is the better choice of the two, because
    shifting alternate rows breaks the symmetry that makes a plain grid
    ambiguous to orient.

    What neither arrangement can do is say which way is up. A dot sheet
    carries no orientation cue, so the ordering is tied to the image
    frame, which stays consistent as long as the sheet keeps roughly the
    same orientation to the camera between captures. Targets that carry
    their own cue (ChArUco or an ArUco marker grid) are unambiguous, so
    prefer them when the printed sheet allows.
    """

    target_type = CalibrationTargetType.DOT_GRID
    config_class = DotGridConfig
    TARGET_SPACING_MM = 20.0
    DOT_TO_SPACING_RATIO = 0.5
    # Vertical pitch that gives equilateral triangles for a given
    # horizontal pitch, used only when suggesting a staggered sheet.
    HEX_ROW_SPACING_RATIO = 0.866
    MIN_DOT_PIXELS = 5
    MAX_COLUMNS = 20
    MAX_ROWS = 20
    MARGIN_RATIO = 0.12
    MIN_DOT_AREA_PX = 12
    MIN_CIRCULARITY = 0.6
    MIN_LEVEL_SEPARATION = 0.35
    # Resolution of the search for the row direction. One degree is well
    # inside the tolerance of any printable sheet, and the sweep only
    # runs on the small set of detected dot centres.
    ANGLE_STEP_DEGREES = 1.0
    # How far either side of the last good row direction to look before
    # falling back to the full sweep.
    ANGLE_WINDOW_DEGREES = 12.0
    # How many shortlisted orientations get the full homography check.
    MAX_FINALISTS = 8
    # Mean pixel residual above which no orientation is accepted. A
    # correct reading on a flat render lands well under a pixel, but a
    # real lens bends straight lines, which a homography cannot model.
    # The threshold therefore tolerates a few pixels of distortion
    # while still rejecting mislabelled lattices, whose residuals are
    # typically tens of pixels.
    MAX_HOMOGRAPHY_ERROR = 8.0

    def __init__(self, config: DotGridConfig):
        if config.spacing_mm <= 0:
            raise ValueError("Dot spacing must be positive")
        if config.dot_diameter_mm <= 0:
            raise ValueError("Dot diameter must be positive")
        if config.dot_diameter_mm >= config.spacing_mm:
            raise ValueError(
                "Dot diameter "
                f"({config.dot_diameter_mm}) must be smaller than "
                f"spacing ({config.spacing_mm}) or neighbouring "
                "dots merge into one blob"
            )
        self.config = config
        self._angle_hint: float | None = None

    @property
    def grid_size(self) -> tuple[int, int]:
        """Return the number of calibration points across and down."""
        return self.config.dots_x, self.config.dots_y

    @property
    def row_pitch_mm(self) -> float:
        """Vertical distance between rows, defaulting to the column pitch."""
        return self.config.row_spacing_mm or self.config.spacing_mm

    @property
    def row_offset_mm(self) -> float:
        """Horizontal shift applied to every second row."""
        return self.config.row_offset_mm

    @property
    def is_staggered(self) -> bool:
        return self.row_offset_mm > 0.0

    @property
    def point_count(self) -> int:
        cols, rows = self.grid_size
        return cols * rows

    @property
    def card_size_mm(self) -> tuple[float, float]:
        width, height = self._lattice_span_mm()
        diameter = self.config.dot_diameter_mm
        return width + diameter, height + diameter

    def object_points(self) -> np.ndarray:
        """Return the dot centres as a row-major (N, 3) lattice.

        Row 0 is the first row and sits at the origin; the lattice grows
        along +x and +y, with every second row shifted by
        :attr:`row_offset_mm`. The absolute offset is irrelevant to the
        solve, which absorbs it into the extrinsics; keeping the origin
        at a dot means the renderer can map millimetres straight onto
        pixels.
        """
        cols, rows = self.grid_size
        pitch = self.config.spacing_mm
        row_pitch = self.row_pitch_mm
        offset = self.row_offset_mm
        points = np.zeros((cols * rows, 3), dtype=np.float32)
        for row in range(rows):
            shift = offset if row % 2 else 0.0
            for col in range(cols):
                points[row * cols + col] = (
                    col * pitch + shift,
                    row * row_pitch,
                    0.0,
                )
        return points

    def _lattice_span_mm(self) -> tuple[float, float]:
        """Return the extent of the dot centres, ignoring dot diameter."""
        cols, rows = self.grid_size
        return (
            (cols - 1) * self.config.spacing_mm + self.row_offset_mm,
            (rows - 1) * self.row_pitch_mm,
        )

    @classmethod
    def recommend_config(
        cls,
        card_width_mm: float,
        card_height_mm: float,
        camera_resolution: tuple[int, int] = (640, 480),
        surface_size_mm: tuple[float, float] | None = None,
    ) -> DotGridConfig:
        min_dim = min(camera_resolution)
        if surface_size_mm:
            mm_per_pixel = min(surface_size_mm) / min_dim
        else:
            mm_per_pixel = card_width_mm / min_dim

        spacing = cls.TARGET_SPACING_MM
        if mm_per_pixel > 0:
            estimated_dot_pixels = (
                spacing * cls.DOT_TO_SPACING_RATIO
            ) / mm_per_pixel
            if estimated_dot_pixels < cls.MIN_DOT_PIXELS:
                spacing *= cls.MIN_DOT_PIXELS / estimated_dot_pixels

        diameter = spacing * cls.DOT_TO_SPACING_RATIO
        row_pitch = spacing * cls.HEX_ROW_SPACING_RATIO
        usable_width = card_width_mm * (1 - 2 * cls.MARGIN_RATIO)
        usable_height = card_height_mm * (1 - 2 * cls.MARGIN_RATIO)

        # Stagger alternate rows by half a pitch, which is the
        # arrangement most printed dot sheets use.
        dots_x = max(2, min(cls.MAX_COLUMNS, int(usable_width / spacing)))
        dots_y = max(2, min(cls.MAX_ROWS, int(usable_height / row_pitch)))

        return DotGridConfig(
            dots_x=dots_x,
            dots_y=dots_y,
            spacing_mm=round(spacing, 1),
            dot_diameter_mm=round(diameter, 1),
            row_spacing_mm=round(row_pitch, 1),
            row_offset_mm=round(spacing / 2, 1),
        )

    def _pattern_size(self) -> tuple[int, int]:
        return self.grid_size

    def generate_image(
        self,
        output_size: tuple[int, int] | None = None,
        margin_px: int | None = None,
    ) -> np.ndarray:
        """Render the sheet.

        ``output_size`` is ``(width, height)`` in pixels, matching the
        other targets; the returned array is a 2D image so it has shape
        ``(height, width)``.
        """
        width_mm, height_mm = self.card_size_mm
        if margin_px is None:
            margin_px = max(10, int(max(width_mm, height_mm) * 0.1))
        if output_size is None:
            px_per_mm = 10
            output_size = (
                int(width_mm * px_per_mm) + 2 * margin_px,
                int(height_mm * px_per_mm) + 2 * margin_px,
            )

        width_px, height_px = output_size
        image = np.full((height_px, width_px), 255, dtype=np.uint8)

        scale = min(
            width_px / max(width_mm, 1e-6),
            height_px / max(height_mm, 1e-6),
        )
        radius_px = max(1, round(self.config.dot_diameter_mm * scale / 2))

        # Keep a clear white border so the outermost dots survive
        # thresholding as separate blobs. A single scale in both axes
        # preserves the configured pitches even when the requested
        # output aspect does not match the card; the lattice is
        # centred in the surplus direction.
        origin_x = margin_px + radius_px
        origin_y = margin_px + radius_px
        span_x, span_y = self._lattice_span_mm()
        usable_w = max(1.0, width_px - 2 * origin_x)
        usable_h = max(1.0, height_px - 2 * origin_y)
        step = min(
            usable_w / span_x if span_x > 0 else float("inf"),
            usable_h / span_y if span_y > 0 else float("inf"),
        )
        if step == float("inf"):
            step = 0.0
        offset_x = origin_x + (usable_w - span_x * step) / 2
        offset_y = origin_y + (usable_h - span_y * step) / 2

        for x_mm, y_mm, _ in self.object_points():
            pixel_x = round(offset_x + x_mm * step)
            pixel_y = round(offset_y + y_mm * step)
            cv2.circle(image, (pixel_x, pixel_y), radius_px, 0, -1)

        return image

    def detect(
        self, image: np.ndarray
    ) -> tuple[list[tuple[float, float]], list[int]] | None:
        cols, rows = self._pattern_size()
        if cols < 2 or rows < 2:
            return None

        centers = self._find_dot_centers(
            to_gray(image), expected_count=cols * rows
        )
        if centers is None:
            return None

        ordered = self._order_as_lattice(centers, cols, rows)
        if ordered is None:
            return None

        ids = list(range(cols * rows))
        return normalize_detection(ordered, ids)

    def _find_dot_centers(
        self, gray: np.ndarray, expected_count: int | None = None
    ) -> np.ndarray | None:
        """Return the sub-pixel centre of every dot, or None.

        The sheet almost never fills the frame: usually it is a small
        patch of paper on a much larger bed. Thresholding is therefore
        tried in both polarities. Dark ink on light paper makes the
        dots the dark class, but when the bed itself is dark the paper
        can end up as the minority class instead. The polarity whose
        dot-shaped component count matches the configured sheet wins;
        otherwise the minority-class result is kept.
        """
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _binary, dark_mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        light_mask = cv2.bitwise_not(dark_mask)
        if np.count_nonzero(light_mask) <= np.count_nonzero(dark_mask):
            ordered_masks = (light_mask, dark_mask)
        else:
            ordered_masks = (dark_mask, light_mask)

        first = self._extract_centers(ordered_masks[0], blurred)
        if expected_count is not None:
            if first is not None and len(first) == expected_count:
                return first
            second = self._extract_centers(ordered_masks[1], blurred)
            if second is not None and len(second) == expected_count:
                return second
            # Neither polarity yields a whole sheet. Reading part of a
            # sheet would poison the solve, so refuse it.
            logger.debug(
                "Found %s/%s dots, expected %d",
                None if first is None else len(first),
                None if second is None else len(second),
                expected_count,
            )
            return None
        return first

    def _extract_centers(
        self, mask: np.ndarray, blurred: np.ndarray
    ) -> np.ndarray | None:
        """Collect dot centres from one threshold polarity."""
        height_px, width_px = mask.shape[:2]
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask, 8
        )
        if count <= 1:
            return None

        centers = []
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            if area < self.MIN_DOT_AREA_PX:
                continue
            left = stats[index, cv2.CC_STAT_LEFT]
            top = stats[index, cv2.CC_STAT_TOP]
            width = stats[index, cv2.CC_STAT_WIDTH]
            height = stats[index, cv2.CC_STAT_HEIGHT]
            if (
                left <= 0
                or top <= 0
                or left + width >= width_px
                or top + height >= height_px
            ):
                # The bed, the paper edge, or a shadow: a real dot is
                # fully surrounded by paper and never touches the
                # frame border.
                continue
            # Work inside the component's own bounding box. Slicing the
            # whole label image once per dot would make detection cost
            # grow with the frame area times the dot count, which is far
            # too slow for the live preview.
            window = (slice(top, top + height), slice(left, left + width))
            component = (labels[window] == index).astype(np.uint8) * 255
            if not self._is_dot_shaped(component, area):
                continue
            center = self._refine_center(blurred[window], component)
            centers.append((center[0] + left, center[1] + top))

        if not centers:
            return None
        return np.asarray(centers, dtype=np.float32)

    def _is_dot_shaped(self, component: np.ndarray, area: int) -> bool:
        """Reject elongated blobs and anything with a hole."""
        contours, _hierarchy = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter <= 0:
            return False
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        return circularity >= self.MIN_CIRCULARITY

    def _refine_center(
        self, gray: np.ndarray, component: np.ndarray
    ) -> tuple[float, float]:
        """Locate a dot centre from the intensity, not the binary mask.

        An intensity-weighted centroid of the dot's footprint is a
        sub-pixel estimate that does not depend on thresholding, which
        matters because a printed dot edge is soft.
        """
        mask = component > 0
        weights = np.where(mask, 255.0 - gray, 0.0).astype(np.float64)
        total = float(weights.sum())
        if total <= 0:
            ys, xs = np.nonzero(mask)
            if not len(xs):
                return 0.0, 0.0
            return float(xs.mean()), float(ys.mean())
        ys, xs = np.nonzero(mask)
        picked = weights[ys, xs]
        return (
            float((xs * picked).sum() / picked.sum()),
            float((ys * picked).sum() / picked.sum()),
        )

    def _order_as_lattice(
        self, centers: np.ndarray, cols: int, rows: int
    ) -> np.ndarray | None:
        """Sort detected centres into row-major lattice order.

        Rows are separated along a direction perpendicular to the row
        itself, which is what keeps a staggered sheet readable: the two
        lattice directions of a hexagonal sheet meet at 60 degrees, so
        neither principal axis is a row normal and projecting onto one
        would smear a row across its neighbours.

        A perspective view also spaces the rows unevenly, so levels are
        split by count rather than by gap size. Both candidate
        orientations are built and the one whose recovered pitches match
        the configured ones wins, which is what decides whether the sheet
        was read along its rows or across them.

        Getting this assignment wrong is silent: the solve returns
        numbers, just the wrong ones.
        """
        if len(centers) != cols * rows:
            logger.debug(
                "Found %d dots, expected %d", len(centers), cols * rows
            )
            return None

        centroid = centers.mean(axis=0)
        centered = centers - centroid

        # Shortlist cheaply, then decide properly. A planar target's
        # image is related to its object points by a homography, so the
        # arrangement that fits one is the right one. Ranking by a
        # homography for every angle would be too slow to run on every
        # preview frame, so a closed-form affine fit ranks the
        # candidates and only the best few get the full check.
        ordered = self._best_orientation(centers, centered, cols, rows)
        if ordered is None and self._angle_hint is not None:
            # The hint missed, so the sheet moved further than the local
            # window allows. Fall back to the full search.
            self._angle_hint = None
            ordered = self._best_orientation(centers, centered, cols, rows)
        if ordered is None:
            return None
        return ordered

    def _best_orientation(
        self,
        centers: np.ndarray,
        centered: np.ndarray,
        cols: int,
        rows: int,
    ) -> np.ndarray | None:
        angles = self._candidate_angles()
        local = self._angle_hint is not None
        best_angle = None
        finalists: list[tuple[float, np.ndarray, float]] = []

        for angle in angles:
            radians = np.radians(angle)
            along = np.array([np.cos(radians), np.sin(radians)])
            normal = np.array([-along[1], along[0]])
            # Rows must run top to bottom, so the row axis has to point
            # down the image.
            if normal[1] < 0:
                normal = -normal

            arranged = self._arrange(
                centers,
                centered @ along,
                centered @ normal,
                cols,
                rows,
            )
            if arranged is None:
                continue
            points = centers[arranged]
            # A wrong axis produces a badly shaped affine fit even though
            # its level spacing can look deceptively regular, so the fit
            # is what ranks candidates.
            finalists.append((self._affine_residual(points), points, angle))

        if not finalists:
            return None

        finalists.sort(key=lambda item: item[0])
        best_residual = None
        best_points = None
        for _rank, points, angle in finalists[: self.MAX_FINALISTS]:
            residual = self._homography_residual(points)
            if best_residual is None or residual < best_residual:
                best_residual = residual
                best_points = points
                best_angle = angle

        if best_points is None or best_residual > self.MAX_HOMOGRAPHY_ERROR:
            if not local:
                logger.debug(
                    "No candidate orientation fits the configured lattice"
                )
            return None

        self._angle_hint = best_angle
        return best_points

    def _affine_residual(self, ordered_points: np.ndarray) -> float:
        """Return the mean pixel error of the best affine fit.

        Cheap enough to rank every candidate angle, and good enough to
        sort them: a mislabelled lattice cannot be straightened by any
        affine map, while perspective only bends the correct one a
        little.
        """
        object_points = self.object_points()
        if len(ordered_points) != len(object_points):
            return float("inf")

        source = object_points[:, :2].astype(np.float64)
        destination = ordered_points.astype(np.float64)
        design = np.column_stack([source, np.ones(len(source))])
        solution, *_rest = np.linalg.lstsq(design, destination, rcond=None)
        predicted = design @ solution
        return float(np.linalg.norm(predicted - destination, axis=1).mean())

    def _homography_residual(self, ordered_points: np.ndarray) -> float:
        """Return how far the ordering sits from a planar projection.

        A sheet of dots is planar, so its object points and its image
        points must be related by a single homography. An ordering that
        is read along the wrong axis, or with rows swapped, cannot be,
        and shows up here as a large residual. This holds under any
        viewpoint, which a pitch comparison does not: perspective
        foreshortens the two directions by different amounts.
        """
        object_points = self.object_points()
        if len(ordered_points) != len(object_points):
            return float("inf")

        source = np.asarray(object_points[:, :2], dtype=np.float32)
        destination = np.asarray(ordered_points, dtype=np.float32)
        homography, _mask = cv2.findHomography(source, destination, method=0)
        if homography is None:
            return float("inf")

        homogeneous = np.column_stack([source, np.ones(len(source))]).astype(
            np.float64
        )
        projected = homogeneous @ homography.T
        denominator = projected[:, 2:3]
        if not np.all(np.abs(denominator) > 1e-9):
            return float("inf")
        predicted = projected[:, :2] / denominator
        return float(np.linalg.norm(predicted - destination, axis=1).mean())

    def _candidate_angles(self) -> np.ndarray:
        """Yield the row directions to try, in image space.

        Principal axes are not usable here. A hexagonal sheet is close to
        six-fold symmetric, so its point cloud has a nearly isotropic
        covariance and the eigenvectors point wherever the noise takes
        them; a roughly square rectangular sheet is nearly as bad. The
        row direction is a property of the pattern, not of the second
        moments, so it is searched for instead.

        The sweep runs over the half turn that keeps the direction
        pointing right, because the sign has to be fixed to the image
        frame: two captures of the same sheet must be read the same way.
        Once a direction has been found the search narrows around it,
        which keeps the per-frame cost low while a fallback to the full
        sweep still catches a sheet that has been moved.
        """
        if self._angle_hint is not None:
            return np.arange(
                self._angle_hint - self.ANGLE_WINDOW_DEGREES,
                self._angle_hint + self.ANGLE_WINDOW_DEGREES,
                self.ANGLE_STEP_DEGREES,
            )
        return np.arange(-90.0, 90.0, self.ANGLE_STEP_DEGREES)

    def _arrange(
        self,
        centers: np.ndarray,
        across: np.ndarray,
        down: np.ndarray,
        cols: int,
        rows: int,
    ) -> np.ndarray | None:
        """Build a row-major ordering for one candidate orientation.

        Returns the index order, or None when this orientation does not
        group the centres into clean rows and columns.
        """
        row_order = np.argsort(down, kind="stable")
        row_blocks = np.array_split(row_order, rows)

        ordered: list[np.ndarray] = []
        for block in row_blocks:
            if len(block) != cols:
                return None
            ordered.append(block[np.argsort(across[block], kind="stable")])

        column_order = np.concatenate(ordered)
        # One representative per level: the whole first row spans every
        # column, and the first entry of each row block spans every row.
        column_levels = across[column_order[:cols]]
        row_levels = np.array(
            [down[block[0]] for block in row_blocks if len(block)]
        )
        separation = min(
            self._separation(column_levels),
            self._separation(row_levels),
        )
        if separation < self.MIN_LEVEL_SEPARATION:
            return None
        return column_order

    def _separation(self, values: np.ndarray) -> float:
        """Return the smallest level gap relative to the mean gap.

        A genuine lattice has every gap at least a sizeable fraction of
        the average, so a small value means the split is cutting across
        a level rather than between levels.
        """
        ordered = np.sort(values)
        if len(ordered) < 2:
            return 0.0
        gaps = np.diff(ordered)
        mean_gap = float(gaps.mean())
        if mean_gap <= 0:
            return 0.0
        return float(gaps.min() / mean_gap)

    def summary(self) -> list[SummaryRow]:
        cols, rows = self.grid_size
        rows_summary = [
            SummaryRow(key="grid_size", value=f"{cols} x {rows} dots"),
            SummaryRow(
                key="dot_spacing",
                measurement_mm=self.config.spacing_mm,
            ),
        ]
        if self.is_staggered:
            rows_summary.append(
                SummaryRow(key="row_spacing", measurement_mm=self.row_pitch_mm)
            )
        rows_summary.append(
            SummaryRow(
                key="dot_diameter",
                measurement_mm=self.config.dot_diameter_mm,
            )
        )
        return rows_summary


__all__ = ["DotGridConfig", "DotGridTarget"]
