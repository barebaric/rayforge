from __future__ import annotations
from typing import Optional, List, Callable
from ..context import RayforgeContext
from ..core.step import Step
from ..core.capability import CUT, SCORE, ENGRAVE
from .modifier import MakeTransparent, ToGrayscale
from .producer import (
    DepthEngraver,
    ContourProducer,
    FrameProducer,
    MaterialTestGridProducer,
    Rasterizer,
    ShrinkWrapProducer,
)
from .transformer import (
    MultiPassTransformer,
    Optimize,
    Smooth,
    TabOpsTransformer,
    OverscanTransformer,
)


def create_contour_step(
    context: RayforgeContext, name: Optional[str] = None, optimize: bool = True
) -> Step:
    """Factory to create and configure a Contour step."""
    machine = context.machine
    assert machine is not None
    default_head = machine.get_default_head()
    step = Step(
        typelabel=_("Contour"),
        name=name,
    )
    step.capabilities = {CUT, SCORE}
    step.opsproducer_dict = ContourProducer().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.per_workpiece_transformers_dicts = [
        Smooth(enabled=False, amount=20).to_dict(),
        TabOpsTransformer().to_dict(),
    ]
    if optimize:
        step.per_workpiece_transformers_dicts.append(
            Optimize().to_dict(),
        )
    step.per_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.selected_laser_uid = default_head.uid
    step.kerf_mm = default_head.spot_size_mm[0]
    step.max_cut_speed = machine.max_cut_speed
    step.max_travel_speed = machine.max_travel_speed
    return step


def create_raster_step(
    context: RayforgeContext, name: Optional[str] = None
) -> Step:
    """Factory to create and configure a Rasterize step."""
    machine = context.machine
    assert machine is not None
    default_head = machine.get_default_head()

    # Calculate auto overscan distance based on machine capabilities
    # Use a reasonable default cut speed for calculation if not specified
    auto_distance = OverscanTransformer.calculate_auto_distance(
        machine.max_cut_speed, machine.acceleration
    )

    step = Step(
        typelabel=_("Engrave (Raster)"),
        name=name,
    )
    step.capabilities = {ENGRAVE}
    step.opsproducer_dict = Rasterizer().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.per_workpiece_transformers_dicts = [
        OverscanTransformer(
            enabled=True, distance_mm=auto_distance, auto=True
        ).to_dict(),
        Optimize().to_dict(),
    ]
    step.per_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.selected_laser_uid = default_head.uid
    step.max_cut_speed = machine.max_cut_speed
    step.max_travel_speed = machine.max_travel_speed
    return step


def create_depth_engrave_step(
    context: RayforgeContext, name: Optional[str] = None
) -> Step:
    """Factory to create and configure a Depth Engrave step."""
    machine = context.machine
    assert machine is not None
    default_head = machine.get_default_head()

    # Calculate auto overscan distance based on machine capabilities
    # Use a reasonable default cut speed for calculation if not specified
    default_cut_speed = 500
    auto_distance = OverscanTransformer.calculate_auto_distance(
        default_cut_speed, machine.acceleration
    )

    step = Step(
        typelabel=_("Engrave (Depth-Aware)"),
        name=name,
    )
    step.capabilities = {ENGRAVE}
    step.opsproducer_dict = DepthEngraver().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.per_workpiece_transformers_dicts = [
        OverscanTransformer(
            enabled=True, distance_mm=auto_distance, auto=True
        ).to_dict(),
        Optimize().to_dict(),
    ]
    step.per_step_transformers_dicts = [
        # MultiPassTransformer(passes=1, z_step_down=0.0).to_dict()
    ]
    step.selected_laser_uid = default_head.uid
    step.max_cut_speed = machine.max_cut_speed
    step.max_travel_speed = machine.max_travel_speed
    return step


def create_shrinkwrap_step(
    context: RayforgeContext, name: Optional[str] = None
) -> Step:
    """Factory to create and configure a Shrinkwrap (concave hull) step."""
    machine = context.machine
    assert machine is not None
    default_head = machine.get_default_head()
    step = Step(
        typelabel=_("Shrink Wrap"),
        name=name,
    )
    step.capabilities = {CUT, SCORE}
    step.opsproducer_dict = ShrinkWrapProducer().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.per_workpiece_transformers_dicts = [
        Smooth(enabled=False, amount=20).to_dict(),
        TabOpsTransformer().to_dict(),
        Optimize().to_dict(),
    ]
    step.per_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.selected_laser_uid = default_head.uid
    step.kerf_mm = default_head.spot_size_mm[0]
    step.max_cut_speed = machine.max_cut_speed
    step.max_travel_speed = machine.max_travel_speed
    return step


def create_frame_step(
    context: RayforgeContext, name: Optional[str] = None
) -> Step:
    """Factory to create and configure a Frame step."""
    machine = context.machine
    assert machine is not None
    default_head = machine.get_default_head()
    step = Step(
        typelabel=_("Frame Outline"),
        name=name,
    )
    step.capabilities = {CUT, SCORE}
    step.opsproducer_dict = FrameProducer().to_dict()
    # FrameProducer does not use image data, so no modifiers are needed.
    step.modifiers_dicts = []
    step.per_workpiece_transformers_dicts = [
        TabOpsTransformer().to_dict(),
        Optimize().to_dict(),
    ]
    step.per_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.selected_laser_uid = default_head.uid
    step.kerf_mm = default_head.spot_size_mm[0]
    step.max_cut_speed = machine.max_cut_speed
    step.max_travel_speed = machine.max_travel_speed
    return step


def create_material_test_step(
    context: RayforgeContext, name: Optional[str] = None
) -> Step:
    """Factory to create a Material Test step."""
    machine = context.machine
    assert machine is not None
    step = Step(
        typelabel=_("Material Test Grid"),
        name=name,
    )
    step.capabilities = set()
    step.opsproducer_dict = MaterialTestGridProducer().to_dict()
    # Material test doesn't use image modifiers
    step.modifiers_dicts = []
    # No transformers - ops are already optimally ordered
    step.per_workpiece_transformers_dicts = []
    # No post-step transformers - we don't want path optimization
    step.per_step_transformers_dicts = []
    step.selected_laser_uid = machine.get_default_head().uid
    step.max_cut_speed = machine.max_cut_speed
    step.max_travel_speed = machine.max_travel_speed

    return step


STEP_FACTORIES: List[Callable[[RayforgeContext, Optional[str]], Step]] = [
    create_contour_step,
    create_raster_step,
    create_depth_engrave_step,
    create_shrinkwrap_step,
    create_frame_step,
]
