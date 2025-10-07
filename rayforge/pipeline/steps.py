from __future__ import annotations
from typing import Optional, List, Callable
from .. import config
from ..core.step import Step
from .modifier import MakeTransparent, ToGrayscale
from .producer import (
    DepthEngraver,
    EdgeTracer,
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
    name: Optional[str] = None, optimize: bool = True
) -> Step:
    """Factory to create and configure a Contour step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Contour"),
        name=name,
    )
    step.opsproducer_dict = EdgeTracer().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.opstransformers_dicts = [
        Smooth(enabled=False, amount=20).to_dict(),
        TabOpsTransformer().to_dict(),
    ]
    if optimize:
        step.opstransformers_dicts.append(
            Optimize().to_dict(),
        )
    step.post_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.kerf_mm = config.config.machine.get_default_head().spot_size_mm[0]
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_raster_step(name: Optional[str] = None) -> Step:
    """Factory to create and configure a Rasterize step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Engrave (Raster)"),
        name=name,
    )
    step.opsproducer_dict = Rasterizer().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.opstransformers_dicts = [
        OverscanTransformer().to_dict(),
        Optimize().to_dict(),
    ]
    step.post_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_depth_engrave_step(name: Optional[str] = None) -> Step:
    """Factory to create and configure a Depth Engrave step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Engrave (Depth-Aware)"),
        name=name,
    )
    step.opsproducer_dict = DepthEngraver().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.opstransformers_dicts = [
        OverscanTransformer().to_dict(),
        Optimize().to_dict(),
    ]
    step.post_step_transformers_dicts = [
        # MultiPassTransformer(passes=1, z_step_down=0.0).to_dict()
    ]
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_shrinkwrap_step(name: Optional[str] = None) -> Step:
    """Factory to create and configure a Shrinkwrap (concave hull) step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Shrink Wrap"),
        name=name,
    )
    step.opsproducer_dict = ShrinkWrapProducer().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.opstransformers_dicts = [
        Smooth(enabled=False, amount=20).to_dict(),
        TabOpsTransformer().to_dict(),
        Optimize().to_dict(),
    ]
    step.post_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.kerf_mm = config.config.machine.get_default_head().spot_size_mm[0]
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_frame_step(name: Optional[str] = None) -> Step:
    """Factory to create and configure a Frame step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Frame Outline"),
        name=name,
    )
    step.opsproducer_dict = FrameProducer().to_dict()
    # FrameProducer does not use image data, so no modifiers are needed.
    step.modifiers_dicts = []
    step.opstransformers_dicts = [
        TabOpsTransformer().to_dict(),
        Optimize().to_dict(),
    ]
    step.post_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.kerf_mm = config.config.machine.get_default_head().spot_size_mm[0]
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_material_test_step(name: Optional[str] = None) -> Step:
    """Factory to create a Material Test step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Material Test Grid"),
        name=name,
    )
    step.opsproducer_dict = MaterialTestGridProducer().to_dict()
    # Material test doesn't use image modifiers
    step.modifiers_dicts = []
    # No transformers - ops are already optimally ordered
    step.opstransformers_dicts = []
    # No post-step transformers - we don't want path optimization
    step.post_step_transformers_dicts = []
    step.selected_laser_uid = config.config.machine.get_default_head().uid
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed

    return step


STEP_FACTORIES: List[Callable[[Optional[str]], Step]] = [
    create_contour_step,
    create_raster_step,
    create_depth_engrave_step,
    create_shrinkwrap_step,
    create_frame_step,
]
