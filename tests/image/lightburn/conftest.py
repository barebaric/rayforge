"""
Pytest configuration for LightBurn importer tests.

Registers ``laser_essentials`` steps and ``post_processors``
transformers so ``ItemAssembler._apply_settings`` can construct real
``ContourStep`` / ``EngraveStep`` instances during ``get_doc_items()``,
letting the end-to-end dispatch be exercised (image layer → EngraveStep,
vector layer → ContourStep) rather than only the soft-check path.
"""

# pyright: reportMissingImports=false

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_LASER_ADDON = _REPO_ROOT / "rayforge/builtin_addons/rayforge-addon-laser"
_POST_ADDON = _REPO_ROOT / "rayforge/builtin_addons/rayforge-addon-post"

for _p in (_LASER_ADDON, _POST_ADDON):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


@pytest.fixture(scope="session", autouse=True)
def _register_laser_addons():
    from rayforge import worker_init
    from rayforge.core.step_registry import step_registry
    from rayforge.pipeline.transformer.registry import transformer_registry

    # Prevent the auto-loader from clobbering these registrations.
    worker_init._worker_addons_loaded = True

    from laser_essentials.steps import ContourStep, EngraveStep
    from post_processors.transformers import (
        BidirScanOffsetTransformer,
        CropTransformer,
        LeadInOutTransformer,
        MergeLinesTransformer,
        MultiPassTransformer,
        Optimize,
        OverscanTransformer,
        Smooth,
        TabOpsTransformer,
    )

    if step_registry.get("ContourStep") is None:
        step_registry.register(ContourStep)
    if step_registry.get("EngraveStep") is None:
        step_registry.register(EngraveStep)

    for _t in (
        BidirScanOffsetTransformer,
        CropTransformer,
        LeadInOutTransformer,
        MergeLinesTransformer,
        MultiPassTransformer,
        Optimize,
        OverscanTransformer,
        Smooth,
        TabOpsTransformer,
    ):
        if transformer_registry.get(_t.__name__) is None:
            transformer_registry.register(_t, addon_name="post_processors")

    yield
