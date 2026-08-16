"""Scene visibility state shared by all renderers.

The visibility toggles decide which parts of the scene participate in a
frame.  They are a scene-assembly concern, so they live outside the
camera (which is pure math) and are applied by the SceneRenderer when it
builds the per-frame draw list.
"""

from dataclasses import dataclass


@dataclass
class SceneVisibility:
    """Which scene categories are visible in the 3D view.

    The ScenePresenter owns a single instance; the canvas flips these
    flags through typed setter methods and the SceneRenderer filters its
    draw list by them.
    """

    show_travel_moves: bool = False
    show_grid: bool = True
    show_nogo_zones: bool = True
    show_models: bool = True
    show_ops_underlay: bool = True
    show_stock: bool = True
    show_workpiece_image: bool = True
