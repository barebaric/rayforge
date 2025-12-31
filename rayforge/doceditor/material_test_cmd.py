import json
import logging
from typing import TYPE_CHECKING

from ..core.item import DocItem
from ..core.step import Step
from ..core.undo import ListItemCommand
from ..core.workpiece import WorkPiece
from ..image.procedural import ProceduralImporter
from ..pipeline.producer.material_test_grid import (
    MaterialTestGridProducer,
    draw_material_test_preview,
    get_material_test_proportional_size,
)
from ..pipeline.steps import create_material_test_step

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class MaterialTestCmd:
    """Handles creation and updates for material test grids."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor
        self._doc = editor.doc
        self._history_manager = editor.history_manager
        self._doc.descendant_updated.connect(self._on_step_updated)

    def create_test_grid(self):
        """
        Creates a new material test grid, including its Step and WorkPiece,
        and adds them to the document.
        """
        with self._history_manager.transaction(_("Add Material Test")) as t:
            # The producer holds all parameters for the grid.
            producer = MaterialTestGridProducer()
            opsproducer_dict = producer.to_dict()
            params = opsproducer_dict.get("params", {})
            name = _("Material Test Grid")

            # Instantiate a new Step, then assign its producer dictionary.
            step = create_material_test_step(self._editor.context)
            step.name = name
            step.opsproducer_dict = opsproducer_dict

            # Get function paths programmatically for type safety.
            draw_func_path = (
                f"{draw_material_test_preview.__module__}."
                f"{draw_material_test_preview.__name__}"
            )
            size_func_path = (
                f"{get_material_test_proportional_size.__module__}."
                f"{get_material_test_proportional_size.__name__}"
            )

            # Use the generic importer to create the procedural content.
            importer = ProceduralImporter(
                drawing_function_path=draw_func_path,
                size_function_path=size_func_path,
                params=params,
                name=name,
            )
            payload = importer.get_doc_items()
            if not payload:
                logger.error("Failed to create material test grid.")
                return

            source = payload.source
            workpiece = payload.items[0]
            assert isinstance(workpiece, WorkPiece)

            self._doc.add_asset(source)
            step.generated_workpiece_uid = workpiece.uid  # Link step to WP
            width_mm, height_mm = workpiece.size

            machine_dims = self._editor.machine_dimensions
            if machine_dims:
                ws_width, ws_height = machine_dims
                workpiece.pos = (
                    ws_width / 2 - width_mm / 2,
                    ws_height / 2 - height_mm / 2,
                )

            active_layer = self._doc.active_layer
            if active_layer.workflow:
                t.execute(
                    ListItemCommand(
                        owner_obj=active_layer.workflow,
                        item=step,
                        undo_command="remove_step",
                        redo_command="add_step",
                    )
                )
            t.execute(
                ListItemCommand(
                    owner_obj=active_layer,
                    item=workpiece,
                    undo_command="remove_child",
                    redo_command="add_child",
                )
            )
        logger.info(
            f"Created material test grid ({width_mm:.1f}x{height_mm:.1f} mm)"
        )

    def _on_step_updated(
        self, sender: DocItem, *, origin: DocItem, parent_of_origin: DocItem
    ):
        if not isinstance(origin, Step) or not origin.opsproducer_dict:
            return
        if origin.opsproducer_dict.get("type") != "MaterialTestGridProducer":
            return
        self.sync_preview_from_step(origin)

    def sync_preview_from_step(self, step: Step):
        if not step.generated_workpiece_uid:
            return

        item = self._doc.find_descendant_by_uid(step.generated_workpiece_uid)
        if not isinstance(item, WorkPiece):
            logger.warning("Could not find workpiece owned by step.")
            return

        workpiece_to_update = item
        if not workpiece_to_update.source:
            return

        opsproducer_dict = step.opsproducer_dict
        if opsproducer_dict is None:
            return

        source = workpiece_to_update.source
        new_params = opsproducer_dict.get("params", {})

        # Re-create the recipe with the updated geometric parameters.
        try:
            old_recipe = json.loads(source.original_data)
            new_recipe_dict = {
                "drawing_function_path": old_recipe["drawing_function_path"],
                "size_function_path": old_recipe["size_function_path"],
                "params": new_params,
            }
            new_recipe_data = json.dumps(new_recipe_dict).encode("utf-8")
            source.original_data = new_recipe_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Could not update procedural source data: {e}")
            return

        # Invalidate render cache and notify UI of content change.
        workpiece_to_update.clear_render_cache()
        workpiece_to_update.updated.send(workpiece_to_update)

        # Recalculate size and update the workpiece.
        new_width_mm, new_height_mm = get_material_test_proportional_size(
            new_params
        )
        workpiece_to_update.set_size(new_width_mm, new_height_mm)
