import logging
from pathlib import Path
from rayforge.core.hooks import hookimpl

logger = logging.getLogger(__name__)


@hookimpl
def register_material_libraries(library_manager):
    materials_dir = Path(__file__).parent.parent / "materials"
    logger.debug(f"Registering materials from {materials_dir}")
    if materials_dir.exists():
        library_manager.add_library_from_path(
            materials_dir, read_only=True, addon_name="core_materials"
        )
