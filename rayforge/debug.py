import json
import yaml
import logging
import tempfile
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, TYPE_CHECKING

from . import const

if TYPE_CHECKING:
    from .doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class DebugDumpManager:
    """
    Orchestrates the creation of comprehensive debug dump files using the
    new logging system.
    """

    def create_dump_archive(
        self, editor: Optional["DocEditor"] = None
    ) -> Optional[Path]:
        """
        Gathers all debug information, writes it to a temporary directory,
        and creates a ZIP archive.

        If editor is given, the current project is serialized and included
        in the archive (regardless of whether it has been saved to disk).
        """
        from .config import LOG_DIR
        from .context import get_context
        from .ui_gtk.about import get_dependency_info
        from . import __version__

        logger.info("Creating debug dump archive...")
        try:
            context = get_context()
            config = context.config
            machine_mgr = context.machine_mgr

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                # 1. Copy the latest session log file
                session_logs = sorted(
                    LOG_DIR.glob("session-*.log"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if session_logs:
                    latest_log = session_logs[0]
                    shutil.copy(latest_log, tmp_path / latest_log.name)
                else:
                    logger.warning(
                        "No session log file found to include in dump."
                    )

                # 2. Write system info to system_info.txt
                dep_info = get_dependency_info()
                with open(tmp_path / "system_info.txt", "w") as f:
                    f.write(
                        f"## {const.APP_NAME} {__version__ or 'Unknown'}\n\n"
                    )
                    for category, deps in dep_info.items():
                        f.write(f"### {category}\n")
                        for name, ver in deps:
                            f.write(f"{name}: {ver}\n")
                        f.write("\n")

                # 3. Write configs to YAML files
                if config and config.machine:
                    with open(tmp_path / "active_machine.yaml", "w") as f:
                        yaml.safe_dump(config.machine.to_dict(), f)
                with open(tmp_path / "app_config.yaml", "w") as f:
                    yaml.safe_dump(config.to_dict(), f)

                all_machines_dict: Dict[str, Dict[str, Any]] = {
                    machine_id: machine.to_dict()
                    for machine_id, machine in machine_mgr.machines.items()
                }
                with open(tmp_path / "all_machines.yaml", "w") as f:
                    yaml.safe_dump(all_machines_dict, f)

                # 4. Write custom dialects
                custom_dialects = [
                    d.to_dict()
                    for d in context.dialect_mgr.get_all()
                    if d.is_custom
                ]
                if custom_dialects:
                    with open(tmp_path / "custom_dialects.yaml", "w") as f:
                        yaml.safe_dump(custom_dialects, f)

                # 5. Copy addons.yaml if it exists
                addon_config_file = context.addon_config.config_file
                if addon_config_file.exists():
                    shutil.copy(addon_config_file, tmp_path / "addons.yaml")

                # 6. Serialize and include project if requested
                if editor is not None:
                    doc_dict = editor.doc.to_dict()
                    json_bytes = json.dumps(doc_dict, indent=2).encode("utf-8")
                    project_file = tmp_path / "project.ryp"
                    with zipfile.ZipFile(
                        project_file,
                        "w",
                        compression=zipfile.ZIP_DEFLATED,
                    ) as zf:
                        zf.writestr("project.json", json_bytes)

                # 7. Create ZIP archive
                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                archive_name = f"rayforge_debug_{timestamp_str}"
                # Use a system-wide temp dir for the final archive to ensure
                # it survives the 'with' block of the temporary directory.
                final_archive_base = Path(tempfile.gettempdir()) / archive_name

                shutil.make_archive(
                    str(final_archive_base), "zip", root_dir=tmpdir
                )
                archive_path = final_archive_base.with_suffix(".zip")
                logger.info(f"Debug dump archive created at {archive_path}")
                return archive_path

        except Exception:
            logger.error("Failed to create debug dump archive", exc_info=True)
            return None

    @staticmethod
    def save_archive_to(archive_path: Path, destination: Path):
        """
        Moves a previously created dump archive to the given destination.
        Cleans up the temporary archive regardless of success.
        """
        try:
            shutil.move(str(archive_path), str(destination))
        finally:
            if archive_path.exists():
                archive_path.unlink()
