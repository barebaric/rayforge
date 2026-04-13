import logging
import re
import shutil
import tempfile
import yaml
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from ...core.model import ModelLibrary
from .package import (
    DevicePackage,
    export_machine_to_dir,
    parse_meta,
    MANIFEST_FILENAME,
)

if TYPE_CHECKING:
    from ...core.model_manager import ModelManager
    from ...context import RayforgeContext
    from ...machine.models.machine import Machine

logger = logging.getLogger(__name__)

_UNSAFE_FILENAME_RE = re.compile(r"[^\w\-. ]")


def _safe_filename(name: str) -> str:
    return _UNSAFE_FILENAME_RE.sub("_", name) or "device"


def _find_manifest_in_zip(zf: zipfile.ZipFile) -> str:
    for name in zf.namelist():
        if Path(name).name == MANIFEST_FILENAME:
            parts = Path(name).parts
            if len(parts) <= 2:
                return name
    raise ValueError(f"No {MANIFEST_FILENAME} found in zip archive")


def _extract_zip_to(
    zf: zipfile.ZipFile,
    dest_dir: Path,
    prefix: str,
):
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if prefix:
            if not name.startswith(prefix + "/"):
                continue
            relative = name[len(prefix) + 1 :]
        else:
            relative = name
        if not relative:
            continue
        dest_file = dest_dir / relative
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src, open(dest_file, "wb") as dst:
            shutil.copyfileobj(src, dst)


class DevicePackageManager:
    """
    Discovers, loads, and manages device packages from one or more
    source directories.

    When a device package contains a ``models/`` directory, it is
    automatically registered as a read-only library in the
    application's :class:`ModelManager`.
    """

    def __init__(
        self,
        source_dirs: Optional[List[Path]] = None,
        install_dir: Optional[Path] = None,
    ):
        self._source_dirs: List[Path] = source_dirs or []
        self._install_dir: Optional[Path] = install_dir
        self._packages: Dict[str, DevicePackage] = {}
        self._load_errors: Dict[str, str] = {}

    @property
    def source_dirs(self) -> List[Path]:
        return list(self._source_dirs)

    @property
    def install_dir(self) -> Optional[Path]:
        return self._install_dir

    def add_source_dir(self, directory: Path):
        if directory not in self._source_dirs:
            self._source_dirs.append(directory)

    def discover(
        self,
        context: Optional["RayforgeContext"] = None,
    ) -> List[DevicePackage]:
        """
        Scan all source directories for device packages and return
        the loaded list.

        Packages with the same name from later source directories
        override earlier ones (user packages override built-in).

        If *context* is provided, device packages that contain a
        ``models/`` directory are registered as read-only libraries
        in ``context.model_mgr``.
        """
        self._packages.clear()
        self._load_errors.clear()

        for source_dir in self._source_dirs:
            if not source_dir.exists():
                continue
            self._scan_directory(source_dir)

        if context is not None:
            self._register_model_libraries(context)

        return self.get_all()

    def get_all(self) -> List[DevicePackage]:
        return sorted(self._packages.values(), key=lambda p: p.name.lower())

    def get(self, name: str) -> Optional[DevicePackage]:
        return self._packages.get(name)

    def get_load_errors(self) -> Dict[str, str]:
        return dict(self._load_errors)

    def load_package(self, path: Path) -> DevicePackage:
        """
        Load a single device package from the given directory.

        Also registers it in the internal package map, replacing any
        existing package with the same name.
        """
        pkg = DevicePackage.from_path(path)
        self._packages[pkg.name] = pkg
        return pkg

    def install_from_zip(self, zip_path: Path) -> DevicePackage:
        """
        Install a device package from a ``.zip`` file.

        Validates that the zip contains a ``device.yaml`` with a
        compatible ``api_version``, then extracts to the install
        directory.
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        if self._install_dir is None:
            raise RuntimeError("No install directory configured")

        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest_name = _find_manifest_in_zip(zf)

            with zf.open(manifest_name) as f:
                data = yaml.safe_load(f)

            meta = parse_meta(data, zip_path)
            dest_dir = self._install_dir / _safe_filename(meta.name)

            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)

            prefix = str(Path(manifest_name).parent)
            if prefix == ".":
                prefix = ""
            _extract_zip_to(zf, dest_dir, prefix)

        return self.load_package(dest_dir)

    def export_to_zip(self, package: DevicePackage, dest: Path) -> Path:
        """
        Zip a device package directory.

        Returns the path to the created ``.rfdevice.zip`` file.
        """
        if package.source_dir is None:
            raise ValueError("Package has no source directory")

        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / f"{_safe_filename(package.name)}.rfdevice.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(package.source_dir.rglob("*")):
                if file_path.is_file():
                    arcname = file_path.relative_to(package.source_dir)
                    zf.write(file_path, arcname)

        return zip_path

    def export_machine(
        self,
        machine: "Machine",
        dest: Path,
        model_mgr: Optional["ModelManager"] = None,
    ) -> Path:
        """
        Export a :class:`Machine` as a shareable ``.rfdevice.zip``.

        Creates a temporary device package directory from the
        machine's current configuration, zips it, and cleans up.
        """
        safe = _safe_filename(machine.name)
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / f"{safe}.rfdevice.zip"

        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir = Path(tmp) / safe
            export_machine_to_dir(machine, pkg_dir, model_mgr)

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in sorted(pkg_dir.rglob("*")):
                    if file_path.is_file():
                        arcname = file_path.relative_to(pkg_dir)
                        zf.write(file_path, arcname)

        return zip_path

    def _scan_directory(self, directory: Path):
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            manifest = child / MANIFEST_FILENAME
            if not manifest.exists():
                continue
            try:
                pkg = DevicePackage.from_path(child)
                self._packages[pkg.name] = pkg
                logger.debug(f"Loaded device package: {pkg.name} from {child}")
            except Exception as e:
                key = str(child)
                self._load_errors[key] = str(e)
                logger.error(
                    f"Failed to load device package from {child}: {e}"
                )

    def _register_model_libraries(self, context: "RayforgeContext"):
        model_mgr = context.model_mgr
        for pkg in self._packages.values():
            if pkg.source_dir is None:
                continue
            models_dir = pkg.source_dir / "models"
            if not models_dir.is_dir():
                continue
            lib_id = f"device:{pkg.name}"
            lib = ModelLibrary(
                library_id=lib_id,
                display_name=pkg.name,
                path=models_dir,
                read_only=True,
            )
            if model_mgr.add_library(lib):
                logger.debug(
                    f"Registered model library for device: {pkg.name}"
                )
