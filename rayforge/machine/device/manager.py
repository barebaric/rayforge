import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from ...core.model import ModelLibrary
from .package import DevicePackage, load as load_package, MANIFEST_FILENAME

if TYPE_CHECKING:
    from ...core.model_manager import ModelManager

logger = logging.getLogger(__name__)


@runtime_checkable
class _HasModelManager(Protocol):
    @property
    def model_mgr(self) -> "ModelManager": ...


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
    ):
        self._source_dirs: List[Path] = source_dirs or []
        self._packages: Dict[str, DevicePackage] = {}
        self._load_errors: Dict[str, str] = {}

    @property
    def source_dirs(self) -> List[Path]:
        return list(self._source_dirs)

    def add_source_dir(self, directory: Path):
        if directory not in self._source_dirs:
            self._source_dirs.append(directory)

    def discover(
        self,
        context: Optional[_HasModelManager] = None,
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
        pkg = load_package(path)
        self._packages[pkg.name] = pkg
        return pkg

    def _scan_directory(self, directory: Path):
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            manifest = child / MANIFEST_FILENAME
            if not manifest.exists():
                continue
            try:
                pkg = load_package(child)
                self._packages[pkg.name] = pkg
                logger.debug(f"Loaded device package: {pkg.name} from {child}")
            except Exception as e:
                key = str(child)
                self._load_errors[key] = str(e)
                logger.error(
                    f"Failed to load device package from {child}: {e}"
                )

    def _register_model_libraries(self, context: _HasModelManager):
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
