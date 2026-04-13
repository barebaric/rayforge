import logging
from pathlib import Path
from typing import Dict, List, Optional

from .package import DevicePackage, load as load_package, MANIFEST_FILENAME

logger = logging.getLogger(__name__)


class DevicePackageManager:
    """
    Discovers, loads, and manages device packages from one or more
    source directories.
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

    def discover(self) -> List[DevicePackage]:
        """
        Scan all source directories for device packages and return
        the loaded list.

        Packages with the same name from later source directories
        override earlier ones (user packages override built-in).
        """
        self._packages.clear()
        self._load_errors.clear()

        for source_dir in self._source_dirs:
            if not source_dir.exists():
                continue
            self._scan_directory(source_dir)

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
