import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type, Dict, Tuple

from .base_exporter import BaseExporter
from .base_importer import Importer, ImporterFeature
from .base_renderer import Renderer


@dataclass(frozen=True)
class FileFilter:
    label: str
    extensions: Tuple[str, ...]
    mime_types: Tuple[str, ...]


logger = logging.getLogger(__name__)


class ImporterRegistry:
    """Registry for file importers."""

    def __init__(self):
        self._importers_by_ext: Dict[str, Type[Importer]] = {}
        self._importers_by_mime: Dict[str, Type[Importer]] = {}
        self._importers_by_name: Dict[str, Type[Importer]] = {}
        self._addon_items: Dict[str, set[str]] = {}

    def register(
        self,
        importer_cls: Type[Importer],
        addon_name: Optional[str] = None,
    ) -> None:
        """Register an importer for its extensions and MIME types."""
        name = importer_cls.__name__
        self._importers_by_name[name] = importer_cls
        for ext in importer_cls.extensions:
            self._importers_by_ext[ext] = importer_cls

        for mime_type in importer_cls.mime_types:
            self._importers_by_mime[mime_type] = importer_cls

        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(name)

        addon_str = f" from {addon_name}" if addon_name else ""
        logger.debug(f"Registered importer {importer_cls.__name__}{addon_str}")

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all importers registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of importers unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for name in items:
            if name in self._importers_by_name:
                del self._importers_by_name[name]
                count += 1
        for ext in list(self._importers_by_ext.keys()):
            if self._importers_by_ext[ext].__name__ in items:
                del self._importers_by_ext[ext]
        for mime in list(self._importers_by_mime.keys()):
            if self._importers_by_mime[mime].__name__ in items:
                del self._importers_by_mime[mime]
        return count

    def get_by_extension(self, file_ext: str) -> Optional[Type[Importer]]:
        """Get importer class for a file extension."""
        return self._importers_by_ext.get(file_ext)

    def get_by_mime_type(self, mime_type: str) -> Optional[Type[Importer]]:
        """Get importer class for a MIME type."""
        return self._importers_by_mime.get(mime_type)

    def get_supported_extensions(self) -> list[str]:
        """Get all supported file extensions."""
        return list(self._importers_by_ext.keys())

    def get_by_name(self, name: str) -> Optional[Type[Importer]]:
        """Get importer class by its class name."""
        return self._importers_by_name.get(name)

    def get_for_file(self, file_path: Path) -> Optional[Type[Importer]]:
        """Get the appropriate importer for a file path."""
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            importer_cls = self.get_by_mime_type(mime_type)
            if importer_cls:
                return importer_cls
        file_ext = file_path.suffix.lower()
        return self.get_by_extension(file_ext)

    def get_all_filters(self) -> List[FileFilter]:
        """
        Get all supported import filters.

        Returns:
            A list of FileFilter objects.
        """
        seen_exts = set()
        filters = []
        for importer_cls in set(self._importers_by_ext.values()):
            if importer_cls.extensions[0] not in seen_exts:
                seen_exts.update(importer_cls.extensions)
                filters.append(
                    FileFilter(
                        label=importer_cls.label,
                        extensions=importer_cls.extensions,
                        mime_types=importer_cls.mime_types,
                    )
                )
        return filters

    def get_all(self) -> List[Type[Importer]]:
        """
        Get all registered importer classes.

        Returns:
            A list of unique importer classes.
        """
        return list(self._importers_by_name.values())

    def by_feature(self, feature: ImporterFeature) -> List[Type[Importer]]:
        """
        Get all importer classes that support a specific feature.

        Args:
            feature: The ImporterFeature to filter by.

        Returns:
            A list of importer classes that support the feature.
        """
        return [
            imp
            for imp in self._importers_by_name.values()
            if feature in imp.features
        ]

    def mime_types_by_feature(self, feature: ImporterFeature) -> set[str]:
        """
        Get all MIME types supported by importers with a specific feature.

        Args:
            feature: The ImporterFeature to filter by.

        Returns:
            A set of MIME type strings.
        """
        return {
            mime
            for imp in self._importers_by_name.values()
            if feature in imp.features
            for mime in imp.mime_types
        }


class RendererRegistry:
    """Registry for asset renderers."""

    def __init__(self):
        self._renderers: Dict[str, Renderer] = {}
        self._addon_items: Dict[str, set[str]] = {}

    def register(
        self,
        renderer: Renderer,
        addon_name: Optional[str] = None,
    ) -> None:
        """Register a renderer for an asset type."""
        key = renderer.__class__.__name__
        self._renderers[key] = renderer
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(key)
        addon_str = f" from {addon_name}" if addon_name else ""
        logger.debug(f"Registered renderer for {key}{addon_str}")

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all renderers registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of renderers unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for key in items:
            if key in self._renderers:
                del self._renderers[key]
                count += 1
        return count

    def get(self, asset_type: str) -> Optional[Renderer]:
        """Get renderer for an asset type."""
        from rayforge.worker_init import ensure_addons_loaded

        ensure_addons_loaded()
        return self._renderers.get(asset_type)

    def all(self) -> Dict[str, Renderer]:
        """Return all registered renderers."""
        return self._renderers.copy()

    def get_by_name(self, name: str) -> Optional[Renderer]:
        """Get renderer by its class name."""
        return self._renderers.get(name)


class ExporterRegistry:
    """Registry for file exporters."""

    def __init__(self):
        self._exporters_by_ext: Dict[str, Type[BaseExporter]] = {}
        self._exporters_by_mime: Dict[str, Type[BaseExporter]] = {}
        self._addon_items: Dict[str, set[str]] = {}

    def register(
        self,
        exporter_cls: Type[BaseExporter],
        addon_name: Optional[str] = None,
    ) -> None:
        """Register an exporter for its extensions and MIME types."""
        name = exporter_cls.__name__
        for ext in exporter_cls.extensions:
            self._exporters_by_ext[ext] = exporter_cls

        for mime_type in exporter_cls.mime_types:
            self._exporters_by_mime[mime_type] = exporter_cls

        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(name)

        addon_str = f" from {addon_name}" if addon_name else ""
        logger.debug(f"Registered exporter {exporter_cls.__name__}{addon_str}")

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all exporters registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of exporters unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for ext in list(self._exporters_by_ext.keys()):
            if self._exporters_by_ext[ext].__name__ in items:
                del self._exporters_by_ext[ext]
                count += 1
        for mime in list(self._exporters_by_mime.keys()):
            if self._exporters_by_mime[mime].__name__ in items:
                del self._exporters_by_mime[mime]
        return count

    def get_by_extension(self, file_ext: str) -> Optional[Type[BaseExporter]]:
        """Get exporter class for a file extension."""
        return self._exporters_by_ext.get(file_ext)

    def get_by_mime_type(self, mime_type: str) -> Optional[Type[BaseExporter]]:
        """Get exporter class for a MIME type."""
        return self._exporters_by_mime.get(mime_type)

    def get_all_filters(self) -> List[FileFilter]:
        """
        Get all supported export filters.

        Returns:
            A list of FileFilter objects.
        """
        seen_exts = set()
        filters = []
        for exporter_cls in set(self._exporters_by_ext.values()):
            if exporter_cls.extensions[0] not in seen_exts:
                seen_exts.update(exporter_cls.extensions)
                filters.append(
                    FileFilter(
                        label=exporter_cls.label,
                        extensions=exporter_cls.extensions,
                        mime_types=exporter_cls.mime_types,
                    )
                )
        return filters


importer_registry = ImporterRegistry()
renderer_registry = RendererRegistry()
exporter_registry = ExporterRegistry()
