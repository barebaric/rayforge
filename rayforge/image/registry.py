import logging
from dataclasses import dataclass
from typing import List, Optional, Type, Dict, Tuple

from .base_exporter import BaseExporter
from .base_importer import Importer
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
        self._addon_items: Dict[str, set[str]] = {}

    def register(
        self,
        importer_cls: Type[Importer],
        addon_name: Optional[str] = None,
    ) -> None:
        """Register an importer for its extensions and MIME types."""
        name = importer_cls.__name__
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
        for ext in list(self._importers_by_ext.keys()):
            if self._importers_by_ext[ext].__name__ in items:
                del self._importers_by_ext[ext]
                count += 1
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
