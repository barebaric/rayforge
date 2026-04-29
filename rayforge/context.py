import logging
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .addon_mgr.addon_manager import AddonManager
    from .camera.manager import CameraManager
    from .core.addon_config import AddonConfig
    from .core.ai.ai_service import AIService
    from .core.ai.config import AIConfigManager
    from .core.config import Config, ConfigManager
    from .core.library_manager import LibraryManager
    from .core.model_manager import ModelManager
    from .core.recipe_manager import RecipeManager
    from .debug import DebugDumpManager
    from .license import LicenseValidator
    from .machine.device.manager import DeviceProfileManager
    from .machine.models.machine import Machine
    from .machine.models.manager import MachineManager
    from .machine.models.dialect_manager import DialectManager
    import pluggy


logger = logging.getLogger(__name__)

_context_instance: Optional["RayforgeContext"] = None
_context_lock = threading.Lock()


class RayforgeContext:
    """
    A central, singleton context for managing the lifecycle of major
    application services.
    """

    def __init__(self):
        """
        Initializes the context. This constructor is lightweight and safe
        to call from any process. Only ArtifactStore is created eagerly
        to ensure it's available before any subtask starts.
        """
        from .pipeline.artifact.store import ArtifactStore
        from .debug import DebugDumpManager
        from .shared.util.localized import get_system_language

        self.artifact_store = ArtifactStore()
        self._debug_dump_manager = DebugDumpManager()
        self._language: str = get_system_language()
        self.exit_after_settle = False
        self.exit_pending = False
        self._headless: bool = False

        self._dialect_mgr: Optional["DialectManager"] = None
        self._plugin_mgr: Optional["pluggy.PluginManager"] = None
        self._addon_config: Optional["AddonConfig"] = None
        self._license_validator: Optional["LicenseValidator"] = None
        self._addon_mgr: Optional["AddonManager"] = None
        self._ai_service: Optional["AIService"] = None
        self._ai_config_mgr: Optional["AIConfigManager"] = None
        self._machine_mgr: Optional["MachineManager"] = None
        self._config_mgr: Optional["ConfigManager"] = None
        self._config: Optional["Config"] = None
        self._camera_mgr: Optional["CameraManager"] = None
        self._material_mgr: Optional["LibraryManager"] = None
        self._model_mgr: Optional["ModelManager"] = None
        self._recipe_mgr: Optional["RecipeManager"] = None
        self._device_profile_mgr: Optional["DeviceProfileManager"] = None

    @property
    def machine(self) -> Optional["Machine"]:
        """
        Returns the active machine from the config, or None if
        the config or machine is not set.
        """
        return self.config.machine

    @property
    def dialect_mgr(self) -> "DialectManager":
        """Returns the dialect manager."""
        if self._dialect_mgr is None:
            from .config import DIALECT_DIR
            from .machine.models.dialect_manager import DialectManager

            self._dialect_mgr = DialectManager(DIALECT_DIR)
        return self._dialect_mgr

    @property
    def plugin_mgr(self) -> "pluggy.PluginManager":
        """Returns the plugin manager."""
        if self._plugin_mgr is None:
            import pluggy
            from .core.hooks import RayforgeSpecs

            self._plugin_mgr = pluggy.PluginManager("rayforge")
            self._plugin_mgr.add_hookspecs(RayforgeSpecs)
        return self._plugin_mgr

    @property
    def license_validator(self) -> "LicenseValidator":
        """Returns the license validator."""
        if self._license_validator is None:
            from .config import LICENSES_DIR, PATREON_CLIENT_ID
            from .license import LicenseValidator

            self._license_validator = LicenseValidator(
                LICENSES_DIR, PATREON_CLIENT_ID
            )
        return self._license_validator

    @property
    def addon_config(self) -> "AddonConfig":
        """Returns the addon configuration."""
        if self._addon_config is None:
            from .config import CONFIG_DIR
            from .core.addon_config import AddonConfig

            self._addon_config = AddonConfig(CONFIG_DIR)
            self._addon_config.load()
        return self._addon_config

    @property
    def addon_mgr(self) -> "AddonManager":
        """Returns the addon manager."""
        if self._addon_mgr is None:
            from .config import (
                ADDONS_DIR,
                BUILTIN_ADDONS_DIR,
                PRIVATE_ADDONS_DIR,
            )
            from .addon_mgr.addon_manager import AddonManager

            self._addon_mgr = AddonManager(
                [BUILTIN_ADDONS_DIR, PRIVATE_ADDONS_DIR, ADDONS_DIR],
                ADDONS_DIR,
                self.plugin_mgr,
                self.addon_config,
                license_validator=self.license_validator,
            )
            self._load_addons_and_call_hooks()
        return self._addon_mgr

    def _load_addons_and_call_hooks(self):
        """
        Loads addons and calls registration hooks.

        This is called automatically when addon_mgr is first accessed.
        In headless mode, only worker entry points are loaded.
        """
        if self._addon_mgr is None:
            return

        from .core.registration import (
            call_registration_hooks,
            get_registries,
        )
        from .doceditor.layout.registry import (
            register_builtin_layout_strategies,
        )

        register_builtin_layout_strategies()

        registries = get_registries(headless=self._headless)
        self._addon_mgr.set_registries(registries)
        self._addon_mgr.load_installed_addons(worker_only=self._headless)
        call_registration_hooks(
            self.plugin_mgr,
            headless=self._headless,
            registries=registries,
        )
        self.plugin_mgr.hook.rayforge_init(context=self)

        logger.info(f"Addons loaded (headless={self._headless})")

    @property
    def ai_service(self) -> "AIService":
        """Returns the AI service."""
        if self._ai_service is None:
            from .core.ai.ai_service import AIService

            self._ai_service = AIService()
            _ = self.ai_config_mgr
        return self._ai_service

    @property
    def ai_config_mgr(self) -> "AIConfigManager":
        """Returns the AI configuration manager."""
        if self._ai_config_mgr is None:
            from .config import AI_CONFIG_FILE
            from .core.ai.config import AIConfigManager

            self._ai_config_mgr = AIConfigManager(
                AI_CONFIG_FILE, self.ai_service
            )
            self._ai_config_mgr.load()
        return self._ai_config_mgr

    @property
    def machine_mgr(self) -> "MachineManager":
        """Returns the machine manager."""
        if self._machine_mgr is None:
            from .config import MACHINE_DIR
            from .machine.models.manager import MachineManager

            logger.info("Lazy loading machine manager")
            self._machine_mgr = MachineManager(MACHINE_DIR)
            if not self._machine_mgr.machines:
                self._machine_mgr.create_default_machine()
        return self._machine_mgr

    @property
    def config_mgr(self) -> "ConfigManager":
        """Returns the config manager."""
        if self._config_mgr is None:
            from .config import CONFIG_FILE
            from .core.config import ConfigManager as CoreConfigManager

            logger.info("Lazy loading config manager")
            self._config_mgr = CoreConfigManager(CONFIG_FILE, self.machine_mgr)
            self._config = self._config_mgr.config
            if not self._config.machine:
                machine = list(
                    sorted(
                        self.machine_mgr.machines.values(), key=lambda m: m.id
                    )
                )[0]
                self._config.set_machine(machine)
        return self._config_mgr

    @property
    def config(self) -> "Config":
        """Returns the config."""
        return self.config_mgr.config

    @property
    def camera_mgr(self) -> "CameraManager":
        """Returns the camera manager."""
        if self._camera_mgr is None:
            from .camera.manager import CameraManager

            logger.info("Lazy loading camera manager")
            self._camera_mgr = CameraManager(self)
            self._camera_mgr.initialize()
        return self._camera_mgr

    @property
    def material_mgr(self) -> "LibraryManager":
        """Returns the material manager."""
        if self._material_mgr is None:
            from .config import USER_MATERIALS_DIR
            from .core.library_manager import LibraryManager

            logger.info("Lazy loading material manager")
            self._material_mgr = LibraryManager(USER_MATERIALS_DIR)
            self._material_mgr.load_all_libraries()

            if not self._headless:
                self.addon_mgr.registries["library_manager"] = (
                    self._material_mgr
                )
                self.plugin_mgr.hook.register_material_libraries(
                    library_manager=self._material_mgr
                )
        return self._material_mgr

    @property
    def model_mgr(self) -> "ModelManager":
        """Returns the model manager."""
        if self._model_mgr is None:
            from .core.model_manager import ModelManager

            logger.info("Lazy loading model manager")
            self._model_mgr = ModelManager()
            self._model_mgr.register_bundled_library()

            if not self._headless:
                self.addon_mgr.registries["model_manager"] = self._model_mgr
                self.plugin_mgr.hook.register_model_libraries(
                    model_manager=self._model_mgr
                )
        return self._model_mgr

    @property
    def recipe_mgr(self) -> "RecipeManager":
        """Returns the recipe manager."""
        if self._recipe_mgr is None:
            from .config import USER_RECIPES_DIR
            from .core.recipe_manager import RecipeManager

            logger.info("Lazy loading recipe manager")
            self._recipe_mgr = RecipeManager(USER_RECIPES_DIR)
        return self._recipe_mgr

    @property
    def device_profile_mgr(self) -> "DeviceProfileManager":
        """Returns the device profile manager."""
        if self._device_profile_mgr is None:
            from .config import BUILTIN_DEVICES_DIR, USER_DEVICES_DIR
            from .machine.device.manager import DeviceProfileManager

            logger.info("Lazy loading device profile manager")
            self._device_profile_mgr = DeviceProfileManager(
                [BUILTIN_DEVICES_DIR, USER_DEVICES_DIR],
                install_dir=USER_DEVICES_DIR,
            )
            self._device_profile_mgr.discover(context=self)
        return self._device_profile_mgr

    @property
    def debug_dump_manager(self) -> "DebugDumpManager":
        """Returns the debug dump manager."""
        return self._debug_dump_manager

    @property
    def language(self) -> str:
        """
        Get the current language code for localized content.

        Returns:
            Language code (e.g., 'en', 'de', 'zh_CN')
        """
        return self._language

    @language.setter
    def language(self, value: str):
        """
        Set the current language for localized content.

        Args:
            value: Language code (will be normalized)
        """
        from .shared.util.localized import normalize_language_code

        normalized = normalize_language_code(value)
        if normalized:
            self._language = normalized

    def initialize_lite_context(self, machine_dir):
        """
        Initializes a minimal context for testing. Sets up MachineManager
        and Config without cameras, materials, recipes, or addons.
        """
        from pathlib import Path
        from .core.config import ConfigManager as CoreConfigManager
        from .machine.models.manager import MachineManager

        self._headless = True
        self._machine_mgr = MachineManager(machine_dir)

        if not self._machine_mgr.machines:
            self._machine_mgr.create_default_machine()

        config_file = Path(machine_dir) / ".." / "config.yaml"
        self._config_mgr = CoreConfigManager(config_file, self._machine_mgr)
        self._config = self._config_mgr.config
        if not self._config.machine:
            machine = list(
                sorted(self._machine_mgr.machines.values(), key=lambda m: m.id)
            )[0]
            self._config.set_machine(machine)

    async def shutdown(self):
        """
        Shuts down all managed services in the correct order.
        """
        logger.info("RayforgeContext shutting down...")
        if self._camera_mgr:
            self._camera_mgr.shutdown()
        if self._machine_mgr:
            await self._machine_mgr.shutdown()
        if self._ai_service:
            await self._ai_service.close_all()
        self.artifact_store.shutdown()
        logger.info("RayforgeContext shutdown complete.")


def get_context() -> "RayforgeContext":
    """
    A thread-safe, lazy-initializing accessor for the global RayforgeContext
    singleton.
    """
    global _context_instance
    if _context_instance is None:
        with _context_lock:
            if _context_instance is None:
                _context_instance = RayforgeContext()
    return _context_instance
