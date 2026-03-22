import pluggy

hookspec = pluggy.HookspecMarker("rayforge")
hookimpl = pluggy.HookimplMarker("rayforge")

MINIMUM_API_VERSION = 1
PLUGIN_API_VERSION = 9


"""
API Changelog
============

Version 9
---------
Added ``main_window_ready`` hook to allow addons to register UI pages,
commands, and other components that require access to the main window.
Added ``register_exporters`` hook to allow addons to register file
exporters.
Added ``register_importers`` hook to allow addons to register file
importers.
Added ``register_renderers`` hook to allow addons to register custom
renderers for their asset types.

Version 8
---------
Added ``register_asset_types`` hook to allow addons to register custom
asset types. Assets are registered via the asset_type_registry and can
be deserialized from document files dynamically.

Version 7
---------
Added ``register_material_libraries`` hook to allow addons to register
material libraries. Addons can return a list of paths to directories
containing material YAML files.

Version 6
---------
Added ``register_transformers`` hook to allow addons to register custom
OpsTransformer classes for post-processing operations. Transformers are
now registered via the transformer_registry instead of introspection.

Version 5
---------
Replaced ``register_step_widgets`` hook with ``step_settings_loaded``
and ``transformer_settings_loaded`` hooks. Addons now add their widgets
directly to the settings dialog when the hook is called, instead of
registering widget classes in advance. This gives addons full control
over widget instantiation and lifecycle.

Version 4
---------
Consolidated menu and action registration. The ``register_menu_items``
hook has been removed. Use ``register_actions`` with the action_registry
to register actions with optional menu and toolbar placement.

The ``register_layout_strategies`` hook now only registers strategy
classes. Layout actions should be registered via ``register_actions``.

Version 3
---------
Added AI provider support to the core API. No changes to hook
specifications - existing addons remain compatible. The core
RayforgeContext now includes AI provider management capabilities.

Version 2
---------
Added ``register_layout_strategies`` hook to allow addons to register
custom layout strategies for the UI. This enables addons to define
how content is arranged and displayed in different contexts.

Version 1
-------
Initial plugin API release. Includes core hooks for addon lifecycle,
resource registration, and UI integration:

- ``rayforge_init``: Called when application context is initialized
- ``on_unload``: Called when addon is disabled or unloaded
- ``register_machines``: Register new machine drivers
- ``register_steps``: Register custom step types
- ``register_producers``: Register custom ops producers
- ``register_step_widgets``: Register step settings widgets (removed v5)
- ``register_menu_items``: Register menu items (removed in v4)
- ``register_commands``: Register editor commands
- ``register_actions``: Register window actions
"""


class RayforgeSpecs:
    """
    Core hook specifications.
    Addons implement these methods to extend functionality.
    """

    @hookspec
    def rayforge_init(self, context):
        """
        Called when the application context is fully initialized.
        Use this for general setup, logging, or UI injection.

        .. versionadded:: 1

        Args:
            context: The global RayforgeContext.
        """

    @hookspec
    def on_unload(self):
        """
        Called when an addon is being disabled or unloaded.
        Use this to clean up resources, close connections, unregister
        handlers, etc.

        .. versionadded:: 1
        """

    @hookspec
    def register_machines(self, machine_manager):
        """
        Called to allow addons to register new machine drivers.

        .. versionadded:: 1

        Args:
            machine_manager: The application's MachineManager instance.
        """

    @hookspec
    def register_steps(self, step_registry):
        """
        Called to allow addons to register custom step types.

        .. versionadded:: 1

        Args:
            step_registry: The global StepRegistry instance.
        """

    @hookspec
    def register_producers(self, producer_registry):
        """
        Called to allow addons to register custom ops producers.

        .. versionadded:: 1

        Args:
            producer_registry: The global ProducerRegistry instance.
        """

    @hookspec
    def register_transformers(self, transformer_registry):
        """
        Called to allow addons to register custom ops transformers.

        .. versionadded:: 6

        Args:
            transformer_registry: The global TransformerRegistry instance.
        """

    @hookspec
    def register_asset_types(self, asset_type_registry):
        """
        Called to allow addons to register custom asset types.

        .. versionadded:: 8

        Args:
            asset_type_registry: The global AssetTypeRegistry instance.
        """

    @hookspec
    def step_settings_loaded(self, dialog, step, producer):
        """
        Called when a step settings dialog is being populated.
        Addons can add custom widgets to the dialog based on the
        step's producer type.

        .. versionadded:: 5

        Args:
            dialog: The GeneralStepSettingsView instance to add widgets to.
            step: The Step instance being configured.
            producer: The OpsProducer instance, or None if not available.
        """

    @hookspec
    def transformer_settings_loaded(self, dialog, step, transformer):
        """
        Called when post-processing settings are being populated.
        Addons can add custom widgets for their transformers.

        .. versionadded:: 5

        Args:
            dialog: The PostProcessingSettingsView instance to add widgets to.
            step: The Step instance being configured.
            transformer: The OpsTransformer instance.
        """

    @hookspec
    def register_commands(self, command_registry):
        """
        Called to allow addons to register editor commands.

        .. versionadded:: 1

        Args:
            command_registry: The global CommandRegistry instance.
        """

    @hookspec
    def register_actions(self, action_registry):
        """
        Called to allow addons to register window actions.

        .. versionadded:: 1
        .. versionchanged:: 4
            Now receives action_registry instead of window. Use
            action_registry.register() with optional menu and toolbar
            placement parameters.

        Args:
            action_registry: The global ActionRegistry instance.
        """

    @hookspec
    def register_layout_strategies(self, layout_registry):
        """
        Called to allow addons to register custom layout strategies.

        .. versionadded:: 2
        .. versionchanged:: 4
            Only registers strategy classes. Layout actions should be
            registered via the ``register_actions`` hook with menu and
            toolbar placement.

        Args:
            layout_registry: Registry for layout strategy classes.
        """

    @hookspec
    def register_material_libraries(self, library_manager):
        """
        Called to allow addons to register material libraries.

        Addons should call ``library_manager.add_library_from_path(path)`` to
        register directories containing material YAML files. By default,
        registered libraries are read-only.

        .. versionadded:: 7

        Args:
            library_manager: The global LibraryManager instance.
        """

    @hookspec
    def register_exporters(self, exporter_registry):
        """
        Called to allow addons to register file exporters.

        Addons should call ``exporter_registry.register(exporter_cls)`` to
        register exporter classes for their supported file formats.

        .. versionadded:: 9

        Args:
            exporter_registry: The global ExporterRegistry instance.
        """

    @hookspec
    def register_importers(self, importer_registry):
        """
        Called to allow addons to register file importers.

        Addons should call ``importer_registry.register(importer_cls)`` to
        register importer classes for their supported file formats.

        .. versionadded:: 10

        Args:
            importer_registry: The global ImporterRegistry instance.
        """

    @hookspec
    def register_renderers(self, renderer_registry):
        """
        Called to allow addons to register custom renderers.

        Addons should call ``renderer_registry.register(renderer)`` to
        register renderer instances. The renderer's class name is used as
        the registry key.

        .. versionadded:: 9

        Args:
            renderer_registry: The global RendererRegistry instance.
        """

    @hookspec
    def main_window_ready(self, main_window):
        """
        Called when the main window is fully initialized.

        Addons can use this hook to register custom UI pages, commands,
        or other components that require access to the main window.

        .. versionadded:: 9

        Args:
            main_window: The MainWindow instance.
        """
