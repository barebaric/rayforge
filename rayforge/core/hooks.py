import pluggy

hookspec = pluggy.HookspecMarker("rayforge")
hookimpl = pluggy.HookimplMarker("rayforge")

MINIMUM_API_VERSION = 1
PLUGIN_API_VERSION = 3


"""
API Changelog
=============

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
- ``register_step_widgets``: Register custom step settings widgets
- ``register_menu_items``: Register menu items
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
    def register_step_widgets(self, widget_registry):
        """
        Called to allow addons to register custom step settings widgets.

        .. versionadded:: 1

        Args:
            widget_registry: The global StepWidgetRegistry instance.
        """

    @hookspec
    def register_menu_items(self, menu_registry):
        """
        Called to allow addons to register menu items.

        .. versionadded:: 1

        Args:
            menu_registry: The global MenuRegistry instance.
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
    def register_actions(self, window):
        """
        Called to allow addons to register window actions.

        .. versionadded:: 1

        Args:
            window: The MainWindow instance. Use
                window.action_registry.register() to register actions with
                addon tracking for proper cleanup.
        """

    @hookspec
    def register_layout_strategies(self, layout_registry):
        """
        Called to allow addons to register custom layout strategies.

        .. versionadded:: 2

        Args:
            layout_registry: Registry for layout strategies.
                Supports registering strategies with optional metadata.
        """
