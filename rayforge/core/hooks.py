import pluggy

hookspec = pluggy.HookspecMarker("rayforge")
hookimpl = pluggy.HookimplMarker("rayforge")

MINIMUM_API_VERSION = 1
PLUGIN_API_VERSION = 4


"""
API Changelog
=============

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
- ``register_step_widgets``: Register custom step settings widgets
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
    def register_step_widgets(self, widget_registry):
        """
        Called to allow addons to register custom step settings widgets.

        .. versionadded:: 1

        Args:
            widget_registry: The global StepWidgetRegistry instance.
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
