import pluggy

hookspec = pluggy.HookspecMarker("rayforge")
hookimpl = pluggy.HookimplMarker("rayforge")

PLUGIN_API_VERSION = 1


class RayforgeSpecs:
    """
    Core hook specifications.
    Plugins (Rayforge packages) implement these methods to extend
    functionality.
    """

    @hookspec
    def rayforge_init(self, context):
        """
        Called when the application context is fully initialized.
        Use this for general setup, logging, or UI injection.

        Args:
            context: The global RayforgeContext.
        """

    @hookspec
    def on_unload(self):
        """
        Called when an addon is being disabled or unloaded.
        Use this to clean up resources, close connections, unregister
        handlers, etc.
        """

    @hookspec
    def register_machines(self, machine_manager):
        """
        Called to allow plugins to register new machine drivers.

        Args:
            machine_manager: The application's MachineManager instance.
        """

    @hookspec
    def register_steps(self, step_registry):
        """
        Called to allow plugins to register custom step types.

        Args:
            step_registry: The global StepRegistry instance.
        """

    @hookspec
    def register_producers(self, producer_registry):
        """
        Called to allow plugins to register custom ops producers.

        Args:
            producer_registry: The global ProducerRegistry instance.
        """

    @hookspec
    def register_step_widgets(self, widget_registry):
        """
        Called to allow plugins to register custom step settings widgets.

        Args:
            widget_registry: The global StepWidgetRegistry instance.
        """

    @hookspec
    def register_menu_items(self, menu_registry):
        """
        Called to allow plugins to register menu items.

        Args:
            menu_registry: The global MenuRegistry instance.
        """
