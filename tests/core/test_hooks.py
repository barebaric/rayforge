"""Tests for the hook system using synthetic plugins."""

from rayforge.context import RayforgeContext
from rayforge.core.hooks import hookimpl


class MockPlugin:
    """A fake plugin defined in memory."""

    def __init__(self):
        self.init_called = False
        self.registered_machines = False

    @hookimpl
    def rayforge_init(self, context):
        self.init_called = True
        assert isinstance(context, RayforgeContext)

    @hookimpl
    def register_machines(self, machine_manager):
        self.registered_machines = True


class TestHookLifecycle:
    """Test cases for hook lifecycle execution."""

    def test_hook_lifecycle_execution(self):
        """Test that hooks are triggered in the correct order."""
        context = RayforgeContext()
        plugin = MockPlugin()
        context.plugin_mgr.register(plugin)

        context._machine_mgr = "mock_machine_mgr"  # type: ignore

        context.plugin_mgr.hook.register_machines(
            machine_manager=context._machine_mgr
        )
        context.plugin_mgr.hook.rayforge_init(context=context)

        assert plugin.init_called is True
        assert plugin.registered_machines is True

    def test_multiple_plugins_hooks_execution(self):
        """Test that multiple plugins have their hooks executed."""
        plugin1 = MockPlugin()
        plugin2 = MockPlugin()
        context = RayforgeContext()

        context.plugin_mgr.register(plugin1)
        context.plugin_mgr.register(plugin2)

        context._machine_mgr = "mock_machine_mgr"  # type: ignore

        context.plugin_mgr.hook.register_machines(
            machine_manager=context._machine_mgr
        )
        context.plugin_mgr.hook.rayforge_init(context=context)

        assert plugin1.init_called is True
        assert plugin1.registered_machines is True
        assert plugin2.init_called is True
        assert plugin2.registered_machines is True

    def test_plugin_without_hookimpl_not_registered(self):
        """Test that plugins without hookimpl are not called."""

        class NoHookImplPlugin:
            """A plugin without any hookimpl methods."""

            def __init__(self):
                self.init_called = False

            def rayforge_init(self, context):
                self.init_called = True

        plugin = NoHookImplPlugin()
        context = RayforgeContext()

        context.plugin_mgr.register(plugin)

        context.plugin_mgr.hook.rayforge_init(context=context)

        assert plugin.init_called is False

    def test_hook_receives_correct_context(self):
        """Test that hooks receive the correct context instance."""
        received_context = []

        class ContextPlugin:
            """Plugin that captures the received context."""

            @hookimpl
            def rayforge_init(self, context):
                received_context.append(context)

        context = RayforgeContext()
        plugin = ContextPlugin()

        context.plugin_mgr.register(plugin)
        context.plugin_mgr.hook.rayforge_init(context=context)

        assert len(received_context) == 1
        assert received_context[0] is context

    def test_hook_receives_correct_machine_manager(self):
        """Test that register_machines hook receives the correct manager."""
        received_manager = []

        class MachineManagerPlugin:
            """Plugin that captures the received machine manager."""

            @hookimpl
            def register_machines(self, machine_manager):
                received_manager.append(machine_manager)

        context = RayforgeContext()
        plugin = MachineManagerPlugin()
        mock_manager = "mock_machine_mgr"

        context.plugin_mgr.register(plugin)
        context.plugin_mgr.hook.register_machines(machine_manager=mock_manager)

        assert len(received_manager) == 1
        assert received_manager[0] == mock_manager
