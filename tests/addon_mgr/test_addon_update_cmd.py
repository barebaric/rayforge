from unittest.mock import MagicMock, patch, call
import pytest
from gettext import gettext as _
from rayforge.addon_mgr.update_cmd import UpdateCommand
from rayforge.addon_mgr.addon import Addon, AddonMetadata

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


# Helper to create a properly structured mock addon
def create_mock_addon(name="addon", version="1.0.0"):
    mock_addon = MagicMock(spec=Addon)
    # Configure the nested metadata structure
    mock_addon.metadata = MagicMock()
    mock_addon.metadata.name = name
    mock_addon.metadata.version = version
    return mock_addon


# Helper to create mock remote metadata
def create_mock_remote_meta(
    name="addon", version="1.1.0", url="http://a.b/c.git"
):
    # Using a spec is good, but requires explicit attribute configuration
    # For a simple data object, a basic MagicMock is often easier.
    mock_meta = MagicMock(spec=AddonMetadata)
    mock_meta.name = name
    mock_meta.display_name = f"Display {name}"
    mock_meta.version = version
    mock_meta.url = url
    return mock_meta


@pytest.fixture
def mock_context():
    """Provides a mock RayforgeContext with mocked managers."""
    context = MagicMock()
    context.addon_mgr = MagicMock()
    return context


@pytest.fixture
def update_command(mock_context):
    """Provides an instance of UpdateCommand with mocked dependencies."""
    # Create a mock TaskManager that executes callbacks immediately
    mock_task_mgr = MagicMock()
    mock_task_mgr.schedule_on_main_thread = MagicMock(
        side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
    )
    return UpdateCommand(mock_task_mgr, mock_context)


class TestUpdateCheck:
    async def test_check_for_updates_worker_no_updates(
        self, update_command, mock_context
    ):
        """
        Verify that no signal is sent when the addon manager finds no
        updates.
        """
        # Arrange
        mock_context.addon_mgr.check_for_updates.return_value = []
        mock_ctx_task = MagicMock()

        # Act
        with patch.object(
            update_command.notification_requested, "send"
        ) as mock_send:
            await update_command._check_for_updates_worker(mock_ctx_task)

            # Assert
            mock_context.addon_mgr.check_for_updates.assert_called_once()
            mock_send.assert_not_called()
            mock_ctx_task.set_message.assert_has_calls(
                [
                    call(_("Checking for addon updates...")),
                    call(_("Addons are up to date.")),
                ]
            )

    async def test_check_for_updates_worker_finds_updates(
        self, update_command, mock_context
    ):
        """
        Verify the correct signal is sent when updates are found.
        """
        # Arrange
        updates_payload = [(create_mock_addon(), create_mock_remote_meta())]
        mock_context.addon_mgr.check_for_updates.return_value = updates_payload
        mock_ctx_task = MagicMock()

        # Act
        with patch.object(
            update_command.notification_requested, "send"
        ) as mock_send:
            await update_command._check_for_updates_worker(mock_ctx_task)

            # Assert
            mock_context.addon_mgr.check_for_updates.assert_called_once()
            mock_send.assert_called_once()
            # Check the keyword arguments of the signal send
            sent_kwargs = mock_send.call_args.kwargs
            assert "message" in sent_kwargs
            assert "persistent" in sent_kwargs and sent_kwargs["persistent"]
            assert "action_label" in sent_kwargs
            assert "action_callback" in sent_kwargs
            mock_ctx_task.set_message.assert_called_with(
                _("Addon updates found.")
            )

    async def test_check_for_updates_worker_handles_exception(
        self, update_command, mock_context
    ):
        """
        Verify that an exception during the check is handled gracefully.
        """
        # Arrange
        mock_context.addon_mgr.check_for_updates.side_effect = Exception(
            "Network error"
        )
        mock_ctx_task = MagicMock()

        # Act
        with patch.object(
            update_command.notification_requested, "send"
        ) as mock_send:
            await update_command._check_for_updates_worker(mock_ctx_task)

            # Assert
            mock_send.assert_not_called()
            mock_ctx_task.set_message.assert_called_with(
                _("Update check failed.")
            )


class TestUpdateInstallation:
    async def test_install_updates_worker_all_succeed(
        self, update_command, mock_context
    ):
        """
        Test the installation worker when all addon updates succeed.
        """
        # Arrange
        updates_to_install = [
            (
                create_mock_addon("addon1"),
                create_mock_remote_meta("addon1", "1.1.0"),
            ),
            (
                create_mock_addon("addon2"),
                create_mock_remote_meta("addon2", "2.1.0"),
            ),
        ]
        # Mock install_addon to return a dummy path on success
        mock_context.addon_mgr.install_addon.return_value = "/fake/path"
        mock_ctx_task = MagicMock()

        # Act
        with patch.object(
            update_command.notification_requested, "send"
        ) as mock_send:
            await update_command._install_updates_worker(
                mock_ctx_task, updates_to_install
            )

            # Assert
            assert mock_context.addon_mgr.install_addon.call_count == 2
            mock_send.assert_called_once()
            # Check message of the notification
            sent_kwargs = mock_send.call_args.kwargs
            assert "message" in sent_kwargs
            assert "2 addons successfully updated" in sent_kwargs["message"]
            mock_ctx_task.set_message.assert_called_with(
                _("All addon updates installed!")
            )

    async def test_install_updates_worker_some_fail(
        self, update_command, mock_context
    ):
        """
        Test the installation worker when one update fails and one succeeds.
        """
        # Arrange
        updates_to_install = [
            (
                create_mock_addon("addon1"),
                create_mock_remote_meta("addon1", "1.1.0"),
            ),
            (
                create_mock_addon("addon2"),
                create_mock_remote_meta("addon2", "2.1.0"),
            ),
        ]
        # First call succeeds, second call fails (returns None)
        mock_context.addon_mgr.install_addon.side_effect = [
            "/fake/path",
            None,
        ]
        mock_ctx_task = MagicMock()

        # Act
        with patch.object(
            update_command.notification_requested, "send"
        ) as mock_send:
            await update_command._install_updates_worker(
                mock_ctx_task, updates_to_install
            )

            # Assert
            assert mock_context.addon_mgr.install_addon.call_count == 2
            mock_send.assert_called_once()
            # Check message of the notification
            sent_kwargs = mock_send.call_args.kwargs
            assert "message" in sent_kwargs
            assert "1 addons updated" in sent_kwargs["message"]
            assert "1 failed" in sent_kwargs["message"]
            mock_ctx_task.set_message.assert_called_with(
                _("Finished with {num_failed} errors.").format(num_failed=1)
            )

    async def test_install_updates_worker_all_fail(
        self, update_command, mock_context
    ):
        """
        Test the installation worker when all updates fail.
        """
        # Arrange
        updates_to_install = [
            (
                create_mock_addon("addon1"),
                create_mock_remote_meta("addon1", "1.1.0"),
            ),
        ]
        # Simulate failure by returning None
        mock_context.addon_mgr.install_addon.return_value = None
        mock_ctx_task = MagicMock()

        # Act
        with patch.object(
            update_command.notification_requested, "send"
        ) as mock_send:
            await update_command._install_updates_worker(
                mock_ctx_task, updates_to_install
            )

            # Assert
            mock_send.assert_called_once()
            sent_kwargs = mock_send.call_args.kwargs
            assert "message" in sent_kwargs
            assert "Failed to update addon." in sent_kwargs["message"]
            mock_ctx_task.set_message.assert_called_with(
                _("Finished with {num_failed} errors.").format(num_failed=1)
            )

    async def test_install_updates_worker_no_updates(
        self, update_command, mock_context
    ):
        """
        Test that calling the worker with an empty list does nothing.
        """
        # Arrange
        mock_ctx_task = MagicMock()

        # Act
        await update_command._install_updates_worker(mock_ctx_task, [])

        # Assert
        mock_context.addon_mgr.install_addon.assert_not_called()
