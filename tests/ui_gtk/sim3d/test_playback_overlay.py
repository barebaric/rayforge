"""UI tests for the PlaybackOverlay playback glue."""

# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest

from rayforge.ui_gtk.sim3d.playback_overlay import PlaybackOverlay


class FakePlayer:
    """Minimal OpPlayer stand-in exposing the playback surface."""

    def __init__(self, n_ops: int = 0):
        class FakeOps:
            def __init__(self, n):
                self._n = n

            def len(self) -> int:
                return self._n

            def __len__(self) -> int:
                return self._n

        self.ops = FakeOps(n_ops)
        self._current_index = -1

    @property
    def current_index(self) -> int:
        return self._current_index

    def seek(self, index: int):
        self._current_index = index

    def seek_to_fraction(self, fraction: float):
        self._current_index = int(self.ops.len() * fraction)


class FakeCanvas:
    """Minimal Canvas3D stand-in recording queue_render calls."""

    def __init__(self):
        self.render_queued = 0

    def queue_render(self):
        self.render_queued += 1


@pytest.mark.ui
def test_command_count_no_player(ui_context_initializer):
    overlay = PlaybackOverlay()
    assert overlay.command_count == 0
    assert overlay.current_index == -1


@pytest.mark.ui
def test_command_count_with_player(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_player(FakePlayer(n_ops=10))
    assert overlay.command_count == 10


@pytest.mark.ui
def test_set_player_none_disables_range(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_player(FakePlayer(n_ops=10))
    overlay.set_player(None)
    assert overlay.command_count == 0
    assert not overlay._slider.get_sensitive()


@pytest.mark.ui
def test_seek_seeks_player_and_queues_render(ui_context_initializer):
    overlay = PlaybackOverlay()
    canvas = FakeCanvas()
    overlay.set_canvas(canvas)
    player = FakePlayer(n_ops=10)
    overlay.set_player(player)
    canvas.render_queued = 0
    overlay.seek(4)
    assert player.current_index == 4
    assert canvas.render_queued == 1


@pytest.mark.ui
def test_seek_without_player_noop(ui_context_initializer):
    overlay = PlaybackOverlay()
    canvas = FakeCanvas()
    overlay.set_canvas(canvas)
    overlay.seek(4)
    assert canvas.render_queued == 0


@pytest.mark.ui
def test_seek_to_fraction_seeks_and_syncs_slider(ui_context_initializer):
    overlay = PlaybackOverlay()
    canvas = FakeCanvas()
    overlay.set_canvas(canvas)
    player = FakePlayer(n_ops=10)
    overlay.set_player(player)
    canvas.render_queued = 0
    overlay.seek_to_fraction(0.5)
    assert player.current_index == 5
    assert canvas.render_queued == 1


@pytest.mark.ui
def test_handle_space_toggles_only_when_playable(ui_context_initializer):
    overlay = PlaybackOverlay()
    overlay.set_canvas(FakeCanvas())
    overlay.set_player(FakePlayer(n_ops=0))
    assert not overlay._play_button.get_sensitive()
    overlay.handle_space()
    assert not overlay._playing

    overlay.set_player(FakePlayer(n_ops=10))
    assert overlay._play_button.get_sensitive()
    overlay.handle_space()
    assert overlay._playing
