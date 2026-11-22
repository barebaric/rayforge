import sys
from unittest.mock import MagicMock

from rayforge.shared.util import sleep_inhibitor as module


class FakeBackend:
    def __init__(self, acquire_result: bool = True):
        self.acquire_result = acquire_result
        self.acquire_calls = 0
        self.release_calls = 0

    def acquire(self) -> bool:
        self.acquire_calls += 1
        return self.acquire_result

    def release(self) -> None:
        self.release_calls += 1


class TestSleepInhibitor:
    def test_acquire_and_release(self, monkeypatch):
        backend = FakeBackend()
        monkeypatch.setattr(module, "_create_backend", lambda: backend)
        inhibitor = module.SleepInhibitor()

        inhibitor.acquire()

        assert inhibitor.active is True
        assert backend.acquire_calls == 1

        inhibitor.release()

        assert inhibitor.active is False
        assert backend.release_calls == 1

    def test_reference_counting(self, monkeypatch):
        backend = FakeBackend()
        monkeypatch.setattr(module, "_create_backend", lambda: backend)
        inhibitor = module.SleepInhibitor()

        inhibitor.acquire()
        inhibitor.acquire()
        assert backend.acquire_calls == 1

        inhibitor.release()
        assert inhibitor.active is True
        assert backend.release_calls == 0

        inhibitor.release()
        assert inhibitor.active is False
        assert backend.release_calls == 1

    def test_release_without_acquire_is_a_no_op(self):
        inhibitor = module.SleepInhibitor()
        inhibitor.release()
        assert inhibitor.active is False

    def test_failed_acquire_stays_inactive(self, monkeypatch):
        backend = FakeBackend(acquire_result=False)
        monkeypatch.setattr(module, "_create_backend", lambda: backend)
        inhibitor = module.SleepInhibitor()

        inhibitor.acquire()

        assert inhibitor.active is False
        inhibitor.release()
        assert backend.release_calls == 0

    def test_release_all(self, monkeypatch):
        backend = FakeBackend()
        monkeypatch.setattr(module, "_create_backend", lambda: backend)
        inhibitor = module.SleepInhibitor()

        inhibitor.acquire()
        inhibitor.acquire()
        inhibitor.release_all()

        assert inhibitor.active is False
        assert backend.release_calls == 1

        inhibitor.release()
        assert backend.release_calls == 1


class TestProcessBackend:
    def test_missing_command_returns_false(self, monkeypatch):
        popen = MagicMock(side_effect=FileNotFoundError)
        monkeypatch.setattr(module.subprocess, "Popen", popen)
        backend = module._ProcessBackend(["does-not-exist"], True)

        assert backend.acquire() is False

    def test_process_that_exits_immediately_returns_false(self, monkeypatch):
        monkeypatch.setattr(
            module.subprocess,
            "Popen",
            lambda *args, **kwargs: MagicMock(poll=lambda: 0),
        )
        backend = module._ProcessBackend(["true"], release_by_stdin=False)

        assert backend.acquire() is False


class TestWindowsBackend:
    def test_sets_and_clears_execution_state(self, monkeypatch):
        kernel32 = MagicMock()
        kernel32.SetThreadExecutionState.return_value = 1
        fake_ctypes = MagicMock()
        fake_ctypes.windll.kernel32 = kernel32
        monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

        backend = module._WindowsBackend()
        assert backend.acquire() is True

        backend.release()

        assert kernel32.SetThreadExecutionState.call_count == 2
        first_call, last_call = kernel32.SetThreadExecutionState.call_args_list
        assert first_call.args[0] == (
            module._WindowsBackend.ES_CONTINUOUS
            | module._WindowsBackend.ES_SYSTEM_REQUIRED
        )
        assert last_call.args[0] == module._WindowsBackend.ES_CONTINUOUS
