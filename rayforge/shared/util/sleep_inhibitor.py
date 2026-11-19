"""Prevent automatic system sleep while a machine job is running.

Losing the machine connection because the host suspended in the middle
of a job stops the job and can leave the controller in an alarm state.
This module takes an OS-level idle inhibitor for the duration of a job
so that only an explicit suspend request (or lid close on most systems)
still goes through.

The inhibitor is reference counted because several jobs may overlap in
the future, and it is released automatically on interpreter shutdown.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import threading
from gettext import gettext as _

logger = logging.getLogger(__name__)

_REASON = _("Rayforge is running a machine job")


class _ProcessBackend:
    """Backend that holds an inhibitor by keeping a child process alive.

    On Linux the child is ``systemd-inhibit`` running ``cat``; the
    inhibitor is held while our end of the pipe stays open, so the
    child also exits if this process dies. On macOS the child is
    ``caffeinate``, which is simply terminated on release.
    """

    def __init__(self, argv: list[str], release_by_stdin: bool):
        self._argv = argv
        self._release_by_stdin = release_by_stdin
        self._proc: subprocess.Popen | None = None

    def acquire(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                self._argv,
                stdin=(
                    subprocess.PIPE
                    if self._release_by_stdin
                    else subprocess.DEVNULL
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, ValueError) as e:
            logger.warning("Could not inhibit system sleep: %s", e)
            return False
        if self._proc.poll() is not None:
            logger.warning(
                "Sleep inhibitor command %r exited immediately",
                self._argv[0],
            )
            self._proc = None
            return False
        return True

    def release(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if self._release_by_stdin and proc.stdin is not None:
                proc.stdin.close()
                proc.wait(timeout=2)
            else:
                proc.terminate()
                proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.warning("Sleep inhibitor process did not exit")
        except OSError as e:
            logger.debug("Error while releasing sleep inhibitor: %s", e)


class _WindowsBackend:
    """Backend using SetThreadExecutionState on Windows.

    The execution state is per thread, so the flag is set from a
    dedicated worker thread that stays alive until release. This keeps
    acquire and release correct even when they are called from
    different threads.
    """

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None

    def acquire(self) -> bool:
        try:
            import ctypes

            windll = getattr(ctypes, "windll", None)
            if windll is None:
                logger.warning("Could not inhibit system sleep: no windll")
                return False
            kernel32 = windll.kernel32
        except (AttributeError, OSError) as e:
            logger.warning("Could not inhibit system sleep: %s", e)
            return False

        stop = threading.Event()
        started = threading.Event()
        result: list[bool] = []

        def hold() -> None:
            try:
                state = kernel32.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
                )
                result.append(bool(state))
                started.set()
                if not state:
                    return
                stop.wait()
                kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            except (AttributeError, OSError) as e:
                logger.debug("Error while inhibiting sleep: %s", e)
                started.set()

        thread = threading.Thread(
            target=hold, name="rayforge-sleep-inhibitor", daemon=True
        )
        thread.start()
        started.wait(timeout=5)
        if not result or not result[0]:
            stop.set()
            thread.join(timeout=2)
            return False
        self._thread = thread
        self._stop = stop
        return True

    def release(self) -> None:
        thread, self._thread = self._thread, None
        stop, self._stop = self._stop, None
        if thread is None or stop is None:
            return
        stop.set()
        thread.join(timeout=2)


def _create_backend():
    """Creates the platform-specific inhibitor backend."""
    if sys.platform == "win32":
        return _WindowsBackend()
    if sys.platform == "darwin":
        return _ProcessBackend(
            ["caffeinate", "-i", "-w", str(os.getpid())],
            release_by_stdin=False,
        )
    return _ProcessBackend(
        [
            "systemd-inhibit",
            "--what=sleep",
            "--mode=block",
            f"--why={_REASON}",
            "cat",
        ],
        release_by_stdin=True,
    )


class SleepInhibitor:
    """Reference counted OS idle inhibitor."""

    def __init__(self):
        self._lock = threading.Lock()
        self._users = 0
        self._backend = None

    @property
    def active(self) -> bool:
        """Whether an inhibitor is currently held."""
        with self._lock:
            return self._backend is not None

    def acquire(self) -> None:
        """Requests that automatic system sleep be blocked.

        The first caller takes the inhibitor; later callers only bump
        the reference count. Failure to acquire is logged by the
        backend and is not fatal: the job still runs without the
        inhibitor.
        """
        with self._lock:
            self._users += 1
            if self._users > 1:
                return
            backend = _create_backend()
            if backend.acquire():
                self._backend = backend
                logger.info("Inhibited automatic system sleep for job")
            else:
                self._backend = None

    def release(self) -> None:
        """Releases one reference, dropping the inhibitor at zero."""
        with self._lock:
            if self._users == 0:
                return
            self._users -= 1
            if self._users > 0:
                return
            backend, self._backend = self._backend, None
            if backend is not None:
                backend.release()
                logger.info("Released automatic system sleep inhibition")

    def release_all(self) -> None:
        """Releases every reference, e.g. during interpreter shutdown."""
        with self._lock:
            self._users = 0
            backend, self._backend = self._backend, None
            if backend is not None:
                backend.release()


sleep_inhibitor = SleepInhibitor()

atexit.register(sleep_inhibitor.release_all)
