"""Tests for the progress context abstraction."""

from typing import List

from rayforge.pipeline.progress import (
    CallbackProgressContext,
    NoOpProgressContext,
    ProgressContext,
)


class MockProgressContext(ProgressContext):
    """Mock ProgressContext for unit testing.

    Captures all progress/cancellation calls for assertions.
    """

    def __init__(self, cancelled: bool = False):
        self._cancelled = cancelled
        self._base = 0.0
        self._range = 1.0
        self._total = 1.0
        self.progress_calls: List[float] = []
        self.message_calls: List[str] = []
        self.total_calls: List[float] = []
        self.flush_calls: int = 0
        self.sub_contexts: List["MockProgressContext"] = []

    def is_cancelled(self) -> bool:
        return self._cancelled

    def set_progress(self, progress: float) -> None:
        self.progress_calls.append(progress)

    def set_message(self, message: str) -> None:
        self.message_calls.append(message)

    def set_total(self, total: float) -> None:
        self.total_calls.append(total)

    def sub_context(
        self,
        base_progress: float,
        progress_range: float,
        total: float = 1.0,
    ) -> "MockProgressContext":
        sub_ctx = MockProgressContext(cancelled=self._cancelled)
        sub_ctx._base = base_progress
        sub_ctx._range = progress_range
        sub_ctx._total = total
        self.sub_contexts.append(sub_ctx)
        return sub_ctx

    def flush(self) -> None:
        self.flush_calls += 1

    def set_cancelled(self, cancelled: bool) -> None:
        """Helper to change cancellation state during tests."""
        self._cancelled = cancelled


class TestNoOpProgressContext:
    def test_is_cancelled_returns_false(self):
        ctx = NoOpProgressContext()
        assert ctx.is_cancelled() is False

    def test_set_progress_does_nothing(self):
        ctx = NoOpProgressContext()
        ctx.set_progress(0.5)

    def test_set_message_does_nothing(self):
        ctx = NoOpProgressContext()
        ctx.set_message("test message")

    def test_set_total_does_nothing(self):
        ctx = NoOpProgressContext()
        ctx.set_total(100)

    def test_sub_context_returns_self(self):
        ctx = NoOpProgressContext()
        sub = ctx.sub_context(0.0, 0.5)
        assert sub is ctx

    def test_flush_does_nothing(self):
        ctx = NoOpProgressContext()
        ctx.flush()


class TestCallbackProgressContext:
    def test_is_cancelled_calls_function(self):
        cancelled = [False]

        def is_cancelled_func():
            return cancelled[0]

        ctx = CallbackProgressContext(is_cancelled_func)
        assert ctx.is_cancelled() is False
        cancelled[0] = True
        assert ctx.is_cancelled() is True

    def test_set_progress_normalizes_and_calls_callback(self):
        progress_values = []

        def progress_callback(value):
            progress_values.append(value)

        ctx = CallbackProgressContext(
            is_cancelled_func=lambda: False,
            progress_callback=progress_callback,
        )
        ctx.set_total(200)
        ctx.set_progress(20)
        assert len(progress_values) == 1
        assert progress_values[0] == 0.1

    def test_set_message_calls_callback(self):
        messages = []

        def message_callback(msg):
            messages.append(msg)

        ctx = CallbackProgressContext(
            is_cancelled_func=lambda: False,
            message_callback=message_callback,
        )
        ctx.set_message("test message")
        assert messages == ["test message"]

    def test_sub_context_creates_nested_context(self):
        progress_values = []

        def progress_callback(value):
            progress_values.append(value)

        ctx = CallbackProgressContext(
            is_cancelled_func=lambda: False,
            progress_callback=progress_callback,
        )
        sub = ctx.sub_context(0.0, 0.5)
        sub.set_total(100)
        sub.set_progress(50)
        assert len(progress_values) == 1
        assert progress_values[0] == 0.25

    def test_flush_does_nothing(self):
        ctx = CallbackProgressContext(is_cancelled_func=lambda: False)
        ctx.flush()


class TestMockProgressContext:
    def test_is_cancelled_returns_initial_value(self):
        ctx = MockProgressContext(cancelled=True)
        assert ctx.is_cancelled() is True

    def test_is_cancelled_can_be_changed(self):
        ctx = MockProgressContext(cancelled=False)
        assert ctx.is_cancelled() is False
        ctx.set_cancelled(True)
        assert ctx.is_cancelled() is True

    def test_set_progress_captures_calls(self):
        ctx = MockProgressContext()
        ctx.set_progress(0.1)
        ctx.set_progress(0.5)
        assert ctx.progress_calls == [0.1, 0.5]

    def test_set_message_captures_calls(self):
        ctx = MockProgressContext()
        ctx.set_message("message 1")
        ctx.set_message("message 2")
        assert ctx.message_calls == ["message 1", "message 2"]

    def test_set_total_captures_calls(self):
        ctx = MockProgressContext()
        ctx.set_total(100)
        ctx.set_total(200)
        assert ctx.total_calls == [100, 200]

    def test_sub_context_creates_nested_context(self):
        ctx = MockProgressContext()
        sub = ctx.sub_context(0.0, 0.5)
        assert sub is not ctx
        assert sub in ctx.sub_contexts
        assert sub._base == 0.0
        assert sub._range == 0.5

    def test_flush_counts_calls(self):
        ctx = MockProgressContext()
        assert ctx.flush_calls == 0
        ctx.flush()
        ctx.flush()
        assert ctx.flush_calls == 2
