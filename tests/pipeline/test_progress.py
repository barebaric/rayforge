"""Tests for the progress context abstraction."""

from rayforge.shared.tasker.progress import (
    CallbackProgressContext,
    NoOpProgressContext,
)
from conftest import MockProgressContext


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

    def test_sub_context_creates_new_instance(self):
        ctx = NoOpProgressContext()
        sub = ctx.sub_context(0.0, 0.5, 1.0)
        assert sub is not ctx

    def test_flush_does_nothing(self):
        ctx = NoOpProgressContext()
        ctx.flush()


class TestCallbackProgressContext:
    def test_is_cancelled_calls_function(self):
        cancelled = [False]

        def is_cancelled_func():
            return cancelled[0]

        ctx = CallbackProgressContext(
            is_cancelled_func,
            lambda x: None,
            lambda m: None,
        )
        assert ctx.is_cancelled() is False
        cancelled[0] = True
        assert ctx.is_cancelled() is True

    def test_set_progress_normalizes_and_calls_callback(self):
        progress_values = []

        def progress_callback(value):
            progress_values.append(value)

        ctx = CallbackProgressContext(
            lambda: False,
            progress_callback,
            lambda m: None,
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
            lambda: False,
            lambda x: None,
            message_callback,
        )
        ctx.set_message("test message")
        assert messages == ["test message"]

    def test_sub_context_creates_nested_context(self):
        progress_values = []

        def progress_callback(value):
            progress_values.append(value)

        ctx = CallbackProgressContext(
            lambda: False,
            progress_callback,
            lambda m: None,
        )
        sub = ctx.sub_context(0.0, 0.5, 1.0)
        sub.set_total(100)
        sub.set_progress(50)
        assert len(progress_values) == 1
        assert progress_values[0] == 0.25

    def test_flush_does_nothing(self):
        ctx = CallbackProgressContext(
            lambda: False,
            lambda x: None,
            lambda m: None,
        )
        ctx.flush()


class TestMockProgressContext:
    def test_is_cancelled_returns_initial_value(self):
        ctx = MockProgressContext(cancelled=True)
        ctx._inner._progress_context.set_wrapper(ctx._inner)
        assert ctx.is_cancelled() is True

    def test_is_cancelled_can_be_changed(self):
        ctx = MockProgressContext(cancelled=False)
        ctx._inner._progress_context.set_wrapper(ctx._inner)
        assert ctx.is_cancelled() is False
        ctx.set_cancelled(True)
        assert ctx.is_cancelled() is True

    def test_set_progress_captures_calls(self):
        ctx = MockProgressContext()
        ctx._inner._progress_context.set_wrapper(ctx._inner)
        ctx.set_progress(0.1)
        ctx.set_progress(0.5)
        assert ctx.progress_calls == [0.1, 0.5]

    def test_set_message_captures_calls(self):
        ctx = MockProgressContext()
        ctx._inner._progress_context.set_wrapper(ctx._inner)
        ctx.set_message("message 1")
        ctx.set_message("message 2")
        assert ctx.message_calls == ["message 1", "message 2"]

    def test_set_total_captures_calls(self):
        ctx = MockProgressContext()
        ctx._inner._progress_context.set_wrapper(ctx._inner)
        ctx.set_total(100)
        ctx.set_total(200)
        assert ctx.total_calls == [100, 200]

    def test_sub_context_creates_nested_context(self):
        ctx = MockProgressContext()
        ctx._inner._progress_context.set_wrapper(ctx._inner)
        sub = ctx.sub_context(0.0, 0.5, 1.0)
        sub._inner._progress_context.set_wrapper(sub._inner)
        assert sub is not ctx
        assert len(ctx.sub_contexts) == 1
        assert sub._inner._progress_context._base == 0.0
        assert sub._inner._progress_context._range == 0.5

    def test_flush_counts_calls(self):
        ctx = MockProgressContext()
        ctx._inner._progress_context.set_wrapper(ctx._inner)
        assert ctx.flush_calls == 0
        ctx.flush()
        ctx.flush()
        assert ctx.flush_calls == 2
