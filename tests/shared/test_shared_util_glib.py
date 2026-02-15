from rayforge.shared.util.glib import falsify, DebounceMixin


class TestFalsify:
    def test_calls_function_with_args(self):
        result = []

        def callback(a, b):
            result.append((a, b))

        falsify(callback, 1, 2)

        assert result == [(1, 2)]

    def test_calls_function_with_kwargs(self):
        result = []

        def callback(a, b=None):
            result.append((a, b))

        falsify(callback, 1, b=2)

        assert result == [(1, 2)]

    def test_returns_false(self):
        def callback():
            pass

        result = falsify(callback)

        assert result is False

    def test_returns_false_regardless_of_callback_return(self):
        def callback():
            return "something"

        result = falsify(callback)

        assert result is False


class TestDebounceMixin:
    def test_init_sets_defaults(self):
        class MyClass(DebounceMixin):
            pass

        obj = MyClass()

        assert obj._debounce_timer == 0
        assert obj._debounced_callback is None
        assert obj._debounced_args == ()

    def test_can_be_combined_with_other_init(self):
        class MyClass(DebounceMixin):
            def __init__(self, value):
                super().__init__()
                self.value = value

        obj = MyClass(42)

        assert obj.value == 42
        assert obj._debounce_timer == 0
