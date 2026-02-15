import logging

from rayforge.shared.util.debug import get_caller_stack, safe_caller_stack


def inner_function():
    return get_caller_stack(depth=2)


def middle_function():
    return inner_function()


def outer_function():
    return middle_function()


class TestGetCallerStack:
    def test_basic_stack(self):
        stack = outer_function()
        assert stack is not None
        parts = stack.split(" <- ")
        assert len(parts) == 2

    def test_depth_parameter(self):
        def level_4():
            return get_caller_stack(depth=4)

        def level_3():
            return level_4()

        def level_2():
            return level_3()

        def level_1():
            return level_2()

        stack = level_1()
        parts = stack.split(" <- ")
        assert len(parts) == 4

    def test_format(self):
        stack = middle_function()
        assert ":" in stack
        assert " <- " in stack


class TestSafeCallerStack:
    def test_returns_none_when_not_debug(self):
        logging.getLogger().setLevel(logging.INFO)
        try:
            result = safe_caller_stack()
            assert result is None
        finally:
            logging.getLogger().setLevel(logging.NOTSET)

    def test_returns_stack_when_debug(self):
        logging.getLogger().setLevel(logging.DEBUG)
        try:
            result = safe_caller_stack()
            assert result is not None
            assert " <- " in result
        finally:
            logging.getLogger().setLevel(logging.NOTSET)

    def test_respects_depth(self):
        logging.getLogger().setLevel(logging.DEBUG)
        try:

            def inner():
                return safe_caller_stack(depth=3)

            def middle():
                return inner()

            def outer():
                return middle()

            result = outer()
            assert result is not None
            parts = result.split(" <- ")
            assert len(parts) == 3
        finally:
            logging.getLogger().setLevel(logging.NOTSET)
