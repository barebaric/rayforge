from rayforge.shared.util.once import once_per_object


class TestOncePerObject:
    def test_single_object_called_once(self):
        call_count = 0

        @once_per_object
        def track_call(obj):
            nonlocal call_count
            call_count += 1

        class MyObject:
            pass

        obj = MyObject()
        track_call(obj)
        track_call(obj)
        track_call(obj)
        assert call_count == 1

    def test_different_objects_each_called(self):
        call_count = 0

        @once_per_object
        def track_call(obj):
            nonlocal call_count
            call_count += 1

        class MyObject:
            pass

        obj1 = MyObject()
        obj2 = MyObject()
        obj3 = MyObject()
        track_call(obj1)
        track_call(obj1)
        track_call(obj2)
        track_call(obj2)
        track_call(obj3)
        assert call_count == 3

    def test_returns_value_on_first_call(self):
        @once_per_object
        def get_value(obj):
            return "result"

        class MyObject:
            pass

        obj = MyObject()
        result = get_value(obj)
        assert result == "result"

    def test_returns_none_on_subsequent_calls(self):
        @once_per_object
        def get_value(obj):
            return "result"

        class MyObject:
            pass

        obj = MyObject()
        get_value(obj)
        result = get_value(obj)
        assert result is None

    def test_passes_args_and_kwargs(self):
        received_args = []

        @once_per_object
        def track_call(obj, arg1, arg2=None):
            received_args.append((arg1, arg2))

        class MyObject:
            pass

        obj = MyObject()
        track_call(obj, "value1", arg2="value2")
        assert received_args == [("value1", "value2")]

    def test_preserves_function_name(self):
        @once_per_object
        def my_function(obj):
            pass

        assert my_function.__name__ == "my_function"
