from rayforge.shared.util.cache import lru_cache_unless_forced


class TestLruCacheUnlessForced:
    def test_caches_result(self):
        call_count = 0

        @lru_cache_unless_forced()
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_func(5)
        result2 = expensive_func(5)
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1

    def test_force_bypasses_cache(self):
        call_count = 0

        @lru_cache_unless_forced()
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive_func(5)
        expensive_func(5, force=True)
        assert call_count == 2

    def test_different_args_different_cache_entries(self):
        call_count = 0

        @lru_cache_unless_forced()
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive_func(5)
        expensive_func(10)
        assert call_count == 2

    def test_respects_maxsize(self):
        call_count = 0

        @lru_cache_unless_forced(maxsize=2)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive_func(1)
        expensive_func(2)
        expensive_func(3)
        expensive_func(1)
        assert call_count == 4

    def test_returns_cached_value(self):
        @lru_cache_unless_forced()
        def get_value(x):
            return {"value": x}

        result1 = get_value(5)
        result2 = get_value(5)
        assert result1 is result2

    def test_force_returns_new_value(self):
        @lru_cache_unless_forced()
        def get_value(x):
            return {"value": x}

        result1 = get_value(5)
        result2 = get_value(5, force=True)
        assert result1 is not result2

    def test_with_keyword_args(self):
        call_count = 0

        @lru_cache_unless_forced()
        def func(a, b=10):
            nonlocal call_count
            call_count += 1
            return a + b

        result1 = func(5, b=10)
        result2 = func(5, b=10)
        assert result1 == 15
        assert result2 == 15
        assert call_count == 1

    def test_force_false_uses_cache(self):
        call_count = 0

        @lru_cache_unless_forced()
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive_func(5)
        expensive_func(5, force=False)
        assert call_count == 1

    def test_preserves_function_name(self):
        @lru_cache_unless_forced()
        def my_function(x):
            return x

        assert my_function.__name__ == "my_function"
