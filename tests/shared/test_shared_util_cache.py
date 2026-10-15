from rayforge.shared.util.cache import LruCache, lru_cache_unless_forced


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


class TestLruCache:
    def test_get_returns_stored_values(self):
        cache = LruCache(max_entries=2)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.get("missing") is None
        assert "a" in cache

    def test_evicts_least_recently_used_entry(self):
        cache = LruCache(max_entries=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")
        cache.put("c", 3)
        assert set(cache.keys()) == {"a", "c"}

    def test_byte_budget_evicts_least_recently_used(self):
        cache = LruCache[str, int](
            max_entries=10, max_bytes=100, sizeof=lambda v: v
        )
        cache.put("a", 40)
        cache.put("b", 40)
        cache.get("a")
        cache.put("c", 40)
        assert set(cache.keys()) == {"a", "c"}
        assert cache.nbytes == 80

    def test_keeps_newest_entry_even_over_budget(self):
        cache = LruCache[str, int](
            max_entries=10, max_bytes=100, sizeof=lambda v: v
        )
        cache.put("a", 40)
        cache.put("big", 500)
        assert list(cache.keys()) == ["big"]
        assert cache.nbytes == 500

    def test_replacing_a_key_updates_the_size(self):
        cache = LruCache[str, int](
            max_entries=10, max_bytes=100, sizeof=lambda v: v
        )
        cache.put("a", 40)
        cache.put("a", 10)
        assert len(cache) == 1
        assert cache.nbytes == 10

    def test_clear_empties_the_cache(self):
        cache = LruCache[str, int](
            max_entries=10, max_bytes=100, sizeof=lambda v: v
        )
        cache.put("a", 40)
        cache.clear()
        assert len(cache) == 0
        assert cache.nbytes == 0
