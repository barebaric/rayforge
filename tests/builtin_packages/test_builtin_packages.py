from rayforge.builtin_packages import ensure_loaded


class TestBuiltinPackages:
    def test_ensure_loaded_can_be_called(self):
        """Test that ensure_loaded() can be called without error."""
        ensure_loaded()

    def test_ensure_loaded_is_idempotent(self):
        """Test that ensure_loaded() can be called multiple times."""
        ensure_loaded()
        ensure_loaded()
        ensure_loaded()
