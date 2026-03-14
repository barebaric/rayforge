"""
Pytest configuration for post_processors builtin addon tests.
"""

import pytest


@pytest.fixture
def mock_progress_context():
    """
    Provides a mock ProgressContext for testing compute functions.
    """

    class _SimpleMockProgressContext:
        def __init__(self):
            self.progress_calls: list[tuple[float, str]] = []
            self.message_calls: list[str] = []
            self._is_cancelled = False
            self._total = 1.0
            self._sub_contexts: list["_SimpleMockProgressContext"] = []

        def is_cancelled(self) -> bool:
            return self._is_cancelled

        def set_progress(self, progress: float) -> None:
            normalized = (
                progress / self._total if self._total > 0 else progress
            )
            self.progress_calls.append((normalized, ""))

        def set_message(self, message: str) -> None:
            self.message_calls.append(message)

        def set_total(self, total: float) -> None:
            if total <= 0:
                self._total = 1.0
            else:
                self._total = float(total)

        def sub_context(
            self,
            base_progress: float,
            progress_range: float,
            total: float = 1.0,
        ) -> "_SimpleMockProgressContext":
            sub_ctx = _SimpleMockProgressContext()
            sub_ctx._total = total
            self._sub_contexts.append(sub_ctx)
            return sub_ctx

        def flush(self) -> None:
            pass

    return _SimpleMockProgressContext()


@pytest.fixture(scope="session", autouse=True)
def register_post_processors():
    """
    Automatically register post_processors transformers for all tests.
    """
    from rayforge import worker_init
    from rayforge.pipeline.transformer.registry import transformer_registry

    worker_init._worker_addons_loaded = True

    # Import and register transformers directly from the addon
    from post_processors.transformers import (
        Smooth,
        TabOpsTransformer,
        CropTransformer,
        Optimize,
        MultiPassTransformer,
        OverscanTransformer,
    )

    ADDON_NAME = "post_processors"
    transformer_registry.register(Smooth, addon_name=ADDON_NAME)
    transformer_registry.register(TabOpsTransformer, addon_name=ADDON_NAME)
    transformer_registry.register(CropTransformer, addon_name=ADDON_NAME)
    transformer_registry.register(Optimize, addon_name=ADDON_NAME)
    transformer_registry.register(MultiPassTransformer, addon_name=ADDON_NAME)
    transformer_registry.register(OverscanTransformer, addon_name=ADDON_NAME)

    yield
