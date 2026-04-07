import time
import logging
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .elements.workpiece import WorkPieceElement

logger = logging.getLogger(__name__)

MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


class OpsCacheRegistry:
    """Global registry that caps total ops surface cache memory
    across all WorkPieceElements using LRU eviction."""

    def __init__(self, max_bytes: int = MAX_CACHE_BYTES):
        self._max_bytes = max_bytes
        self._total_bytes: int = 0
        self._wp_bytes: Dict[str, int] = {}
        self._wp_last_draw: Dict[str, float] = {}
        self._wp_elements: Dict[str, "WorkPieceElement"] = {}
        self._evicting: bool = False

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def register(self, element: "WorkPieceElement"):
        uid = element.data.uid
        self._wp_elements[uid] = element
        self._wp_bytes.setdefault(uid, 0)
        self._wp_last_draw.setdefault(uid, 0.0)

    def unregister(self, wp_uid: str):
        freed = self._wp_bytes.pop(wp_uid, 0)
        self._total_bytes = max(0, self._total_bytes - freed)
        self._wp_elements.pop(wp_uid, None)
        self._wp_last_draw.pop(wp_uid, None)

    def touch(self, wp_uid: str):
        if wp_uid in self._wp_last_draw:
            self._wp_last_draw[wp_uid] = time.monotonic()

    def add(self, wp_uid: str, step_uid: str, byte_size: int):
        self._total_bytes += byte_size
        self._wp_bytes[wp_uid] = self._wp_bytes.get(wp_uid, 0) + byte_size
        logger.debug(
            f"OpsCache: +{byte_size >> 20}MB "
            f"({wp_uid[:8]}/{step_uid[:8]}) "
            f"total={self._total_bytes >> 20}MB"
        )
        if not self._evicting:
            self._evict_if_needed(wp_uid)

    def remove(self, wp_uid: str, step_uid: str, byte_size: int):
        if byte_size <= 0:
            return
        self._total_bytes = max(0, self._total_bytes - byte_size)
        self._wp_bytes[wp_uid] = max(
            0, self._wp_bytes.get(wp_uid, 0) - byte_size
        )

    def _evict_if_needed(self, protect_uid: str):
        self._evicting = True
        try:
            while self._total_bytes > self._max_bytes:
                lru_uid = self._find_lru(protect_uid)
                if lru_uid is None:
                    logger.warning(
                        f"OpsCache: budget exceeded "
                        f"({self._total_bytes >> 20}MB/"
                        f"{self._max_bytes >> 20}MB), "
                        f"cannot evict"
                    )
                    break
                elem = self._wp_elements.get(lru_uid)
                if elem:
                    logger.info(
                        f"OpsCache: evicting {lru_uid[:8]}, "
                        f"freeing "
                        f"~{self._wp_bytes.get(lru_uid, 0) >> 20}MB"
                    )
                    elem.clear_all_ops_caches()
                else:
                    freed = self._wp_bytes.pop(lru_uid, 0)
                    self._total_bytes = max(0, self._total_bytes - freed)
                    self._wp_last_draw.pop(lru_uid, None)
        finally:
            self._evicting = False

    def _find_lru(self, protect_uid: str) -> Optional[str]:
        lru_uid = None
        lru_time = float("inf")
        for uid, t in self._wp_last_draw.items():
            if (
                uid != protect_uid
                and t < lru_time
                and self._wp_bytes.get(uid, 0) > 0
            ):
                lru_time = t
                lru_uid = uid
        return lru_uid


registry = OpsCacheRegistry()
