from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

from .handle import BaseArtifactHandle


class ArtifactLifecycle(Enum):
    """Lifecycle states for artifacts in the ledger."""

    INITIAL = "initial"
    PROCESSING = "processing"
    DONE = "done"
    STALE = "stale"
    ERROR = "error"


@dataclass
class LedgerEntry:
    """Entry in the artifact ledger tracking state and metadata."""

    state: ArtifactLifecycle
    handle: Optional[BaseArtifactHandle] = None
    generation_id: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
