from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .handle import BaseArtifactHandle


@dataclass
class LedgerEntry:
    """
    Entry in the artifact ledger for caching handles.

    The ledger is a pure cache - it only stores handles for artifacts.
    State tracking is now handled by the DAG scheduler via ArtifactNode.
    """

    handle: Optional[BaseArtifactHandle] = None
    generation_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
