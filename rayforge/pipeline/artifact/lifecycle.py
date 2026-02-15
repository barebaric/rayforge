from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING

from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from ..dag.node import NodeState


def _default_state() -> "NodeState":
    from ..dag.node import NodeState

    return NodeState.DIRTY


@dataclass
class LedgerEntry:
    """
    Entry in the artifact ledger for caching handles.

    The ledger stores handles for artifacts and tracks their state.
    State tracking is now the single source of truth in the ledger.
    """

    handle: Optional[BaseArtifactHandle] = None
    generation_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: "NodeState" = field(default_factory=_default_state)
