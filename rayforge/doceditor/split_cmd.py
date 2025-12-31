import uuid
import logging
from typing import TYPE_CHECKING, List
from abc import ABC, abstractmethod
from ..core.geo import Geometry
from ..core.item import DocItem
from ..core.undo import ListItemCommand
from ..core.workpiece import WorkPiece

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class SplitStrategy(ABC):
    """
    Abstract base class for strategies that determine how to split a
    WorkPiece's geometry into multiple fragments.
    """

    @abstractmethod
    def calculate_fragments(self, workpiece: "WorkPiece") -> List[Geometry]:
        """
        Calculates the geometric fragments for the split operation.

        Args:
            workpiece: The WorkPiece to split.

        Returns:
            A list of Geometry objects. Each geometry should represent a
            fragment in the same normalized coordinate space (0-1 box, Y-up)
            as the original workpiece's boundaries.
        """
        pass


class ConnectivitySplitStrategy(SplitStrategy):
    """
    Splits a workpiece by separating disjoint vector components (islands).
    This is the standard "Split" behavior for vector shapes.
    """

    def calculate_fragments(self, workpiece: "WorkPiece") -> List[Geometry]:
        if not workpiece.boundaries or workpiece.boundaries.is_empty():
            return []
        return workpiece.boundaries.split_into_components()


class SplitCmd:
    """Handles splitting of document items."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def split_items(
        self,
        items: List[WorkPiece],
        strategy: SplitStrategy = ConnectivitySplitStrategy(),
    ) -> List[DocItem]:
        """
        Splits the provided items into multiple fragments based on the given
        strategy. Replaces the original items with the new fragments in the
        document.

        Args:
            items: The list of items to split.
            strategy: The strategy to use for calculating fragments.
                      Defaults to splitting disjoint components.

        Returns:
            A list of the newly created items.
        """
        if not items:
            return []

        history = self._editor.history_manager
        newly_created_items = []

        with history.transaction(_("Split item(s)")) as t:
            for item in items:
                # Capture the parent before any modification/removal occurs.
                # Executing remove_cmd may set item.parent to None.
                parent = item.parent
                if not isinstance(item, WorkPiece) or not parent:
                    continue

                fragments = strategy.calculate_fragments(item)
                new_pieces = item.apply_split(fragments)

                # If splitting didn't produce multiple pieces, do nothing for
                # this item.
                if len(new_pieces) <= 1:
                    continue

                # Remove the original
                remove_cmd = ListItemCommand(
                    owner_obj=parent,
                    item=item,
                    undo_command="add_child",
                    redo_command="remove_child",
                    name=_("Remove original item"),
                )
                t.execute(remove_cmd)

                # Add the new pieces
                for piece in new_pieces:
                    # Assign a unique ID to each new piece
                    piece.uid = str(uuid.uuid4())

                add_cmd = ListItemCommand(
                    owner_obj=parent,
                    item=new_pieces,
                    undo_command="remove_children",
                    redo_command="add_children",
                    name=_("Add split fragments"),
                )
                t.execute(add_cmd)
                newly_created_items.extend(new_pieces)

        return newly_created_items
