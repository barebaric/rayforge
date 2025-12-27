from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from blinker import Signal
import uuid
import logging


logger = logging.getLogger(__name__)


@dataclass
class ResettableCounter:
    """
    A resettable counter for tracking maintenance intervals.

    Attributes:
        uid: Unique identifier for this counter.
        name: Display name for the counter (e.g., "Laser Tube", "Lubrication").
        value: Current counter value in hours.
        notify_at: Optional threshold value (hours) for notification.
        notification_sent: True if the notification has already been triggered.
    """

    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Counter"
    value: float = 0.0
    notify_at: Optional[float] = None
    notification_sent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize counter to dictionary."""
        return {
            "uid": self.uid,
            "name": self.name,
            "value": self.value,
            "notify_at": self.notify_at,
            "notification_sent": self.notification_sent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResettableCounter":
        """Deserialize counter from dictionary."""
        return cls(
            uid=data.get("uid", str(uuid.uuid4())),
            name=data.get("name", "Counter"),
            value=data.get("value", 0.0),
            notify_at=data.get("notify_at"),
            notification_sent=data.get("notification_sent", False),
        )

    def add_hours(self, hours: float) -> None:
        """Add hours to the counter value."""
        self.value += hours
        logger.debug(
            f"Counter '{self.name}' added {hours}h, now {self.value}h"
        )

    def reset(self) -> None:
        """Reset the counter value to 0."""
        self.value = 0.0
        self.notification_sent = False
        logger.debug(f"Counter '{self.name}' reset to {self.value}h")

    def is_due_for_notification(self) -> bool:
        """Check if counter has reached notification threshold."""
        if self.notify_at is None:
            return False
        return self.value >= self.notify_at


class MachineHours:
    """
    Tracks machine operating hours for maintenance purposes.

    This class maintains a cumulative total of machine hours and provides
    resettable counters for tracking specific maintenance intervals.
    """

    def __init__(self):
        self.total_hours: float = 0.0
        self.counters: Dict[str, ResettableCounter] = {}
        self.changed = Signal()

    def add_hours(self, hours: float) -> None:
        """
        Add hours to total and all counters.

        Args:
            hours: Hours to add (can be fractional).
        """
        if hours <= 0:
            return

        self.total_hours += hours
        for counter in self.counters.values():
            counter.add_hours(hours)

        logger.info(
            f"Added {hours}h to machine hours (total: {self.total_hours}h)"
        )
        self.changed.send(self)

    def add_counter(self, counter: ResettableCounter) -> None:
        """Add a new resettable counter."""
        if counter.uid in self.counters:
            logger.warning(f"Counter with uid {counter.uid} already exists")
            return
        self.counters[counter.uid] = counter
        self.changed.send(self)

    def update_counter(self, counter: ResettableCounter) -> None:
        """
        Notify that a counter has been modified.
        This signals listeners (like the UI) to refresh.
        """
        if counter.uid in self.counters:
            self.changed.send(self)
        else:
            logger.warning(
                f"Attempted to update counter {counter.uid} which does not "
                "exist."
            )

    def consume_due_notifications(self) -> list[ResettableCounter]:
        """
        Identifies counters that have reached their limit but haven't been
        notified yet. Marks them as notified and returns the list.
        """
        due = []
        state_changed = False

        for counter in self.counters.values():
            if (
                counter.is_due_for_notification()
                and not counter.notification_sent
            ):
                counter.notification_sent = True
                due.append(counter)
                state_changed = True

        if state_changed:
            # Emit changed signal so the updated 'notification_sent' flags
            # are persisted to disk.
            self.changed.send(self)

        return due

    def remove_counter(self, counter_uid: str) -> None:
        """Remove a counter by its UID."""
        if counter_uid in self.counters:
            del self.counters[counter_uid]
            self.changed.send(self)

    def get_counter(self, counter_uid: str) -> Optional[ResettableCounter]:
        """Get a counter by its UID."""
        return self.counters.get(counter_uid)

    def reset_counter(self, counter_uid: str) -> None:
        """Reset a specific counter."""
        counter = self.get_counter(counter_uid)
        if counter:
            counter.reset()
            self.changed.send(self)

    def reset_total_hours(self) -> None:
        """Reset the total accumulated machine hours to zero."""
        self.total_hours = 0.0
        logger.info("Reset total machine hours to 0")
        self.changed.send(self)

    def reset_all_counters(self) -> None:
        """Reset all counters."""
        for counter in self.counters.values():
            counter.reset()
        self.changed.send(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_hours": self.total_hours,
            "counters": {
                uid: counter.to_dict()
                for uid, counter in self.counters.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MachineHours":
        """Deserialize from dictionary."""
        machine_hours = cls()
        machine_hours.total_hours = data.get("total_hours", 0.0)

        counters_data = data.get("counters", {})
        for uid, counter_data in counters_data.items():
            counter = ResettableCounter.from_dict(counter_data)
            machine_hours.counters[uid] = counter

        return machine_hours

    def get_counters_due_for_notification(self) -> list[ResettableCounter]:
        """Get list of counters that are due for notification."""
        return [
            counter
            for counter in self.counters.values()
            if counter.is_due_for_notification()
        ]
