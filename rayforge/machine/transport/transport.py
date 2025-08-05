from abc import ABC, abstractmethod
from enum import Enum, auto
from blinker import Signal


class TransportStatus(Enum):
    UNKNOWN = auto()
    IDLE = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()
    CLOSING = auto()
    DISCONNECTED = auto()
    SLEEPING = auto()


class Transport(ABC):
    """
    Abstract base class for asynchronous data transports.
    """

    def __init__(self):
        """
        Initialize transport with callbacks and notification handler.

        Signals:
            received: Function to handle received data
            status_changed: Function to handle connection status changes
        """
        self.received = Signal()
        self.status_changed = Signal()

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection and start data flow.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Gracefully terminate connection and cleanup resources.
        """
        pass

    @abstractmethod
    async def send(self, data: bytes) -> None:
        """
        Send binary data through the transport.

        Raises:
            ConnectionError: If transport is not connected
        """
        pass
