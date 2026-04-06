import sys

from .transport import Transport, TransportStatus
from .http import HttpTransport
from .serial import SerialTransport
from .telnet import TelnetTransport
from .udp import UdpTransport
from .udp_server import UdpServerTransport
from .websocket import WebSocketTransport

if sys.platform != "win32":
    from .serial_server import SerialServerTransport
else:
    SerialServerTransport = None


__all__ = [
    "Transport",
    "TransportStatus",
    "HttpTransport",
    "SerialTransport",
    "SerialServerTransport",
    "TelnetTransport",
    "UdpTransport",
    "UdpServerTransport",
    "WebSocketTransport",
]
