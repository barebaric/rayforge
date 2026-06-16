"""
Real USB connection for EzCad2/LMC Galvo controllers.

Uses pyusb/libusb. Communicates with VID 0x9588 / PID 0x9899 (JCZ/LMC)
or VID 0x1A86 / PID 0x5512 (CH341 clone boards).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import usb.core
import usb.util

from .galvo_consts import (
    USB_PID_CH341,
    USB_PID_JCZ,
    USB_VID_CH341,
    USB_VID_JCZ,
    USB_READ_ENDPOINT,
    USB_WRITE_ENDPOINT,
)

logger = logging.getLogger(__name__)


class GalvoUSBConnection:
    """
    USB connection for EzCad2/LMC galvo controller boards.
    """

    def __init__(self) -> None:
        self._devices: Dict[int, Any] = {}
        self._interfaces: Dict[int, Any] = {}
        self._timeout = 100

    def find_device(self, index: int = 0) -> Any:
        found = usb.core.find(
            idVendor=USB_VID_JCZ,
            idProduct=USB_PID_JCZ,
            find_all=True,
        )
        devices = list(found) if found else []
        if not devices:
            found = usb.core.find(
                idVendor=USB_VID_CH341,
                idProduct=USB_PID_CH341,
                find_all=True,
            )
            devices = list(found) if found else []
        if not devices:
            raise ConnectionRefusedError(
                "No EzCad2 controller found."
            )
        try:
            return devices[index]
        except IndexError:
            raise ConnectionRefusedError(
                "EzCad2 controller found but index out of range."
            )

    @property
    def is_connected(self) -> bool:
        return bool(self._devices)

    def open(self, index: int = 0) -> int:
        try:
            device = self.find_device(index)
            self._devices[index] = device
            device.set_configuration()
            interface = device.get_active_configuration()[(0, 0)]
            self._interfaces[index] = interface
            try:
                if device.is_kernel_driver_active(
                    interface.bInterfaceNumber
                ):
                    device.detach_kernel_driver(
                        interface.bInterfaceNumber
                    )
            except (NotImplementedError, usb.core.USBError):
                pass
            try:
                usb.util.claim_interface(device, interface)
            except usb.core.USBError:
                usb.util.release_interface(device, interface)
                usb.util.claim_interface(device, interface)
            return index
        except Exception as e:
            logger.error(f"Failed to open USB device: {e}")
            raise ConnectionRefusedError(str(e))

    def close(self, index: int = 0) -> None:
        device = self._devices.pop(index, None)
        if device is None:
            return
        interface = self._interfaces.pop(index, None)
        try:
            if interface is not None:
                usb.util.release_interface(device, interface)
            usb.util.dispose_resources(device)
            device.reset()
        except Exception:
            pass

    def write(
        self, index: int = 0, packet: Optional[bytes] = None
    ) -> None:
        if packet is None:
            return
        device = self._devices.get(index)
        if device is None:
            raise ConnectionError("Not connected")
        device.write(
            endpoint=USB_WRITE_ENDPOINT,
            data=packet,
            timeout=self._timeout,
        )

    def read(self, index: int = 0) -> bytes:
        device = self._devices.get(index)
        if device is None:
            raise ConnectionError("Not connected")
        return bytes(
            device.read(
                endpoint=USB_READ_ENDPOINT,
                size_or_buffer=8,
                timeout=self._timeout,
            )
        )
