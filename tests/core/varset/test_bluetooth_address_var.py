"""Tests for the BluetoothAddressVar validator."""

import pytest

from rayforge.core.varset import BluetoothAddressVar, ValidationError


def test_accepts_colon_and_dash_addresses():
    var = BluetoothAddressVar(key="address", label="Address")
    var.value = "aa:bb:cc:dd:ee:ff"
    var.validate()
    var.value = "aa-bb-cc-dd-ee-ff"
    var.validate()


def test_rejects_empty_address():
    var = BluetoothAddressVar(key="address", label="Address")
    var.value = ""
    with pytest.raises(ValidationError, match="empty"):
        var.validate()


def test_rejects_malformed_address():
    var = BluetoothAddressVar(key="address", label="Address")
    var.value = "aa:bb:cc"
    with pytest.raises(ValidationError, match="Invalid Bluetooth address"):
        var.validate()
