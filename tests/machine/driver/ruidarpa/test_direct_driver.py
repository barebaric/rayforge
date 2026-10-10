"""
Unit tests for the RpaDirectDriver live-command wrappers.

The wrappers delegate to the wrapped RdDriver and discard the returned
lines, which the driver auto-sends when connected. Calling ``run()`` on
those lines would double-send every live command, so the tests pin
``run`` as never-called and the wrapper return value as None.
"""

from unittest.mock import Mock

import pytest
from rpalib.version import __version__
from ruidadriver.ruida_driver import RdDriver

from rayforge.machine.driver.ruidarpa.rpa_direct_driver import RpaDirectDriver

_LIVE_METHODS = [
    ("home", ()),
    ("home_z", ()),
    ("focus_z", ()),
    ("jog_xy_to", (10.0, 20.0)),
    ("jog_xy_rel", (5.0, 5.0)),
    ("jog_x_rel", (5.0,)),
    ("jog_y_rel", (5.0,)),
    ("jog_z_rel", (5.0,)),
    ("jog_u_rel", (5.0,)),
    ("pause", ()),
    ("resume", ()),
    ("stop_job", ()),
    ("reset", ()),
]

_SPEED_METHODS = [
    ("jog_set_xy_speed", (100.0,)),
    ("jog_set_z_speed", (100.0,)),
    ("jog_set_u_speed", (100.0,)),
]


def _direct_driver(connected: bool) -> tuple[RpaDirectDriver, Mock]:
    """Return an RpaDirectDriver wrapping a mock RdDriver."""
    driver = RpaDirectDriver()
    mock_driver = Mock(spec=RdDriver)
    mock_driver.is_connected = connected
    driver._driver = mock_driver
    return driver, mock_driver


class TestLiveWrapperDelegation:
    """Live-command wrappers delegate and never run the returned lines."""

    @pytest.mark.parametrize("method,args", _LIVE_METHODS)
    def test_delegates_and_never_runs_returned_lines(self, method, args):
        """The wrapper calls the RdDriver method and discards its lines."""
        driver, mock_driver = _direct_driver(connected=True)
        mock_method = getattr(mock_driver, method)
        mock_method.return_value = ["SPEED_LASER_1 100.0"]

        result = getattr(driver, method)(*args)

        mock_method.assert_called_once_with(*args)
        mock_driver.run.assert_not_called()
        assert result is None

    @pytest.mark.parametrize("method,args", _LIVE_METHODS)
    def test_raises_when_disconnected(self, method, args):
        """Jog, home, and job-control commands must fail loudly instead
        of silently no-oping.
        """
        driver, _mock_driver = _direct_driver(connected=False)

        with pytest.raises(RuntimeError, match="not connected"):
            getattr(driver, method)(*args)


class TestStartDelegation:
    """RpaDirectDriver.start forwards connection parameters to RdDriver."""

    def test_start_forwards_protocol(self):
        """A protocol argument must reach the wrapped RdDriver."""
        driver = RpaDirectDriver()
        mock_driver = Mock(spec=RdDriver)
        mock_driver.start.return_value = True
        driver._driver = mock_driver

        result = driver.start("192.168.1.10", None, None, "tcp")

        assert result is True
        mock_driver.start.assert_called_once_with(
            udp_host="192.168.1.10", usb_device=None, protocol="tcp"
        )

    def test_start_omits_protocol_when_none(self):
        """Without a protocol the wrapped driver keeps its previous one."""
        driver = RpaDirectDriver()
        mock_driver = Mock(spec=RdDriver)
        mock_driver.start.return_value = True
        driver._driver = mock_driver

        driver.start("192.168.1.10", None)

        mock_driver.start.assert_called_once_with(
            udp_host="192.168.1.10", usb_device=None
        )

    def test_start_forwards_magic_and_protocol_together(self):
        """magic and protocol are independent optional kwargs."""
        driver = RpaDirectDriver()
        mock_driver = Mock(spec=RdDriver)
        mock_driver.start.return_value = True
        driver._driver = mock_driver

        driver.start("192.168.1.10", None, 0x88, "udp")

        mock_driver.start.assert_called_once_with(
            udp_host="192.168.1.10",
            usb_device=None,
            magic=0x88,
            protocol="udp",
        )

    def test_get_version_returns_installed_ruida_pa(self):
        """get_version reflects the locally installed ruida-pa version."""
        driver = RpaDirectDriver()
        assert driver.get_version() == __version__


class TestRequireConnected:
    """_require_connected gates jog/home on a live session."""

    def test_raises_when_disconnected(self):
        """A disconnected driver raises rather than returning None lines."""
        driver, _mock_driver = _direct_driver(connected=False)

        with pytest.raises(RuntimeError, match="not connected"):
            driver._require_connected()

    def test_returns_driver_when_connected(self):
        """A connected driver is returned for direct delegation."""
        driver, mock_driver = _direct_driver(connected=True)

        assert driver._require_connected() is mock_driver


class TestSpeedSetterDelegation:
    """jog_set_*_speed setters delegate without requiring a connection."""

    @pytest.mark.parametrize("method,args", _SPEED_METHODS)
    def test_speed_setter_delegates_while_disconnected(self, method, args):
        """Speed is session-less state, so the setter works disconnected."""
        driver, mock_driver = _direct_driver(connected=False)

        getattr(driver, method)(*args)

        getattr(mock_driver, method).assert_called_once_with(*args)

    @pytest.mark.parametrize("method,args", _SPEED_METHODS)
    def test_speed_setter_never_runs_lines(self, method, args):
        """A speed setter must never touch the script runner."""
        driver, mock_driver = _direct_driver(connected=True)

        getattr(driver, method)(*args)

        mock_driver.run.assert_not_called()


class TestGluescriptProperty:
    """The ``gluescript`` property exposes the wrapped RdDriver."""

    def test_returns_wrapped_driver(self):
        """gluescript must hand back the wrapped RdDriver instance."""
        driver, mock_driver = _direct_driver(connected=True)
        assert driver.gluescript is mock_driver

    def test_creates_driver_on_first_use(self):
        """gluescript must create the RdDriver when none exists yet."""
        driver = RpaDirectDriver()
        assert driver._driver is None
        gs = driver.gluescript
        assert gs is not None
        assert driver._driver is gs


class TestRunJobDelegation:
    """run_job delegates to the connected wrapped driver."""

    def test_delegates_to_wrapped_driver(self):
        """run_job must call the wrapped driver's run_job."""
        driver, mock_driver = _direct_driver(connected=True)
        driver.run_job()
        mock_driver.run_job.assert_called_once_with(None, auto_checksum=False)

    def test_passes_job_and_auto_checksum(self):
        """run_job must forward job lines and auto_checksum."""
        driver, mock_driver = _direct_driver(connected=True)
        driver.run_job(["START_JOB"], auto_checksum=True)
        mock_driver.run_job.assert_called_once_with(
            ["START_JOB"], auto_checksum=True
        )

    def test_raises_when_disconnected(self):
        """run_job must fail loudly when not connected."""
        driver, _mock_driver = _direct_driver(connected=False)
        with pytest.raises(RuntimeError, match="not connected"):
            driver.run_job()
