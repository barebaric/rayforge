"""
Tests for the Ruida server.
"""

from rayforge.machine.driver.ruida.ruida_server import RuidaServer
from rayforge.machine.driver.ruida.ruida_protocol import RuidaState
from rayforge.machine.driver.ruida.ruida_util import encode35


class TestMemoryWrite:
    """Test DA memory write commands."""

    def test_write_overrides_default(self):
        state = RuidaState()
        server = RuidaServer(state=state)
        default_name, default_val = state.mem_lookup(0x0026)
        write_val = encode35(default_val + 1000) + encode35(0)
        cmd = b"\xda\x01\x00\x26" + write_val
        server._process_single_command(cmd)
        _, new_val = state.mem_lookup(0x0026)
        assert new_val == default_val + 1000
