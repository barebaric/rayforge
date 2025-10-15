import pytest
import struct
from pathlib import Path
from pytest_mock import MockerFixture  # Import the mocker fixture type

from rayforge.image.ruida.parser import RuidaParser, _unscramble


def _scramble(byte_val: int) -> int:
    """Encodes a single byte for a Ruida file, inverse of _unscramble."""
    b7 = (byte_val >> 7) & 1
    b0 = byte_val & 1
    result = (byte_val & 0x7E) | (b0 << 7) | b7
    result ^= 0x88
    return (result + 1) & 0xFF


def _encode_abs_coords(x_mm: float, y_mm: float) -> bytes:
    """Encodes an absolute XY coordinate pair into 10 bytes."""

    def _encode_val(val_mm: float) -> bytes:
        """
        Encodes a single coordinate value into 5 big-endian bytes,
        matching the parser's decoder.
        """
        val_um = int(val_mm * 1000)

        # Handle negative numbers using 35-bit two's complement, to match the
        # decoder's logic.
        if val_um < 0:
            val_um += 0x800000000  # 1 << 35

        # This loop creates the bytes in little-endian order (LSB first).
        byte_array = bytearray(5)
        for i in range(5):
            byte_array[i] = (val_um >> (i * 7)) & 0x7F

        # The decoder expects big-endian, so we reverse the byte array
        # in-place.
        byte_array.reverse()
        return bytes(byte_array)

    return _encode_val(x_mm) + _encode_val(y_mm)


def create_test_rd_file(filepath: Path) -> None:
    """
    Generates a simple binary .rd file for a 10mm square at 20mm/s, 50% pwr.
    """
    content = bytearray()
    content += b"\xc9\x04" + (b"\x00" + struct.pack("<f", 20.0))
    content += b"\xc6\x32" + (b"\x00" + struct.pack("<H", 500))
    content += b"\xca\x06" + (b"\x00\x00\x00\x00\x00")
    content += b"\x88" + _encode_abs_coords(0, 0)
    content += b"\xa8" + _encode_abs_coords(10, 0)
    content += b"\xa8" + _encode_abs_coords(10, 10)
    content += b"\xa8" + _encode_abs_coords(0, 10)
    content += b"\xa8" + _encode_abs_coords(0, 0)
    content += b"\xd7"
    scrambled_content = b"RDWORKV8.01" + bytes([_scramble(b) for b in content])
    filepath.write_bytes(scrambled_content)


@pytest.fixture
def simple_square_rd_file(tmp_path: Path) -> Path:
    """Pytest fixture to create and provide a temporary .rd file."""
    rd_file = tmp_path / "test.rd"
    create_test_rd_file(rd_file)
    return rd_file


def test_scramble_unscramble_bijection():
    """Verify that _unscramble is the perfect inverse of _scramble."""
    for byte_val in range(256):
        scrambled = _scramble(byte_val)
        unscrambled = _unscramble(scrambled)
        assert unscrambled == byte_val


def test_parser_on_simple_square(
    simple_square_rd_file: Path, mocker: MockerFixture
):
    """
    Tests the parser by mocking RuidaCommand to verify constructor calls.
    This test is now independent of the RuidaCommand class implementation.
    """
    # Mock the RuidaCommand class within the parser's namespace
    mock_cmd_cls = mocker.patch("rayforge.image.ruida.parser.RuidaCommand")

    data = simple_square_rd_file.read_bytes()
    parser = RuidaParser(data)
    job = parser.parse()

    # Verify layer settings (this part is unaffected)
    assert len(job.layers) == 1
    layer0 = job.layers[0]
    assert layer0.speed == pytest.approx(20.0)
    assert layer0.power == pytest.approx(50.0)

    # Verify that RuidaCommand was called with the correct arguments
    calls = mock_cmd_cls.call_args_list
    assert len(calls) == 6

    # Check arguments for each call
    # Call 1: Move_Abs(0, 0)
    assert calls[0].args[0] == "Move_Abs"
    assert calls[0].args[1] == pytest.approx([0, 0])
    assert calls[0].args[2] == 0

    # Call 2: Cut_Abs(10, 0)
    assert calls[1].args[0] == "Cut_Abs"
    assert calls[1].args[1] == pytest.approx([10, 0])
    assert calls[1].args[2] == 0

    # Call 3: Cut_Abs(10, 10)
    assert calls[2].args[0] == "Cut_Abs"
    assert calls[2].args[1] == pytest.approx([10, 10])

    # Call 4: Cut_Abs(0, 10)
    assert calls[3].args[0] == "Cut_Abs"
    assert calls[3].args[1] == pytest.approx([0, 10])

    # Call 5: Cut_Abs(0, 0)
    assert calls[4].args[0] == "Cut_Abs"
    assert calls[4].args[1] == pytest.approx([0, 0])

    # Call 6: End
    assert calls[5].args[0] == "End"


TEST_FILES_DIR = Path(__file__).parent

"""
TODO: I have no Ruida files with the right license to be allowed to
put it into this repo. But it passed for me :-)
@pytest.mark.parametrize("filename", ["one.rd", "two.rd", "three.rd"])
def test_real_world_files(filename: str):
    ""
    Tests the parser against sample real-world files. Skips if not found.
    ""
    filepath = TEST_FILES_DIR / filename
    if not filepath.exists():
        pytest.skip(f"Test file not found: {filepath}")

    data = filepath.read_bytes()
    parser = RuidaParser(data)
    job = parser.parse()

    assert job is not None
    # These assertions will still work with a mocked or real command object
    assert len(job.commands) > 0, f"{filename} produced no commands"
    assert len(job.layers) > 0, f"{filename} produced no layers"
"""
