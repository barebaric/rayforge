"""Golden-file regression tests for the ruidarpa encoder transcript.

The committed fixture ``golden/transcript.cglu`` locks the encoder's
GlueScript transcript byte-for-byte. The representative job covers
layer attribute blocks and per-op action lines across vector cuts and
moves, a linearized arc, a scan line, raw power pass-through, and the
air-assist path.

Regenerate the fixture with ``golden/regen_golden.py`` when the encoder
or upstream GlueScript legitimately changes the transcript; do not
weaken these tests to mask drift.
"""

import importlib.util
from pathlib import Path
from typing import Any

from ruidadriver.rd_gluescript import GlueScript

from rayforge.machine.driver.ruidarpa.rpa_encoder import RuidaRPAEncoder

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_FILE = GOLDEN_DIR / "transcript.cglu"


def _load_regen_builder():
    """Load build_representative_job from the regen script module."""
    module_path = GOLDEN_DIR / "regen_golden.py"
    spec = importlib.util.spec_from_file_location("regen_golden", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load regen script: {module_path}")
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_representative_job


build_representative_job = _load_regen_builder()


def _encode_representative_job():
    """Encode the representative job exactly as the fixture was built."""
    doc, ops, machine = build_representative_job()
    result = RuidaRPAEncoder().encode(ops, machine, doc)
    return result.text


class TestGoldenFixture:
    """Byte-level lock on the committed transcript fixture."""

    def test_staged_output_matches_fixture(self):
        """Encoder output must equal the fixture, byte for byte."""
        expected = GOLDEN_FILE.read_text(encoding="utf-8")
        assert _encode_representative_job() + "\n" == expected

    def test_staged_output_is_deterministic(self):
        """Two encodes of the same job must produce identical bytes."""
        first = _encode_representative_job()
        second = _encode_representative_job()
        assert first == second


class TestGoldenStructure:
    """Structural invariants independent of exact fixture bytes."""

    def test_no_enable_block_cutting_leaks_into_output(self):
        """ENABLE_BLOCK_CUTTING must not leak into the transcript."""
        assert "ENABLE_BLOCK_CUTTING" not in _encode_representative_job()

    def test_single_job_framing(self):
        """The transcript is self-contained: one declare_job/end_job."""
        lines = _encode_representative_job().split("\n")
        assert sum(1 for line in lines if line.startswith("declare_job(")) == 1
        assert (
            sum(1 for line in lines if line.startswith("declare_layer(")) == 3
        )
        assert lines.count("end_job()") == 1

    def test_ends_with_end_job(self):
        """The transcript terminates with the end_job() line."""
        lines = _encode_representative_job().split("\n")
        assert lines[-2] == "end_job()"

    def test_layer_attribute_blocks(self):
        """Layer attrs stage with workflow settings, raw power, defaults."""
        text = _encode_representative_job()
        assert (
            "declare_layer('Cut', '#ff6600', 'VECTOR', 'NONE', "
            "5.0, 20.0, 50.0, 50.0)" in text
        )
        assert (
            "declare_layer('Engrave', '#33cc33', 'IMAGE', 'X_BI', "
            "2.5, 30.0, 50.0, 50.0)" in text
        )
        assert (
            "declare_layer('Default', '#00ccff', 'VECTOR', 'NONE', "
            "100.0, 20.0, 20.0, 20.0)" in text
        )

    def test_layer_action_blocks(self):
        """Per-op settings emit their transcript lines."""
        text = _encode_representative_job()
        assert "power(5.0)" in text
        assert "power(0.0)" not in text
        assert "cut_speed(4.166666666666667)" in text
        assert "frequency(25.0)" in text
        assert "pwm(50.0)" in text
        assert "air_assist_on()" in text
        assert "air_assist_off()" in text

    def test_moves_cuts_arc_and_scan_present(self):
        """Near/far moves and cuts, an arc, and a scan line all stage."""
        text = _encode_representative_job()
        assert "move_xy_to(0.0, 0.0)" in text
        assert "move_xy_to(20.0, 20.0)" in text
        assert "cut_xy_to(5.0, 5.0)" in text
        # The engrave layer is X_BI overscan, so its horizontal fill
        # emits the single-axis X form.
        assert "cut_x_to(50.0)" in text
        assert text.count("cut_xy_to(") > 5
        # Per-pixel scan power carries the 8% image power bias, so the
        # 128/255 and 255/255 pixels emit 58.196...% and 100.0% (clamped).
        assert "power(58.19607843137254)" in text
        assert "power(100.0)" in text


class TestTranscriptStaging:
    """The transcript compiles to controller-facing rpascript."""

    def test_transcript_stages_to_rpascript(self):
        """stage_gluescript on the transcript yields near/far forms,
        LAST_LAYER, and END_JOB/EOF framing."""
        transcript = _encode_representative_job().split("\n")
        gs = GlueScript()
        gs.stage_gluescript(transcript)
        rpascript = gs.rpascript

        assert any(line.startswith("MOVE_NEAR_XY") for line in rpascript)
        assert any(line.startswith("MOVE_FAR_XY") for line in rpascript)
        assert any(line.startswith("CUT_NEAR_XY") for line in rpascript)
        assert any(line.startswith("CUT_FAR_XY") for line in rpascript)
        assert any(line.startswith("LAST_LAYER") for line in rpascript)
        assert rpascript[-2] == "END_JOB"
        assert rpascript[-1] == "EOF"
