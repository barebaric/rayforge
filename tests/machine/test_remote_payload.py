"""Tests for the remote job payload serialization."""

import numpy as np
import pytest
from raygeo.ops import Ops

from rayforge.machine.job_status import JobKind
from rayforge.machine.models.machine import Machine
from rayforge.machine.remote import JobPayload
from rayforge.pipeline.artifact.job import JobArtifact
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap


def _make_encoded() -> EncodedOutput:
    return EncodedOutput(
        text="G0 X10\nG1 X20\nG1 X30\n",
        op_map=MachineCodeOpMap.from_lists(
            [(0, 1), (1, 1), (2, 1)], [0, 1, 2]
        ),
        driver_data={
            "binary": b"\x01\x02\xff\x00",
            "array": np.arange(4, dtype=np.int32),
            "nested": {"list": [1, 2.5, "x", None, True]},
        },
    )


def _make_payload() -> JobPayload:
    ops = Ops()
    ops.move_to(10, 10, 0)
    ops.line_to(20, 10, 0)
    ops.line_to(20, 20, 0)
    return JobPayload(
        encoded=_make_encoded(),
        ops=ops,
        doc_name="test.rfz",
        kind=JobKind.FRAME,
        distance_mm=123.4,
        estimated_seconds=42.5,
        command_times=[0.1, 1.5, 3.25],
    )


@pytest.fixture
def machine(lite_context):
    m = Machine(lite_context)
    lite_context.machine_mgr.add_machine(m)
    return m


def test_roundtrip_preserves_all_fields():
    payload = _make_payload()
    restored = JobPayload.from_bytes(payload.to_bytes())

    assert restored.doc_name == "test.rfz"
    assert restored.kind == JobKind.FRAME
    assert restored.distance_mm == 123.4
    assert restored.estimated_seconds == 42.5
    assert restored.command_times == [0.1, 1.5, 3.25]
    assert restored.op_count == 3


def test_roundtrip_preserves_ops():
    payload = _make_payload()
    restored = JobPayload.from_bytes(payload.to_bytes())

    assert payload.ops is not None
    assert restored.ops is not None
    assert len(restored.ops) == len(payload.ops)
    assert restored.ops.distance() == payload.ops.distance()
    assert restored.ops.category(1) == payload.ops.category(1)


def test_roundtrip_preserves_encoded_output():
    payload = _make_payload()
    restored = JobPayload.from_bytes(payload.to_bytes())

    encoded = payload.encoded
    other = restored.encoded
    assert other.text == encoded.text
    assert other.op_map.op_count == encoded.op_map.op_count
    assert other.op_map.line_count == encoded.op_map.line_count
    for i in range(encoded.op_map.op_count):
        assert other.op_map.span_for_op(i) == (encoded.op_map.span_for_op(i))
    for line in range(encoded.op_map.line_count):
        assert other.op_map.op_for_line(line) == (
            encoded.op_map.op_for_line(line)
        )
    assert other.driver_data["binary"] == encoded.driver_data["binary"]
    np.testing.assert_array_equal(
        other.driver_data["array"], encoded.driver_data["array"]
    )
    assert other.driver_data["nested"] == encoded.driver_data["nested"]


def test_roundtrip_without_optional_data():
    encoded = EncodedOutput(text="", op_map=MachineCodeOpMap(), driver_data={})
    payload = JobPayload(encoded=encoded)
    restored = JobPayload.from_bytes(payload.to_bytes())

    assert restored.doc_name == ""
    assert restored.kind == JobKind.SEND
    assert restored.distance_mm == 0.0
    assert restored.estimated_seconds is None
    assert restored.command_times is None
    assert restored.ops is None
    assert restored.op_count == 0
    assert restored.encoded.text == ""


def test_rejects_bad_magic():
    data = bytearray(_make_payload().to_bytes())
    data[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="Not a RayForge job payload"):
        JobPayload.from_bytes(bytes(data))


def test_rejects_unsupported_version():
    data = bytearray(_make_payload().to_bytes())
    data[4:8] = (999).to_bytes(4, "big")
    with pytest.raises(ValueError, match="Unsupported job payload"):
        JobPayload.from_bytes(bytes(data))


def test_rejects_truncated_data():
    payload = _make_payload()
    with pytest.raises(ValueError, match="truncated"):
        JobPayload.from_bytes(payload.to_bytes()[:8])
    with pytest.raises(ValueError, match="truncated"):
        JobPayload.from_bytes(payload.to_bytes()[:-2])


def test_rejects_unserializable_driver_data():
    encoded = EncodedOutput(
        text="G0 X0\n",
        op_map=MachineCodeOpMap.from_lists([(0, 1)], [0]),
        driver_data={"bad": object()},
    )
    with pytest.raises(TypeError, match="Cannot serialize"):
        JobPayload(encoded=encoded).to_bytes()


def test_from_job_artifact_with_machine(machine):
    ops = Ops()
    ops.move_to(10, 10, 0)
    ops.line_to(20, 10, 0)
    artifact = JobArtifact(
        ops=ops,
        distance=10.0,
        generation_id=1,
        time_estimate=5.0,
        encoded_output=_make_encoded(),
    )

    payload = JobPayload.from_job_artifact(
        artifact, doc_name="doc.rfz", machine=machine
    )

    assert payload.doc_name == "doc.rfz"
    assert payload.distance_mm == 10.0
    assert payload.estimated_seconds == 5.0
    assert payload.ops is not None
    assert payload.command_times is not None
    assert len(payload.command_times) == len(ops)

    restored = JobPayload.from_bytes(payload.to_bytes())
    assert restored.command_times == payload.command_times
    assert restored.ops is not None
    assert len(restored.ops) == len(ops)


def test_from_job_artifact_without_machine():
    ops = Ops()
    ops.move_to(10, 10, 0)
    artifact = JobArtifact(
        ops=ops,
        distance=0.0,
        generation_id=1,
        encoded_output=_make_encoded(),
    )

    payload = JobPayload.from_job_artifact(artifact)

    assert payload.command_times is None
    assert payload.estimated_seconds is None
    assert payload.ops is not None


def test_from_job_artifact_requires_encoded_output():
    ops = Ops()
    artifact = JobArtifact(ops=ops, distance=0.0, generation_id=1)
    with pytest.raises(ValueError, match="no encoded output"):
        JobPayload.from_job_artifact(artifact)
