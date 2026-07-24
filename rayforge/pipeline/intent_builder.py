"""
Intent construction for the raygeo-backed pipeline.

The :class:`IntentBuilder` walks a :class:`~rayforge.core.doc.Doc` and
produces a flat list of :class:`~raygeo.pipeline.request.NodeRequest`
objects with **stable keys** and **deterministic version tokens**.

Stable keys
-----------
* ``workpiece:{wp_uid}:{step_uid}``  — one compute node per
  workpiece / step pair.
* ``step:{step_uid}``  — one aggregate node per step that concatenates
  the workpiece compute outputs and applies per-step transformers.
* ``job``  — one final aggregate node linking all step outputs with
  job-level markers and machine parameters.

Version tokens
--------------
raygeo's cache is keyed by node key only; the ``version_token`` is the
sole invalidation signal.  Tokens are SHA-1 digests of a canonical
representation of the inputs that affect a node's output:

* **Compute tokens** hash
  ``(geometry_revision, step.params, transformer_params)``.
  For step scopes declaring a position-sensitive transformer (see
  :meth:`Step.is_position_sensitive`),
  ``transform_revision`` of the workpiece is folded into the token;
  otherwise it is omitted so pure moves do not invalidate workpiece
  compute results.

* **Aggregate tokens** hash
  ``(upstream compute tokens, placement, markers,
  transformer_params + position_sensitive())``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Sequence,
    Tuple,
)

from raygeo.cnc.execution.specs import (
    AggregateGroup,
    AggregateInput,
    AggregateSpec,
    ComputePayload,
    MachineParams,
    Marker,
)
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.contour import ContourSpec
from raygeo.ops.part import Part
from raygeo.pipeline.request import NodeRequest
from raygeo.pipeline.stage import StageSpec

if TYPE_CHECKING:
    from ..core.doc import Doc
    from ..core.layer import Layer
    from ..core.step import Step
    from ..core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


# Stable key formats.  Centralised here so the producer and the DOM
# reattachment map (see IntentController) always agree.
WORKPIECE_KEY_FMT = "workpiece:{wp_uid}:{step_uid}"
STEP_KEY_FMT = "step:{step_uid}"
JOB_KEY = "job"


def workpiece_key(wp_uid: str, step_uid: str) -> str:
    return WORKPIECE_KEY_FMT.format(wp_uid=wp_uid, step_uid=step_uid)


def step_key(step_uid: str) -> str:
    return STEP_KEY_FMT.format(step_uid=step_uid)


def job_key() -> str:
    return JOB_KEY


class IntentBuilder:
    """
    Builds a flat :class:`NodeRequest` list from a :class:`Doc`.

    The builder is stateless: each call to :meth:`build` produces a
    fresh, self-contained list suitable for wrapping in a raygeo
    :class:`Intent`.
    """

    def __init__(self, generation_id: int = 0):
        self._generation_id = generation_id

    @property
    def generation_id(self) -> int:
        return self._generation_id

    def build(self, doc: "Doc") -> List[NodeRequest]:
        """
        Walk *doc* and produce one NodeRequest per workpiece-step pair,
        one per step, and one final job aggregate.
        """
        nodes: List[NodeRequest] = []
        # Map each step's key to the list of upstream workpiece compute
        # tokens — the step aggregate token depends on all of them.
        step_compute_tokens: Dict[str, List[Tuple[str, int]]] = {}

        for layer in doc.layers:
            if not layer.workflow or not layer.workflow.steps:
                continue
            workpieces = list(layer.all_workpieces)
            if not workpieces:
                continue
            for step in layer.workflow.steps:
                if not step.visible:
                    continue
                tokens = self._build_workpiece_nodes(step, workpieces, nodes)
                step_compute_tokens[step.uid] = tokens

        for layer in doc.layers:
            if not layer.workflow:
                continue
            for step in layer.workflow.steps:
                if not step.visible:
                    continue
                upstream = step_compute_tokens.get(step.uid, [])
                if not upstream:
                    continue
                self._build_step_node(step, layer, upstream, nodes)

        self._build_job_node(doc, nodes)
        return nodes

    # ------------------------------------------------------------------
    # Workpiece compute nodes
    # ------------------------------------------------------------------

    def _build_workpiece_nodes(
        self,
        step: "Step",
        workpieces: "Sequence[WorkPiece]",
        out: List[NodeRequest],
    ) -> List[Tuple[str, int]]:
        """
        Append one compute NodeRequest per workpiece for *step* and
        return the list of ``(node_key, version_token)`` pairs the
        step aggregate will hash over.
        """
        pos_sensitive = step.is_position_sensitive()
        tokens: List[Tuple[str, int]] = []
        for wp in workpieces:
            key = workpiece_key(wp.uid, step.uid)
            token = self._compute_token(step, wp, pos_sensitive)
            tokens.append((key, token))
            out.append(self._make_request(key, token, _wp_stage(wp)))
        return tokens

    def _build_step_node(
        self,
        step: "Step",
        layer: "Layer",
        upstream: List[Tuple[str, int]],
        out: List[NodeRequest],
    ) -> None:
        key = step_key(step.uid)
        token = self._aggregate_token(step, layer, upstream)
        stage = _step_stage(step, upstream)
        out.append(self._make_request(key, token, stage))

    def _build_job_node(self, doc: "Doc", out: List[NodeRequest]) -> None:
        key = job_key()
        token = self._job_token(doc)
        out.append(self._make_request(key, token, _job_stage(doc)))

    # ------------------------------------------------------------------
    # Token computation
    # ------------------------------------------------------------------

    def _compute_token(
        self, step: "Step", wp: "WorkPiece", pos_sensitive: bool
    ) -> int:
        payload = {
            "kind": "compute",
            "step_uid": step.uid,
            "wp_uid": wp.uid,
            "geo_rev": wp.geometry_revision,
            "step_params": _step_compute_params(step),
            "wpxf": _canonical(step.per_workpiece_transformers_dicts),
        }
        if pos_sensitive:
            payload["xf_rev"] = wp.transform_revision
        return _hash_int(payload)

    def _aggregate_token(
        self,
        step: "Step",
        layer: "Layer",
        upstream: List[Tuple[str, int]],
    ) -> int:
        payload = {
            "kind": "step_aggregate",
            "step_uid": step.uid,
            "upstream": [list(t) for t in upstream],
            "step_params": _step_compute_params(step),
            "spxf": _canonical(step.per_step_transformers_dicts),
            "position_sensitive": step.is_position_sensitive(),
        }
        return _hash_int(payload)

    def _job_token(self, doc: "Doc") -> int:
        payloads = []
        for layer in doc.layers:
            if not layer.workflow:
                continue
            for step in layer.workflow.steps:
                if not step.visible:
                    continue
                payloads.append(
                    {
                        "step_uid": step.uid,
                        "step_params": _step_compute_params(step),
                        "spxf": _canonical(step.per_step_transformers_dicts),
                    }
                )
        payload: Dict[str, Any] = {"kind": "job", "steps": payloads}
        return _hash_int(payload)

    # ------------------------------------------------------------------
    # Node construction
    # ------------------------------------------------------------------

    def _make_request(self, key: str, token: int, stage: Any) -> NodeRequest:
        return NodeRequest(
            key=key,
            generation_id=self._generation_id,
            stage=stage,
            version_token=token,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _step_compute_params(step: "Step") -> Dict[str, Any]:
    """
    Return a JSON-serialisable dict of step attributes that influence
    compute output.  UIDs and cosmetic fields are intentionally omitted
    so the token only changes when the actual compute inputs change.
    """
    return {
        "type": type(step).__name__,
        "visible": step.visible,
        "power": step.power,
        "max_power": step.max_power,
        "cut_speed": step.cut_speed,
        "max_cut_speed": step.max_cut_speed,
        "travel_speed": step.travel_speed,
        "max_travel_speed": step.max_travel_speed,
        "air_assist": step.air_assist,
        "kerf_mm": step.kerf_mm,
        "tab_power": step.tab_power,
        "frequency": step.frequency,
        "pulse_width": step.pulse_width,
        "pixels_per_mm": list(step.pixels_per_mm),
        "opsproducer": step.opsproducer_dict,
    }


def _canonical(obj: Any) -> Any:
    """
    Return *obj* in a form suitable for JSON round-tripping so that
    structurally-equal inputs produce identical serialisations.
    """
    try:
        return json.loads(json.dumps(obj, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(obj)


def _hash_int(payload: Mapping[str, Any]) -> int:
    """
    Produce a 63-bit positive integer hash of *payload*.

    Uses SHA-1 of a canonical JSON encoding so the value is stable
    across Python processes (unlike :func:`hash`, which is randomised
    per process for strings).
    """
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(blob).digest()
    # Take the first 8 bytes, mask the sign bit.
    value = int.from_bytes(digest[:8], "big")
    return value & 0x7FFFFFFFFFFFFFFF


# ----------------------------------------------------------------------
# Stage construction
# ----------------------------------------------------------------------
# Build minimal but valid raygeo :class:`StageSpec` instances.  For
# B0 nothing actually dispatches; the assembler / aggregate contents
# do not need to be meaningful, only valid enough for raygeo's
# converter (``convert_node_request`` validates StageSpec instances at
# :func:`create_intent_from_nodes` time).  Per-step-type assemblers
# (Contour, Raster, ...) replace this wiring during the contour and
# raster cutover slices.


_IDENTITY_4X4: List[List[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _wp_stage(wp: "WorkPiece") -> StageSpec.Compute:
    """
    Build a compute :class:`StageSpec.Compute` for the workpiece node.

    Uses an empty :class:`Part` and a default :class:`ContourSpec`
    assembler as the smallest valid payload.  Concrete assemblers for
    each step kind replace this during the per-step cutover.
    """
    part = Part()
    payload = ComputePayload(assembler=Assembler(ContourSpec()))
    return StageSpec.Compute(part=part, params=payload)


def _step_stage(
    step: "Step",
    upstream: List[Tuple[str, int]],
) -> StageSpec.Aggregate:
    """
    Build an aggregate :class:`StageSpec.Aggregate` for the step node.

    One :class:`AggregateGroup` per upstream workpiece compute node,
    wrapped by that workpiece's start / end markers.  The placement
    matrix here is identity; concrete per-workpiece placements replace
    this during the cutover.
    """
    groups: List[AggregateGroup] = []
    for wp_key, _token in upstream:
        # Extract the workpiece uid from the key
        # ``workpiece:{wp_uid}:{step_uid}``.
        wp_uid = wp_key.split(":", 1)[1].rsplit(":", 1)[0]
        start = Marker.WorkpieceStart(uid=wp_uid, _tag=True)
        end = Marker.WorkpieceEnd(uid=wp_uid, _tag=True)
        inp = AggregateInput(
            source_key=wp_key,
            placement_matrix=_IDENTITY_4X4,
            uid=wp_uid,
        )
        groups.append(
            AggregateGroup(
                start_markers=[start],
                inputs=[inp],
                end_markers=[end],
            )
        )
    spec = AggregateSpec(
        wrap_start=[],
        groups=groups,
        wrap_end=[],
        machine=MachineParams(),
    )
    return StageSpec.Aggregate(spec=spec)


def _job_stage(doc: "Doc") -> StageSpec.Aggregate:
    """
    Build the final job aggregate :class:`StageSpec.Aggregate`.

    Wraps all visible step aggregates with ``JobStart`` / ``JobEnd``
    markers.  Encoder output (G-code / vertex arrays) is a separate
    compute stage appended after the job aggregate during the
    encoding cutover.
    """
    groups: List[AggregateGroup] = []
    for layer in doc.layers:
        if not layer.workflow:
            continue
        for step in layer.workflow.steps:
            if not step.visible:
                continue
            sk = step_key(step.uid)
            inp = AggregateInput(
                source_key=sk,
                placement_matrix=_IDENTITY_4X4,
                uid=step.uid,
            )
            groups.append(
                AggregateGroup(
                    start_markers=[],
                    inputs=[inp],
                    end_markers=[],
                )
            )
    spec = AggregateSpec(
        wrap_start=[Marker.JobStart(_tag=True)],
        groups=groups,
        wrap_end=[Marker.JobEnd(_tag=True)],
        machine=MachineParams(),
    )
    return StageSpec.Aggregate(spec=spec)
