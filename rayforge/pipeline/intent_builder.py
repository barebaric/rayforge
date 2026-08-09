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

import asyncio
import hashlib
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
from raygeo.cnc.execution.specs import (
    AggregateGroup,
    AggregateInput,
    AggregateSpec,
    EncodeSpec,
    LinkMode,
    MachineParams,
    MachineTransformSpec,
    Marker,
    RotaryMappingSpec,
)
from raygeo.geo import Geometry, Matrix
from raygeo.ops import Ops
from raygeo.ops.convert import (
    EncodeOutput,
    Encoder,
    GcodeSpec,
    PythonEncoder,
)
from raygeo.pipeline.request import NodeRequest
from raygeo.pipeline.stage import StageSpec

from ..machine.driver import get_driver_cls
from ..machine.driver.dummy import NoDeviceDriver
from ..machine.kinematic_math import KinematicMath
from ..machine.models.dialect import GRBL_DIALECT
from ..machine.models.rotary_module import RotaryMode, RotaryType
from .coordspace import MachineSpace
from .encoder.base import EncodedOutput
from .encoder.rust_helpers import build_encode_context, dialect_to_spec
from .transformer import OpsTransformer
from .transformer.registry import transformer_registry

if TYPE_CHECKING:
    from ..core.doc import Doc
    from ..core.layer import Layer
    from ..core.step import Step
    from ..core.workpiece import WorkPiece
    from ..machine.models.dialect import GcodeDialect
    from ..machine.models.machine import Machine

logger = logging.getLogger(__name__)


# Stable key formats.  Centralised here so the producer and the DOM
# reattachment map (see IntentController) always agree.
WORKPIECE_KEY_FMT = "workpiece:{wp_uid}:{step_uid}"
STEP_KEY_FMT = "step:{step_uid}"
JOB_KEY = "job"
JOB_ENCODE_KEY = "job:encode"
JOB_MACHINEXFORM_KEY = "job:machinexform"


def workpiece_key(wp_uid: str, step_uid: str) -> str:
    return WORKPIECE_KEY_FMT.format(wp_uid=wp_uid, step_uid=step_uid)


def parse_workpiece_key(key: str) -> tuple[str, str] | None:
    """Parse a ``workpiece:{wp_uid}:{step_uid}`` key.

    Returns ``(wp_uid, step_uid)`` or ``None`` if the key does not
    match the expected format.
    """
    if not key.startswith("workpiece:"):
        return None
    rest = key[len("workpiece:") :]
    idx = rest.find(":")
    if idx == -1 or rest.find(":", idx + 1) != -1:
        return None
    return (rest[:idx], rest[idx + 1 :])


def step_key(step_uid: str) -> str:
    return STEP_KEY_FMT.format(step_uid=step_uid)


def job_key() -> str:
    return JOB_KEY


def job_encode_key() -> str:
    return JOB_ENCODE_KEY


def job_machinexform_key() -> str:
    return JOB_MACHINEXFORM_KEY


class IntentBuilder:
    """
    Builds a flat :class:`NodeRequest` list from a :class:`Doc`.

    The builder is stateless: each call to :meth:`build` produces a
    fresh, self-contained list suitable for wrapping in a raygeo
    :class:`Intent`.
    """

    def __init__(
        self,
        machine: Machine,
        generation_id: int = 0,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self._machine = machine
        self._generation_id = generation_id
        self._loop = loop
        self._doc: Doc | None = None

    @property
    def generation_id(self) -> int:
        return self._generation_id

    def build(self, doc: Doc) -> list[NodeRequest]:
        """
        Walk *doc* and produce one NodeRequest per workpiece-step pair,
        one per step, and one final job aggregate.
        """
        self._doc = doc
        nodes: list[NodeRequest] = []
        # Map each step's key to the list of upstream workpiece compute
        # inputs — the step aggregate token and placement depend on all
        # of them.
        step_compute_inputs: dict[str, list[tuple[str, int, WorkPiece]]] = {}
        # Per-step aggregate version tokens, used by the job aggregate
        # token so a position change that invalidates one step's
        # aggregate also invalidates the job aggregate (and encode).
        step_tokens: dict[str, int] = {}

        for layer in doc.layers:
            if not layer.workflow or not layer.workflow.steps:
                continue
            workpieces = list(layer.all_workpieces)
            if not workpieces:
                continue
            for step in layer.workflow.steps:
                if not step.visible:
                    continue
                inputs = self._build_workpiece_nodes(step, workpieces, nodes)
                step_compute_inputs[step.uid] = inputs

        for layer in doc.layers:
            if not layer.workflow:
                continue
            for step in layer.workflow.steps:
                if not step.visible:
                    continue
                upstream = step_compute_inputs.get(step.uid, [])
                if not upstream:
                    continue
                self._build_step_node(step, layer, upstream, nodes)
                step_tokens[step.uid] = self._aggregate_token(
                    step, layer, upstream
                )

        if step_tokens:
            self._build_job_node(doc, nodes, step_tokens)
            self._build_machine_transform_node(doc, nodes, step_tokens)
            self._build_encoder_node(doc, nodes, step_tokens)
        return nodes

    # ------------------------------------------------------------------
    # Workpiece compute nodes
    # ------------------------------------------------------------------

    def _build_workpiece_nodes(
        self,
        step: Step,
        workpieces: Sequence[WorkPiece],
        out: list[NodeRequest],
    ) -> list[tuple[str, int, WorkPiece]]:
        """
        Append one compute NodeRequest per workpiece for *step* and
        return the list of ``(node_key, version_token, workpiece)``
        triples the step aggregate consumes.
        """
        pos_sensitive = step.is_position_sensitive()
        inputs: list[tuple[str, int, WorkPiece]] = []

        # Parallelise per-workpiece Part construction (rendering + image
        # preprocessing) when we have a reference to the TaskManager's
        # event loop and more than one workpiece.  The heavy work
        # (dithering, grayscale, auto-levels) delegates to
        # raygeo Rust code which releases the GIL, so threads yield
        # real parallelism.
        stages: list[StageSpec.Compute]
        loop = self._loop
        if loop is not None and len(workpieces) > 1:

            async def _gather():
                coros = [
                    loop.run_in_executor(
                        None,
                        self._wp_stage,
                        step,
                        wp,
                    )
                    for wp in workpieces
                ]
                return await asyncio.gather(*coros)

            future = asyncio.run_coroutine_threadsafe(_gather(), loop)
            stages = future.result()
        else:
            stages = [self._wp_stage(step, wp) for wp in workpieces]

        for wp, stage in zip(workpieces, stages):
            key = workpiece_key(wp.uid, step.uid)
            token = self._compute_token(step, wp, pos_sensitive)
            inputs.append((key, token, wp))
            out.append(self._make_request(key, token, stage))
        return inputs

    def _build_step_node(
        self,
        step: Step,
        layer: Layer,
        upstream: list[tuple[str, int, WorkPiece]],
        out: list[NodeRequest],
    ) -> None:
        key = step_key(step.uid)
        token = self._aggregate_token(step, layer, upstream)
        stage = self._step_stage(step, upstream)
        # The step aggregate output is only consumed by the job
        # aggregate during a single run, so it is not worth caching:
        # caching would retain a full extra copy of the step's command
        # buffer between builds.
        out.append(self._make_request(key, token, stage, cacheable=False))

    def _build_job_node(
        self,
        doc: Doc,
        out: list[NodeRequest],
        step_tokens: dict[str, int],
    ) -> None:
        key = job_key()
        token = self._job_token(doc, step_tokens)
        stage = self._job_stage(doc, step_tokens)
        out.append(self._make_request(key, token, stage))

    def _build_machine_transform_node(
        self,
        doc: Doc,
        out: list[NodeRequest],
        step_tokens: dict[str, int],
    ) -> None:
        """Append the machine-transform compute node between the job
        aggregate and the encoder.

        This node consumes the job aggregate's world-space Ops and
        produces machine-space Ops by applying curve linearization,
        rotary axis mapping, world→machine coordinate transforms,
        WCS offsets, Z-flip, and AXIS_REPLACEMENT downstream.
        The encoder then reads from this node instead of directly
        from the job aggregate.
        """
        if self._machine is None:
            return
        key = job_machinexform_key()
        token = self._machine_transform_token(doc, step_tokens)
        stage = self._build_machine_transform_stage(doc)
        # The machine-space ops are consumed solely by the encoder
        # during the same run, so they are not cached: caching would
        # retain a full extra copy of the job's command buffer between
        # builds.
        out.append(self._make_request(key, token, stage, cacheable=False))

    def _build_encoder_node(
        self,
        doc: Doc,
        out: list[NodeRequest],
        step_tokens: dict[str, int],
    ) -> None:
        """Append the encoder compute node that consumes the
        machine-transform node's machine-space Ops and produces
        the machine code (G-code / vertex / texture).

        The encoder runs through raygeo's ``EncoderCompute`` stage.
        For Grbl the native Rust ``GcodeSpec`` is used directly; for
        any other machine the driver-specific encoder is wrapped in a
        :class:`PythonEncoder` so it runs under the GIL on a rayon
        worker thread — off the GTK main thread.
        """
        if self._machine is None:
            return
        key = job_encode_key()
        token = self._encode_token(doc, step_tokens)
        stage = self._encode_stage(doc)
        out.append(self._make_request(key, token, stage))

    # ------------------------------------------------------------------
    # Token computation
    # ------------------------------------------------------------------

    def _stock_revision(self) -> int:
        """Hash of visible stock items' world transforms and asset UIDs.

        Ensures that moving, adding, or removing a stock item
        invalidates crop-dependent compute caches.
        """
        if self._doc is None:
            return 0
        payload = []
        for item in self._doc.stock_items:
            if not item.visible:
                continue
            payload.append(
                {
                    "uid": item.uid,
                    "matrix": item.matrix.to_list(),
                    "asset_uid": item.stock_asset_uid,
                }
            )
        return _hash_int({"kind": "stock", "items": payload})

    def _compute_token(
        self, step: Step, wp: WorkPiece, pos_sensitive: bool
    ) -> int:
        payload = {
            "kind": "compute",
            "step_uid": step.uid,
            "wp_uid": wp.uid,
            "geo_rev": wp.geometry_revision,
            "wp_size": list(wp.size) if wp.size else [0, 0],
            "step_params": step.get_cache_params(),
            "assembler_params": _canonical(self._assembler_params(step, wp)),
            "wpxf": _canonical(step.per_workpiece_transformers_dicts),
        }
        if pos_sensitive:
            payload["xf_rev"] = wp.transform_revision
            payload["stock_rev"] = self._stock_revision()
        if step.uses_global_state and step.layer and step.layer.workflow:
            chain = [
                s for s in step.layer.workflow.steps if s.uses_global_state
            ]
            if step in chain:
                idx = chain.index(step)
                if idx > 0:
                    prev = chain[idx - 1]
                    payload["predecessor_token"] = self._compute_token(
                        prev, wp, prev.is_position_sensitive()
                    )
        return _hash_int(payload)

    def _aggregate_token(
        self,
        step: Step,
        layer: Layer,
        upstream: list[tuple[str, int, WorkPiece]],
    ) -> int:
        # Fold the per-workpiece placement matrix and target dimensions
        # into the token. The aggregate applies the placement matrix
        # to the (possibly cached) workpiece compute output, so a move
        # that leaves the compute cache untouched must still invalidate
        # the aggregate — otherwise the cached step ops are displayed
        # at their previous world position.
        placements: list[Any] = []
        for _k, _t, wp in upstream:
            placements.append(
                {
                    "matrix": _workpiece_placement_matrix(wp),
                    "size": list(wp.size) if wp.size else [0, 0],
                }
            )
        payload = {
            "kind": "step_aggregate",
            "step_uid": step.uid,
            "upstream": [[k, t] for k, t, _wp in upstream],
            "step_params": step.get_cache_params(),
            "spxf": _canonical(step.per_step_transformers_dicts),
            "wpxf": _canonical(step.per_workpiece_transformers_dicts),
            "position_sensitive": step.is_position_sensitive(),
            "placements": placements,
        }
        if step.is_position_sensitive():
            payload["stock_rev"] = self._stock_revision()
        return _hash_int(payload)

    def _job_token(self, doc: Doc, step_tokens: dict[str, int]) -> int:
        # The job aggregate concatenates the step aggregates' outputs
        # verbatim (identity placement at the job level). Its token
        # therefore folds in the per-step aggregate tokens so that any
        # upstream change (workpiece move, transformer edit, step
        # param change) propagates through to the job/encode cache.
        payloads = []
        for layer in doc.layers:
            if not layer.workflow:
                continue
            for step in layer.workflow.steps:
                if not step.visible:
                    continue
                if step.uid not in step_tokens:
                    continue
                payloads.append(
                    {
                        "step_uid": step.uid,
                        "step_token": step_tokens.get(step.uid, 0),
                        "step_params": step.get_cache_params(),
                        "spxf": _canonical(step.per_step_transformers_dicts),
                    }
                )
        payload: dict[str, Any] = {"kind": "job", "steps": payloads}
        return _hash_int(payload)

    # ------------------------------------------------------------------
    # Node construction
    # ------------------------------------------------------------------

    def _make_request(
        self, key: str, token: int, stage: Any, cacheable: bool = True
    ) -> NodeRequest:
        return NodeRequest(
            key=key,
            generation_id=self._generation_id,
            stage=stage,
            version_token=token,
            cacheable=cacheable,
        )

    # ------------------------------------------------------------------
    # Compute stage construction
    # ------------------------------------------------------------------

    def _wp_stage(self, step: Step, wp: WorkPiece) -> StageSpec.Compute:
        """
        Build a compute :class:`StageSpec.Compute` for the workpiece
        node by delegating to :meth:`Step.build_compute_payload`.

        Step kinds that wire a real raygeo assembler override
        ``build_compute_payload`` (e.g. :class:`ContourStep`,
        :class:`EngraveStep`) to return both the :class:`Part`
        (carrying vector geometry or an image source) and the
        :class:`ComputePayload` (carrying the assembler spec).

        Per-workpiece transformers (e.g. ``OverscanTransformer``,
        ``BidirScanOffsetTransformer``) are resolved into typed Rust
        specs and attached to the payload so the Rust compute stage
        applies them after assembly.
        """
        part, payload = step.build_compute_payload(self._machine, wp)
        step.populate_payload(payload, self._machine)
        payload.transformers = self._build_transformer_specs(
            step.per_workpiece_transformers_dicts,
            workpiece=wp,
        )

        if step.uses_global_state and step.layer:
            workflow = step.layer.workflow
            if workflow:
                chain = [s for s in workflow.steps if s.uses_global_state]
                if step in chain:
                    idx = chain.index(step)
                    if idx > 0:
                        prev = chain[idx - 1]
                        payload.state_source_keys = [
                            workpiece_key(wp.uid, prev.uid)
                        ]
        return StageSpec.Compute(part=part, params=payload)

    def _assembler_params(self, step: Step, wp: WorkPiece) -> Any:
        """
        Return a JSON-serialisable representation of the assembler spec
        parameters that the step resolves for its machine.

        Delegates to :meth:`Step.assembler_token_params`.  Returns
        :data:`None` when the step exposes no assembler params; the
        compute token is unaffected in that case.
        """
        try:
            return step.assembler_token_params(self._machine, wp)
        except Exception:
            logger.debug(
                "Step %s has no assembler token params",
                step.uid,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Transformer spec construction
    # ------------------------------------------------------------------

    def _build_transformer_specs(
        self,
        transformer_dicts: list[dict[str, Any]],
        *,
        workpiece: WorkPiece | None = None,
    ) -> list[Any]:
        """Build typed Rust ``*Spec`` pyclasses from a list of
        serialised transformer dicts.

        Instantiates each enabled transformer via the registry and
        calls ``to_spec`` to produce the typed spec the Rust compute
        and aggregate stages consume.  ``workpiece`` is forwarded so
        that position-sensitive transformers (e.g. CropTransformer)
        can resolve their regions.
        """
        transformers: list[OpsTransformer] = []
        for t_dict in transformer_dicts:
            if not t_dict.get("enabled", True):
                continue
            name = t_dict.get("name")
            if not name or not isinstance(name, str):
                continue
            cls = transformer_registry.get(name)
            if cls is None:
                logger.warning(
                    "Transformer %r not found in registry; skipping",
                    name,
                )
                continue
            try:
                transformers.append(cls.from_dict(t_dict))
            except Exception:
                logger.exception(
                    "Failed to instantiate transformer %r; skipping",
                    name,
                )
        if not transformers:
            return []
        stock = self._resolve_stock_geometries()
        settings = self._transformer_settings()

        specs: list = []
        for t in transformers:
            if not t.enabled:
                continue
            specs.append(t.to_spec(workpiece, stock, settings))
        return specs

    def _transformer_settings(self) -> dict[str, Any] | None:
        """Return the settings dict forwarded to ``to_spec``.

        Currently this carries the ``driver_native_overscan`` flag so
        :class:`OverscanTransformer` can short-circuit when the
        machine driver handles overscan itself.
        """
        if self._machine is None:
            return None
        try:
            native = bool(self._machine.driver.native_overscan)
        except AttributeError:
            native = False
        return {"driver_native_overscan": native}

    def _resolve_stock_geometries(self) -> list[Any] | None:
        """Return the world-space stock boundary geometries.

        Transformers such as CropTransformer use this to clip
        per-workpiece ops to the machine's work area or to explicit
        StockItems.

        Doc-owned :class:`StockItem` entries take precedence. The
        machine workarea rectangle is used as a fallback only when
        no doc stock exists.
        """
        geos: list[Any] = []

        if self._doc is not None:
            for item in self._doc.stock_items:
                if not item.visible:
                    continue
                try:
                    geo = item.get_world_rect_geometry()
                except Exception:
                    logger.debug(
                        "Failed to resolve stock geometry for %s",
                        item.uid,
                        exc_info=True,
                    )
                    continue
                if geo is not None and not geo.is_empty():
                    geos.append(geo)

        if self._machine is not None and not geos:
            try:
                space = MachineSpace.from_machine(self._machine)
                wx, wy, w, h = space.get_workarea_world_rect()
                geo = Geometry()
                geo.move_to(wx, wy)
                geo.line_to(wx + w, wy)
                geo.line_to(wx + w, wy + h)
                geo.line_to(wx, wy + h)
                geo.close_path()
                geos.append(geo)
            except Exception:
                logger.debug(
                    "Failed to resolve machine workarea for stock",
                    exc_info=True,
                )

        return geos

    # ------------------------------------------------------------------
    # Step aggregate stage
    # ------------------------------------------------------------------

    def _step_stage(
        self,
        step: Step,
        upstream: list[tuple[str, int, WorkPiece]],
    ) -> StageSpec.Aggregate:
        """
        Build an aggregate :class:`StageSpec.Aggregate` for the step
        node.

        One :class:`AggregateGroup` per upstream workpiece compute node,
        wrapped by that workpiece's start / end markers.  Each input
        carries the workpiece's world placement matrix (scale normalised
        to ±1, sign preserved — absolute scale is handled via
        ``target_dimensions`` for scalable artifacts) and the
        workpiece's physical size as ``target_dimensions``.

        Per-step transformers (e.g. ``MultiPassTransformer``,
        ``Optimize``) are resolved into typed Rust specs and attached
        to :attr:`AggregateSpec.transformers` so the Rust aggregate
        stage applies them after concatenation.  ``MachineParams`` is
        populated from the resolved machine so the aggregate's time
        estimate is correct.
        """
        groups: list[AggregateGroup] = []
        for wp_key, _token, wp in upstream:
            placement = _workpiece_placement_matrix(wp)
            target = wp.size
            inp = AggregateInput(
                source_key=wp_key,
                placement_matrix=placement,
                uid=wp.uid,
                target_dimensions=target,
            )
            start = Marker.WorkpieceStart(uid=wp.uid, _tag=True)
            end = Marker.WorkpieceEnd(uid=wp.uid, _tag=True)
            link_mode = LinkMode.none()
            if step.uses_global_state:
                link_mode = LinkMode.sequential(
                    safe_z=getattr(step, "safe_z", 2.0)
                )
            groups.append(
                AggregateGroup(
                    start_markers=[start],
                    inputs=[inp],
                    end_markers=[end],
                    link_mode=link_mode,
                )
            )
        spec = AggregateSpec(
            wrap_start=[],
            groups=groups,
            wrap_end=[],
            machine=self._machine_params(),
            transformers=self._build_transformer_specs(
                step.per_step_transformers_dicts
            ),
        )
        return StageSpec.Aggregate(spec=spec)

    def _machine_params(self) -> MachineParams:
        """Build :class:`MachineParams` from the resolved machine."""
        return MachineParams(
            default_feed_rate=float(self._machine.max_cut_speed),
            default_rapid_rate=float(self._machine.max_travel_speed),
            acceleration=float(self._machine.acceleration),
        )

    # ------------------------------------------------------------------
    # Job aggregate stage
    # ------------------------------------------------------------------

    def _job_stage(
        self, doc: Doc, step_tokens: dict[str, int]
    ) -> StageSpec.Aggregate:
        """
        Build the final job aggregate :class:`StageSpec.Aggregate`.

        One :class:`AggregateGroup` per layer, wrapped by
        ``LayerStart`` / ``LayerEnd`` markers, containing one
        :class:`AggregateInput` per visible step in that layer that has
        workpiece compute nodes upstream (i.e. is present in
        *step_tokens*).  The whole aggregate is wrapped by
        ``JobStart`` / ``JobEnd`` markers.  ``MachineParams`` is
        populated from the resolved machine so the aggregate's time
        estimate is correct.
        """
        groups: list[AggregateGroup] = []
        for layer in doc.layers:
            if not layer.workflow:
                continue
            step_inputs: list[AggregateInput] = []
            for step in layer.workflow.steps:
                if not step.visible:
                    continue
                if step.uid not in step_tokens:
                    continue
                sk = step_key(step.uid)
                step_inputs.append(
                    AggregateInput(
                        source_key=sk,
                        placement_matrix=_IDENTITY_4X4,
                        uid=step.uid,
                    )
                )
            if not step_inputs:
                continue
            groups.append(
                AggregateGroup(
                    start_markers=[
                        Marker.LayerStart(uid=layer.uid, _tag=True)
                    ],
                    inputs=step_inputs,
                    end_markers=[Marker.LayerEnd(uid=layer.uid, _tag=True)],
                )
            )
        spec = AggregateSpec(
            wrap_start=[Marker.JobStart(_tag=True)],
            groups=groups,
            wrap_end=[Marker.JobEnd(_tag=True)],
            machine=self._machine_params(),
        )
        return StageSpec.Aggregate(spec=spec)

    # ------------------------------------------------------------------
    # Encoder stage
    # ------------------------------------------------------------------

    def _encode_stage(self, doc: Doc) -> EncodeSpec:
        """Build the encoder :class:`EncodeSpec` for the job encode
        node.

        The encoder receives machine-space ops from the upstream
        ``job:machinexform`` node (the machine transform stage).
        For Grbl machines the native Rust ``GcodeSpec`` is used
        directly; for any other machine a
        :class:`PythonEncoder` wraps the driver-specific encoder
        callable.
        """
        encoder = self._build_encoder(doc)
        return EncodeSpec(
            source_key=job_machinexform_key(), encoder=Encoder(encoder)
        )

    def _build_encoder(self, doc: Doc) -> Any:
        """Resolve the encoder for the configured machine.

        Routes Grbl machines to the native Rust ``GcodeSpec`` and
        every other machine to a :class:`PythonEncoder` wrapping the
        driver-specific encoder callable.  The pre-processing
        transforms are handled by the upstream machine-transform
        stage.
        """
        machine = self._machine
        assert machine is not None

        dialect = machine.dialect
        if dialect is not None and _is_grbl(dialect):
            return self._grbl_encoder_spec(doc)

        return PythonEncoder(
            self._make_python_encoder_callable(machine, doc),
            "driver.encode",
        )

    def _grbl_encoder_spec(self, doc: Doc) -> GcodeSpec:
        """Build a native ``GcodeSpec`` for a Grbl machine.

        Receives machine-space ops from the upstream
        ``job:machinexform`` node and encodes them directly on a
        rayon thread without crossing the GIL.
        """
        machine = self._machine
        assert machine is not None
        dialect = machine.dialect
        assert dialect is not None
        # Build a minimal Ops with estimated extents so path variables
        # like ``job.extents[0..3]`` used in dialect templates are
        # populated with reasonable values.  The exact extents are
        # computed later from the real ops at encode time (the
        # machine-transform stage preserves bounding-box metadata).
        approx_ops = _approximate_job_ops(doc)
        context = build_encode_context(approx_ops, machine, doc)
        return GcodeSpec(
            dialect=dialect_to_spec(dialect, machine),
            context_json=json.dumps(context),
        )

    def _build_machine_transform_stage(self, doc: Doc) -> MachineTransformSpec:
        """Build the :class:`MachineTransformSpec` for the machine-
        transform pipeline node.

        Collects the world→machine matrix, WCS offsets, and per-layer
        rotary mapping config from the machine and document and
        packages them into a serialisable spec that the Rust
        ``MachineTransformCompute`` stage consumes.
        """
        machine = self._machine
        assert machine is not None

        space = MachineSpace.from_machine(machine)

        # World→machine 4x4 matrix.
        w2m = space.get_world_to_machine_matrix()

        # Default WCS command offset.
        default_wcs_offset = list(
            space.get_command_offset(
                wcs_offset=machine.get_active_wcs_offset(),
                wcs_is_workarea_origin=machine.wcs_origin_is_workarea_origin,
            )
        )

        # Per-layer WCS offsets.
        layer_wcs_offsets: list[tuple[str, list[float]]] = []
        for layer in doc.layers:
            effective_wcs = layer.get_effective_wcs(machine)
            wcs_off = machine.get_wcs_offset(effective_wcs)
            cmd_offset = space.get_command_offset(
                wcs_offset=wcs_off,
                wcs_is_workarea_origin=machine.wcs_origin_is_workarea_origin,
            )
            layer_wcs_offsets.append((layer.uid, list(cmd_offset)))

        # Per-layer rotary mappings.
        rotary_mappings = self._build_rotary_mappings(doc, machine)

        return MachineTransformSpec(
            source_key=job_key(),
            linearize_curves=not machine.supports_curves,
            world_to_machine=w2m.tolist(),
            default_wcs_offset=default_wcs_offset,
            layer_wcs_offsets=layer_wcs_offsets,
            reverse_z=machine.reverse_z_axis,
            rotary_mappings=rotary_mappings,
        )

    @staticmethod
    def _build_rotary_mappings(
        doc: Doc,
        machine: Machine,
    ) -> list:
        """Build per-layer :class:`RotaryMappingSpec` entries."""
        mappings: list = []
        for layer in doc.layers:
            if not layer.rotary_enabled:
                continue
            module = machine.get_rotary_module_for_layer(layer)
            if module is None:
                continue

            diameter = layer.rotary_diameter
            gear_ratio = KinematicMath.gear_ratio(
                module.rotary_type == RotaryType.ROLLERS,
                diameter,
                module.roller_diameter,
            )

            # Extract axis position and cylinder direction.
            rot3 = module.transform[:3, :3].astype(np.float64).copy()
            for col in range(3):
                norm = np.linalg.norm(rot3[:, col])
                if norm > 1e-12:
                    rot3[:, col] /= norm

            mod_pos = module.transform[:3, 3].astype(np.float64)
            axis_position_3d = mod_pos + rot3 @ module.axis_position
            cylinder_dir = rot3[:, 0].copy()
            norm = np.linalg.norm(cylinder_dir)
            if norm > 1e-12:
                cylinder_dir /= norm

            if module.mode == RotaryMode.TRUE_4TH_AXIS:
                rotary_axis = module.axis.name
                replaced_axis = None
            else:
                rotary_axis = "Y"
                replaced_axis = module.axis.name
            mappings.append(
                RotaryMappingSpec(
                    layer_uid=layer.uid,
                    diameter=diameter,
                    gear_ratio=gear_ratio,
                    reverse=module.reverse_axis,
                    axis_position_3d=axis_position_3d.tolist(),
                    cylinder_dir=cylinder_dir.tolist(),
                    rotary_axis=rotary_axis,
                    replaced_axis=replaced_axis,
                    mm_per_rotation=module.mm_per_rotation,
                )
            )
        return mappings

    def _make_python_encoder_callable(
        self, machine: Machine, doc: Doc
    ) -> Callable[[Any], Any]:
        """Build a Python callable ``(ops) -> EncodeOutput`` that
        invokes the driver-specific encoder directly on
        machine-space ops.

        The pre-processing transforms (linearization, rotary mapping,
        world→machine, WCS offset, Z-flip, AXIS_REPLACEMENT) are
        handled by the upstream machine-transform stage, so this
        callable only applies the final driver encoding step.
        """
        if machine.driver_name:
            try:
                driver_cls = get_driver_cls(machine.driver_name)
            except (ValueError, ImportError):
                driver_cls = NoDeviceDriver
        else:
            driver_cls = NoDeviceDriver

        driver_encoder = driver_cls.create_encoder(machine)

        def encode(ops: Any) -> EncodeOutput:
            encoded = driver_encoder.encode(ops, machine, doc)
            if not isinstance(encoded, EncodedOutput):
                raise TypeError(
                    "encoder must return EncodedOutput, "
                    f"got {type(encoded).__name__}"
                )
            return EncodeOutput.MachineCode(
                text=encoded.text,
                op_to_machine_code=encoded.op_map.op_to_machine_code,
                machine_code_to_op=encoded.op_map.machine_code_to_op,
            )

        return encode

    # ------------------------------------------------------------------
    # Encoder token
    # ------------------------------------------------------------------

    def _encode_token(self, doc: Doc, step_tokens: dict[str, int]) -> int:
        """Compute the version token for the job encode node.

        Folds in the machine-transform node's token plus the encoder
        identity so the cache invalidates when either the machine
        transforms or the encoder config change.
        """
        payload = {
            "kind": "encode",
            "mxform_token": self._machine_transform_token(doc, step_tokens),
            "machine": _machine_token_payload(self._machine),
        }
        return _hash_int(payload)

    def _machine_transform_token(
        self, doc: Doc, step_tokens: dict[str, int]
    ) -> int:
        """Compute the version token for the machine-transform node.

        Folds in the job aggregate's token plus the machine identity
        (supports_curves, reverse_z, WCS config, rotary module config)
        so any change to the machine or job invalidates the cache.
        """
        payload = {
            "kind": "machine_transform",
            "job_token": self._job_token(doc, step_tokens),
            "machine": _machine_token_payload(self._machine),
        }
        if self._machine is not None:
            cfg = _machine_transform_config_payload(self._machine, doc)
            payload.update(cfg)
        return _hash_int(payload)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _workpiece_placement_matrix(wp: WorkPiece) -> list[list[float]]:
    """
    Build the 4×4 placement matrix for a workpiece's aggregate input.

    The workpiece's world transform is decomposed and re-composed with
    the absolute scale normalised to ``1.0`` (the sign of the Y scale
    is preserved to keep flips).  Absolute scaling is handled by the
    aggregate via ``target_dimensions`` for scalable artifacts, so the
    placement matrix only carries translation, rotation, flip, and skew.
    """
    world = wp.get_world_transform()
    tx, ty, angle, _sx, sy, skew = world.decompose()
    placement = Matrix.compose(
        tx, ty, angle, 1.0, math.copysign(1.0, sy), skew
    )
    return placement.to_4x4_list()


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
# Build raygeo :class:`StageSpec` instances for the step aggregate and
# job aggregate nodes.  The per-workpiece compute stage is built by
# :meth:`IntentBuilder._wp_stage`, which selects the assembler spec
# from the step's ``ASSEMBLER_NAME``.


_IDENTITY_4X4: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _is_grbl(dialect: GcodeDialect) -> bool:
    """Return True if *dialect* is the Grbl G-code dialect."""
    return dialect.uid == GRBL_DIALECT.uid


def _machine_token_payload(machine: Machine | None) -> Any:
    """Build a JSON-serialisable representation of the machine
    identity for the encode token."""
    if machine is None:
        return None
    return {
        "driver_name": machine.driver_name,
        "active_wcs": machine.active_wcs,
        "gcode_precision": machine.gcode_precision,
        "supports_curves": machine.supports_curves,
        "supports_arcs": machine.supports_arcs,
        "reverse_z_axis": machine.reverse_z_axis,
        "max_cut_speed": machine.max_cut_speed,
        "max_travel_speed": machine.max_travel_speed,
        "acceleration": machine.acceleration,
        "axis_extents": list(machine.axis_extents),
    }


def _machine_transform_config_payload(
    machine: Machine, doc: Doc
) -> dict[str, Any]:
    """Build a JSON-serialisable payload of machine transform config
    for the machine-transform token."""

    payload: dict[str, Any] = {
        "wcs_origin_is_workarea_origin": machine.wcs_origin_is_workarea_origin,
    }
    # Rotary module UIDs per layer (to detect rotary config changes).
    for layer in doc.layers:
        uid = layer.uid
        if layer.rotary_enabled:
            module = machine.get_rotary_module_for_layer(layer)
            if module is not None:
                payload[f"rotary:{uid}"] = {
                    "module_uid": module.uid,
                    "mode": module.mode.value,
                    "axis": module.axis.name,
                    "mm_per_rotation": module.mm_per_rotation,
                    "diameter": layer.rotary_diameter,
                    "roller_diameter": module.roller_diameter,
                    "rotary_type": module.rotary_type.value,
                    "reverse_axis": module.reverse_axis,
                }
    return payload


def _approximate_job_ops(doc: Doc) -> Ops:
    """Build a minimal Ops spanning the estimated job extents.

    Used by :meth:`IntentBuilder._grbl_encoder_spec` so that
    path variables like ``job.extents[0..3]`` are populated with
    reasonable values before the real ops are available from the
    pipeline.

    The extents are estimated from workpiece positions and sizes
    in world space.
    """
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")

    for layer in doc.layers:
        for wp in layer.all_workpieces:
            tx, ty = wp.pos
            sx, sy = wp.size if wp.size else (0, 0)
            if sx > 0 and sy > 0:
                xmin = min(xmin, tx)
                ymin = min(ymin, ty)
                xmax = max(xmax, tx + sx)
                ymax = max(ymax, ty + sy)

    if xmin == float("inf"):
        return Ops()

    ops = Ops()
    ops.job_start()
    ops.move_to(xmin, ymin, 0.0)
    ops.line_to(xmax, ymax, 0.0)
    ops.job_end()
    return ops
