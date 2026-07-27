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
import math
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from raygeo.cnc.execution.specs import (
    AggregateGroup,
    AggregateInput,
    AggregateSpec,
    EncodeSpec,
    MachineParams,
    Marker,
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

from ..machine.models.dialect import GRBL_DIALECT
from .encoder.base import EncodedOutput
from .encoder.rust_helpers import build_encode_context, dialect_to_spec
from .stage.assembler_helpers import (
    MachineDefaults,
    resolve_machine_defaults,
)
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


def workpiece_key(wp_uid: str, step_uid: str) -> str:
    return WORKPIECE_KEY_FMT.format(wp_uid=wp_uid, step_uid=step_uid)


def step_key(step_uid: str) -> str:
    return STEP_KEY_FMT.format(step_uid=step_uid)


def job_key() -> str:
    return JOB_KEY


def job_encode_key() -> str:
    return JOB_ENCODE_KEY


class IntentBuilder:
    """
    Builds a flat :class:`NodeRequest` list from a :class:`Doc`.

    The builder is stateless: each call to :meth:`build` produces a
    fresh, self-contained list suitable for wrapping in a raygeo
    :class:`Intent`.
    """

    def __init__(
        self,
        machine: "Optional[Machine]" = None,
        generation_id: int = 0,
    ):
        self._machine = machine
        self._generation_id = generation_id
        self._stock_geometries: Optional[List[Any]] = None
        self._doc: Optional["Doc"] = None

    @property
    def generation_id(self) -> int:
        return self._generation_id

    def build(self, doc: "Doc") -> List[NodeRequest]:
        """
        Walk *doc* and produce one NodeRequest per workpiece-step pair,
        one per step, and one final job aggregate.
        """
        self._doc = doc
        nodes: List[NodeRequest] = []
        # Map each step's key to the list of upstream workpiece compute
        # inputs — the step aggregate token and placement depend on all
        # of them.
        step_compute_inputs: Dict[str, List[Tuple[str, int, WorkPiece]]] = {}
        # Per-step aggregate version tokens, used by the job aggregate
        # token so a position change that invalidates one step's
        # aggregate also invalidates the job aggregate (and encode).
        step_tokens: Dict[str, int] = {}

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

        self._build_job_node(doc, nodes, step_tokens)
        self._build_encoder_node(doc, nodes, step_tokens)
        return nodes

    # ------------------------------------------------------------------
    # Workpiece compute nodes
    # ------------------------------------------------------------------

    def _build_workpiece_nodes(
        self,
        step: "Step",
        workpieces: "Sequence[WorkPiece]",
        out: List[NodeRequest],
    ) -> List[Tuple[str, int, WorkPiece]]:
        """
        Append one compute NodeRequest per workpiece for *step* and
        return the list of ``(node_key, version_token, workpiece)``
        triples the step aggregate consumes.
        """
        pos_sensitive = step.is_position_sensitive()
        inputs: List[Tuple[str, int, WorkPiece]] = []
        for wp in workpieces:
            key = workpiece_key(wp.uid, step.uid)
            token = self._compute_token(step, wp, pos_sensitive)
            inputs.append((key, token, wp))
            stage = self._wp_stage(step, wp)
            out.append(self._make_request(key, token, stage))
        return inputs

    def _build_step_node(
        self,
        step: "Step",
        layer: "Layer",
        upstream: List[Tuple[str, int, WorkPiece]],
        out: List[NodeRequest],
    ) -> None:
        key = step_key(step.uid)
        token = self._aggregate_token(step, layer, upstream)
        stage = self._step_stage(step, upstream)
        out.append(self._make_request(key, token, stage))

    def _build_job_node(
        self,
        doc: "Doc",
        out: List[NodeRequest],
        step_tokens: Dict[str, int],
    ) -> None:
        key = job_key()
        token = self._job_token(doc, step_tokens)
        out.append(self._make_request(key, token, self._job_stage(doc)))

    def _build_encoder_node(
        self,
        doc: "Doc",
        out: List[NodeRequest],
        step_tokens: Dict[str, int],
    ) -> None:
        """Append the encoder compute node that consumes the job
        aggregate's Ops and produces the machine code (G-code /
        vertex / texture).

        The encoder runs through raygeo's ``EncoderCompute`` stage.
        For Grbl the native Rust ``GcodeSpec`` is used directly; for
        any other machine the full ``machine.encode_ops`` Python path
        (including rotary mapping, world→machine, and WCS offset
        pre-processing) is wrapped in a :class:`PythonEncoder` so it
        runs under the GIL on a rayon worker thread — off the GTK
        main thread.
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
        self, step: "Step", wp: "WorkPiece", pos_sensitive: bool
    ) -> int:
        payload = {
            "kind": "compute",
            "step_uid": step.uid,
            "wp_uid": wp.uid,
            "geo_rev": wp.geometry_revision,
            "wp_size": list(wp.size) if wp.size else [0, 0],
            "step_params": _step_compute_params(step),
            "assembler_params": _canonical(self._assembler_params(step, wp)),
            "wpxf": _canonical(step.per_workpiece_transformers_dicts),
        }
        if pos_sensitive:
            payload["xf_rev"] = wp.transform_revision
            payload["stock_rev"] = self._stock_revision()
        return _hash_int(payload)

    def _aggregate_token(
        self,
        step: "Step",
        layer: "Layer",
        upstream: List[Tuple[str, int, "WorkPiece"]],
    ) -> int:
        # Fold the per-workpiece placement matrix and target dimensions
        # into the token. The aggregate applies the placement matrix
        # to the (possibly cached) workpiece compute output, so a move
        # that leaves the compute cache untouched must still invalidate
        # the aggregate — otherwise the cached step ops are displayed
        # at their previous world position.
        placements: List[Any] = []
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
            "step_params": _step_compute_params(step),
            "spxf": _canonical(step.per_step_transformers_dicts),
            "wpxf": _canonical(step.per_workpiece_transformers_dicts),
            "position_sensitive": step.is_position_sensitive(),
            "placements": placements,
        }
        if step.is_position_sensitive():
            payload["stock_rev"] = self._stock_revision()
        return _hash_int(payload)

    def _job_token(self, doc: "Doc", step_tokens: Dict[str, int]) -> int:
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
                payloads.append(
                    {
                        "step_uid": step.uid,
                        "step_token": step_tokens.get(step.uid, 0),
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

    # ------------------------------------------------------------------
    # Compute stage construction
    # ------------------------------------------------------------------

    def _wp_stage(self, step: "Step", wp: "WorkPiece") -> StageSpec.Compute:
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
        machine_defaults = self._resolve_machine_defaults(step)
        part, payload = step.build_compute_payload(machine_defaults, wp)
        payload.power = step.power
        payload.cut_speed = step.cut_speed
        if self._machine is not None:
            try:
                laser = step.get_selected_laser(self._machine)
                payload.head_uid = laser.uid if laser else None
            except ValueError:
                logger.debug(
                    "Step %s has no laser heads on machine; "
                    "head_uid left unset",
                    step.uid,
                )
        payload.transformers = self._build_transformer_specs(
            step.per_workpiece_transformers_dicts,
            workpiece=wp,
        )
        return StageSpec.Compute(part=part, params=payload)

    def _resolve_machine_defaults(self, step: "Step") -> MachineDefaults:
        """
        Resolve :class:`MachineDefaults` for *step*.

        When no machine is available (e.g. in tests that construct a
        bare :class:`IntentBuilder`), fall back to the step's own
        parameters and conservative defaults so the assembler still
        receives a valid spec.
        """
        if self._machine is not None:
            try:
                laser = step.get_selected_laser(self._machine)
            except ValueError:
                logger.debug(
                    "Step %s has no laser heads; using bare defaults",
                    step.uid,
                )
                laser = None
            settings = step.to_dict()
            settings["arc_tolerance"] = self._machine.arc_tolerance
            settings["machine_supports_arcs"] = self._machine.supports_arcs
            settings["machine_supports_curves"] = self._machine.supports_curves
            if laser is not None:
                return resolve_machine_defaults(laser, settings)
        return MachineDefaults(
            kerf_mm=step.kerf_mm,
            arc_tolerance=0.03,
            allow_arcs=True,
            supports_curves=False,
            line_interval_mm=0.1,
            step_power=step.power,
            tool_radius=step.kerf_mm / 2.0,
            step_over=step.kerf_mm,
            cut_speed=step.cut_speed,
        )

    def _assembler_params(self, step: "Step", wp: "WorkPiece") -> Any:
        """
        Return a JSON-serialisable representation of the assembler spec
        parameters that the step resolves for its machine.

        Delegates to :meth:`Step.assembler_token_params`.  Returns
        :data:`None` when no machine is configured or the step exposes
        no assembler params; the compute token is unaffected in that
        case.
        """
        if self._machine is None:
            return None
        try:
            laser = step.get_selected_laser(self._machine)
        except ValueError:
            logger.debug(
                "Step %s has no laser heads; skipping assembler params",
                step.uid,
            )
            return None
        settings = step.to_dict()
        settings["arc_tolerance"] = self._machine.arc_tolerance
        settings["machine_supports_arcs"] = self._machine.supports_arcs
        settings["machine_supports_curves"] = self._machine.supports_curves
        defaults = resolve_machine_defaults(laser, settings)
        try:
            return step.assembler_token_params(defaults, wp)
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
        transformer_dicts: List[Dict[str, Any]],
        *,
        workpiece: "Optional[WorkPiece]" = None,
    ) -> List[Any]:
        """Build typed Rust ``*Spec`` pyclasses from a list of
        serialised transformer dicts.

        Instantiates each enabled transformer via the registry and
        calls ``to_spec`` to produce the typed spec the Rust compute
        and aggregate stages consume.  ``workpiece`` is forwarded so
        that position-sensitive transformers (e.g. CropTransformer)
        can resolve their regions.
        """
        transformers: List[OpsTransformer] = []
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

    def _transformer_settings(self) -> Optional[Dict[str, Any]]:
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

    def _resolve_stock_geometries(self) -> Optional[List[Any]]:
        """Return the world-space stock boundary geometries.

        Resolved once per :meth:`build` call and cached on the
        builder. Transformers such as CropTransformer use this to
        clip per-workpiece ops to the machine's work area or to
        explicit StockItems.

        Doc-owned :class:`StockItem` entries take precedence. The
        machine workarea rectangle is used as a fallback only when
        no doc stock exists.
        """
        if self._stock_geometries is not None:
            return self._stock_geometries

        geos: List[Any] = []

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
                from .coordspace import MachineSpace

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

        self._stock_geometries = geos
        return self._stock_geometries

    # ------------------------------------------------------------------
    # Step aggregate stage
    # ------------------------------------------------------------------

    def _step_stage(
        self,
        step: "Step",
        upstream: List[Tuple[str, int, "WorkPiece"]],
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
        groups: List[AggregateGroup] = []
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
            machine=self._machine_params(),
            transformers=self._build_transformer_specs(
                step.per_step_transformers_dicts
            ),
        )
        return StageSpec.Aggregate(spec=spec)

    def _machine_params(self) -> MachineParams:
        """Build :class:`MachineParams` from the resolved machine.

        Falls back to zero rates (which disables time estimation) when
        no machine is configured.
        """
        if self._machine is None:
            return MachineParams()
        return MachineParams(
            default_feed_rate=float(self._machine.max_cut_speed),
            default_rapid_rate=float(self._machine.max_travel_speed),
            acceleration=float(self._machine.acceleration),
        )

    # ------------------------------------------------------------------
    # Job aggregate stage
    # ------------------------------------------------------------------

    def _job_stage(self, doc: "Doc") -> StageSpec.Aggregate:
        """
        Build the final job aggregate :class:`StageSpec.Aggregate`.

        One :class:`AggregateGroup` per layer, wrapped by
        ``LayerStart`` / ``LayerEnd`` markers, containing one
        :class:`AggregateInput` per visible step in that layer.  The
        whole aggregate is wrapped by ``JobStart`` / ``JobEnd``
        markers.  ``MachineParams`` is populated from the resolved
        machine so the aggregate's time estimate is correct.
        """
        groups: List[AggregateGroup] = []
        for layer in doc.layers:
            if not layer.workflow:
                continue
            step_inputs: List[AggregateInput] = []
            for step in layer.workflow.steps:
                if not step.visible:
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

    def _encode_stage(self, doc: "Doc") -> EncodeSpec:
        """Build the encoder :class:`EncodeSpec` for the job encode
        node.

        Routes Grbl machines to the native Rust ``GcodeSpec`` and
        every other machine to a :class:`PythonEncoder` wrapping the
        full ``machine.encode_ops`` path (rotary mapping,
        world→machine, WCS offset, and the driver-specific encoder).
        """
        encoder = self._build_encoder(doc)
        return EncodeSpec(source_key=job_key(), encoder=Encoder(encoder))

    def _build_encoder(self, doc: "Doc") -> Any:
        """Resolve the encoder for the configured machine.

        Routes Grbl machines to the native Rust ``GcodeSpec`` and
        every other machine to a :class:`PythonEncoder` wrapping the
        full ``machine.encode_ops`` path.
        """
        machine = self._machine
        assert machine is not None

        dialect = machine.dialect
        if dialect is not None and _is_grbl(dialect):
            return self._grbl_encoder_spec(doc)

        return PythonEncoder(
            self._make_python_encoder_callable(machine, doc),
            "machine.encode_ops",
        )

    def _grbl_encoder_spec(self, doc: "Doc") -> GcodeSpec:
        """Build a native ``GcodeSpec`` for a Grbl machine.

        The Rust ``GcodeSpec`` encoder expects pre-processed
        machine-space ops; the world→machine, rotary mapping, and
        WCS pre-processing that ``machine.encode_ops`` performs are
        not yet ported, so for the staged migration we route even
        Grbl through the Python-callable encoder for byte-identical
        output.  This method is retained for the future cutover.
        """
        machine = self._machine
        assert machine is not None
        dialect = machine.dialect
        assert dialect is not None
        context = build_encode_context(Ops(), machine, doc)
        return GcodeSpec(
            dialect=dialect_to_spec(dialect, machine),
            context_json=json.dumps(context),
        )

    def _make_python_encoder_callable(
        self, machine: "Machine", doc: "Doc"
    ) -> Callable[[Any], Any]:
        """Build a Python callable ``(ops) -> EncodeOutput`` that
        invokes the full ``machine.encode_ops`` path and adapts the
        rayforge :class:`EncodedOutput` to raygeo's
        :class:`EncodeOutput`.
        """

        def encode(ops: Any) -> EncodeOutput:
            encoded = machine.encode_ops(ops, doc)
            if not isinstance(encoded, EncodedOutput):
                raise TypeError(
                    "machine.encode_ops must return EncodedOutput, "
                    f"got {type(encoded).__name__}"
                )
            return EncodeOutput.MachineCode(
                text=encoded.text,
                op_to_machine_code=dict(encoded.op_map.op_to_machine_code),
                machine_code_to_op=dict(encoded.op_map.machine_code_to_op),
            )

        return encode

    # ------------------------------------------------------------------
    # Encoder token
    # ------------------------------------------------------------------

    def _encode_token(self, doc: "Doc", step_tokens: Dict[str, int]) -> int:
        """Compute the version token for the job encode node.

        Folds in the job aggregate's token plus the machine identity
        (driver name, dialect, gcode precision, WCS, rotary config)
        so a machine swap invalidates the encoded output cache.
        """
        payload = {
            "kind": "encode",
            "job_token": self._job_token(doc, step_tokens),
            "machine": _machine_token_payload(self._machine),
        }
        return _hash_int(payload)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _workpiece_placement_matrix(wp: "WorkPiece") -> List[List[float]]:
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
# Build raygeo :class:`StageSpec` instances for the step aggregate and
# job aggregate nodes.  The per-workpiece compute stage is built by
# :meth:`IntentBuilder._wp_stage`, which selects the assembler spec
# from the step's ``ASSEMBLER_NAME``.


_IDENTITY_4X4: List[List[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _is_grbl(dialect: "GcodeDialect") -> bool:
    """Return True if *dialect* is the Grbl G-code dialect."""
    return dialect.uid == GRBL_DIALECT.uid


def _machine_token_payload(machine: "Optional[Machine]") -> Any:
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
