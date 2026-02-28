"""
Stress test for the pipeline with random invalidations.

This module contains black-box stress tests that subject the pipeline
to random invalidations over several minutes with cyclic pauses to
detect memory leaks and hung tasks.

Run with: pixi run test -m stress
Skip with: pixi run test -m "not stress"
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import pytest

from rayforge.core.doc import Doc
from rayforge.core.geo import Geometry
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.steps import ContourStep, EngraveStep
from rayforge.pipeline.view.view_manager import ViewManager

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.pipeline.artifact.store import ArtifactStore
    from rayforge.shared.tasker.manager import TaskManager


logger = logging.getLogger(__name__)


@dataclass
class StressTestConfig:
    """Configuration for stress test parameters."""

    total_duration_sec: int = 90
    chaos_phase_sec: int = 19
    settle_phase_sec: int = 7
    min_invalidation_interval_ms: int = 2
    max_invalidation_interval_ms: int = 777
    max_settle_wait_sec: int = 30
    leak_threshold: int = 2
    initial_workpiece_count: int = 2
    max_workpieces: int = 5


class InvalidationType(Enum):
    """All invalidation types to test."""

    ADD_WORKPIECE = auto()
    REMOVE_WORKPIECE = auto()
    ADD_STEP = auto()
    REMOVE_STEP = auto()
    UPDATE_STEP_POWER = auto()
    UPDATE_STEP_SPEED = auto()
    CHANGE_WORKPIECE_POS = auto()
    CHANGE_WORKPIECE_SIZE = auto()
    CHANGE_WORKPIECE_ROTATION = auto()
    INVALIDATE_JOB_ASSEMBLY = auto()


@dataclass
class MetricsSnapshot:
    """Captures pipeline state at a point in time."""

    timestamp: float
    shm_block_count: int
    total_refcount: int
    dag_node_count: int
    is_busy: bool
    active_task_count: int
    generation_id: int
    workpiece_count: int
    step_count: int


@dataclass
class StressTestController:
    """
    Main controller for the pipeline stress test.

    This controller manages the chaos/settle cycle, collects metrics,
    and detects memory leaks and hung tasks.
    """

    config: StressTestConfig
    pipeline: Pipeline
    doc: Doc
    task_mgr: "TaskManager"
    artifact_store: "ArtifactStore"
    context: "RayforgeContext"

    _snapshots: List[MetricsSnapshot] = field(default_factory=list)
    _workpieces: List[WorkPiece] = field(default_factory=list)
    _workpiece_counter: int = 0
    _view_manager: Optional["ViewManager"] = None
    _svg_data: bytes = field(
        default_factory=lambda: b"""
        <svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
        <rect width="50" height="30" />
        </svg>"""
    )

    def capture_metrics(self) -> MetricsSnapshot:
        """Capture current pipeline state."""
        scheduler = self.pipeline._scheduler
        graph = scheduler.graph

        workpiece_count = sum(
            1 for wp in self.doc.all_workpieces if wp.layer is not None
        )
        step_count = sum(
            len(layer.workflow.steps)
            for layer in self.doc.layers
            if layer.workflow is not None
        )

        return MetricsSnapshot(
            timestamp=time.time(),
            shm_block_count=len(self.artifact_store._handles),
            total_refcount=sum(
                h.refcount for h in self.artifact_store._handles.values()
            ),
            dag_node_count=len(graph.get_all_nodes()),
            is_busy=self.pipeline.is_busy,
            active_task_count=len(self.task_mgr),
            generation_id=self.pipeline._data_generation_id,
            workpiece_count=workpiece_count,
            step_count=step_count,
        )

    async def wait_for_settle(self) -> bool:
        """
        Wait for pipeline to reach idle state.

        Returns True if settled within timeout, False if timeout exceeded.
        """
        deadline = time.time() + self.config.max_settle_wait_sec
        while time.time() < deadline:
            if not self.pipeline.is_busy and not self.task_mgr.has_tasks():
                return True
            await asyncio.sleep(0.1)
        return False

    def _create_workpiece(self) -> WorkPiece:
        """Create a new workpiece with source asset."""
        self._workpiece_counter += 1
        name = f"stress_wp_{self._workpiece_counter}.svg"
        workpiece = WorkPiece(name=name)

        source = SourceAsset(
            Path(name),
            original_data=self._svg_data,
            renderer=SVG_RENDERER,
        )
        self.doc.add_asset(source)

        gen_config = SourceAssetSegment(
            source_asset_uid=source.uid,
            pristine_geometry=Geometry(),
            vectorization_spec=PassthroughSpec(),
        )
        workpiece.source_segment = gen_config
        workpiece.set_size(50, 30)
        workpiece.pos = (
            random.uniform(10, 100),
            random.uniform(10, 100),
        )

        return workpiece

    def _get_available_steps(self) -> List:
        """Get all available steps in the document."""
        steps = []
        for layer in self.doc.layers:
            if layer.workflow:
                steps.extend(layer.workflow.steps)
        return steps

    async def _execute_invalidation(
        self, invalid_type: InvalidationType
    ) -> None:
        """Execute a single invalidation operation."""
        layer = self.doc.active_layer

        if invalid_type == InvalidationType.ADD_WORKPIECE:
            if len(self._workpieces) < self.config.max_workpieces:
                wp = self._create_workpiece()
                layer.add_workpiece(wp)
                self._workpieces.append(wp)
                logger.debug(
                    f"Added workpiece '{wp.name}', "
                    f"total: {len(self._workpieces)}"
                )

        elif invalid_type == InvalidationType.REMOVE_WORKPIECE:
            if len(self._workpieces) > 1:
                wp = random.choice(self._workpieces)
                wp_layer = wp.layer
                if wp_layer:
                    wp_layer.remove_workpiece(wp)
                self._workpieces.remove(wp)
                logger.debug(
                    f"Removed workpiece '{wp.name}', "
                    f"remaining: {len(self._workpieces)}"
                )

        elif invalid_type == InvalidationType.ADD_STEP:
            if layer.workflow and len(self._get_available_steps()) < 5:
                step_type = random.choice(
                    [ContourStep.create, EngraveStep.create]
                )
                step = step_type(self.context)
                layer.workflow.add_step(step)
                logger.debug(f"Added step '{step.name}'")

        elif invalid_type == InvalidationType.REMOVE_STEP:
            steps = self._get_available_steps()
            if len(steps) > 1:
                step = random.choice(steps)
                if step.layer and step.layer.workflow:
                    step.layer.workflow.remove_step(step)
                    logger.debug(f"Removed step '{step.name}'")

        elif invalid_type == InvalidationType.UPDATE_STEP_POWER:
            steps = self._get_available_steps()
            if steps:
                step = random.choice(steps)
                new_power = random.uniform(0.3, 1.0)
                step.set_power(new_power)
                logger.debug(
                    f"Updated step '{step.name}' power to {new_power:.2f}"
                )

        elif invalid_type == InvalidationType.UPDATE_STEP_SPEED:
            steps = self._get_available_steps()
            if steps:
                step = random.choice(steps)
                new_speed = random.uniform(100, 1000)
                step.cut_speed = new_speed
                logger.debug(
                    f"Updated step '{step.name}' speed to {new_speed:.0f}"
                )

        elif invalid_type == InvalidationType.CHANGE_WORKPIECE_POS:
            if self._workpieces:
                wp = random.choice(self._workpieces)
                new_pos = (
                    random.uniform(10, 100),
                    random.uniform(10, 100),
                )
                wp.pos = new_pos
                logger.debug(f"Moved workpiece '{wp.name}' to {new_pos}")

        elif invalid_type == InvalidationType.CHANGE_WORKPIECE_SIZE:
            if self._workpieces:
                wp = random.choice(self._workpieces)
                new_size = (
                    random.uniform(30, 80),
                    random.uniform(20, 60),
                )
                wp.set_size(*new_size)
                logger.debug(f"Resized workpiece '{wp.name}' to {new_size}")

        elif invalid_type == InvalidationType.CHANGE_WORKPIECE_ROTATION:
            if self._workpieces:
                wp = random.choice(self._workpieces)
                new_angle = random.uniform(0, 360)
                wp.angle = new_angle
                logger.debug(
                    f"Rotated workpiece '{wp.name}' to {new_angle:.1f}°"
                )

        elif invalid_type == InvalidationType.INVALIDATE_JOB_ASSEMBLY:
            self.doc.job_assembly_invalidated.send(self.doc)
            logger.debug("Invalidated job assembly")

    async def run_chaos_phase(self) -> None:
        """Execute random invalidations for the chaos period."""
        start_time = time.time()
        invalidation_count = 0

        while time.time() - start_time < self.config.chaos_phase_sec:
            invalid_types = list(InvalidationType)
            invalid_type = random.choice(invalid_types)

            await self._execute_invalidation(invalid_type)
            invalidation_count += 1

            interval_ms = random.randint(
                self.config.min_invalidation_interval_ms,
                self.config.max_invalidation_interval_ms,
            )
            await asyncio.sleep(interval_ms / 1000.0)

        logger.info(
            f"Chaos phase complete: {invalidation_count} invalidations"
        )

    async def run_settle_phase(self) -> Optional[MetricsSnapshot]:
        """
        Wait for pipeline to settle and capture metrics.

        Returns None if settle timeout exceeded.
        """
        settled = await self.wait_for_settle()
        if not settled:
            logger.error("Pipeline failed to settle within timeout!")
            return None

        snapshot = self.capture_metrics()
        self._snapshots.append(snapshot)
        logger.info(
            f"Settle phase complete: SHM={snapshot.shm_block_count}, "
            f"refcount={snapshot.total_refcount}, "
            f"tasks={snapshot.active_task_count}, "
            f"busy={snapshot.is_busy}"
        )
        return snapshot

    def _get_artifact_breakdown(self) -> Dict[str, int]:
        """
        Get a breakdown of SHM blocks by artifact type.
        Calculates exact unique SHM names across all tracking registries
        to confidently identify untracked (leaked) blocks.
        """
        breakdown: Dict[str, int] = {}
        tracked_shms = set()

        # Tracked by ledger
        ledger = self.pipeline._artifact_manager._ledger
        for key, entry in ledger.items():
            if entry.handle is not None:
                tracked_shms.add(entry.handle.shm_name)

        # Tracked by step renders
        for (
            handle
        ) in self.pipeline._artifact_manager._step_render_handles.values():
            tracked_shms.add(handle.shm_name)

        # Tracked by step stage retained handles
        # Handles are now tracked by task ID in _retained_handles_by_task
        for (
            handles
        ) in self.pipeline._step_stage._retained_handles_by_task.values():
            for handle in handles:
                tracked_shms.add(handle.shm_name)

        # Tracked by ViewManager
        if self._view_manager is not None:
            vm = self._view_manager
            for entry in vm._view_entries.values():
                if entry.handle:
                    tracked_shms.add(entry.handle.shm_name)
                if entry.source_handle:
                    tracked_shms.add(entry.source_handle.shm_name)
            for handle in vm._source_artifact_handles.values():
                tracked_shms.add(handle.shm_name)
            for handle in vm._inflight_chunk_handles:
                tracked_shms.add(handle.shm_name)

        all_shms = set(self.artifact_store._handles.keys())
        untracked_shms = all_shms - tracked_shms

        breakdown["total_shm"] = len(all_shms)
        breakdown["tracked_unique"] = len(tracked_shms)
        breakdown["untracked"] = len(untracked_shms)

        if untracked_shms:
            logger.warning(
                f"Untracked SHM blocks ({len(untracked_shms)}): "
                f"{list(untracked_shms)[:20]}"
            )

            # Log holders for untracked blocks
            for name in list(untracked_shms)[:5]:
                handle = self.artifact_store._handles.get(name)
                if handle:
                    holders = handle.holders if handle.holders else []
                    logger.warning(
                        f"Handle {name}: refcount={handle.refcount}, "
                        f"all_holders={holders}"
                    )
                else:
                    logger.warning(f"Handle {name}: not in _handles dict")

            # Log ledger contents for debugging
            ledger_shm_names = set()
            for k, e in ledger.items():
                if e.handle is not None:
                    ledger_shm_names.add(e.handle.shm_name)
            logger.warning(f"Ledger SHM names: {sorted(ledger_shm_names)}")

            # Log which handles in ledger have matching refcounts
            for k, e in ledger.items():
                if e.handle is not None:
                    canonical = self.artifact_store._handles.get(
                        e.handle.shm_name
                    )
                    if canonical and canonical.refcount != e.handle.refcount:
                        logger.warning(
                            f"Refcount mismatch for {e.handle.shm_name}: "
                            f"ledger handle={e.handle.refcount}, "
                            f"canonical={canonical.refcount}"
                        )

            # Log all ViewManager tracking for debugging
            if self._view_manager is not None:
                vm = self._view_manager
                vm_view_handles = set()
                vm_source_handles = set()
                vm_chunk_handles = set()
                for entry in vm._view_entries.values():
                    if entry.handle:
                        vm_view_handles.add(entry.handle.shm_name)
                    if entry.source_handle:
                        vm_source_handles.add(entry.source_handle.shm_name)
                for h in vm._source_artifact_handles.values():
                    vm_source_handles.add(h.shm_name)
                for h in vm._inflight_chunk_handles:
                    vm_chunk_handles.add(h.shm_name)
                logger.warning(f"VM view handles: {sorted(vm_view_handles)}")
                logger.warning(
                    f"VM source handles: {sorted(vm_source_handles)}"
                )
                logger.warning(f"VM chunk handles: {sorted(vm_chunk_handles)}")

        return breakdown

    def check_for_leaks(self) -> bool:
        """
        Analyze metrics for memory leak patterns.

        Detects leaks primarily by checking for 'untracked' SHM blocks, which
        indicate handles that have been lost by the system (the original bug).

        Also monitors SHM growth, but since document size can fluctuate
        randomly in the chaos phase, continuous growth is not strictly a
        failure condition unless it's extreme or accompanied by untracked
        blocks.

        Returns True if leak detected, False otherwise.
        """
        breakdown = self._get_artifact_breakdown()

        # DEFINITIVE LEAK: Untracked blocks.
        # This means ArtifactStore has handles that no manager claims.
        if breakdown["untracked"] > self.config.leak_threshold:
            logger.error(
                f"Memory leak detected: {breakdown['untracked']} untracked "
                f"SHM blocks found! (Total: {breakdown['total_shm']})"
            )
            return True

        if len(self._snapshots) < 3:
            return False

        settle_snapshots = self._snapshots[1:]

        # Monitor growth for informational purposes/extreme bounds check
        growth_count = 0
        for i in range(1, len(settle_snapshots)):
            prev_shm = settle_snapshots[i - 1].shm_block_count
            curr_shm = settle_snapshots[i].shm_block_count
            growth = curr_shm - prev_shm
            if growth > self.config.leak_threshold:
                growth_count += 1
                logger.info(
                    f"SHM growth observed: {growth} blocks "
                    f"(prev={prev_shm}, curr={curr_shm})"
                )

        # If SHM count explodes beyond reasonable bounds for the document size
        # (max 5 workpieces + steps + job ~= 25-30 blocks active), flag it.
        # We use a safe upper bound of 100 to account for buffering/zombies.
        last_shm = settle_snapshots[-1].shm_block_count
        if last_shm > 100:
            logger.error(
                f"Potential leak detected: SHM count {last_shm} exceeds "
                "reasonable bound (100) for stress test parameters."
            )
            return True

        return False

    def check_for_state_corruption(self) -> bool:
        """
        Check for busy state corruption.

        Returns True if corruption detected, False otherwise.
        """
        for snapshot in self._snapshots:
            if not snapshot.is_busy and snapshot.active_task_count > 0:
                logger.error(
                    f"State corruption: is_busy=False but "
                    f"active_task_count={snapshot.active_task_count}"
                )
                return True
        return False

    async def run(self) -> bool:
        """
        Run the complete stress test cycle.

        Returns True if test passed, False if failures detected.
        """
        logger.info(
            f"Starting stress test: {self.config.total_duration_sec}s total, "
            f"{self.config.chaos_phase_sec}s chaos / "
            f"{self.config.settle_phase_sec}s settle cycles"
        )

        for i in range(self.config.initial_workpiece_count):
            wp = self._create_workpiece()
            self.doc.active_layer.add_workpiece(wp)
            self._workpieces.append(wp)

        if self.doc.active_layer.workflow:
            step = ContourStep.create(self.context)
            self.doc.active_layer.workflow.add_step(step)

        await asyncio.sleep(0.5)
        settled = await self.wait_for_settle()
        if not settled:
            logger.error("Initial settle failed!")
            return False

        baseline = self.capture_metrics()
        self._snapshots.append(baseline)
        logger.info(
            f"Baseline metrics: SHM={baseline.shm_block_count}, "
            f"refcount={baseline.total_refcount}, "
            f"nodes={baseline.dag_node_count}"
        )

        cycle = 0
        elapsed = 0.0
        test_passed = True

        while elapsed < self.config.total_duration_sec:
            cycle += 1
            logger.info(f"=== Cycle {cycle} ===")

            await self.run_chaos_phase()

            snapshot = await self.run_settle_phase()
            if snapshot is None:
                logger.error(f"Cycle {cycle}: Failed to settle!")
                test_passed = False
                break

            elapsed = time.time() - baseline.timestamp

        if test_passed:
            if self.check_for_leaks():
                test_passed = False
            if self.check_for_state_corruption():
                test_passed = False

        final = self.capture_metrics()
        logger.info(
            f"Final metrics: SHM={final.shm_block_count}, "
            f"refcount={final.total_refcount}, "
            f"nodes={final.dag_node_count}, "
            f"cycles={cycle}"
        )

        return test_passed


@pytest.mark.stress
@pytest.mark.asyncio
async def test_pipeline_stress_random_invalidations(
    task_mgr, context_initializer
):
    """
    Stress test the pipeline with random invalidations over 3 minutes.

    Black-box test that:
    - Creates a document with workpieces and steps
    - Runs chaos phases with random invalidations
    - Runs settle phases to check for leaks/hangs
    - Reports metrics at each phase boundary

    Verifies:
    1. No memory leaks (SHM blocks released after settle)
    2. No hung tasks (pipeline reaches idle state within timeout)
    3. No busy state corruption (is_busy reflects actual state)
    """
    doc = Doc()
    layer = doc.active_layer
    if layer.workflow:
        layer.workflow.set_steps([])

    pipeline = Pipeline(
        doc=doc,
        task_manager=task_mgr,
        artifact_store=context_initializer.artifact_store,
        machine=context_initializer.machine,
    )

    view_manager = ViewManager(
        pipeline=pipeline,
        artifact_store=context_initializer.artifact_store,
        machine=context_initializer.machine,
    )

    config = StressTestConfig()
    controller = StressTestController(
        config=config,
        pipeline=pipeline,
        doc=doc,
        task_mgr=task_mgr,
        artifact_store=context_initializer.artifact_store,
        context=context_initializer,
    )
    controller._view_manager = view_manager

    try:
        test_passed = await controller.run()
        assert test_passed, "Stress test failed - see logs for details"
    finally:
        view_manager.shutdown()
        pipeline.shutdown()
