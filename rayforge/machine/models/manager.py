import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from blinker import Signal

from ...context import get_context
from ...shared.tasker import task_mgr
from ..driver.driver import ResourceBusyError
from .machine import Machine


logger = logging.getLogger(__name__)


class MachineManager:
    def __init__(self, base_dir: Path):
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self.machines: Dict[str, Machine] = dict()
        self._machine_ref_for_pyreverse: Machine
        self.machine_added = Signal()
        self.machine_removed = Signal()
        self.machine_updated = Signal()
        self.load()

    async def shutdown(self):
        """
        Shuts down all managed machines and their drivers gracefully.
        """
        logger.info("Shutting down all machines.")
        tasks = [machine.shutdown() for machine in self.machines.values()]
        if tasks:
            await asyncio.gather(*tasks)
        logger.info("All machines shut down.")

    def initialize_connections(self):
        """
        Initializes machine connections on startup by rebuilding all drivers
        and then connecting, prioritizing the active machine.
        """
        context = get_context()
        active_machine = context.config.machine

        # Define a lambda to use with add_coroutine that captures the machine
        def connect_task(m):
            return lambda ctx: self._rebuild_and_connect_machine(m)

        # First, schedule the task for the active machine
        if active_machine:
            task_mgr.add_coroutine(connect_task(active_machine))

        # Then, schedule tasks for the rest
        for machine in self.machines.values():
            if machine is not active_machine:
                task_mgr.add_coroutine(connect_task(machine))

    async def _rebuild_and_connect_machine(self, machine: "Machine"):
        """
        A single, sequenced task that rebuilds a machine's driver and then
        connects if auto_connect is on.
        """
        await machine._rebuild_driver_instance()
        if machine.auto_connect:
            await self._safe_connect(machine)

    async def _safe_connect(self, machine: "Machine"):
        """
        Attempts to connect a machine, suppressing ResourceBusyErrors.
        """
        try:
            await machine.connect()
        except ResourceBusyError:
            context = get_context()
            if machine is context.config.machine:
                logger.warning(
                    f"Active machine '{machine.name}' could not connect "
                    "because resource is busy."
                )
            else:
                logger.debug(
                    f"Inactive machine '{machine.name}' deferred connection: "
                    "resource busy."
                )
        except Exception as e:
            logger.error(
                f"Failed to auto-connect machine '{machine.name}': {e}"
            )

    def set_active_machine(self, new_machine: Machine):
        """
        Sets the active machine, handling the connection lifecycle for
        shared resources.
        """
        context = get_context()
        old_machine = context.config.machine

        if old_machine and old_machine.id == new_machine.id:
            return  # No change

        logger.info(f"Switching active machine to '{new_machine.name}'")

        async def switch_routine(ctx):
            # 1. Disconnect the old machine if it's connected
            if old_machine and old_machine.is_connected():
                logger.info(
                    f"Disconnecting previous machine '{old_machine.name}'"
                )
                await old_machine.disconnect()
                # Add a small delay for the OS to release the port
                await asyncio.sleep(0.2)

            # 2. Update the global config. This triggers UI updates.
            context.config.set_machine(new_machine)

            # 3. Connect the new machine if it's set to auto-connect
            if new_machine.auto_connect:
                logger.info(
                    f"Connecting to new active machine '{new_machine.name}'"
                )
                await self._safe_connect(new_machine)

        task_mgr.add_coroutine(switch_routine)

    def filename_from_id(self, machine_id: str) -> Path:
        return self.base_dir / f"{machine_id}.yaml"

    def add_machine(self, machine: Machine):
        if machine.id in self.machines:
            return
        self.machines[machine.id] = machine
        machine.changed.connect(self.on_machine_changed)
        self.save_machine(machine)
        self.machine_added.send(self, machine_id=machine.id)

    def remove_machine(self, machine_id: str):
        machine = self.machines.get(machine_id)
        if not machine:
            return

        machine.changed.disconnect(self.on_machine_changed)
        del self.machines[machine_id]

        machine_file = self.filename_from_id(machine_id)
        try:
            machine_file.unlink()
            logger.info(f"Removed machine file: {machine_file}")
        except OSError as e:
            logger.error(f"Error removing machine file {machine_file}: {e}")

        self.machine_removed.send(self, machine_id=machine_id)

    def get_machine_by_id(self, machine_id):
        return self.machines.get(machine_id)

    def get_machines(self) -> List["Machine"]:
        """Returns a list of all managed machines, sorted by name."""
        return sorted(list(self.machines.values()), key=lambda m: m.name)

    def create_default_machine(self):
        machine = Machine(get_context())
        self.add_machine(machine)
        return machine

    def save_machine(self, machine):
        logger.debug(f"Saving machine {machine.id}")
        machine_file = self.filename_from_id(machine.id)
        with open(machine_file, "w") as f:
            data = machine.to_dict(include_frozen_dialect=False)
            yaml.safe_dump(data, f)

    def load_machine(self, machine_id: str) -> Optional["Machine"]:
        machine_file = self.filename_from_id(machine_id)
        if not machine_file.exists():
            raise FileNotFoundError(f"Machine file {machine_file} not found")
        with open(machine_file, "r") as f:
            data = yaml.safe_load(f)
            if not data:
                msg = f"skipping invalid machine file {f.name}"
                logger.warning(msg)
                return None
        machine = Machine.from_dict(data)
        machine.id = machine_id
        self.machines[machine.id] = machine
        machine.changed.connect(self.on_machine_changed)
        return machine

    def on_machine_changed(self, machine, **kwargs):
        self.save_machine(machine)
        self.machine_updated.send(self, machine_id=machine.id)

    def load(self):
        for file in self.base_dir.glob("*.yaml"):
            try:
                self.load_machine(file.stem)
            except Exception as e:
                logger.error(f"Failed to load machine from {file}: {e}")
