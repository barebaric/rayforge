import { createStore } from "zustand/vanilla";
import type {
  MachineListItem,
  MachineState,
  ViewportContract,
} from "../api/types";

interface MachineStoreState {
  machines: MachineListItem[];
  viewport: ViewportContract | null;
  machineState: MachineState | null;
  activeMachineId: string | null;
}

export const machineStore = createStore<MachineStoreState>()(() => ({
  machines: [],
  viewport: null,
  machineState: null,
  activeMachineId: null,
}));

export function setViewport(vp: ViewportContract): void {
  machineStore.setState({ viewport: vp });
}

export function setMachines(machines: MachineListItem[]): void {
  machineStore.setState((state) => ({
    machines,
    activeMachineId:
      state.activeMachineId ??
      (machines.find((m) => m.is_active)?.id ?? machines[0]?.id ?? null),
  }));
}

export function setMachineState(state: MachineState): void {
  machineStore.setState({ machineState: state });
}

export function setActiveMachine(id: string): void {
  machineStore.setState({
    activeMachineId: id,
    machines: machineStore
      .getState()
      .machines.map((m) => ({ ...m, is_active: m.id === id })),
  });
}
