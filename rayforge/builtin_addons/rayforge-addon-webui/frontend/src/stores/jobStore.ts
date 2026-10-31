import { createStore } from "zustand/vanilla";
import type { JobStatus } from "../api/types";

interface JobStoreState {
  active: boolean;
  phase: string;
  progressFraction: number;
  etaSeconds: number | null;
  elapsedSeconds: number | null;
}

export const jobStore = createStore<JobStoreState>()(() => ({
  active: false,
  phase: "idle",
  progressFraction: 0,
  etaSeconds: null,
  elapsedSeconds: null,
}));

export function setJobStatus(status: JobStatus): void {
  jobStore.setState({
    active: status.active,
    phase: status.phase,
    progressFraction: status.progress_fraction,
    etaSeconds: status.eta_seconds,
    elapsedSeconds: status.elapsed_seconds,
  });
}
