import type { ImportAnalysis, ImportPlacement, ImportResult } from "../api/types";
import { describeError } from "./describe";

export type ImportPhase =
  | "idle"
  | "analyzing"
  | "direct_load"
  | "interactive_config"
  | "unsupported"
  | "importing"
  | "done"
  | "error";

export interface ImportFlowState {
  phase: ImportPhase;
  file: File | null;
  analysis: ImportAnalysis | null;
  result: ImportResult | null;
  error: string | null;
}

export const initialImportState: ImportFlowState = {
  phase: "idle",
  file: null,
  analysis: null,
  result: null,
  error: null,
};

/** Pure transition: file chosen, analyzer answered. */
export function analysisReceived(
  state: ImportFlowState,
  analysis: ImportAnalysis,
): ImportFlowState {
  const phase: ImportPhase =
    !analysis.supported || analysis.action === "unsupported"
      ? "unsupported"
      : analysis.action;
  return { ...state, phase, analysis, error: null };
}

/** Pure transition: the import POST completed. */
export function importFinished(
  state: ImportFlowState,
  result: ImportResult,
): ImportFlowState {
  return { ...state, phase: "done", result, error: null };
}

/** Pure transition: any step failed. */
export function importFailed(
  state: ImportFlowState,
  error: unknown,
): ImportFlowState {
  return { ...state, phase: "error", error: describeError(error) };
}

export interface ImportFlowDeps {
  analyze: (file: File) => Promise<ImportAnalysis>;
  run: (file: File, placement: ImportPlacement) => Promise<ImportResult>;
}

/**
 * Drives the analyze → (optional config) → import sequence. UI layers
 * subscribe to state changes; the transitions themselves are pure
 * functions above.
 */
export class ImportFlow {
  private state: ImportFlowState = initialImportState;

  constructor(private readonly deps: ImportFlowDeps) {}

  get current(): ImportFlowState {
    return this.state;
  }

  async start(file: File): Promise<ImportFlowState> {
    this.state = { ...initialImportState, phase: "analyzing", file };
    try {
      const analysis = await this.deps.analyze(file);
      this.state = analysisReceived(this.state, analysis);
    } catch (err) {
      this.state = importFailed(this.state, err);
    }
    return this.state;
  }

  async run(placement: ImportPlacement = {}): Promise<ImportFlowState> {
    const { file } = this.state;
    if (!file || this.state.phase === "importing") return this.state;
    this.state = { ...this.state, phase: "importing", error: null };
    try {
      const result = await this.deps.run(file, placement);
      this.state = importFinished(this.state, result);
    } catch (err) {
      this.state = importFailed(this.state, err);
    }
    return this.state;
  }
}
