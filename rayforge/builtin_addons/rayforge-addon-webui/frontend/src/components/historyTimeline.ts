import { docStore, setRolling } from "../stores/docStore";
import type { ApiBackend } from "../api/client";
import { button } from "./ui/Button";
import { el } from "./ui/dom";

/** Hard cap on sequential undo/redo requests for one roll. */
export const MAX_ROLL_STEPS = 200;
export const CHIP_LABEL_MAX = 14;

export interface RollPlan {
  undos: number;
  redos: number;
}

/**
 * Plan to move the playhead to an absolute history position: the
 * number of executed commands from the very start of the recorded
 * history, ranging over [0 .. undoCount + redoCount]. The caller
 * caps the returned step counts before executing them.
 */
export function planRollToPosition(
  position: number,
  undoCount: number,
): RollPlan {
  const clamped = Math.max(0, position);
  return {
    undos: Math.max(0, undoCount - clamped),
    redos: Math.max(0, clamped - undoCount),
  };
}

/**
 * Plan to make the chip at `chipIndex` of the undo stack the last
 * executed command. Redo chips continue the numbering after the undo
 * chips. Returns null when the chip does not exist.
 */
export function planRollToChip(
  undoLabels: string[],
  redoLabels: string[],
  chipIndex: number,
): RollPlan | null {
  const total = undoLabels.length + redoLabels.length;
  if (chipIndex < 0 || chipIndex >= total) return null;
  return planRollToPosition(chipIndex + 1, undoLabels.length);
}

/** Truncates a command label for chip display. */
export function chipLabel(label: string, max = CHIP_LABEL_MAX): string {
  if (label.length <= max) return label;
  return `${label.slice(0, max - 1)}…`;
}

/** Index of the chip boundary (history position) nearest to `x`. */
export function snapToBoundary(
  x: number,
  boundaries: number[],
): number {
  let best = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < boundaries.length; i++) {
    const distance = Math.abs(boundaries[i]! - x);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = i;
    }
  }
  return best;
}

export interface HistoryTimelineDeps {
  api: Pick<ApiBackend, "postUndo" | "postRedo">;
  onRollStart: () => void;
  onRollEnd: () => void;
  onError: (message: string) => void;
}

export interface HistoryTimelineRefs {
  root: HTMLElement;
}

/**
 * Solidworks-style horizontal history bar docked above the status
 * chip: jump buttons, one chip per history entry, and a round handle
 * marking the current position. Undo entries are solid chips; redo
 * entries render faded after the handle. Dragging the handle or
 * clicking a chip rolls the document to that point via sequential
 * undo/redo requests.
 */
export function buildHistoryTimeline(
  deps: HistoryTimelineDeps,
): HistoryTimelineRefs {
  const root = el("div", "history-timeline");
  const title = el("span", "history-label");
  title.textContent = "History";

  const jumpStartBtn = button({
    iconName: "stepBack",
    variant: "icon",
    title: "Jump to start (undo everything)",
    onClick: () => void roll(planRollToPosition(0, undoCount())),
  });
  const stepBackBtn = button({
    iconName: "chevronLeft",
    variant: "icon",
    title: "Step back (undo)",
    onClick: () => void roll({ undos: 1, redos: 0 }),
  });
  const stepForwardBtn = button({
    iconName: "chevronRight",
    variant: "icon",
    title: "Step forward (redo)",
    onClick: () => void roll({ undos: 0, redos: 1 }),
  });
  const jumpEndBtn = button({
    iconName: "stepForward",
    variant: "icon",
    title: "Jump to end (redo everything)",
    onClick: () =>
      void roll(planRollToPosition(totalCount(), undoCount())),
  });

  const chips = el("div", "history-chips");
  const handle = el("div", "history-handle");
  handle.title = "Drag to roll the document back or forward";

  let rolling = false;
  let dragBoundaries: number[] = [];

  const undoCount = () => docStore.getState().undoLabels.length;
  const totalCount = () => {
    const state = docStore.getState();
    return state.undoLabels.length + state.redoLabels.length;
  };

  chips.appendChild(handle);

  function renderChips(): void {
    const state = docStore.getState();
    const pieces: HTMLElement[] = [];
    state.undoLabels.forEach((label, index) => {
      const chip = el("button", "history-chip");
      chip.type = "button";
      chip.textContent = chipLabel(label);
      chip.title = label;
      chip.addEventListener("click", () => {
        const plan = planRollToChip(
          state.undoLabels,
          state.redoLabels,
          index,
        );
        if (plan) void roll(plan);
      });
      pieces.push(chip);
    });
    state.redoLabels.forEach((label, index) => {
      const chip = el("button", "history-chip redo");
      chip.type = "button";
      chip.textContent = chipLabel(label);
      chip.title = `Redo: ${label}`;
      chip.addEventListener("click", () => {
        const plan = planRollToChip(
          state.undoLabels,
          state.redoLabels,
          state.undoLabels.length + index,
        );
        if (plan) void roll(plan);
      });
      pieces.push(chip);
    });
    for (const child of [...chips.children]) {
      if (child !== handle) child.remove();
    }
    chips.append(...pieces);
    positionHandle();
  }

  /** Left offset (px, relative to the chip strip) of each boundary. */
  function measureBoundaries(): number[] {
    const chipEls = Array.from(
      chips.querySelectorAll(".history-chip"),
    ) as HTMLElement[];
    const positions: number[] = [0];
    for (const chipEl of chipEls) {
      positions.push(chipEl.offsetLeft + chipEl.offsetWidth);
    }
    return positions;
  }

  function positionHandle(): void {
    const state = docStore.getState();
    dragBoundaries = measureBoundaries();
    const position = state.undoLabels.length;
    const left =
      position === 0
        ? dragBoundaries[0]!
        : dragBoundaries[
            Math.min(position, dragBoundaries.length - 1)
          ];
    handle.style.left = `${left}px`;
    handle.classList.toggle("dragging", false);
  }

  function syncDisabled(): void {
    const state = docStore.getState();
    const busy = rolling;
    jumpStartBtn.disabled = busy || !state.canUndo;
    stepBackBtn.disabled = busy || !state.canUndo;
    stepForwardBtn.disabled = busy || !state.canRedo;
    jumpEndBtn.disabled = busy || !state.canRedo;
    for (const chipEl of Array.from(
      chips.querySelectorAll(".history-chip"),
    ) as HTMLElement[]) {
      (chipEl as HTMLButtonElement).disabled = busy;
    }
    root.classList.toggle("rolling", busy);
  }

  async function roll(plan: RollPlan): Promise<void> {
    if (rolling) return;
    const undos = Math.min(plan.undos, MAX_ROLL_STEPS);
    const redos = Math.min(plan.redos, MAX_ROLL_STEPS);
    if (undos === 0 && redos === 0) return;
    rolling = true;
    setRolling(true);
    syncDisabled();
    deps.onRollStart();
    try {
      for (let i = 0; i < undos; i++) await deps.api.postUndo();
      for (let i = 0; i < redos; i++) await deps.api.postRedo();
    } catch (err) {
      deps.onError(
        `History roll failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      rolling = false;
      setRolling(false);
      syncDisabled();
      deps.onRollEnd();
    }
  }

  handle.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    handle.setPointerCapture(ev.pointerId);
    handle.classList.add("dragging");
  });
  handle.addEventListener("pointermove", (ev) => {
    if (!handle.classList.contains("dragging")) return;
    const rect = chips.getBoundingClientRect();
    const position = snapToBoundary(
      ev.clientX - rect.left,
      dragBoundaries,
    );
    handle.style.left = `${dragBoundaries[position] ?? 0}px`;
  });
  const endDrag = (ev: PointerEvent) => {
    if (!handle.classList.contains("dragging")) return;
    handle.classList.remove("dragging");
    const rect = chips.getBoundingClientRect();
    const position = snapToBoundary(
      ev.clientX - rect.left,
      dragBoundaries,
    );
    const state = docStore.getState();
    const plan = planRollToPosition(position, state.undoLabels.length);
    void roll(plan);
  };
  handle.addEventListener("pointerup", endDrag);
  handle.addEventListener("pointercancel", endDrag);

  root.append(
    title,
    jumpStartBtn,
    stepBackBtn,
    stepForwardBtn,
    jumpEndBtn,
    chips,
  );

  docStore.subscribe(() => {
    renderChips();
    syncDisabled();
  });
  syncDisabled();

  return { root };
}
