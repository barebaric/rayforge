import type { ApiBackend } from "../api/client";
import type { AlignMode, FlipAxis } from "../api/types";
import { docStore, findWorkpiece, selectUids } from "./docStore";

export interface DocActionsDeps {
  api: Pick<
    ApiBackend,
    | "deleteItems"
    | "duplicateItems"
    | "flipItems"
    | "alignItems"
    | "postUndo"
    | "postRedo"
  >;
  refreshDoc: () => Promise<void>;
  onError: (message: string) => void;
}

/** True when keyboard shortcuts should not fire (typing in a field). */
export function isTextInputTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target.isContentEditable
  );
}

/**
 * Selection-scoped document commands (delete/duplicate/flip/align and
 * undo/redo) plus the keyboard dispatch driving them.
 */
export class DocActions {
  constructor(private readonly deps: DocActionsDeps) {}

  private selection(): string[] {
    return docStore.getState().selection.filter((uid) =>
      findWorkpiece(docStore.getState().layers, uid),
    );
  }

  private run(action: () => Promise<unknown>, label: string): void {
    action()
      .then(() => this.deps.refreshDoc())
      .catch((err: unknown) => {
        this.deps.onError(`${label} failed: ${describe(err)}`);
      });
  }

  deleteSelected(): void {
    const uids = this.selection();
    if (uids.length === 0) return;
    this.run(() => this.deps.api.deleteItems(uids), "Delete");
  }

  duplicateSelected(): void {
    const uids = this.selection();
    if (uids.length === 0) return;
    this.run(async () => {
      const result = await this.deps.api.duplicateItems(uids);
      if (result.created.length > 0) {
        selectUids(result.created);
      }
      return result;
    }, "Duplicate");
  }

  flipSelected(axis: FlipAxis): void {
    const uids = this.selection();
    if (uids.length === 0) return;
    this.run(() => this.deps.api.flipItems(uids, axis), "Flip");
  }

  alignSelected(mode: AlignMode): void {
    const uids = this.selection();
    if (uids.length === 0) return;
    this.run(() => this.deps.api.alignItems(uids, mode), "Align");
  }

  undo(): void {
    if (!docStore.getState().canUndo) return;
    this.run(() => this.deps.api.postUndo(), "Undo");
  }

  redo(): void {
    if (!docStore.getState().canRedo) return;
    this.run(() => this.deps.api.postRedo(), "Redo");
  }

  /** Keyboard shortcuts: Delete, Ctrl+D, Ctrl+Z/Ctrl+Shift+Z/Ctrl+Y. */
  handleKeydown(ev: KeyboardEvent): void {
    if (isTextInputTarget(ev.target)) return;
    if (ev.ctrlKey || ev.metaKey) {
      const key = ev.key.toLowerCase();
      if (key === "z" && ev.shiftKey) {
        ev.preventDefault();
        this.redo();
      } else if (key === "z") {
        ev.preventDefault();
        this.undo();
      } else if (key === "y") {
        ev.preventDefault();
        this.redo();
      } else if (key === "d") {
        ev.preventDefault();
        this.duplicateSelected();
      }
      return;
    }
    if (ev.key === "Delete" || ev.key === "Backspace") {
      ev.preventDefault();
      this.deleteSelected();
    }
  }
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
