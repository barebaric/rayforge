import { beforeEach, describe, expect, it } from "vitest";
import type { ApiBackend } from "../api/client";
import type { DocTree } from "../api/types";
import { docStore, setDocHistory, setDocTree } from "./docStore";
import { DocActions, isTextInputTarget } from "./docActions";

interface Calls {
  deleteItems: string[][];
  duplicateItems: string[][];
  flipItems: Array<{ uids: string[]; axis: string }>;
  alignItems: Array<{ uids: string[]; mode: string }>;
  undo: number;
  redo: number;
}

function makeCalls(): Calls {
  return {
    deleteItems: [],
    duplicateItems: [],
    flipItems: [],
    alignItems: [],
    undo: 0,
    redo: 0,
  };
}

function makeApi(calls: Calls): Pick<
  ApiBackend,
  | "deleteItems"
  | "duplicateItems"
  | "flipItems"
  | "alignItems"
  | "postUndo"
  | "postRedo"
> {
  return {
    async deleteItems(uids) {
      calls.deleteItems.push(uids);
    },
    async duplicateItems(uids) {
      calls.duplicateItems.push(uids);
      return { created: uids.map((uid) => `${uid}-copy`) };
    },
    async flipItems(uids, axis) {
      calls.flipItems.push({ uids, axis });
    },
    async alignItems(uids, mode) {
      calls.alignItems.push({ uids, mode });
    },
    async postUndo() {
      calls.undo++;
    },
    async postRedo() {
      calls.redo++;
    },
  };
}

function makeTree(): DocTree {
  return {
    rev: 1,
    layers: [
      {
        uid: "layer-1",
        name: "Cut",
        visible: true,
        color: "#4d9fff",
        is_active: true,
        wcs: null,
        steps: [],
        workpieces: [
          {
            uid: "wp-1",
            name: "Square",
            pos_mm: { x: 10, y: 10 },
            size_mm: { w: 20, h: 20 },
            angle_deg: 0,
            geometry_revision: 1,
            transform_revision: 1,
          },
          {
            uid: "wp-2",
            name: "Circle",
            pos_mm: { x: 60, y: 10 },
            size_mm: { w: 20, h: 20 },
            angle_deg: 0,
            geometry_revision: 1,
            transform_revision: 1,
          },
        ],
      },
    ],
    assets_count: 0,
    stock_items: [],
  };
}

function makeActions(calls: Calls): {
  actions: DocActions;
  refreshed: () => number;
  errors: string[];
} {
  let refreshCount = 0;
  const errors: string[] = [];
  const actions = new DocActions({
    api: makeApi(calls),
    refreshDoc: () => {
      refreshCount++;
      return Promise.resolve();
    },
    onError: (message) => errors.push(message),
  });
  return { actions, refreshed: () => refreshCount, errors };
}

function keydown(
  target: EventTarget | null,
  init: KeyboardEventInit,
): KeyboardEvent {
  const ev = new KeyboardEvent("keydown", { ...init, cancelable: true });
  if (target) {
    Object.defineProperty(ev, "target", { value: target });
  }
  return ev;
}

beforeEach(() => {
  setDocTree(makeTree());
  setDocHistory({
    can_undo: true,
    can_redo: true,
    undo_labels: ["Transform item"],
    redo_labels: [],
    dirty: true,
  });
});

describe("DocActions", () => {
  it("deletes the selected workpieces and refreshes", async () => {
    const calls = makeCalls();
    const { actions, refreshed } = makeActions(calls);
    docStore.setState({ selection: ["wp-1", "wp-2"] });
    actions.deleteSelected();
    await Promise.resolve();
    await Promise.resolve();
    expect(calls.deleteItems).toEqual([["wp-1", "wp-2"]]);
    expect(refreshed()).toBe(1);
  });

  it("ignores commands without a workpiece selection", async () => {
    const calls = makeCalls();
    const { actions, refreshed } = makeActions(calls);
    docStore.setState({ selection: ["layer-1"] });
    actions.deleteSelected();
    actions.duplicateSelected();
    actions.flipSelected("horizontal");
    actions.alignSelected("left");
    expect(calls.deleteItems).toEqual([]);
    expect(calls.duplicateItems).toEqual([]);
    expect(calls.flipItems).toEqual([]);
    expect(calls.alignItems).toEqual([]);
    expect(refreshed()).toBe(0);
  });

  it("skips non-workpiece uids in the selection", async () => {
    const calls = makeCalls();
    const { actions } = makeActions(calls);
    docStore.setState({ selection: ["wp-1", "step-x"] });
    actions.deleteSelected();
    await Promise.resolve();
    expect(calls.deleteItems).toEqual([["wp-1"]]);
  });

  it("selects the created copies after duplicating", async () => {
    const calls = makeCalls();
    const { actions } = makeActions(calls);
    docStore.setState({ selection: ["wp-1"] });
    actions.duplicateSelected();
    await Promise.resolve();
    await Promise.resolve();
    expect(calls.duplicateItems).toEqual([["wp-1"]]);
    expect(docStore.getState().selection).toEqual(["wp-1-copy"]);
  });

  it("passes flip axis and align mode through", async () => {
    const calls = makeCalls();
    const { actions } = makeActions(calls);
    docStore.setState({ selection: ["wp-1", "wp-2"] });
    actions.flipSelected("vertical");
    actions.alignSelected("center_horizontal");
    await Promise.resolve();
    expect(calls.flipItems).toEqual([
      { uids: ["wp-1", "wp-2"], axis: "vertical" },
    ]);
    expect(calls.alignItems).toEqual([
      { uids: ["wp-1", "wp-2"], mode: "center_horizontal" },
    ]);
  });

  it("gates undo/redo on the history state", async () => {
    const calls = makeCalls();
    const { actions } = makeActions(calls);
    actions.undo();
    actions.redo();
    expect(calls.undo).toBe(1);
    expect(calls.redo).toBe(1);
    setDocHistory({
      can_undo: false,
      can_redo: false,
      undo_labels: [],
      redo_labels: [],
      dirty: false,
    });
    actions.undo();
    actions.redo();
    expect(calls.undo).toBe(1);
    expect(calls.redo).toBe(1);
  });

  it("reports failures through onError", async () => {
    const errors: string[] = [];
    const actions = new DocActions({
      api: {
        ...makeApi(makeCalls()),
        async deleteItems() {
          throw new Error("boom");
        },
      },
      refreshDoc: () => Promise.resolve(),
      onError: (message) => errors.push(message),
    });
    docStore.setState({ selection: ["wp-1"] });
    actions.deleteSelected();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(errors).toEqual(["Delete failed: boom"]);
  });
});

describe("DocActions keyboard dispatch", () => {
  it("deletes on the Delete key", async () => {
    const calls = makeCalls();
    const { actions } = makeActions(calls);
    docStore.setState({ selection: ["wp-1"] });
    const ev = keydown(null, { key: "Delete" });
    actions.handleKeydown(ev);
    expect(ev.defaultPrevented).toBe(true);
    await Promise.resolve();
    expect(calls.deleteItems).toEqual([["wp-1"]]);
  });

  it("duplicates on Ctrl+D", async () => {
    const calls = makeCalls();
    const { actions } = makeActions(calls);
    docStore.setState({ selection: ["wp-1"] });
    const ev = keydown(null, { key: "d", ctrlKey: true });
    actions.handleKeydown(ev);
    expect(ev.defaultPrevented).toBe(true);
    await Promise.resolve();
    expect(calls.duplicateItems).toEqual([["wp-1"]]);
  });

  it("dispatches undo/redo with and without shift", async () => {
    const calls = makeCalls();
    const { actions } = makeActions(calls);
    actions.handleKeydown(keydown(null, { key: "z", ctrlKey: true }));
    actions.handleKeydown(
      keydown(null, { key: "Z", ctrlKey: true, shiftKey: true }),
    );
    actions.handleKeydown(keydown(null, { key: "y", ctrlKey: true }));
    expect(calls.undo).toBe(1);
    expect(calls.redo).toBe(2);
  });

  it("ignores keys typed into a text field", async () => {
    const calls = makeCalls();
    const { actions } = makeActions(calls);
    docStore.setState({ selection: ["wp-1"] });
    const input = document.createElement("input");
    for (const key of ["Delete", "Backspace"]) {
      actions.handleKeydown(keydown(input, { key }));
    }
    actions.handleKeydown(
      keydown(input, { key: "d", ctrlKey: true }),
    );
    actions.handleKeydown(keydown(input, { key: "z", ctrlKey: true }));
    expect(calls.deleteItems).toEqual([]);
    expect(calls.duplicateItems).toEqual([]);
    expect(calls.undo).toBe(0);
  });
});

describe("isTextInputTarget", () => {
  it("detects inputs, textareas and contenteditable", () => {
    expect(isTextInputTarget(null)).toBe(false);
    expect(isTextInputTarget(document.createElement("div"))).toBe(false);
    const input = document.createElement("input");
    expect(isTextInputTarget(input)).toBe(true);
    expect(isTextInputTarget(document.createElement("textarea"))).toBe(true);
    const editable = document.createElement("div");
    editable.contentEditable = "true";
    expect(isTextInputTarget(editable)).toBe(true);
  });
});
