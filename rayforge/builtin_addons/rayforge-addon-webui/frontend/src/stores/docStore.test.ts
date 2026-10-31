import { beforeEach, describe, expect, it } from "vitest";
import type { DocTree } from "../api/types";
import {
  docStore,
  geometryStale,
  planDocEvent,
  reorderLayersLocal,
  selectUids,
  setActiveLayerLocal,
  setDocHistory,
  setDocTree,
  setLayerLocal,
} from "./docStore";

function makeTree(): DocTree {
  return {
    rev: 3,
    layers: [
      {
        uid: "layer-1",
        name: "Cut",
        visible: true,
        color: "#4d9fff",
        is_active: true,
        wcs: "G54",
        steps: [],
        workpieces: [
          {
            uid: "wp-1",
            name: "Square",
            pos_mm: { x: 10, y: 20 },
            size_mm: { w: 30, h: 40 },
            angle_deg: 0,
            geometry_revision: 5,
            transform_revision: 2,
          },
        ],
      },
    ],
    assets_count: 0,
    stock_items: [],
  };
}

beforeEach(() => {
  docStore.setState({
    loaded: false,
    rev: -1,
    layers: [],
    assetsCount: 0,
    selection: [],
    expanded: [],
    canUndo: false,
    canRedo: false,
    dirty: false,
    undoLabels: [],
    redoLabels: [],
  });
});

describe("planDocEvent", () => {
  it("refetches when the event carries no tree revision", () => {
    const plan = planDocEvent(
      { type: "doc", kind: "changed", uids: [], revisions: {} },
      3,
    );
    expect(plan.refetchTree).toBe(true);
    expect(plan.refetchHistory).toBe(true);
  });

  it("skips refetch when the revision is already known", () => {
    const plan = planDocEvent(
      { type: "doc", kind: "changed", uids: ["wp-1"], revisions: { tree: 3 } },
      3,
    );
    expect(plan.refetchTree).toBe(false);
  });

  it("refetches when the revision advanced", () => {
    const plan = planDocEvent(
      { type: "doc", kind: "commit", uids: ["wp-1"], revisions: { tree: 4 } },
      3,
    );
    expect(plan.refetchTree).toBe(true);
    expect(plan.changedUids).toEqual(["wp-1"]);
  });
});

describe("geometryStale", () => {
  it("is true for cache misses", () => {
    expect(geometryStale(undefined, makeTree().layers[0]!.workpieces[0]!)).toBe(
      true,
    );
  });

  it("is false when the revision matches", () => {
    expect(geometryStale(5, makeTree().layers[0]!.workpieces[0]!)).toBe(false);
  });

  it("is true when the revision changed", () => {
    expect(geometryStale(4, makeTree().layers[0]!.workpieces[0]!)).toBe(true);
  });
});

describe("selection handling", () => {
  it("replaces the selection for non-additive selects", () => {
    selectUids(["a"]);
    selectUids(["b"]);
    expect(docStore.getState().selection).toEqual(["b"]);
  });

  it("toggles membership for additive selects", () => {
    selectUids(["a"]);
    selectUids(["b"], true);
    expect([...docStore.getState().selection].sort()).toEqual(["a", "b"]);
    selectUids(["a"], true);
    expect(docStore.getState().selection).toEqual(["b"]);
  });

  it("prunes stale uids when a new tree arrives", () => {
    setDocTree(makeTree());
    selectUids(["wp-1", "wp-gone"]);
    const tree = makeTree();
    tree.rev = 9;
    setDocTree(tree);
    expect(docStore.getState().selection).toEqual(["wp-1"]);
  });
});

describe("layer mutations", () => {
  function makeTwoLayers(): void {
    const tree = makeTree();
    tree.layers.push({
      uid: "layer-2",
      name: "Mark",
      visible: true,
      color: "#ff9f40",
      is_active: false,
      wcs: null,
      steps: [],
      workpieces: [],
    });
    setDocTree(tree);
  }

  it("applies partial patches to one layer", () => {
    makeTwoLayers();
    setLayerLocal("layer-1", { name: "Cut rename", color: "#123456" });
    const state = docStore.getState();
    expect(state.layers[0]!.name).toBe("Cut rename");
    expect(state.layers[0]!.color).toBe("#123456");
    expect(state.layers[1]!.name).toBe("Mark");
  });

  it("reorders layers optimistically, keeping unknown uids", () => {
    makeTwoLayers();
    reorderLayersLocal(["layer-2", "layer-1"]);
    expect(docStore.getState().layers.map((l) => l.uid)).toEqual([
      "layer-2",
      "layer-1",
    ]);
    reorderLayersLocal(["layer-2", "ghost"]);
    expect(docStore.getState().layers.map((l) => l.uid)).toEqual([
      "layer-2",
      "layer-1",
    ]);
  });

  it("marks a single layer active", () => {
    makeTwoLayers();
    setActiveLayerLocal("layer-2");
    const state = docStore.getState();
    expect(state.layers.map((l) => l.is_active)).toEqual([false, true]);
    setActiveLayerLocal("layer-1");
    expect(docStore.getState().layers.map((l) => l.is_active)).toEqual([
      true,
      false,
    ]);
  });
});

describe("history state", () => {
  it("stores the /doc/history payload", () => {
    setDocHistory({
      can_undo: true,
      can_redo: true,
      undo_labels: ["Import shape.svg", "Transform item"],
      redo_labels: ["Delete workpiece"],
      dirty: true,
    });
    const state = docStore.getState();
    expect(state.canUndo).toBe(true);
    expect(state.canRedo).toBe(true);
    expect(state.undoLabels).toEqual([
      "Import shape.svg",
      "Transform item",
    ]);
    expect(state.redoLabels).toEqual(["Delete workpiece"]);
    expect(state.dirty).toBe(true);
  });

  it("refetches history for every doc event", () => {
    const plan = planDocEvent(
      { type: "doc", kind: "history", uids: [], revisions: {} },
      -1,
    );
    expect(plan.refetchHistory).toBe(true);
  });
});
