import { createStore } from "zustand/vanilla";
import type {
  DocHistory,
  DocTree,
  DocWsEvent,
  LayerInfo,
  StepInfo,
  StockInfo,
  WorkpieceInfo,
} from "../api/types";

export interface DocState {
  loaded: boolean;
  rev: number;
  layers: LayerInfo[];
  stocks: StockInfo[];
  assetsCount: number;
  selection: string[];
  expanded: string[];
  canUndo: boolean;
  canRedo: boolean;
  dirty: boolean;
  undoLabels: string[];
  redoLabels: string[];
  /** True while the history timeline is replaying undo/redo steps. */
  rolling: boolean;
}

export const docStore = createStore<DocState>()(() => ({
  loaded: false,
  rev: -1,
  layers: [],
  stocks: [],
  assetsCount: 0,
  selection: [],
  expanded: [],
  canUndo: false,
  canRedo: false,
  dirty: false,
  undoLabels: [],
  redoLabels: [],
  rolling: false,
}));

export interface DocEventPlan {
  refetchTree: boolean;
  refetchHistory: boolean;
  changedUids: string[];
}

/**
 * Decides what to refetch for a doc WS event. The tree is refetched
 * whenever the server did not confirm the revision we already have.
 */
export function planDocEvent(
  ev: DocWsEvent,
  currentRev: number,
): DocEventPlan {
  const treeRev = ev.revisions ? ev.revisions["tree"] : undefined;
  return {
    refetchTree: treeRev === undefined || treeRev !== currentRev,
    refetchHistory: true,
    changedUids: ev.uids ?? [],
  };
}

/** True when a cached geometry copy is stale or missing. */
export function geometryStale(
  cachedRevision: number | undefined,
  workpiece: WorkpieceInfo,
): boolean {
  return cachedRevision === undefined || cachedRevision !== workpiece.geometry_revision;
}

export function findLayer(
  layers: LayerInfo[],
  uid: string,
): LayerInfo | undefined {
  return layers.find((layer) => layer.uid === uid);
}

export function findWorkpiece(
  layers: LayerInfo[],
  uid: string,
): { layer: LayerInfo; workpiece: WorkpieceInfo } | undefined {
  for (const layer of layers) {
    const workpiece = layer.workpieces.find((wp) => wp.uid === uid);
    if (workpiece) return { layer, workpiece };
  }
  return undefined;
}

export function findStep(
  layers: LayerInfo[],
  uid: string,
): { layer: LayerInfo; step: StepInfo } | undefined {
  for (const layer of layers) {
    const step = layer.steps.find((s) => s.uid === uid);
    if (step) return { layer, step };
  }
  return undefined;
}

export function findLayerOfWorkpiece(
  layers: LayerInfo[],
  uid: string,
): LayerInfo | undefined {
  return layers.find((layer) =>
    layer.workpieces.some((wp) => wp.uid === uid),
  );
}

export function selectedWorkpieces(
  layers: LayerInfo[],
  selection: string[],
): WorkpieceInfo[] {
  const out: WorkpieceInfo[] = [];
  for (const uid of selection) {
    const found = findWorkpiece(layers, uid);
    if (found) out.push(found.workpiece);
  }
  return out;
}

function pruneSelection(layers: LayerInfo[], selection: string[]): string[] {
  const known = new Set<string>();
  for (const layer of layers) {
    known.add(layer.uid);
    for (const wp of layer.workpieces) known.add(wp.uid);
    for (const step of layer.steps) known.add(step.uid);
  }
  return selection.filter((uid) => known.has(uid));
}

export function setDocTree(tree: DocTree): void {
  docStore.setState((state) => ({
    loaded: true,
    rev: tree.rev,
    layers: tree.layers,
    stocks: tree.stock_items ?? [],
    assetsCount: tree.assets_count,
    selection: pruneSelection(tree.layers, state.selection),
  }));
}

export function setDocHistory(history: DocHistory): void {
  docStore.setState({
    canUndo: history.can_undo,
    canRedo: history.can_redo,
    undoLabels: history.undo_labels,
    redoLabels: history.redo_labels,
    dirty: history.dirty,
  });
}

export function setRolling(rolling: boolean): void {
  docStore.setState({ rolling });
}

export function selectUids(uids: string[], additive = false): void {
  docStore.setState((state) => {
    if (!additive) return { selection: [...uids] };
    const merged = new Set(state.selection);
    for (const uid of uids) {
      if (merged.has(uid)) merged.delete(uid);
      else merged.add(uid);
    }
    return { selection: [...merged] };
  });
}

export function clearSelection(): void {
  docStore.setState({ selection: [] });
}

export function toggleExpanded(uid: string): void {
  docStore.setState((state) => ({
    expanded: state.expanded.includes(uid)
      ? state.expanded.filter((id) => id !== uid)
      : [...state.expanded, uid],
  }));
}

export function isExpanded(state: DocState, uid: string): boolean {
  return state.expanded.includes(uid);
}

export function setLayerVisibleLocal(uid: string, visible: boolean): void {
  setLayerLocal(uid, { visible });
}

export type LayerLocalPatch = Partial<
  Pick<LayerInfo, "name" | "visible" | "color" | "wcs" | "rotary_enabled">
>;

/** Applies a partial layer update optimistically (before the server confirms). */
export function setLayerLocal(uid: string, patch: LayerLocalPatch): void {
  docStore.setState((state) => ({
    layers: state.layers.map((layer) =>
      layer.uid === uid ? { ...layer, ...patch } : layer,
    ),
  }));
}

/** Marks one layer active and clears the flag on the others. */
export function setActiveLayerLocal(uid: string): void {
  docStore.setState((state) => ({
    layers: state.layers.map((layer) => ({
      ...layer,
      is_active: layer.uid === uid,
    })),
  }));
}

/** Reorders layers optimistically to match the given uid order. */
export function reorderLayersLocal(orderedUids: string[]): void {
  docStore.setState((state) => {
    const byUid = new Map(state.layers.map((layer) => [layer.uid, layer]));
    const ordered: LayerInfo[] = [];
    for (const uid of orderedUids) {
      const layer = byUid.get(uid);
      if (layer) ordered.push(layer);
    }
    for (const layer of state.layers) {
      if (!ordered.includes(layer)) ordered.push(layer);
    }
    return { layers: ordered };
  });
}
