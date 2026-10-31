import { api } from "../stores/api";
import {
  docStore,
  isExpanded,
  reorderLayersLocal,
  selectUids,
  setActiveLayerLocal,
  setLayerLocal,
  toggleExpanded,
} from "../stores/docStore";
import { pushToast, uiStore } from "../stores/uiStore";
import type { LayerInfo, StepInfo, WorkpieceInfo } from "../api/types";
import { openStepPickerForLayer } from "./stepPicker";
import { openContextMenu, type ContextMenuItem } from "./ui/ContextMenu";
import { button } from "./ui/Button";
import { el } from "./ui/dom";
import { icon, type IconName } from "./ui/icons";
import type { TreeRow } from "./ui/Tree";
import { replaceTreeContent } from "./ui/Tree";

export interface LayersPanelActions {
  refreshDoc: () => Promise<void>;
}

export interface LayersPanelRefs {
  root: HTMLElement;
}

/** Left-panel Document tab: layer tree with full CRUD wired to the API. */
export function buildLayersPanel(
  actions: LayersPanelActions,
): LayersPanelRefs {
  const root = el("div", "layers-panel");

  const header = el("div", "layers-header");
  const title = el("span", "layers-title");
  title.textContent = "Layers";
  const addBtn = button({
    iconName: "filePlus",
    iconSize: 14,
    variant: "icon",
    title: "Add layer",
    onClick: () => void addLayer(actions),
  });
  header.append(title, addBtn);
  root.appendChild(header);

  const treeHost = el("div", "layers-tree");
  root.appendChild(treeHost);

  const rerender = () => {
    if (uiStore.getState().activeLeftTab === "document") {
      replaceTreeContent(treeHost, buildDocRows(docStore.getState(), actions));
    }
  };
  docStore.subscribe(rerender);
  uiStore.subscribe(rerender);
  rerender();

  return { root };
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

async function addLayer(actions: LayersPanelActions): Promise<void> {
  try {
    await api.postLayer();
    await actions.refreshDoc();
  } catch (err) {
    pushToast("error", `Adding layer failed: ${describe(err)}`);
  }
}

async function deleteLayer(
  layer: LayerInfo,
  actions: LayersPanelActions,
): Promise<void> {
  try {
    await api.deleteLayer(layer.uid);
    await actions.refreshDoc();
  } catch (err) {
    pushToast("error", `Deleting layer failed: ${describe(err)}`);
  }
}

async function activateLayer(
  layer: LayerInfo,
  actions: LayersPanelActions,
): Promise<void> {
  if (layer.is_active) return;
  setActiveLayerLocal(layer.uid);
  try {
    await api.activateLayer(layer.uid);
  } catch (err) {
    pushToast("error", `Activating layer failed: ${describe(err)}`);
    await actions.refreshDoc();
  }
}

async function renameLayer(
  layer: LayerInfo,
  name: string,
): Promise<void> {
  const previous = layer.name;
  setLayerLocal(layer.uid, { name });
  try {
    await api.putLayer(layer.uid, { name });
  } catch (err) {
    setLayerLocal(layer.uid, { name: previous });
    pushToast("error", `Renaming failed: ${describe(err)}`);
  }
}

async function setLayerColor(
  layer: LayerInfo,
  color: string,
): Promise<void> {
  const previous = layer.color;
  if (color === previous) return;
  setLayerLocal(layer.uid, { color });
  try {
    await api.putLayer(layer.uid, { color });
  } catch (err) {
    setLayerLocal(layer.uid, { color: previous });
    pushToast("error", `Color change failed: ${describe(err)}`);
  }
}

function toggleLayerVisible(layer: LayerInfo): void {
  const next = !layer.visible;
  setLayerLocal(layer.uid, { visible: next });
  api.putLayer(layer.uid, { visible: next }).catch((err: unknown) => {
    setLayerLocal(layer.uid, { visible: !next });
    pushToast("error", `Visibility change failed: ${describe(err)}`);
  });
}

async function moveLayer(
  layer: LayerInfo,
  delta: number,
  actions: LayersPanelActions,
): Promise<void> {
  const state = docStore.getState();
  const index = state.layers.findIndex((l) => l.uid === layer.uid);
  const target = index + delta;
  if (index < 0 || target < 0 || target >= state.layers.length) return;
  const uids = state.layers.map((l) => l.uid);
  const [moved] = uids.splice(index, 1);
  uids.splice(target, 0, moved!);
  reorderLayersLocal(uids);
  try {
    await api.reorderLayers(uids);
  } catch (err) {
    pushToast("error", `Reordering failed: ${describe(err)}`);
    await actions.refreshDoc();
  }
}

function layerMenu(
  layer: LayerInfo,
  actions: LayersPanelActions,
): ContextMenuItem[] {
  const last = docStore.getState().layers.length <= 1;
  return [
    { label: "Rename", action: () => focusRowRename(layer.uid) },
    { label: "Move up", action: () => void moveLayer(layer, -1, actions) },
    { label: "Move down", action: () => void moveLayer(layer, 1, actions) },
    { separator: true },
    {
      label: "Add step…",
      action: () => void openStepPickerForLayer(layer, actions.refreshDoc),
    },
    {
      label: "Delete layer",
      danger: true,
      disabled: last,
      action: () => void deleteLayer(layer, actions),
    },
  ];
}

function focusRowRename(uid: string): void {
  const label = document.querySelector<HTMLElement>(
    `.tree-row[data-uid="${uid}"] .tree-label`,
  );
  label?.dispatchEvent(new MouseEvent("dblclick"));
}

function workpieceMenu(
  wp: WorkpieceInfo,
  actions: LayersPanelActions,
): ContextMenuItem[] {
  return [
    {
      label: "Duplicate",
      action: () => void duplicateWorkpiece(wp, actions),
    },
    {
      label: "Delete",
      danger: true,
      action: () => void deleteWorkpieces([wp.uid], actions),
    },
  ];
}

async function duplicateWorkpiece(
  wp: WorkpieceInfo,
  actions: LayersPanelActions,
): Promise<void> {
  try {
    await api.duplicateItems([wp.uid]);
    await actions.refreshDoc();
  } catch (err) {
    pushToast("error", `Duplicating failed: ${describe(err)}`);
  }
}

async function deleteWorkpieces(
  uids: string[],
  actions: LayersPanelActions,
): Promise<void> {
  try {
    await api.deleteItems(uids);
    await actions.refreshDoc();
  } catch (err) {
    pushToast("error", `Deleting failed: ${describe(err)}`);
  }
}

async function deleteStep(
  step: StepInfo,
  actions: LayersPanelActions,
): Promise<void> {
  try {
    await api.deleteStep(step.uid);
    await actions.refreshDoc();
  } catch (err) {
    pushToast("error", `Deleting step failed: ${describe(err)}`);
  }
}

function treeButton(
  iconName: IconName,
  title: string,
  onClick: (ev: MouseEvent) => void,
): HTMLElement {
  const btn = el("button", "tree-btn");
  btn.type = "button";
  btn.title = title;
  btn.appendChild(icon(iconName, 12));
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    onClick(ev);
  });
  return btn;
}

/** Opens a transient native color picker near the clicked chip. */
function pickColor(
  anchorX: number,
  anchorY: number,
  initial: string,
  onCommit: (color: string) => void,
): void {
  const input = el("input", "color-popover") as HTMLInputElement;
  input.type = "color";
  input.value = normalizeHex(initial);
  input.addEventListener("input", () => onCommit(input.value));
  input.addEventListener("change", () => {
    onCommit(input.value);
    window.setTimeout(() => input.remove(), 50);
  });
  input.addEventListener("blur", () => {
    window.setTimeout(() => input.remove(), 50);
  });
  input.style.left = `${anchorX}px`;
  input.style.top = `${anchorY}px`;
  document.body.appendChild(input);
  input.click();
}

function normalizeHex(color: string): string {
  if (/^#[0-9a-fA-F]{6}$/.test(color)) return color;
  if (/^#[0-9a-fA-F]{3}$/.test(color)) {
    return `#${color[1]}${color[1]}${color[2]}${color[2]}${color[3]}${color[3]}`;
  }
  return "#4d9fff";
}

export function buildDocRows(
  state: ReturnType<typeof docStore.getState>,
  actions: LayersPanelActions,
): TreeRow[] {
  const rows: TreeRow[] = [];
  const layerCount = state.layers.length;
  state.layers.forEach((layer, layerIndex) => {
    const expanded = isExpanded(state, layer.uid);
    const layerRow: TreeRow = {
      id: layer.uid,
      depth: 0,
      label: layer.name,
      colorChip: layer.color,
      expandable: true,
      expanded,
      selected: state.selection.includes(layer.uid),
      visible: layer.visible,
      active: layer.is_active,
      renamable: true,
      trailing: [
        treeButton("chevronUp", "Move layer up", () =>
          void moveLayer(layer, -1, actions),
        ),
        treeButton("chevronDown", "Move layer down", () =>
          void moveLayer(layer, 1, actions),
        ),
        treeButton("kebab", "Layer actions", (ev) => {
          openContextMenu(ev.clientX, ev.clientY, layerMenu(layer, actions));
        }),
      ],
      onToggleExpanded: () => toggleExpanded(layer.uid),
      onSelect: (ev) => {
        selectUids([layer.uid], ev.shiftKey);
        void activateLayer(layer, actions);
      },
      onToggleVisible: () => toggleLayerVisible(layer),
      onChipClick: (ev) => {
        pickColor(ev.clientX, ev.clientY, layer.color, (color) =>
          void setLayerColor(layer, color),
        );
      },
      onRename: (name) => void renameLayer(layer, name),
      onContext: (ev) => {
        selectUids([layer.uid]);
        openContextMenu(ev.clientX, ev.clientY, layerMenu(layer, actions));
      },
    };
    if (layerIndex === 0) {
      layerRow.trailing = layerRow.trailing?.slice(1);
    } else if (layerIndex === layerCount - 1) {
      layerRow.trailing = layerRow.trailing?.slice(0, 1);
    }
    rows.push(layerRow);
    if (!expanded) return;

    for (const wp of layer.workpieces) {
      rows.push({
        id: wp.uid,
        depth: 1,
        label: wp.name,
        icon: "box",
        colorChip: layer.color,
        dimmed: !layer.visible,
        selected: state.selection.includes(wp.uid),
        trailing: [
          treeButton("kebab", "Workpiece actions", (ev) => {
            openContextMenu(ev.clientX, ev.clientY, workpieceMenu(wp, actions));
          }),
        ],
        onSelect: (ev) => selectUids([wp.uid], ev.shiftKey),
        onContext: (ev) => {
          selectUids([wp.uid]);
          openContextMenu(ev.clientX, ev.clientY, workpieceMenu(wp, actions));
        },
      });
    }

    for (const step of layer.steps) {
      rows.push({
        id: step.uid,
        depth: 1,
        label: step.name,
        icon: "zap",
        hint: step.typelabel ?? step.step_type,
        statusDot:
          step.visible === undefined ? (step.enabled ? "on" : "off") : step.visible ? "on" : "off",
        dimmed: !layer.visible,
        selected: state.selection.includes(step.uid),
        trailing: [
          treeButton("trash", "Delete step", () =>
            void deleteStep(step, actions),
          ),
        ],
        onSelect: (ev) => selectUids([step.uid], ev.shiftKey),
      });
    }
  });
  return rows;
}
