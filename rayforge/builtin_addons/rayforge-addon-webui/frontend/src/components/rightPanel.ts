import { api } from "../stores/api";
import {
  docStore,
  findLayer,
  findStep,
  selectedWorkpieces,
  setLayerLocal,
} from "../stores/docStore";
import { interactionStore } from "../stores/interactionStore";
import { pushToast } from "../stores/uiStore";
import type { AlignMode, FlipAxis, LayerInfo } from "../api/types";
import { buildStepForm } from "./stepForm";
import { openStepPickerForLayer } from "./stepPicker";
import { buildConsoleSection } from "./control/consolePanel";
import { buildControlSection } from "./control/controlPanel";
import { numberField, type NumberFieldHandle } from "./ui/NumberField";
import { panel } from "./ui/Panel";
import { select } from "./ui/Select";
import { textField } from "./ui/TextField";
import { floatingCard, type FloatingCardRefs } from "./ui/FloatingCard";
import { tabs, type TabsHandle } from "./ui/Tabs";
import { toggle } from "./ui/Toggle";
import { button } from "./ui/Button";
import { clearElement, el } from "./ui/dom";

export interface RightPanelRefs extends FloatingCardRefs {
  content: HTMLElement;
  machineContent: HTMLElement;
}

export interface RightPanelActions {
  refreshDoc: () => Promise<void>;
  newDoc: () => void;
  deleteSelected: () => void;
  duplicateSelected: () => void;
  flip: (axis: FlipAxis) => void;
  align: (mode: AlignMode) => void;
}

interface LiveTransformFields {
  uid: string;
  doc: { x: number; y: number; w: number; h: number; a: number };
  fields: {
    x: NumberFieldHandle;
    y: NumberFieldHandle;
    w: NumberFieldHandle;
    h: NumberFieldHandle;
    a: NumberFieldHandle;
  };
}

let liveTransformFields: LiveTransformFields | null = null;

export function buildRightPanel(actions: RightPanelActions): RightPanelRefs {
  const content = el("div", "panel-content");

  const bottom = el("div", "right-panel-bottom");
  const tabHost = el("div", "panel-tabs");
  const machineContent = el("div", "panel-content machine-panel-content");

  const controlSection = buildControlSection();
  const consoleSection = buildConsoleSection();

  let machineTabs: TabsHandle | null = null;
  function showMachineTab(id: string): void {
    machineTabs?.setActive(id);
    clearElement(machineContent);
    machineContent.appendChild(
      id === "console" ? consoleSection.root : controlSection.root,
    );
  }
  machineTabs = tabs(
    [
      { id: "control", label: "Control" },
      { id: "console", label: "Console" },
    ],
    "control",
    showMachineTab,
  );
  tabHost.appendChild(machineTabs.root);
  showMachineTab("control");
  bottom.append(tabHost, machineContent);

  const body = el("div", "floating-panel-body");
  body.append(content, bottom);
  const card = floatingCard("right", "Properties", "sliders", body);

  docStore.subscribe((state) => render(content, state, actions));
  render(content, docStore.getState(), actions);

  interactionStore.subscribe((state) => {
    const live = liveTransformFields;
    if (!live) return;
    const preview = state.transformPreview;
    if (!preview) {
      live.fields.x.setValue(live.doc.x);
      live.fields.y.setValue(live.doc.y);
      live.fields.w.setValue(live.doc.w);
      live.fields.h.setValue(live.doc.h);
      live.fields.a.setValue(live.doc.a);
      return;
    }
    if (live.uid !== preview.uid) return;
    live.fields.x.setValue(preview.pos.x);
    live.fields.y.setValue(preview.pos.y);
    live.fields.w.setValue(preview.size.w);
    live.fields.h.setValue(preview.size.h);
    live.fields.a.setValue(preview.angleDeg);
  });

  return {
    root: card.root,
    stub: card.stub,
    content,
    machineContent,
  };
}

function render(
  content: HTMLElement,
  state: ReturnType<typeof docStore.getState>,
  actions: RightPanelActions,
): void {
  clearElement(content);
  liveTransformFields = null;

  const workpieces = selectedWorkpieces(state.layers, state.selection);
  if (workpieces.length === 1) {
    content.appendChild(buildWorkpiecePanel(workpieces[0]!, actions));
    return;
  }
  if (workpieces.length > 1) {
    content.appendChild(buildMultiSelectionPanel(workpieces.length, actions));
    return;
  }

  const stepSel = state.selection
    .map((uid) => findStep(state.layers, uid))
    .find((s) => s !== undefined);
  if (stepSel) {
    content.appendChild(buildStepForm(stepSel.step, actions));
    return;
  }

  const layerSel = state.selection
    .map((uid) => findLayer(state.layers, uid))
    .find((l) => l !== undefined);
  if (layerSel) {
    content.appendChild(buildLayerPanel(layerSel, actions));
    return;
  }

  content.appendChild(buildEmptyPanel());
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function buildWorkpiecePanel(
  wp: ReturnType<typeof selectedWorkpieces>[number],
  actions: RightPanelActions,
): HTMLElement {
  const p = panel("Transform");

  const commit = (x: number, y: number, w: number, h: number, a: number) => {
    api
      .putItemTransform(wp.uid, {
        pos_mm: { x, y },
        size_mm: { w, h },
        angle_deg: a,
      })
      .then(() => actions.refreshDoc())
      .catch((err: unknown) => {
        pushToast("error", `Transform failed: ${String(err)}`);
      });
  };

  const fields: Record<string, NumberFieldHandle> = {};
  const mk = (label: string, value: number, key: string, unit?: string) => {
    const handle = numberField({
      label,
      value,
      unit,
      onCommit: () => {
        commit(
          fields.x ? read(fields.x, wp.pos_mm.x) : wp.pos_mm.x,
          fields.y ? read(fields.y, wp.pos_mm.y) : wp.pos_mm.y,
          fields.w ? read(fields.w, wp.size_mm.w) : wp.size_mm.w,
          fields.h ? read(fields.h, wp.size_mm.h) : wp.size_mm.h,
          fields.a ? read(fields.a, wp.angle_deg) : wp.angle_deg,
        );
      },
    });
    fields[key] = handle;
    return handle.root;
  };

  const grid = el("div", "field-grid");
  grid.append(
    mk("X", wp.pos_mm.x, "x", "mm"),
    mk("Y", wp.pos_mm.y, "y", "mm"),
    mk("W", wp.size_mm.w, "w", "mm"),
    mk("H", wp.size_mm.h, "h", "mm"),
    mk("Angle", wp.angle_deg, "a", "°"),
  );
  p.body.appendChild(grid);
  p.body.appendChild(buildWorkpieceActions(actions));

  liveTransformFields = {
    uid: wp.uid,
    doc: {
      x: wp.pos_mm.x,
      y: wp.pos_mm.y,
      w: wp.size_mm.w,
      h: wp.size_mm.h,
      a: wp.angle_deg,
    },
    fields: {
      x: fields.x!,
      y: fields.y!,
      w: fields.w!,
      h: fields.h!,
      a: fields.a!,
    },
  };

  const geo = panel("Workpiece");
  const rows: Array<[string, string]> = [
    ["Name", wp.name],
    [
      "Revisions",
      `geo ${wp.geometry_revision} · tf ${wp.transform_revision}`,
    ],
  ];
  for (const [key, value] of rows) {
    const row = el("div", "kv-row");
    const k = el("span", "kv-key");
    k.textContent = key;
    const v = el("span", "kv-value");
    v.textContent = value;
    row.append(k, v);
    geo.body.appendChild(row);
  }

  const wrap = el("div");
  wrap.append(p.root, geo.root);
  return wrap;
}

function read(handle: NumberFieldHandle, fallback: number): number {
  const input = handle.root.querySelector("input") as HTMLInputElement | null;
  const parsed = input ? Number.parseFloat(input.value) : Number.NaN;
  return Number.isNaN(parsed) ? fallback : parsed;
}

function buildWorkpieceActions(actions: RightPanelActions): HTMLElement {
  const row = el("div", "panel-actions");
  row.append(
    button({ label: "Flip H", onClick: () => actions.flip("horizontal") }),
    button({ label: "Flip V", onClick: () => actions.flip("vertical") }),
    button({
      label: "Duplicate",
      iconName: "copy",
      onClick: actions.duplicateSelected,
    }),
    button({
      label: "Delete",
      iconName: "trash",
      variant: "danger",
      onClick: actions.deleteSelected,
    }),
  );
  return row;
}

const ALIGN_MODES: Array<{ mode: AlignMode; label: string }> = [
  { mode: "left", label: "Left" },
  { mode: "center_horizontal", label: "Center" },
  { mode: "right", label: "Right" },
  { mode: "top", label: "Top" },
  { mode: "center_vertical", label: "Middle" },
  { mode: "bottom", label: "Bottom" },
];

function buildMultiSelectionPanel(
  count: number,
  actions: RightPanelActions,
): HTMLElement {
  const p = panel("Transform");
  const note = el("div", "panel-note");
  note.textContent = `${count} workpieces selected`;
  p.body.appendChild(note);

  const alignLabel = el("div", "num-field-label");
  alignLabel.textContent = "Align";
  const alignGrid = el("div", "align-grid");
  for (const { mode, label } of ALIGN_MODES) {
    alignGrid.appendChild(
      button({ label, onClick: () => actions.align(mode) }),
    );
  }
  p.body.append(alignLabel, alignGrid);
  p.body.appendChild(buildWorkpieceActions(actions));
  return p.root;
}

const WCS_SLOTS = [
  "G54",
  "G55",
  "G56",
  "G57",
  "G58",
  "G59",
  "G59.1",
  "G59.2",
  "G59.3",
];

function buildLayerPanel(
  layer: LayerInfo,
  actions: RightPanelActions,
): HTMLElement {
  const p = panel("Layer settings");

  const nameField = textField({
    label: "Name",
    value: layer.name,
    onCommit: (name) => {
      if (!name) return;
      setLayerLocal(layer.uid, { name });
      api.putLayer(layer.uid, { name }).catch((err: unknown) => {
        pushToast("error", `Renaming failed: ${describe(err)}`);
      });
    },
  });

  const colorWrap = el("div", "schema-field-row layer-color-row");
  const colorLabel = el("span", "num-field-label");
  colorLabel.textContent = "Color";
  const colorInput = el("input", "layer-color-input") as HTMLInputElement;
  colorInput.type = "color";
  colorInput.value = normalizeHex(layer.color);
  colorInput.addEventListener("input", () => {
    setLayerLocal(layer.uid, { color: colorInput.value });
    api.putLayer(layer.uid, { color: colorInput.value }).catch(
      (err: unknown) => {
        pushToast("error", `Color change failed: ${describe(err)}`);
      },
    );
  });
  colorWrap.append(colorLabel, colorInput);

  const wcs = select(
    "WCS",
    [{ value: "", label: "None" }, ...WCS_SLOTS.map((slot) => ({ value: slot, label: slot }))],
    layer.wcs ?? "",
    (value) => {
      const next = value === "" ? null : value;
      setLayerLocal(layer.uid, { wcs: next });
      api.putLayer(layer.uid, { wcs: next }).catch((err: unknown) => {
        pushToast("error", `WCS change failed: ${describe(err)}`);
      });
    },
  );
  wcs.root.classList.add("in-panel");

  const rotaryRow = el("div", "schema-field-row");
  const rotaryLabel = el("span", "num-field-label");
  rotaryLabel.textContent = "Rotary";
  const rotary = toggle(Boolean(layer.rotary_enabled), (checked) => {
    setLayerLocal(layer.uid, { rotary_enabled: checked });
    api.putLayer(layer.uid, { rotary_enabled: checked }).catch(
      (err: unknown) => {
        pushToast("error", `Rotary toggle failed: ${describe(err)}`);
      },
    );
  });
  rotaryRow.append(rotaryLabel, rotary.root);

  const actionsRow = el("div", "panel-actions layer-actions");
  actionsRow.appendChild(
    button({
      label: "Add step",
      iconName: "zap",
      variant: "primary",
      onClick: () => void openStepPickerForLayer(layer, actions.refreshDoc),
    }),
  );

  p.body.append(nameField.root, colorWrap, wcs.root, rotaryRow, actionsRow);
  return p.root;
}

function normalizeHex(color: string): string {
  if (/^#[0-9a-fA-F]{6}$/.test(color)) return color;
  if (/^#[0-9a-fA-F]{3}$/.test(color)) {
    return `#${color[1]}${color[1]}${color[2]}${color[2]}${color[3]}${color[3]}`;
  }
  return "#4d9fff";
}

function buildEmptyPanel(): HTMLElement {
  const p = panel("Properties");
  const note = el("div", "panel-note");
  note.textContent = "Nothing selected";
  p.body.appendChild(note);
  return p.root;
}
