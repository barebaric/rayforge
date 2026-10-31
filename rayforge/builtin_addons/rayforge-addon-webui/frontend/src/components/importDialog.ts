import { api } from "../stores/api";
import { pushToast } from "../stores/uiStore";
import type { ImportAnalysis } from "../api/types";
import {
  buildPassthroughSpec,
  buildTraceSpec,
} from "../steps/schemaForm";
import { ImportFlow } from "../import/importFlow";
import { button } from "./ui/Button";
import { clearElement, el } from "./ui/dom";
import { numberField } from "./ui/NumberField";
import { slider } from "./ui/Slider";
import { toggle } from "./ui/Toggle";

const DEFAULT_THRESHOLD = 128;
const DEFAULT_PPI = 96;

export interface ImportDialogActions {
  refreshDoc: () => Promise<void>;
}

/**
 * True when the file needs bitmap tracing instead of vector
 * passthrough, matching the desktop dialog's default.
 */
export function prefersTrace(features: string[]): boolean {
  return (
    features.includes("BITMAP_TRACING") &&
    !features.includes("DIRECT_VECTOR")
  );
}

/**
 * Import flow UI: file picker → analyze → direct load or a small
 * config dialog (TraceSpec for bitmaps, passthrough for vectors) →
 * import with optional x/y placement.
 */
export function openImportDialog(actions: ImportDialogActions): void {
  const input = el("input") as HTMLInputElement;
  input.type = "file";
  input.style.display = "none";
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    input.remove();
    if (file) void runDialog(file, actions);
  });
  document.body.appendChild(input);
  input.click();
}

/** Runs the import flow for an already-chosen file (dialog entry point). */
export function openImportDialogForFile(
  file: File,
  actions: ImportDialogActions,
): Promise<void> {
  return runDialog(file, actions);
}

async function runDialog(
  file: File,
  actions: ImportDialogActions,
): Promise<void> {
  const backdrop = buildBackdrop();
  const body = backdrop.querySelector(".dialog-body") as HTMLElement;
  body.appendChild(pendingNote(`Analyzing ${file.name}…`));
  document.body.appendChild(backdrop);

  const flow = new ImportFlow({
    analyze: (f) => api.analyzeImport(f),
    run: (f, placement) => api.importFile(f, placement),
  });

  const state = await flow.start(file);
  if (state.phase === "unsupported") {
    closeBackdrop(backdrop);
    pushToast("error", `Unsupported file type: ${file.name}`);
    return;
  }
  if (state.phase === "error") {
    closeBackdrop(backdrop);
    pushToast("error", `Import analysis failed: ${state.error}`);
    return;
  }
  if (state.phase === "direct_load") {
    closeBackdrop(backdrop);
    const done = await flow.run({});
    if (done.phase === "done") {
      await finishImport(done.result?.created.length ?? 0, actions, file.name);
    } else {
      pushToast("error", `Import failed: ${done.error}`);
    }
    return;
  }
  if (state.phase !== "interactive_config" || !state.analysis) {
    closeBackdrop(backdrop);
    pushToast("error", `Import failed: unexpected analyzer result`);
    return;
  }
  renderConfigForm(body, backdrop, flow, state.analysis, actions, file.name);
}

async function finishImport(
  count: number,
  actions: ImportDialogActions,
  name: string,
): Promise<void> {
  pushToast(
    "success",
    count === 1
      ? `Imported ${name} (1 workpiece)`
      : `Imported ${name} (${count} workpieces)`,
  );
  await actions.refreshDoc();
}

function renderConfigForm(
  body: HTMLElement,
  backdrop: HTMLElement,
  flow: ImportFlow,
  analysis: ImportAnalysis,
  actions: ImportDialogActions,
  fileName: string,
): void {
  clearElement(body);
  body.appendChild(summaryNote(analysis, fileName));

  const wantsTrace = prefersTrace(analysis.features);
  const specValues = { threshold: DEFAULT_THRESHOLD, invert: false };
  let ppi = DEFAULT_PPI;

  if (wantsTrace) {
    const traceForm = el("div", "import-spec-form");
    const threshold = slider(
      "Threshold",
      0,
      255,
      1,
      specValues.threshold,
      (value) => {
        specValues.threshold = value;
      },
    );
    const invert = toggle(false, (checked) => {
      specValues.invert = checked;
    });
    const invertRow = el("div", "schema-field-row");
    const invertLabel = el("span", "num-field-label");
    invertLabel.textContent = "Invert";
    invertRow.append(invertLabel, invert.root);
    const ppiField = numberField({
      label: "Pixels per inch",
      value: ppi,
      min: 1,
      onCommit: (value) => {
        ppi = value;
      },
    });
    traceForm.append(threshold.root, invertRow, ppiField.root);
    body.appendChild(traceForm);
  } else {
    const note = el("div", "panel-note");
    note.textContent =
      "Vectors are imported as-is (passthrough).";
    body.appendChild(note);
  }

  let x = 0;
  let y = 0;
  const placement = el("div", "field-grid import-placement");
  const xField = numberField({
    label: "X (mm)",
    value: x,
    unit: "mm",
    onCommit: (value) => {
      x = value;
    },
  });
  const yField = numberField({
    label: "Y (mm)",
    value: y,
    unit: "mm",
    onCommit: (value) => {
      y = value;
    },
  });
  placement.append(xField.root, yField.root);
  body.appendChild(placement);

  const footer = el("div", "dialog-footer");
  const cancelBtn = button({
    label: "Cancel",
    onClick: () => closeBackdrop(backdrop),
  });
  const importBtn = button({
    label: "Import",
    variant: "primary",
    onClick: () => {
      void (async () => {
        importBtn.disabled = true;
        const spec = wantsTrace
          ? buildTraceSpec(specValues.threshold, specValues.invert, ppi)
          : buildPassthroughSpec();
        const done = await flow.run({
          spec,
          x,
          y,
        });
        if (done.phase === "done") {
          closeBackdrop(backdrop);
          await finishImport(done.result?.created.length ?? 0, actions, fileName);
        } else {
          importBtn.disabled = false;
          pushToast("error", `Import failed: ${done.error}`);
        }
      })();
    },
  });
  footer.append(cancelBtn, importBtn);
  body.appendChild(footer);
}

function summaryNote(analysis: ImportAnalysis, fileName: string): HTMLElement {
  const wrap = el("div", "import-summary");
  const title = el("div", "import-summary-title");
  title.textContent = fileName;
  wrap.appendChild(title);
  const info = el("div", "panel-note");
  const bits: string[] = [analysis.importer ?? "unknown importer"];
  const manifest = analysis.manifest;
  if (manifest?.natural_size_mm) {
    const [w, h] = manifest.natural_size_mm;
    bits.push(`${w} × ${h} mm`);
  }
  if (manifest && manifest.layers.length > 0) {
    bits.push(`${manifest.layers.length} layer(s)`);
  }
  info.textContent = bits.join(" · ");
  wrap.appendChild(info);
  for (const warning of manifest?.warnings ?? []) {
    const row = el("div", "panel-note import-warning");
    row.textContent = warning;
    wrap.appendChild(row);
  }
  for (const error of manifest?.errors ?? []) {
    const row = el("div", "panel-note import-warning");
    row.textContent = error;
    wrap.appendChild(row);
  }
  return wrap;
}

function pendingNote(text: string): HTMLElement {
  const note = el("div", "panel-note");
  note.textContent = text;
  return note;
}

function buildBackdrop(): HTMLElement {
  const backdrop = el("div", "modal-backdrop");
  const dialog = el("div", "dialog");
  const header = el("header", "dialog-header");
  const title = el("span", "dialog-title");
  title.textContent = "Import";
  const closeBtn = button({
    iconName: "x",
    variant: "icon",
    title: "Close",
    onClick: () => closeBackdrop(backdrop),
  });
  header.append(title, closeBtn);
  const body = el("div", "dialog-body");
  dialog.append(header, body);
  backdrop.appendChild(dialog);
  backdrop.addEventListener("click", (ev) => {
    if (ev.target === backdrop) closeBackdrop(backdrop);
  });
  backdrop.dataset.role = "import-dialog";
  return backdrop;
}

function closeBackdrop(backdrop: HTMLElement): void {
  backdrop.remove();
}
