import { api } from "../stores/api";
import { docStore } from "../stores/docStore";
import type { StepInfo, StepSchema } from "../api/types";
import { pushToast } from "../stores/uiStore";
import {
  buildFormModel,
  sliderStep,
  type FormFieldModel,
  type FormGroupModel,
} from "../steps/schemaForm";
import { button } from "./ui/Button";
import { clearElement, el } from "./ui/dom";
import { numberField } from "./ui/NumberField";
import { panel } from "./ui/Panel";
import { select } from "./ui/Select";
import { slider } from "./ui/Slider";
import { textField } from "./ui/TextField";
import { toggle } from "./ui/Toggle";

const SAVE_DEBOUNCE_MS = 300;

const schemaCache = new Map<string, StepSchema>();

export interface StepFormActions {
  refreshDoc: () => Promise<void>;
}

export function buildStepForm(
  step: StepInfo,
  actions: StepFormActions,
): HTMLElement {
  const wrap = el("div", "step-form");
  const savedHint = el("span", "saved-hint");

  const headerBtn = button({
    iconName: "trash",
    variant: "icon",
    title: "Delete step",
    onClick: () => void deleteStep(step, actions),
  });
  const header = panel(step.name, headerBtn);
  header.header.appendChild(savedHint);

  const meta = el("div", "panel-note");
  meta.textContent = `Type: ${step.typelabel ?? step.step_type}`;
  header.body.appendChild(meta);

  const body = el("div", "step-form-body");
  header.body.appendChild(body);
  wrap.appendChild(header.root);

  const cached = schemaCache.get(step.uid);
  if (cached) {
    renderForm(body, step, cached, savedHint);
  } else {
    body.appendChild(loadingNote());
    void api
      .getStepSchema(step.uid)
      .then((schema) => {
        schemaCache.set(step.uid, schema);
        if (!body.isConnected) return;
        clearElement(body);
        renderForm(body, step, schema, savedHint);
      })
      .catch((err: unknown) => {
        if (!body.isConnected) return;
        clearElement(body);
        body.appendChild(errorNote(err));
      });
  }
  return wrap;
}

function loadingNote(): HTMLElement {
  const note = el("div", "panel-note");
  note.textContent = "Loading schema…";
  return note;
}

function errorNote(err: unknown): HTMLElement {
  const note = el("div", "panel-note");
  note.textContent = `Schema unavailable: ${describe(err)}`;
  return note;
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

async function deleteStep(
  step: StepInfo,
  actions: StepFormActions,
): Promise<void> {
  try {
    await api.deleteStep(step.uid);
    await actions.refreshDoc();
  } catch (err) {
    pushToast("error", `Deleting step failed: ${describe(err)}`);
  }
}

function renderForm(
  body: HTMLElement,
  step: StepInfo,
  schema: StepSchema,
  savedHint: HTMLElement,
): void {
  const values: Record<string, unknown> = { ...step.params };
  const dirty = new Set<string>();
  let timer: number | null = null;
  let saving = false;

  const setHint = (text: string, cls: string) => {
    savedHint.textContent = text;
    savedHint.className = `saved-hint ${cls}`;
  };

  const flush = async () => {
    timer = null;
    if (dirty.size === 0 || saving) return;
    const params: Record<string, unknown> = {};
    for (const key of dirty) params[key] = values[key];
    dirty.clear();
    saving = true;
    setHint("Saving…", "saving");
    try {
      await api.putStep(step.uid, { params });
      setHint("Saved", "done");
      window.setTimeout(() => {
        if (savedHint.textContent === "Saved") setHint("", "");
      }, 1600);
    } catch (err) {
      setHint("", "");
      pushToast("error", `Saving step failed: ${describe(err)}`);
    } finally {
      saving = false;
      if (dirty.size > 0 && timer === null) schedule();
    }
  };

  const schedule = () => {
    setHint("Editing…", "dirty");
    if (timer !== null) window.clearTimeout(timer);
    timer = window.setTimeout(() => void flush(), SAVE_DEBOUNCE_MS);
  };

  const commitValue = (key: string, value: unknown) => {
    values[key] = value;
    dirty.add(key);
    schedule();
  };

  const groups: FormGroupModel[] = buildFormModel(schema, step.params);
  for (const group of groups) {
    if (group.fields.length === 0) continue;
    const groupEl = el("div", "schema-group");
    if (group.title) {
      const heading = el("div", "schema-group-title");
      heading.textContent = group.title;
      groupEl.appendChild(heading);
    }
    for (const field of group.fields) {
      groupEl.appendChild(buildField(field, commitValue));
    }
    body.appendChild(groupEl);
  }

  const applied = step.workpiece_uids;
  if (applied.length > 0) {
    body.appendChild(buildWorkpieceFilter(step, applied));
  }
}

export function buildField(
  field: FormFieldModel,
  commit: (key: string, value: unknown) => void,
): HTMLElement {
  const host = el("div", "schema-field");
  if (field.description) {
    host.title = field.description;
  }
  switch (field.widget) {
    case "toggle": {
      const handle = toggle(
        Boolean(field.value),
        (checked) => commit(field.key, checked),
      );
      const row = el("div", "schema-field-row");
      const label = el("span", "num-field-label");
      label.textContent = field.label;
      row.append(label, handle.root);
      host.appendChild(row);
      return host;
    }
    case "slider": {
      const min = field.min ?? 0;
      const max = field.max ?? 100;
      const handle = slider(
        field.label,
        min,
        max,
        sliderStep(field),
        Number(field.value) || 0,
        (value) => commit(field.key, value),
      );
      host.appendChild(handle.root);
      return host;
    }
    case "select": {
      const current =
        field.options.find((opt) => opt.value === String(field.value))?.value ??
        field.options[0]?.value ??
        "";
      const handle = select(
        field.label,
        field.options,
        current,
        (value) => commit(field.key, coerceOptionValue(field, value)),
      );
      handle.root.classList.add("in-panel");
      host.appendChild(handle.root);
      return host;
    }
    case "textarea": {
      const label = el("span", "num-field-label");
      label.textContent = field.label;
      const area = el("textarea", "textarea-input") as HTMLTextAreaElement;
      area.value = String(field.value ?? "");
      area.spellcheck = false;
      area.addEventListener("input", () => commit(field.key, area.value));
      host.append(label, area);
      return host;
    }
    case "text": {
      const handle = textField({
        label: field.label,
        value: String(field.value ?? ""),
        onCommit: (next) => commit(field.key, next),
      });
      host.appendChild(handle.root);
      return host;
    }
    default: {
      const value = Number(field.value);
      const handle = numberField({
        label: field.label,
        value: Number.isNaN(value) ? 0 : value,
        unit: field.unit ?? undefined,
        min: field.min ?? undefined,
        max: field.max ?? undefined,
        step: field.step ?? undefined,
        onCommit: (next) => commit(field.key, next),
      });
      host.appendChild(handle.root);
      return host;
    }
  }
}

function coerceOptionValue(field: FormFieldModel, value: string): unknown {
  if (field.type === "number") {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? value : parsed;
  }
  return value;
}

/**
 * Workpiece filter display. PUT /steps/{uid} only accepts name and
 * params, so the filter is shown read-only.
 */
function buildWorkpieceFilter(
  step: StepInfo,
  appliedUids: string[],
): HTMLElement {
  const wrap = el("div", "wp-filter");
  const label = el("span", "num-field-label");
  label.textContent = "Applied to workpieces";
  wrap.appendChild(label);

  const names = appliedUids.map((uid) => {
    const layer = docLayerOf(step.uid);
    return layer?.workpieces.find((wp) => wp.uid === uid)?.name ?? uid;
  });
  const chips = el("div", "chip-list");
  for (const name of names) {
    const chip = el("span", "chip");
    chip.textContent = name;
    chips.appendChild(chip);
  }
  wrap.appendChild(chips);

  const note = el("div", "panel-note");
  note.textContent = "Read-only: this filter is managed by the desktop editor";
  wrap.appendChild(note);
  return wrap;
}

function docLayerOf(stepUid: string): { workpieces: Array<{ uid: string; name: string }> } | null {
  for (const layer of docStore.getState().layers) {
    if (layer.steps.some((s) => s.uid === stepUid)) return layer;
  }
  return null;
}
