import { pushToast } from "../stores/uiStore";
import { button } from "./ui/Button";
import { el } from "./ui/dom";
import { numberField } from "./ui/NumberField";
import { select } from "./ui/Select";
import { tabs, type TabsHandle } from "./ui/Tabs";
import { textField } from "./ui/TextField";
import { toggle } from "./ui/Toggle";

export interface SettingsField {
  /** Dotted path into the settings object, e.g. "axis_extents.0". */
  key: string;
  label: string;
  kind: "text" | "number" | "switch" | "choice";
  options?: Array<{ value: string; label: string }>;
  unit?: string;
}

export interface SettingsSection {
  label: string;
  fields: SettingsField[];
  note?: string;
}

/** Reads a dotted path ("a.b" or "list.0") from a plain object. */
export function getPath(values: Record<string, unknown>, key: string): unknown {
  let cursor: unknown = values;
  for (const part of key.split(".")) {
    if (cursor === null || cursor === undefined) return undefined;
    cursor = (cursor as Record<string, unknown>)[part];
  }
  return cursor;
}

/** Returns a copy of `values` with `key` set to `value` (immutable). */
export function setPath(
  values: Record<string, unknown>,
  key: string,
  value: unknown,
): Record<string, unknown> {
  const [head, ...rest] = key.split(".");
  if (rest.length === 0) {
    return { ...values, [head]: value };
  }
  const current = values[head];
  if (Array.isArray(current)) {
    const copy = [...current];
    const index = Number(rest[0]);
    if (rest.length === 1) {
      copy[index] = value;
    } else {
      const element = (current[index] as Record<string, unknown>) ?? {};
      copy[index] = setPath(element, rest.join("."), value);
    }
    return { ...values, [head]: copy };
  }
  const child = (current as Record<string, unknown> | undefined) ?? {};
  return { ...values, [head]: setPath(child, rest.join("."), value) };
}

export interface SettingsFormEntry {
  key: string;
  initial: unknown;
  read: () => unknown;
}

export interface SettingsForm {
  root: HTMLElement;
  collect(): Record<string, unknown>;
}

/**
 * Builds a change-tracking form for one flat section of settings
 * fields; `collect()` returns only the values that differ from the
 * loaded ones.
 */
export function createSettingsForm(
  fields: SettingsField[],
  values: Record<string, unknown>,
): SettingsForm {
  const readers: SettingsFormEntry[] = [];
  const root = el("div", "settings-section");
  for (const field of fields) {
    const { element, read } = buildSettingsField(field, values);
    readers.push({ key: field.key, initial: getPath(values, field.key), read });
    root.appendChild(element);
  }
  return {
    root,
    collect(): Record<string, unknown> {
      const changed: Record<string, unknown> = {};
      for (const entry of readers) {
        const current = entry.read();
        if (
          current !== undefined &&
          current !== entry.initial &&
          String(current) !== String(entry.initial)
        ) {
          changed[entry.key] = current;
        }
      }
      return changed;
    },
  };
}

/**
 * Folds dotted keys ("a.b", "arr.0") back into nested objects/arrays.
 */
export function foldFlatKeys(
  changed: Record<string, unknown>,
): Record<string, unknown> {
  let out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(changed)) {
    out = setPath(out, key, value);
  }
  return out;
}

/**
 * Folds changed dotted keys onto the base values so partial writes of
 * nested/array fields stay complete (e.g. changing only
 * "axis_extents.0" still sends the full [w, h] pair).
 */
export function mergeFlatKeys(
  base: Record<string, unknown>,
  changed: Record<string, unknown>,
): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(changed)) {
    const applied = setPath(merged, key, value);
    const root = key.split(".")[0];
    merged[root] = applied[root];
  }
  const picked: Record<string, unknown> = {};
  for (const key of Object.keys(changed)) {
    const root = key.split(".")[0];
    picked[root] = merged[root];
  }
  return picked;
}

export interface SettingsDialogOptions {
  title: string;
  sections: SettingsSection[];
  load: () => Promise<Record<string, unknown>>;
  save: (
    changed: Record<string, unknown>,
  ) => Promise<Record<string, unknown>>;
  onSaved?: () => void;
  savedMessage: string;
}

export interface SettingsFormResult {
  root: HTMLElement;
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Modal settings dialog: renders one control per field and, on save,
 * reads every control's current value and hands only the entries that
 * differ from the loaded values to `save`.
 */
export async function openSettingsDialog(
  options: SettingsDialogOptions,
): Promise<void> {
  let values: Record<string, unknown>;
  try {
    values = await options.load();
  } catch (err) {
    pushToast("error", `Loading settings failed: ${describe(err)}`);
    return;
  }
  const forms = options.sections.map((section) =>
    createSettingsForm(section.fields, values),
  );

  const body = el("div", "dialog-body settings-body");
  const sectionHosts = new Map<string, HTMLElement>();
  const tabDefs = options.sections.map((section, index) => ({
    id: String(index),
    label: section.label,
  }));
  const tabHandle: TabsHandle = tabs(
    tabDefs,
    tabDefs[0]?.id ?? "0",
    (id) => {
      for (const [sectionId, host] of sectionHosts) {
        host.hidden = sectionId !== id;
      }
    },
  );
  body.appendChild(tabHandle.root);
  options.sections.forEach((section, index) => {
    const host = forms[index]!.root;
    if (index > 0) host.hidden = true;
    if (section.note) {
      const note = el("div", "panel-note");
      note.textContent = section.note;
      host.appendChild(note);
    }
    sectionHosts.set(String(index), host);
    body.appendChild(host);
  });
  const errorNote = el("div", "panel-note settings-error");
  const saveBtn = button({
    label: "Save",
    iconName: "save",
    variant: "primary",
    onClick: () => void commit(),
  });
  const actionsRow = el("div", "panel-actions");
  actionsRow.appendChild(saveBtn);
  body.appendChild(actionsRow);

  const backdrop = el("div", "modal-backdrop");
  const dialog = el("div", "dialog");
  const header = el("header", "dialog-header");
  const title = el("span", "dialog-title");
  title.textContent = options.title;
  header.append(title, button({
    iconName: "x",
    variant: "icon",
    title: "Close",
    onClick: close,
  }));
  dialog.append(header, body);
  backdrop.appendChild(dialog);
  backdrop.addEventListener("click", (ev) => {
    if (ev.target === backdrop) close();
  });
  document.body.appendChild(backdrop);

  async function commit(): Promise<void> {
    const changed: Record<string, unknown> = {};
    for (const form of forms) {
      Object.assign(changed, form.collect());
    }
    if (Object.keys(changed).length === 0) {
      close();
      return;
    }
    saveBtn.disabled = true;
    try {
      await options.save(changed);
      close();
      pushToast("success", options.savedMessage);
      options.onSaved?.();
    } catch (err) {
      errorNote.textContent = describe(err);
      if (!errorNote.parentElement) body.insertBefore(errorNote, actionsRow);
      saveBtn.disabled = false;
    }
  }

  function close(): void {
    backdrop.remove();
  }
}

export function buildSettingsField(
  field: SettingsField,
  values: Record<string, unknown>,
): { element: HTMLElement; read: () => unknown } {
  const initial = getPath(values, field.key);
  if (field.kind === "switch") {
    const control = toggle(Boolean(initial), () => undefined);
    const row = el("div", "schema-field-row");
    const label = el("span", "num-field-label");
    label.textContent = field.label;
    row.append(label, control.root);
    return {
      element: row,
      read: () => control.root.classList.contains("on"),
    };
  }
  if (field.kind === "choice") {
    const opts = field.options ?? [];
    const current = String(initial ?? opts[0]?.value ?? "");
    const control = select(field.label, opts, current, () => undefined);
    control.root.classList.add("in-panel");
    return {
      element: control.root,
      read: () => {
        const sel = control.root.querySelector(
          "select",
        ) as HTMLSelectElement | null;
        return sel ? sel.value : undefined;
      },
    };
  }
  if (field.kind === "number") {
    const control = numberField({
      label: field.label,
      value: typeof initial === "number" ? initial : 0,
      unit: field.unit,
      onCommit: () => undefined,
    });
    return {
      element: control.root,
      read: () => {
        const input = control.root.querySelector(
          "input",
        ) as HTMLInputElement | null;
        if (!input) return undefined;
        const parsed = Number.parseFloat(input.value);
        return Number.isNaN(parsed) ? undefined : parsed;
      },
    };
  }
  const control = textField({
    label: field.label,
    value: String(initial ?? ""),
    onCommit: () => undefined,
  });
  return {
    element: control.root,
    read: () => {
      const input = control.root.querySelector(
        "input",
      ) as HTMLInputElement | null;
      if (!input) return undefined;
      const text = input.value.trim();
      return text === "" ? undefined : text;
    },
  };
}
