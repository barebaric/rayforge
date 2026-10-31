import type {
  SchemaField,
  SchemaGroup,
  StepSchema,
} from "../api/types";

export type FormWidget =
  | "toggle"
  | "slider"
  | "select"
  | "number"
  | "text"
  | "textarea";

export interface FormOption {
  value: string;
  label: string;
}

export interface FormFieldModel {
  key: string;
  widget: FormWidget;
  kind: SchemaField["kind"];
  type: SchemaField["type"];
  label: string;
  description: string | null;
  value: unknown;
  min: number | null;
  max: number | null;
  step: number | null;
  unit: string | null;
  options: FormOption[];
  optional: boolean;
}

export interface FormGroupModel {
  title: string;
  fields: FormFieldModel[];
}

const WIDGET_BY_KIND: Record<SchemaField["kind"], FormWidget> = {
  switch: "toggle",
  slider: "slider",
  choice: "select",
  textarea: "textarea",
  input: "number",
  length: "number",
  speed: "number",
  angle: "number",
};

export const DISPLAY_UNIT: Record<string, string> = {
  mm: "mm",
  "mm/min": "mm/min",
  deg: "°",
};

/** Kind of control a schema field renders as. */
export function widgetForKind(kind: SchemaField["kind"]): FormWidget {
  return WIDGET_BY_KIND[kind] ?? "number";
}

/**
 * Slider granularity: the schema step, else 1/100th of the range so
 * continuous 0..1 fields keep sub-unit resolution.
 */
export function sliderStep(
  field: Pick<FormFieldModel, "min" | "max" | "step">,
): number {
  const min = field.min ?? 0;
  const max = field.max ?? 100;
  return field.step ?? (max - min) / 100;
}

function typedFallback(type: SchemaField["type"]): unknown {
  if (type === "boolean") return false;
  if (type === "number") return 0;
  return "";
}

/** Current value of a field: live param, schema default, then typed zero. */
export function resolveFieldValue(
  field: SchemaField,
  params: Record<string, unknown>,
): unknown {
  if (field.key in params && params[field.key] !== undefined && params[field.key] !== null) {
    const value = params[field.key] as unknown;
    if (field.type === "number" && typeof value === "string") {
      const parsed = Number.parseFloat(value);
      return Number.isNaN(parsed)
        ? (typedFallback("number") as number)
        : parsed;
    }
    if (field.type === "boolean") {
      return Boolean(value);
    }
    return value;
  }
  if (field.default !== undefined && field.default !== null) {
    return field.default;
  }
  return typedFallback(field.type);
}

/** Numeric option index values arrive as strings; keep selects consistent. */
export function optionValue(value: unknown): string {
  return String(value);
}

function buildFieldModel(field: SchemaField, params: Record<string, unknown>): FormFieldModel {
  const hasOptions = Array.isArray(field.options) && field.options.length > 0;
  const widget = hasOptions
    ? "select"
    : field.type === "boolean"
      ? "toggle"
      : field.kind === "input" && field.type === "string"
        ? "text"
        : widgetForKind(field.kind);
  return {
    key: field.key,
    widget,
    kind: field.kind,
    type: field.type,
    label: field.label,
    description: field.description ?? null,
    value: resolveFieldValue(field, params),
    min: field.min ?? null,
    max: field.max ?? null,
    step: field.step ?? null,
    unit: field.unit ? (DISPLAY_UNIT[field.unit] ?? field.unit) : null,
    options: (field.options ?? []).map((opt) => ({
      value: optionValue(opt.value),
      label: opt.label,
    })),
    optional: Boolean(field.optional),
  };
}

/**
 * Maps a step schema plus the step's live params onto a render-ready
 * form model: one group per schema group, one field per schema field.
 */
export function buildFormModel(
  schema: StepSchema,
  params: Record<string, unknown>,
): FormGroupModel[] {
  return schema.groups.map((group: SchemaGroup) => ({
    title: group.title,
    fields: group.fields
      .filter((field) => field.key !== "")
      .map((field) => buildFieldModel(field, params)),
  }));
}

/** Collects the entries of `values` that differ from `base`. */
export function diffParams(
  base: Record<string, unknown>,
  values: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(values)) {
    if (!(key in base) || base[key] !== value) {
      out[key] = value;
    }
  }
  return out;
}

/**
 * TraceSpec body for a bitmap import. The UI exposes a 0-255 threshold
 * slider; the backend expects 0..1 with an auto-threshold escape hatch.
 */
export function buildTraceSpec(threshold255: number, invert: boolean, ppi: number): Record<string, unknown> {
  return {
    type: "TraceSpec",
    threshold: Math.min(1, Math.max(0, threshold255 / 255)),
    auto_threshold: false,
    invert,
    ppi,
  };
}

/** Default spec for vector sources: parse as-is. */
export function buildPassthroughSpec(): Record<string, unknown> {
  return { type: "PassthroughSpec" };
}
