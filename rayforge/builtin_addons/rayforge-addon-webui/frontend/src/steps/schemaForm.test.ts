import { describe, expect, it } from "vitest";
import type { StepSchema } from "../api/types";
import {
  buildFormModel,
  buildPassthroughSpec,
  buildTraceSpec,
  diffParams,
  resolveFieldValue,
  sliderStep,
  widgetForKind,
} from "./schemaForm";

const schema: StepSchema = {
  step_type: "Contour",
  groups: [
    {
      title: "Laser",
      fields: [
        {
          key: "power",
          type: "number",
          kind: "slider",
          label: "Power",
          description: "Laser power in percent",
          default: 80,
          min: 0,
          max: 100,
          step: 1,
          options: null,
          unit: null,
          optional: false,
        },
        {
          key: "speed",
          type: "number",
          kind: "speed",
          label: "Speed",
          description: null,
          default: 3000,
          min: 1,
          max: 60000,
          step: 10,
          options: null,
          unit: "mm/min",
          optional: false,
        },
        {
          key: "enable",
          type: "boolean",
          kind: "switch",
          label: "Enabled",
          description: null,
          default: true,
          options: null,
          unit: null,
          optional: false,
        },
      ],
    },
    {
      title: "Advanced",
      fields: [
        {
          key: "direction",
          type: "string",
          kind: "choice",
          label: "Direction",
          description: null,
          default: "ccw",
          options: [
            { value: "ccw", label: "Counter-clockwise" },
            { value: "cw", label: "Clockwise" },
          ],
          unit: null,
          optional: false,
        },
        {
          key: "angle",
          type: "number",
          kind: "angle",
          label: "Start angle",
          description: null,
          default: 0,
          min: 0,
          max: 360,
          step: 5,
          options: null,
          unit: "deg",
          optional: false,
        },
        {
          key: "notes",
          type: "string",
          kind: "textarea",
          label: "Notes",
          description: null,
          default: "",
          options: null,
          unit: null,
          optional: true,
        },
      ],
    },
  ],
};

describe("widgetForKind", () => {
  it("maps presentation kinds onto widgets", () => {
    expect(widgetForKind("switch")).toBe("toggle");
    expect(widgetForKind("slider")).toBe("slider");
    expect(widgetForKind("choice")).toBe("select");
    expect(widgetForKind("textarea")).toBe("textarea");
    expect(widgetForKind("input")).toBe("number");
    expect(widgetForKind("length")).toBe("number");
    expect(widgetForKind("speed")).toBe("number");
    expect(widgetForKind("angle")).toBe("number");
  });
});

describe("resolveFieldValue", () => {
  it("prefers the live param over the schema default", () => {
    const power = schema.groups[0]!.fields[0]!;
    expect(resolveFieldValue(power, { power: 42 })).toBe(42);
    expect(resolveFieldValue(power, {})).toBe(80);
  });

  it("coerces numeric strings for number fields", () => {
    const power = schema.groups[0]!.fields[0]!;
    expect(resolveFieldValue(power, { power: "42" })).toBe(42);
    expect(resolveFieldValue(power, { power: "abc" })).toBe(0);
  });

  it("falls back to typed zeros when no default exists", () => {
    const field = {
      key: "count",
      type: "number" as const,
      kind: "input" as const,
      label: "Count",
      default: undefined,
    };
    expect(resolveFieldValue(field, {})).toBe(0);
    expect(
      resolveFieldValue(
        { ...field, type: "boolean" as const, kind: "switch" as const },
        {},
      ),
    ).toBe(false);
    expect(
      resolveFieldValue(
        { ...field, type: "string" as const, kind: "input" as const },
        {},
      ),
    ).toBe("");
  });
});

describe("buildFormModel", () => {
  it("renders fields with options as selects regardless of kind", () => {
    const withOptions = {
      step_type: "X",
      groups: [
        {
          title: "G",
          fields: [
            {
              key: "mode",
              type: "string" as const,
              kind: "input" as const,
              label: "Mode",
              default: "a",
              options: [
                { value: "a", label: "Alpha" },
                { value: "b", label: "Beta" },
              ],
            },
          ],
        },
      ],
    };
    const model = buildFormModel(withOptions, {});
    expect(model[0]!.fields[0]!.widget).toBe("select");
  });

  it("keeps one group per schema group with render-ready fields", () => {
    const model = buildFormModel(schema, { power: 55 });
    expect(model).toHaveLength(2);
    expect(model[0]!.title).toBe("Laser");
    expect(model[0]!.fields.map((f) => f.key)).toEqual([
      "power",
      "speed",
      "enable",
    ]);

    const power = model[0]!.fields[0]!;
    expect(power.widget).toBe("slider");
    expect(power.value).toBe(55);
    expect(power.min).toBe(0);
    expect(power.max).toBe(100);

    const speed = model[0]!.fields[1]!;
    expect(speed.widget).toBe("number");
    expect(speed.unit).toBe("mm/min");
    expect(speed.value).toBe(3000);

    const enable = model[0]!.fields[2]!;
    expect(enable.widget).toBe("toggle");
    expect(enable.value).toBe(true);

    const direction = model[1]!.fields[0]!;
    expect(direction.widget).toBe("select");
    expect(direction.options).toEqual([
      { value: "ccw", label: "Counter-clockwise" },
      { value: "cw", label: "Clockwise" },
    ]);
    expect(direction.value).toBe("ccw");

    const angle = model[1]!.fields[1]!;
    expect(angle.unit).toBe("°");
    expect(angle.widget).toBe("number");

    const notes = model[1]!.fields[2]!;
    expect(notes.widget).toBe("textarea");
    expect(notes.optional).toBe(true);
  });

  it("drops fields without a key", () => {
    const withBlank = {
      step_type: "X",
      groups: [
        {
          title: "G",
          fields: [
            { key: "real", type: "number" as const, kind: "input" as const, label: "Real" },
            { key: "", type: "number" as const, kind: "input" as const, label: "Blank" },
          ],
        },
      ],
    };
    const model = buildFormModel(withBlank, {});
    expect(model[0]!.fields.map((f) => f.key)).toEqual(["real"]);
  });

  it("stringifies option values for selects", () => {
    const numericChoice = {
      step_type: "X",
      groups: [
        {
          title: "G",
          fields: [
            {
              key: "mode",
              type: "number" as const,
              kind: "choice" as const,
              label: "Mode",
              default: 1,
              options: [
                { value: 1, label: "One" },
                { value: 2, label: "Two" },
              ],
            },
          ],
        },
      ],
    } as unknown as StepSchema;
    const model = buildFormModel(numericChoice, {});
    expect(model[0]!.fields[0]!.options).toEqual([
      { value: "1", label: "One" },
      { value: "2", label: "Two" },
    ]);
    expect(model[0]!.fields[0]!.value).toBe(1);
  });
});

describe("diffParams", () => {
  it("collects changed and new keys only", () => {
    expect(diffParams({ a: 1, b: 2 }, { a: 1, b: 3, c: 4 })).toEqual({
      b: 3,
      c: 4,
    });
    expect(diffParams({ a: 1 }, { a: 1 })).toEqual({});
  });
});

describe("spec builders", () => {
  it("maps the 0-255 threshold slider onto TraceSpec", () => {
    expect(buildTraceSpec(128, false, 96)).toEqual({
      type: "TraceSpec",
      threshold: 128 / 255,
      auto_threshold: false,
      invert: false,
      ppi: 96,
    });
    expect(buildTraceSpec(255, true, 300).threshold).toBe(1);
    expect(buildTraceSpec(0, true, 300).threshold).toBe(0);
  });

  it("builds the default passthrough spec", () => {
    expect(buildPassthroughSpec()).toEqual({ type: "PassthroughSpec" });
  });
});

describe("sliderStep", () => {
  it("keeps the schema-declared step", () => {
    expect(sliderStep({ min: 0, max: 1, step: 0.05 })).toBe(0.05);
  });

  it("falls back to a continuous step for unit-interval fields", () => {
    expect(sliderStep({ min: 0, max: 1, step: null })).toBe(0.01);
  });

  it("falls back to 1/100th of the range when bounds are missing", () => {
    expect(sliderStep({ min: null, max: null, step: null })).toBe(1);
  });
});
