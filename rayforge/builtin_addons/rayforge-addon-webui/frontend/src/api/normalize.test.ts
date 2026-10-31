import { describe, expect, it } from "vitest";
import {
  normalizeDocTree,
  normalizeImportAnalysis,
  normalizeJobStatus,
  normalizeMachineState,
  normalizeMachines,
  normalizeOps,
  normalizeStepSchema,
  normalizeViewport,
} from "./normalize";

describe("normalizeViewport", () => {
  it("passes the canonical contract shape through", () => {
    const canonical = {
      panel_orientation: "native",
      world_to_panel: [1, 0, 5, 0, 1, 7],
      x_axis_right: true,
      y_axis_down: false,
      x_axis_negative: false,
      y_axis_negative: false,
      extents: { width_mm: 400, height_mm: 300 },
      margins: { left: 1, right: 2, top: 3, bottom: 4, front: 5, back: 6 },
      workarea: { x_min: 1, y_min: 2, x_max: 3, y_max: 4 },
      workarea_origin_panel: { x: 1, y: 2 },
      active_wcs: { name: "G55", label: "WCS 2", offset_panel: { x: 1, y: 1 } },
      beam_position_panel: null,
      soft_limits: null,
      rotary_active: false,
    };
    const vp = normalizeViewport(canonical);
    expect(vp.extents).toEqual({ width_mm: 400, height_mm: 300 });
    expect(vp.world_to_panel).toEqual([1, 0, 5, 0, 1, 7]);
    expect(vp.workarea).toEqual({ x_min: 1, y_min: 2, x_max: 3, y_max: 4 });
    expect(vp.active_wcs.name).toBe("G55");
    expect(vp.beam_position_panel).toBeNull();
  });

  it("folds the transitional backend layout into the contract", () => {
    const transitional = {
      orientation: "native",
      world_to_panel: [
        [1, 0, 0],
        [0, 1, 0],
      ],
      extents: [200, 200],
      margins: [0, 0, 0, 0],
      workarea: [0, 0, 200, 200],
      workarea_origin: [0, 0],
      soft_limits: [0, 0, 200, 200],
      rotary_active: false,
      wcs: { name: "G54", offset_panel: [0, 0] },
      beam: null,
    };
    const vp = normalizeViewport(transitional);
    expect(vp.panel_orientation).toBe("native");
    expect(vp.world_to_panel).toEqual([1, 0, 0, 0, 1, 0]);
    expect(vp.extents).toEqual({ width_mm: 200, height_mm: 200 });
    expect(vp.workarea).toEqual({
      x_min: 0,
      y_min: 0,
      x_max: 200,
      y_max: 200,
    });
    expect(vp.workarea_origin_panel).toEqual({ x: 0, y: 0 });
    expect(vp.active_wcs.name).toBe("G54");
    expect(vp.soft_limits).toEqual({
      x_min: 0,
      y_min: 0,
      x_max: 200,
      y_max: 200,
    });
    expect(vp.soft_limits_panel).toBeNull();
    expect(vp.work_origin_panel).toEqual({ x: 0, y: 0 });
  });

  it("treats the backend workarea array as x/y/w/h", () => {
    const vp = normalizeViewport({
      extents: [200, 200],
      workarea: [10, 20, 180, 160],
    });
    expect(vp.workarea).toEqual({
      x_min: 10,
      y_min: 20,
      x_max: 190,
      y_max: 180,
    });
  });

  it("folds work origin and panel soft limits for an oriented bed", () => {
    const vp = normalizeViewport({
      orientation: "native",
      extents: [200, 200],
      workarea: [0, 0, 200, 200],
      workarea_origin: [200, 200],
      work_origin: [200, 200],
      soft_limits: [0, 0, 200, 200],
      soft_limits_panel: [50, 40, 140, 140],
    });
    expect(vp.workarea_origin_panel).toEqual({ x: 200, y: 200 });
    expect(vp.work_origin_panel).toEqual({ x: 200, y: 200 });
    expect(vp.soft_limits_panel).toEqual({ x: 50, y: 40, w: 140, h: 140 });
  });
});

describe("normalizeMachineState", () => {
  it("maps array positions with nulls to null vectors", () => {
    const st = normalizeMachineState({
      status: "UNKNOWN",
      connected: false,
      machine_pos: [null, null, null],
      work_pos: [null, null, null],
      wco: [1, 2, 0],
    });
    expect(st.machine_pos).toEqual({ x: null, y: null, z: null });
    expect(st.work_pos).toEqual({ x: null, y: null, z: null });
    expect(st.wco).toEqual({ x: 1, y: 2, z: 0 });
    expect(st.beam_position_panel).toBeNull();
  });

  it("keeps canonical object vectors", () => {
    const st = normalizeMachineState({
      status: "idle",
      connected: true,
      machine_pos: { x: 1, y: 2, z: 3 },
      work_pos: { x: 0, y: 0, z: 0 },
      wco: { x: 0, y: 0, z: 0 },
      beam_position_panel: { x: 5, y: 6 },
    });
    expect(st.machine_pos).toEqual({ x: 1, y: 2, z: 3 });
    expect(st.beam_position_panel).toEqual({ x: 5, y: 6 });
  });

  it("carries the driver error and telemetry fields", () => {
    const st = normalizeMachineState({
      status: "ALARM",
      connected: true,
      machine_pos: [1, 2, 3],
      work_pos: [0, 0, 0],
      wco: [1, 2, 3],
      error: { code: 9, title: "Alarm", description: "Hard limit" },
      feed_rate: 1200,
      spindle_speed: 80,
      active_wcs: "G55",
    });
    expect(st.status).toBe("ALARM");
    expect(st.error).toEqual({
      code: 9,
      title: "Alarm",
      description: "Hard limit",
    });
    expect(st.feed_rate).toBe(1200);
    expect(st.spindle_speed).toBe(80);
    expect(st.active_wcs).toBe("G55");
    expect(normalizeMachineState({}).error).toBeNull();
    expect(normalizeMachineState({}).active_wcs).toBeNull();
  });
});

describe("normalizeMachines", () => {
  it("unwraps the machines envelope and maps active", () => {
    const list = normalizeMachines({
      machines: [
        { id: "a", name: "Alpha", driver: null, connected: false, active: true },
      ],
    });
    expect(list).toEqual([
      { id: "a", name: "Alpha", driver: "", connected: false, is_active: true },
    ]);
  });

  it("keeps a bare canonical array", () => {
    const list = normalizeMachines([
      { id: "b", name: "Beta", driver: "grbl", connected: true, is_active: false },
    ]);
    expect(list[0]!.is_active).toBe(false);
  });
});

describe("normalizeJobStatus", () => {
  it("derives the phase from the transitional shape", () => {
    expect(normalizeJobStatus({ active: false, job: null }).phase).toBe("idle");
    expect(normalizeJobStatus({ active: true, job: {} }).phase).toBe("running");
  });

  it("keeps the canonical shape", () => {
    const job = normalizeJobStatus({
      phase: "paused",
      progress_fraction: 0.5,
      eta_seconds: 12,
    });
    expect(job).toEqual({
      active: true,
      phase: "paused",
      progress_fraction: 0.5,
      eta_seconds: 12,
      elapsed_seconds: null,
    });
  });
});

describe("normalizeDocTree", () => {
  it("tolerates a layer list without embedded items", () => {
    const tree = normalizeDocTree({
      active_layer_uid: "l2",
      layers: [
        { uid: "l1", name: "Layer 1", visible: true, color: "#00ccff" },
        { uid: "l2", name: "Layer 2", visible: false, color: "#ff6600" },
      ],
    });
    expect(tree.rev).toBe(-1);
    expect(tree.layers).toHaveLength(2);
    expect(tree.layers[1]!.is_active).toBe(true);
    expect(tree.layers[1]!.workpieces).toEqual([]);
  });

  it("joins the flat backend layout via content_uids and step_uids", () => {
    const tree = normalizeDocTree({
      active_layer_uid: "l1",
      layers: [
        {
          uid: "l1",
          name: "Cut",
          visible: true,
          color: "#00ccff",
          rotary_enabled: false,
          step_uids: ["s1"],
          content_uids: ["wp1", "wp2"],
        },
      ],
      workpieces: [
        {
          uid: "wp1",
          name: "Square",
          pos_mm: [1, 2],
          size_mm: [10, 10],
          angle_deg: 0,
          geometry_revision: 1,
          transform_revision: 1,
        },
        {
          uid: "wp2",
          name: "Circle",
          pos_mm: [20, 20],
          size_mm: [5, 5],
          angle_deg: 45,
          geometry_revision: 2,
          transform_revision: 3,
        },
      ],
      steps: [
        {
          uid: "s1",
          step_type: "Contour",
          typelabel: "Contour",
          name: "Cut everything",
          params: { power: 80 },
          workpiece_uids: ["wp1", "wp2"],
          visible: true,
        },
      ],
    });
    expect(tree.assets_count).toBe(2);
    const layer = tree.layers[0]!;
    expect(layer.is_active).toBe(true);
    expect(layer.rotary_enabled).toBe(false);
    expect(layer.workpieces.map((wp) => wp.name)).toEqual([
      "Square",
      "Circle",
    ]);
    expect(layer.workpieces[0]!.pos_mm).toEqual({ x: 1, y: 2 });
    expect(layer.steps).toHaveLength(1);
    expect(layer.steps[0]!.typelabel).toBe("Contour");
    expect(layer.steps[0]!.visible).toBe(true);
    expect(layer.steps[0]!.params).toEqual({ power: 80 });
  });

  it("prefers embedded items when the backend nests them", () => {
    const tree = normalizeDocTree({
      active_layer_uid: "l1",
      layers: [
        {
          uid: "l1",
          name: "Cut",
          visible: true,
          color: "#00ccff",
          content_uids: ["wp-missing"],
          workpieces: [
            {
              uid: "wp-embedded",
              name: "Embedded",
              pos_mm: [0, 0],
              size_mm: [1, 1],
              angle_deg: 0,
            },
          ],
          steps: [],
        },
      ],
      workpieces: [],
      steps: [],
    });
    expect(tree.layers[0]!.workpieces[0]!.uid).toBe("wp-embedded");
  });
});

describe("normalizeImportAnalysis", () => {
  it("normalizes the manifest and clamps unknown actions", () => {
    const result = normalizeImportAnalysis({
      supported: true,
      action: "something_new",
      importer: "SvgImporter",
      features: ["SELECTABLE_LAYERS"],
      manifest: {
        title: "drawing.svg",
        natural_size_mm: [210, 297],
        layers: [
          {
            id: "edges",
            name: "Edges",
            default_active: true,
            feature_count: 12,
          },
        ],
        warnings: ["scaled"],
        errors: [],
      },
    });
    expect(result.action).toBe("unsupported");
    expect(result.manifest).toEqual({
      title: "drawing.svg",
      natural_size_mm: [210, 297],
      layers: [
        {
          id: "edges",
          name: "Edges",
          default_active: true,
          feature_count: 12,
        },
      ],
      warnings: ["scaled"],
      errors: [],
    });
    expect(normalizeImportAnalysis(null).action).toBe("unsupported");
  });
});

describe("normalizeStepSchema", () => {
  it("maps varset groups onto form fields with defaults", () => {
    const schema = normalizeStepSchema({
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
              default: 80,
              min: 0,
              max: 100,
              step: 1,
              unit: null,
            },
            {
              key: "mode",
              type: "string",
              kind: "choice",
              label: "Mode",
              options: [
                { value: "a", label: "Alpha" },
                "bad-entry",
              ],
            },
            { key: "", type: "boolean", kind: "switch", label: "Bad" },
          ],
        },
      ],
    });
    expect(schema.groups).toHaveLength(1);
    const fields = schema.groups[0]!.fields;
    expect(fields).toHaveLength(3);
    expect(fields[0]).toMatchObject({
      key: "power",
      kind: "slider",
      min: 0,
      max: 100,
      step: 1,
      options: null,
      unit: null,
    });
    expect(fields[1]!.options).toEqual([{ value: "a", label: "Alpha" }]);
  });
});

describe("normalizeOps", () => {
  it("keeps canonical flat point lists", () => {
    const ops = normalizeOps({
      segments: [
        { points: [0, 0, 10, 0, 10, 10], power: 1, rate: 1200, cutting: true },
        { points: [10, 10, 0, 0], power: null, rate: null, cutting: false },
      ],
      estimated_seconds: 42.5,
      distance_mm: 34.2,
      generation_id: 7,
    });
    expect(ops.segments).toHaveLength(2);
    expect(ops.segments[0]!.points).toEqual([0, 0, 10, 0, 10, 10]);
    expect(ops.segments[0]!.power).toBe(1);
    expect(ops.segments[1]!.power).toBe(0);
    expect(ops.segments[1]!.rate).toBeNull();
    expect(ops.estimated_seconds).toBe(42.5);
    expect(ops.distance_mm).toBe(34.2);
    expect(ops.generation_id).toBe("7");
  });

  it("folds paired point lists into flat coordinates", () => {
    const ops = normalizeOps({
      segments: [
        {
          points: [
            [1, 2],
            [3, 4],
          ],
          power: 0.5,
          cutting: true,
        },
      ],
      estimated_seconds: null,
      distance_mm: 0,
      generation_id: "abc",
    });
    expect(ops.segments[0]!.points).toEqual([1, 2, 3, 4]);
    expect(ops.generation_id).toBe("abc");
  });

  it("drops degenerate segments and malformed payloads", () => {
    const ops = normalizeOps({
      segments: [
        { points: [5, 5], power: 1, cutting: true },
        { points: "bad", power: 1, cutting: true },
        { points: [0, 0, 4, 0], power: 0.2, cutting: true },
      ],
    });
    expect(ops.segments).toHaveLength(1);
    expect(ops.segments[0]!.points).toEqual([0, 0, 4, 0]);
    expect(ops.segments[0]!.power).toBe(0.2);
    expect(ops.estimated_seconds).toBeNull();
  });
});
