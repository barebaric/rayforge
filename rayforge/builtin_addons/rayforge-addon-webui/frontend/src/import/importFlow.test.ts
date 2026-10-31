import { describe, expect, it, vi } from "vitest";
import type { ImportAnalysis, ImportResult } from "../api/types";
import {
  ImportFlow,
  analysisReceived,
  importFailed,
  importFinished,
  initialImportState,
} from "./importFlow";

const bitmapAnalysis: ImportAnalysis = {
  supported: true,
  action: "interactive_config",
  importer: "BitmapImporter",
  features: ["BITMAP_TRACING"],
  manifest: {
    title: "photo.png",
    natural_size_mm: [100, 80],
    layers: [],
    warnings: [],
    errors: [],
  },
};

const vectorAnalysis: ImportAnalysis = {
  supported: true,
  action: "direct_load",
  importer: "DxfImporter",
  features: [],
  manifest: null,
};

const unsupportedAnalysis: ImportAnalysis = {
  supported: false,
  action: "unsupported",
  importer: null,
  features: [],
  manifest: null,
};

const importResult: ImportResult = {
  created: ["wp-1"],
  layer_uid: "layer-1",
};

function makeFile(): File {
  return new File(["data"], "drawing.svg", { type: "image/svg+xml" });
}

describe("pure transitions", () => {
  it("routes supported actions to their phases", () => {
    expect(
      analysisReceived(initialImportState, bitmapAnalysis).phase,
    ).toBe("interactive_config");
    expect(
      analysisReceived(initialImportState, vectorAnalysis).phase,
    ).toBe("direct_load");
    expect(
      analysisReceived(initialImportState, unsupportedAnalysis).phase,
    ).toBe("unsupported");
  });

  it("records results and failures", () => {
    const done = importFinished(initialImportState, importResult);
    expect(done.phase).toBe("done");
    expect(done.result).toEqual(importResult);

    const failed = importFailed(initialImportState, new Error("boom"));
    expect(failed.phase).toBe("error");
    expect(failed.error).toBe("boom");
  });
});

describe("ImportFlow", () => {
  it("analyzes then imports with a placement", async () => {
    const analyze = vi.fn().mockResolvedValue(vectorAnalysis);
    const run = vi.fn().mockResolvedValue(importResult);
    const flow = new ImportFlow({ analyze, run });

    const file = makeFile();
    const afterAnalyze = await flow.start(file);
    expect(afterAnalyze.phase).toBe("direct_load");
    expect(analyze).toHaveBeenCalledWith(file);

    const afterRun = await flow.run({ spec: { type: "PassthroughSpec" } });
    expect(afterRun.phase).toBe("done");
    expect(afterRun.result).toEqual(importResult);
    expect(run).toHaveBeenCalledWith(file, {
      spec: { type: "PassthroughSpec" },
    });
  });

  it("stops with an error state when analysis fails", async () => {
    const flow = new ImportFlow({
      analyze: vi.fn().mockRejectedValue(new Error("offline")),
      run: vi.fn(),
    });
    const state = await flow.start(makeFile());
    expect(state.phase).toBe("error");
    expect(state.error).toBe("offline");
  });

  it("propagates import failures", async () => {
    const flow = new ImportFlow({
      analyze: vi.fn().mockResolvedValue(bitmapAnalysis),
      run: vi.fn().mockRejectedValue(new Error("no space")),
    });
    await flow.start(makeFile());
    const state = await flow.run({});
    expect(state.phase).toBe("error");
    expect(state.error).toBe("no space");
  });

  it("ignores run before analysis produced a file", async () => {
    const run = vi.fn().mockResolvedValue(importResult);
    const flow = new ImportFlow({
      analyze: vi.fn(),
      run,
    });
    const state = await flow.run({});
    expect(state.phase).toBe("idle");
    expect(run).not.toHaveBeenCalled();
  });
});
