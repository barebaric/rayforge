import { describe, expect, it } from "vitest";
import { deriveJobCommands, fmtEta } from "./jobState";

const CONNECTED_WITH_CONTENT = {
  connected: true,
  hasContent: true,
};

describe("deriveJobCommands", () => {
  it("enables start only when connected with content", () => {
    expect(
      deriveJobCommands({
        active: false,
        phase: "idle",
        ...CONNECTED_WITH_CONTENT,
      }).startEnabled,
    ).toBe(true);
    expect(
      deriveJobCommands({
        active: false,
        phase: "idle",
        connected: false,
        hasContent: true,
      }).startEnabled,
    ).toBe(false);
    expect(
      deriveJobCommands({
        active: false,
        phase: "idle",
        connected: true,
        hasContent: false,
      }).startEnabled,
    ).toBe(false);
  });

  it("walks the idle -> running -> paused -> running -> done cycle", () => {
    let state = deriveJobCommands({
      active: false,
      phase: "idle",
      ...CONNECTED_WITH_CONTENT,
    });
    expect(state.kind).toBe("idle");
    expect(state.phaseLabel).toBe("Idle");
    expect(state.holdEnabled).toBe(false);
    expect(state.cancelEnabled).toBe(false);

    state = deriveJobCommands({
      active: true,
      phase: "running",
      ...CONNECTED_WITH_CONTENT,
    });
    expect(state.kind).toBe("running");
    expect(state.phaseLabel).toBe("Running");
    expect(state.startEnabled).toBe(false);
    expect(state.holdEnabled).toBe(true);
    expect(state.holdLabel).toBe("Pause");
    expect(state.holdIcon).toBe("pause");
    expect(state.cancelEnabled).toBe(true);
    expect(state.frameEnabled).toBe(false);

    state = deriveJobCommands({
      active: true,
      phase: "paused",
      ...CONNECTED_WITH_CONTENT,
    });
    expect(state.kind).toBe("paused");
    expect(state.phaseLabel).toBe("Paused");
    expect(state.holdLabel).toBe("Resume");
    expect(state.holdIcon).toBe("play");

    state = deriveJobCommands({
      active: true,
      phase: "running",
      ...CONNECTED_WITH_CONTENT,
    });
    expect(state.holdLabel).toBe("Pause");

    state = deriveJobCommands({
      active: false,
      phase: "idle",
      ...CONNECTED_WITH_CONTENT,
    });
    expect(state.kind).toBe("idle");
    expect(state.startEnabled).toBe(true);
    expect(state.cancelEnabled).toBe(false);
    expect(state.frameEnabled).toBe(true);
  });

  it("treats unknown active phases as running", () => {
    const state = deriveJobCommands({
      active: true,
      phase: "weird",
      ...CONNECTED_WITH_CONTENT,
    });
    expect(state.kind).toBe("running");
    expect(state.phaseLabel).toBe("Running");
  });
});

describe("fmtEta", () => {
  it("formats mm:ss under an hour", () => {
    expect(fmtEta(0)).toBe("00:00");
    expect(fmtEta(65)).toBe("01:05");
    expect(fmtEta(3599)).toBe("59:59");
  });

  it("formats h:mm:ss past an hour", () => {
    expect(fmtEta(3600)).toBe("1:00:00");
    expect(fmtEta(3723)).toBe("1:02:03");
  });

  it("shows dashes for unknown or invalid values", () => {
    expect(fmtEta(null)).toBe("--:--");
    expect(fmtEta(-5)).toBe("--:--");
    expect(fmtEta(Number.NaN)).toBe("--:--");
  });
});
