import { describe, expect, it } from "vitest";
import { CommandHistory, ConsoleLog } from "./consoleHistory";

describe("CommandHistory", () => {
  it("ignores empty commands", () => {
    const history = new CommandHistory();
    history.push("   ");
    expect(history.size).toBe(0);
    expect(history.older("G0")).toBeNull();
  });

  it("recalls commands from newest to oldest", () => {
    const history = new CommandHistory();
    history.push("G0 X10");
    history.push("G0 Y20");
    history.push("M3 S50");

    expect(history.older("")).toBe("M3 S50");
    expect(history.older("M3 S50")).toBe("G0 Y20");
    expect(history.older("G0 Y20")).toBe("G0 X10");
    expect(history.older("G0 X10")).toBe("G0 X10");
  });

  it("moves forward and restores the draft at the end", () => {
    const history = new CommandHistory();
    history.push("G0 X10");
    history.push("G0 Y20");

    expect(history.older("G1 Z5")).toBe("G0 Y20");
    expect(history.older("G0 Y20")).toBe("G0 X10");
    expect(history.newer()).toBe("G0 Y20");
    expect(history.newer()).toBe("G1 Z5");
    expect(history.newer()).toBeNull();
  });

  it("collapses consecutive duplicates", () => {
    const history = new CommandHistory();
    history.push("G0 X10");
    history.push("G0 X10");
    expect(history.size).toBe(1);
    expect(history.older("")).toBe("G0 X10");
  });

  it("respects the capacity limit", () => {
    const history = new CommandHistory(2);
    history.push("G0");
    history.push("G1");
    history.push("G2");
    expect(history.size).toBe(2);
    expect(history.older("")).toBe("G2");
    expect(history.older("G2")).toBe("G1");
    expect(history.older("G1")).toBe("G1");
  });

  it("starts a fresh browse after reset", () => {
    const history = new CommandHistory();
    history.push("G0 X10");
    history.push("G0 Y20");
    history.older("draft");
    history.reset();
    expect(history.newer()).toBeNull();
    expect(history.older("other")).toBe("G0 Y20");
  });
});

describe("ConsoleLog", () => {
  it("appends typed lines with ids", () => {
    const log = new ConsoleLog();
    log.append("cmd", "> G0 X0");
    log.append("ok", "sent");
    const lines = log.all();
    expect(lines).toHaveLength(2);
    expect(lines[0]!.kind).toBe("cmd");
    expect(lines[0]!.text).toBe("> G0 X0");
    expect(lines[1]!.kind).toBe("ok");
    expect(lines[1]!.id).toBeGreaterThan(lines[0]!.id);
  });

  it("caps the buffer at the limit", () => {
    const log = new ConsoleLog(3);
    for (let i = 0; i < 5; i++) {
      log.append("ok", `line ${i}`);
    }
    const lines = log.all();
    expect(lines).toHaveLength(3);
    expect(lines[0]!.text).toBe("line 2");
    expect(lines[2]!.text).toBe("line 4");
  });
});
