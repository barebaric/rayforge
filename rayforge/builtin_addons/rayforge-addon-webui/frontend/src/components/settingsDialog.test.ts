import { describe, expect, it } from "vitest";
import { getPath, setPath } from "./settingsDialog";

describe("getPath", () => {
  it("reads plain keys and nested paths", () => {
    const values = { theme: "dark", machine: { name: "Laser" } };
    expect(getPath(values, "theme")).toBe("dark");
    expect(getPath(values, "machine.name")).toBe("Laser");
    expect(getPath(values, "machine.missing")).toBeUndefined();
  });

  it("reads array indexes via dotted paths", () => {
    const values = { axis_extents: [400, 300] };
    expect(getPath(values, "axis_extents.0")).toBe(400);
    expect(getPath(values, "axis_extents.1")).toBe(300);
  });
});

describe("setPath", () => {
  it("overwrites plain keys immutably", () => {
    const original = { theme: "dark" };
    const next = setPath(original, "theme", "light");
    expect(next).toEqual({ theme: "light" });
    expect(original.theme).toBe("dark");
  });

  it("writes into arrays by index", () => {
    const next = setPath({ axis_extents: [400, 300] }, "axis_extents.1", 250);
    expect(next.axis_extents).toEqual([400, 250]);
  });

  it("creates missing intermediate objects", () => {
    const next = setPath({}, "machine.name", "X");
    expect(next).toEqual({ machine: { name: "X" } });
  });
});
