import { describe, expect, it } from "vitest";
import { docStore } from "./stores/docStore";
import { machineStore } from "./stores/machineStore";
import { uiStore } from "./stores/uiStore";
import { buildShell, type ShellActions } from "./App";

function recordingActions(): ShellActions & { calls: string[] } {
  const calls: string[] = [];
  const actions: ShellActions = {
    newDoc: () => calls.push("new"),
    openProject: () => calls.push("open"),
    saveProject: () => calls.push("save"),
    exportGcode: () => calls.push("export"),
    undo: () => calls.push("undo"),
    redo: () => calls.push("redo"),
    deleteSelected: () => calls.push("delete"),
    duplicateSelected: () => calls.push("duplicate"),
    flip: () => calls.push("flip"),
    align: () => calls.push("align"),
    autoLayout: () => calls.push("auto-layout"),
    addStep: () => calls.push("add-step"),
    machineSettings: () => calls.push("machine-settings"),
    openSettings: () => calls.push("open-settings"),
    importFile: () => calls.push("import"),
    setTool: (tool) => calls.push(`tool:${tool}`),
    viewPreset: (preset) => calls.push(`view:${preset}`),
    frameSelection: () => calls.push("frame"),
    zoom: () => calls.push("zoom"),
    toggleToolpaths: () => calls.push("toolpaths"),
    startJob: () => calls.push("start"),
    homeAll: () => calls.push("home"),
    setOriginHere: () => calls.push("set-origin"),
    goToOrigin: () => calls.push("go-origin"),
    refreshDoc: async () => undefined,
    about: () => undefined,
  };
  return { ...actions, calls };
}

function ribbonButton(
  host: HTMLElement,
  label: string,
): HTMLButtonElement {
  const spans = Array.from(
    host.querySelectorAll<HTMLSpanElement>(
      ".floating-toolbar .ribbon-btn-label",
    ),
  );
  const found = spans.find((span) => span.textContent === label);
  expect(found, `ribbon button ${label}`).toBeDefined();
  return found!.closest("button") as HTMLButtonElement;
}

function activateRibbonTab(
  host: HTMLElement,
  label: string,
): void {
  const tabs = Array.from(
    host.querySelectorAll<HTMLButtonElement>(".ribbon-tab"),
  );
  const tab = tabs.find((el) => el.textContent === label);
  expect(tab, `ribbon tab ${label}`).toBeDefined();
  tab!.click();
}

describe("app shell", () => {
  it("mounts the full layout skeleton", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const shell = buildShell(host);

    expect(host.classList.contains("app")).toBe(true);
    expect(host.querySelector(".topbar")).not.toBeNull();
    expect(host.querySelector(".floating-toolbar")).not.toBeNull();
    expect(shell.leftPanel.root.classList.contains("floating-panel")).toBe(
      true,
    );
    expect(shell.leftPanel.root.querySelector(".tabs")).not.toBeNull();
    expect(host.querySelector(".viewport-host")).not.toBeNull();
    expect(host.querySelector(".transport-pill")).not.toBeNull();
    expect(shell.rightPanel.root.classList.contains("floating-panel")).toBe(
      true,
    );
    expect(shell.rightPanel.root.querySelector(".panel")).not.toBeNull();
    expect(host.querySelector(".statusbar")).not.toBeNull();
    expect(host.querySelector(".command-input")).not.toBeNull();
    expect(host.querySelector(".toasts")).not.toBeNull();
    host.replaceChildren();
  });

  it("shows an empty document tree and connection pill", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const shell = buildShell(host);
    expect(shell.leftPanel.docContent.querySelector(".tree-empty")).not.toBeNull();
    expect(
      shell.topbar.connectionPill.classList.contains("disconnected"),
    ).toBe(true);
    host.replaceChildren();
  });

  it("renders the ribbon tabs", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    buildShell(host);
    const tabs = Array.from(
      host.querySelectorAll<HTMLButtonElement>(".ribbon-tab"),
    ).map((el) => el.textContent);
    expect(tabs).toEqual(["Model", "Machine", "Job", "View", "Help"]);
    expect(
      host.querySelector(".ribbon-tab.active")?.textContent,
    ).toBe("Model");
    host.replaceChildren();
  });

  it("builds the model tab with labeled groups and items", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    buildShell(host);
    const groupLabels = Array.from(
      host.querySelectorAll<HTMLDivElement>(".ribbon-group-label"),
    ).map((el) => el.textContent);
    expect(groupLabels).toEqual([
      "Tools",
      "Project",
      "Modify",
      "Arrange",
      "Layout",
    ]);
    const labels = Array.from(
      host.querySelectorAll<HTMLSpanElement>(".ribbon-btn-label"),
    ).map((span) => span.textContent);
    expect(labels).toEqual([
      "Select",
      "Move",
      "Rotate",
      "Scale",
      "New",
      "Open",
      "Save",
      "Import",
      "Export",
      "Duplicate",
      "Delete",
      "Flip H",
      "Flip V",
      "Left",
      "Center H",
      "Right",
      "Top",
      "Center V",
      "Bottom",
      "Auto layout",
    ]);
    expect(host.querySelectorAll(".ribbon-sep").length).toBe(4);
    host.replaceChildren();
  });

  it("wires model tab actions through the shell actions", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const actions = recordingActions();
    buildShell(host, actions);

    ribbonButton(host, "Select").click();
    expect(actions.calls).toContain("tool:select");
    ribbonButton(host, "New").click();
    ribbonButton(host, "Open").click();
    ribbonButton(host, "Save").click();
    ribbonButton(host, "Import").click();
    ribbonButton(host, "Export").click();
    expect(actions.calls).toEqual([
      "tool:select",
      "new",
      "open",
      "save",
      "import",
      "export",
    ]);

    docStore.setState({ selection: ["uid-1"] });
    ribbonButton(host, "Flip H").click();
    ribbonButton(host, "Left").click();
    expect(actions.calls).toContain("flip");
    expect(actions.calls).toContain("align");
    host.replaceChildren();
  });

  it("exposes machine commands on the machine tab", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const actions = recordingActions();
    machineStore.setState({
      machines: [
        {
          id: "m1",
          name: "Machine",
          driver: "GrblSerialDriver",
          connected: false,
          is_active: true,
        },
      ],
      activeMachineId: "m1",
    });
    buildShell(host, actions);

    activateRibbonTab(host, "Machine");
    expect(
      host.querySelectorAll(".ribbon-btn-label").length,
    ).toBeGreaterThan(0);
    ribbonButton(host, "Home all").click();
    ribbonButton(host, "Set origin").click();
    ribbonButton(host, "Go to origin").click();
    expect(actions.calls).toEqual(["home", "set-origin", "go-origin"]);
    machineStore.setState({ machines: [], activeMachineId: null });
    host.replaceChildren();
  });

  it("disables edit and job buttons until selection and machine allow them", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const actions = recordingActions();
    docStore.setState({ selection: [] });
    buildShell(host, actions);

    expect(ribbonButton(host, "Duplicate").disabled).toBe(true);
    expect(ribbonButton(host, "Delete").disabled).toBe(true);

    docStore.setState({ selection: ["uid-1"] });
    expect(ribbonButton(host, "Delete").disabled).toBe(false);
    ribbonButton(host, "Delete").click();
    expect(actions.calls).toContain("delete");

    docStore.setState({ selection: [] });
    expect(ribbonButton(host, "Delete").disabled).toBe(true);
    host.replaceChildren();
  });

  it("wires the job and view tabs", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const actions = recordingActions();
    buildShell(host, actions);

    activateRibbonTab(host, "Job");
    ribbonButton(host, "Frame").click();
    ribbonButton(host, "Export").click();
    expect(actions.calls).toEqual(["frame", "export"]);

    activateRibbonTab(host, "View");
    ribbonButton(host, "Top").click();
    ribbonButton(host, "Fit").click();
    ribbonButton(host, "Out").click();
    ribbonButton(host, "In").click();
    expect(actions.calls).toEqual([
      "frame",
      "export",
      "view:top",
      "view:fit",
      "zoom",
      "zoom",
    ]);
    host.replaceChildren();
  });

  it("toggles toolpaths from the view tab", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    buildShell(host);
    const initial = uiStore.getState().showToolpaths;
    activateRibbonTab(host, "View");
    let btn = ribbonButton(host, "Toolpaths");
    expect(btn.classList.contains("active")).toBe(initial);

    btn.click();
    btn = ribbonButton(host, "Toolpaths");
    expect(uiStore.getState().showToolpaths).toBe(!initial);
    expect(btn.classList.contains("active")).toBe(!initial);

    btn.click();
    expect(uiStore.getState().showToolpaths).toBe(initial);
    host.replaceChildren();
  });

  it("floating panels collapse to stubs and reopen", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const shell = buildShell(host);

    const collapseLeft = shell.leftPanel.root.querySelector(
      ".panel-collapse",
    ) as HTMLButtonElement;
    collapseLeft.click();
    expect(shell.leftPanel.root.hidden).toBe(true);
    expect(shell.leftPanel.stub.hidden).toBe(false);
    shell.leftPanel.stub.click();
    expect(shell.leftPanel.root.hidden).toBe(false);
    expect(shell.leftPanel.stub.hidden).toBe(true);

    const collapseRight = shell.rightPanel.root.querySelector(
      ".panel-collapse",
    ) as HTMLButtonElement;
    collapseRight.click();
    expect(shell.rightPanel.root.hidden).toBe(true);
    expect(shell.rightPanel.stub.hidden).toBe(false);
    shell.rightPanel.stub.click();
    expect(shell.rightPanel.root.hidden).toBe(false);
    expect(shell.rightPanel.stub.hidden).toBe(true);
    host.replaceChildren();
  });

  it("command palette filters and executes commands", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const actions = recordingActions();
    const shell = buildShell(host, actions);
    const input = shell.topbar.commandInput;

    input.value = "fit";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new FocusEvent("focus"));
    const rows = Array.from(
      host.querySelectorAll<HTMLElement>(
        ".command-popup .menu-item .menu-item-label",
      ),
    ).map((span) => span.textContent);
    expect(rows).toEqual(["Fit view"]);

    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(actions.calls).toContain("view:fit");

    input.value = "undo";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(actions.calls).toContain("undo");

    input.value = "delete";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(actions.calls).toContain("delete");
    host.replaceChildren();
  });
});
