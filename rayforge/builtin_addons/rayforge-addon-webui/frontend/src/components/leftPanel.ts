import { machineStore } from "../stores/machineStore";
import { setLeftTab, uiStore, type LeftTab } from "../stores/uiStore";
import { buildLayersPanel, type LayersPanelActions } from "./layersPanel";
import { clearElement, el } from "./ui/dom";
import { floatingCard, type FloatingCardRefs } from "./ui/FloatingCard";
import { tabs, type TabsHandle } from "./ui/Tabs";

export interface LeftPanelRefs extends FloatingCardRefs {
  docContent: HTMLElement;
  machineContent: HTMLElement;
}

export function buildLeftPanel(
  actions: LayersPanelActions,
): LeftPanelRefs {
  const tabHost = el("div", "panel-tabs");
  const content = el("div", "panel-content");

  const layersPanel = buildLayersPanel(actions);
  const docContent = el("div", "doc-tree-host");
  docContent.appendChild(layersPanel.root);
  const machineContent = el("div", "machine-host");

  let tabsHandle: TabsHandle | null = null;
  let shownTab = uiStore.getState().activeLeftTab;

  function showTab(id: string): void {
    shownTab = id as LeftTab;
    setLeftTab(id as "document" | "machine");
    tabsHandle?.setActive(id);
    clearElement(content);
    content.appendChild(id === "machine" ? machineContent : docContent);
  }

  tabsHandle = tabs(
    [
      { id: "document", label: "Document" },
      { id: "machine", label: "Machine" },
    ],
    uiStore.getState().activeLeftTab,
    showTab,
  );
  tabHost.appendChild(tabsHandle.root);

  const body = el("div", "floating-panel-body");
  body.append(tabHost, content);
  const card = floatingCard("left", "Design", "layers", body);

  const renderMachineTab = () => {
    clearElement(machineContent);
    machineContent.appendChild(buildMachineContent());
  };
  machineStore.subscribe(renderMachineTab);
  uiStore.subscribe((state) => {
    if (state.activeLeftTab !== shownTab) {
      showTab(state.activeLeftTab);
      return;
    }
    if (state.activeLeftTab === "machine") renderMachineTab();
  });

  showTab(uiStore.getState().activeLeftTab);

  return { root: card.root, stub: card.stub, docContent, machineContent };
}

function fmtAxis(value: number | null): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(1)
    : "--";
}

function fmtVec(
  vec: { x: number | null; y: number | null; z: number | null } | null,
): string {
  if (!vec) return "--, --, --";
  return `${fmtAxis(vec.x)}, ${fmtAxis(vec.y)}, ${fmtAxis(vec.z)}`;
}

function buildMachineContent(): HTMLElement {
  const { machines, machineState, activeMachineId, viewport } =
    machineStore.getState();
  const wrap = el("div", "machine-content");
  const active = machines.find((m) => m.id === activeMachineId);

  if (machines.length > 0) {
    const list = el("div", "machine-list");
    list.title =
      machines.length === 1
        ? "The web API controls the active machine; this list is display-only."
        : "The web API controls the active machine (marked); selection is display-only.";
    for (const machine of machines) {
      const row = el("div", "machine-row");
      row.classList.toggle("active-machine", machine.is_active);
      const info = el("div", "machine-row-info");
      const name = el("span", "machine-name");
      name.textContent = machine.name;
      const driver = el("span", "machine-driver");
      driver.textContent = machine.driver || "no driver";
      info.append(name, driver);
      const pill = el("span", `machine-pill ${machine.connected ? "on" : "off"}`);
      pill.textContent = machine.connected ? "Connected" : "Offline";
      row.append(info, pill);
      list.appendChild(row);
    }
    wrap.appendChild(list);
  }

  const info = el("div", "machine-info");
  const wcs =
    machineState?.active_wcs ?? viewport?.active_wcs.name ?? "--";
  const rows: Array<[string, string]> = [
    ["Driver", active?.driver || "--"],
    [
      "Connection",
      machineState
        ? machineState.connected
          ? "Connected"
          : "Offline"
        : "--",
    ],
    ["Status", machineState?.status ?? "--"],
    ["Machine pos", fmtVec(machineState?.machine_pos ?? null)],
    ["Work pos", fmtVec(machineState?.work_pos ?? null)],
    ["WCO", fmtVec(machineState?.wco ?? null)],
    ["Active WCS", wcs],
  ];
  if (machineState?.feed_rate !== null && machineState?.feed_rate !== undefined) {
    rows.push(["Feed", `${machineState.feed_rate}`]);
  }
  if (
    machineState?.spindle_speed !== null &&
    machineState?.spindle_speed !== undefined
  ) {
    rows.push(["Spindle", `${machineState.spindle_speed}`]);
  }
  for (const [key, value] of rows) {
    const row = el("div", "kv-row");
    const k = el("span", "kv-key");
    k.textContent = key;
    const v = el("span", "kv-value mono");
    v.textContent = value;
    row.append(k, v);
    info.appendChild(row);
  }
  wrap.appendChild(info);
  return wrap;
}
