import { api } from "../../stores/api";
import { jobStore } from "../../stores/jobStore";
import { machineStore } from "../../stores/machineStore";
import { pushToast } from "../../stores/uiStore";
import { button } from "../ui/Button";
import { numberField } from "../ui/NumberField";
import { panel } from "../ui/Panel";
import { select, type SelectHandle } from "../ui/Select";
import { el } from "../ui/dom";
import { icon, type IconName } from "../ui/icons";
import {
  jogPadLayout,
  jogTooltip,
  type JogButton,
  type PanelOrientation,
} from "./jogFlags";

const WCS_SLOTS = ["G54", "G55", "G56", "G57", "G58", "G59"];
const JOG_DISTANCES = [1, 5, 10];

export interface ControlSectionRefs {
  root: HTMLElement;
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

const SCREEN_ICON: Record<string, IconName> = {
  up: "chevronUp",
  down: "chevronDown",
  left: "chevronLeft",
  right: "chevronRight",
};

/**
 * The machine "Control" section: jog pad, homing, work origin/WCS,
 * framing and the laser test fire controls. Built once; store
 * subscriptions toggle availability and re-derive the pad layout.
 */
export function buildControlSection(): ControlSectionRefs {
  const root = el("div", "control-section");
  let distanceMm = 10;
  let speedMmMin = 0;
  let powerPercent = 5;
  let layoutSignature = "";

  const wrap = (action: () => Promise<unknown>, what: string): void => {
    action().catch((err: unknown) =>
      pushToast("error", `${what}: ${describe(err)}`),
    );
  };

  const jog = (direction: string): void => {
    wrap(
      () => api.jog(direction, distanceMm, speedMmMin > 0 ? speedMmMin : undefined),
      "Jog failed",
    );
  };

  // -- Jog pad -----------------------------------------------------

  const jogPanel = panel("Jog");
  const jogGrid = el("div", "jog-grid");
  const distanceRow = el("div", "seg-row");
  const distanceButtons = new Map<number, HTMLButtonElement>();
  for (const mm of JOG_DISTANCES) {
    const btn = el("button", "seg-btn");
    btn.type = "button";
    btn.textContent = `${mm}`;
    btn.title = `Jog distance ${mm} mm`;
    btn.addEventListener("click", () => {
      distanceMm = mm;
      for (const [value, other] of distanceButtons) {
        other.classList.toggle("active", value === distanceMm);
      }
    });
    distanceButtons.set(mm, btn);
    distanceRow.appendChild(btn);
  }
  distanceButtons.get(distanceMm)!.classList.add("active");
  const speedField = numberField({
    label: "Speed mm/min (0 = driver default)",
    value: 0,
    min: 0,
    step: 100,
    onCommit: (value) => {
      speedMmMin = value;
    },
  });
  jogPanel.body.append(jogGrid, distanceRow, speedField.root);
  root.appendChild(jogPanel.root);

  const dpadButtons: HTMLButtonElement[] = [];

  function makeDpadButton(
    title: string,
    iconName: IconName,
    axisLabel: string,
  ): HTMLButtonElement {
    const btn = el("button", "btn jog-btn");
    btn.type = "button";
    btn.title = title;
    btn.appendChild(icon(iconName, 14));
    const label = el("span", "jog-axis");
    label.textContent = axisLabel;
    btn.appendChild(label);
    jogGrid.appendChild(btn);
    dpadButtons.push(btn);
    return btn;
  }

  function rebuildDpad(): void {
    const { viewport } = machineStore.getState();
    const orientation: PanelOrientation =
      viewport?.panel_orientation ?? "native";
    const layout = jogPadLayout(
      {
        x_axis_right: viewport?.x_axis_right ?? false,
        y_axis_down: viewport?.y_axis_down ?? false,
        x_axis_negative: viewport?.x_axis_negative ?? false,
        y_axis_negative: viewport?.y_axis_negative ?? false,
      },
      orientation,
    );
    const signature = [
      layout.right.direction,
      layout.up.direction,
      viewport?.x_axis_negative,
      viewport?.y_axis_negative,
    ].join(",");
    if (signature === layoutSignature) return;
    layoutSignature = signature;

    for (const child of [...jogGrid.children]) {
      jogGrid.removeChild(child);
    }
    dpadButtons.length = 0;

    const spots: Array<[JogButton, string, number]> = [
      [layout.up, "2", 1],
      [layout.left, "1", 2],
      [layout.right, "3", 2],
      [layout.down, "2", 3],
    ];
    for (const [spec, col, row] of spots) {
      const polarity = spec.direction.startsWith("+") ? "+" : "-";
      const btn = makeDpadButton(
        jogTooltip(spec),
        SCREEN_ICON[spec.screen] ?? "move",
        `${spec.axis.toUpperCase()}${polarity}`,
      );
      btn.style.gridColumn = col;
      btn.style.gridRow = String(row);
      btn.addEventListener("click", () => jog(spec.direction));
    }

    const zUp = makeDpadButton("Jog up (panel +z)", "chevronUp", "Z+");
    zUp.style.gridColumn = "3";
    zUp.style.gridRow = "1";
    zUp.addEventListener("click", () => jog("+z"));

    const zDown = makeDpadButton("Jog down (panel -z)", "chevronDown", "Z-");
    zDown.style.gridColumn = "3";
    zDown.style.gridRow = "3";
    zDown.addEventListener("click", () => jog("-z"));

    const homeAll = makeDpadButton("Home all axes", "home", "");
    homeAll.classList.add("jog-home");
    homeAll.style.gridColumn = "2";
    homeAll.style.gridRow = "2";
    homeAll.addEventListener("click", () =>
      wrap(() => api.homeMachine(), "Homing failed"),
    );
  }

  // -- Homing ------------------------------------------------------

  const homePanel = panel("Homing");
  const homeRow = el("div", "btn-row");
  const homeAllBtn = button({
    label: "Home all",
    onClick: () => wrap(() => api.homeMachine(), "Homing failed"),
  });
  homeRow.appendChild(homeAllBtn);
  const axisHomeButtons = ["x", "y", "z"].map((axis) =>
    button({
      label: `Home ${axis.toUpperCase()}`,
      onClick: () => wrap(() => api.homeMachine([axis]), "Homing failed"),
    }),
  );
  for (const btn of axisHomeButtons) homeRow.appendChild(btn);
  homePanel.body.appendChild(homeRow);
  root.appendChild(homePanel.root);

  // -- Work origin / WCS --------------------------------------------

  const originPanel = panel("Work origin");
  const originRow = el("div", "btn-row");
  const setOriginBtn = button({
    label: "Set origin here",
    title: "Zero the active WCS at the current position",
    onClick: () =>
      wrap(() => api.setWorkOriginHere(), "Set work origin failed"),
  });
  const gotoOriginBtn = button({
    label: "Go to origin",
    title: "Move to the work-area origin",
    onClick: () => {
      const origin =
        machineStore.getState().viewport?.workarea_origin_panel;
      if (!origin) {
        pushToast("warning", "Work-area origin unknown");
        return;
      }
      wrap(() => api.moveTo(origin.x, origin.y), "Move failed");
    },
  });
  originRow.append(setOriginBtn, gotoOriginBtn);
  const wcsRow = el("div", "wcs-row");
  const wcsSelect: SelectHandle = select(
    "Coordinate system",
    WCS_SLOTS.map((slot) => ({ value: slot, label: slot })),
    "G54",
    (slot) => wrap(() => api.selectWcs(slot), "Select WCS failed"),
  );
  const syncWcsBtn = button({
    label: "Sync",
    title: "Re-read WCS offsets from the device",
    onClick: () => wrap(() => api.syncWcs(), "WCS sync failed"),
  });
  wcsRow.append(wcsSelect.root, syncWcsBtn);
  originPanel.body.append(originRow, wcsRow);
  root.appendChild(originPanel.root);

  // -- Frame & laser -------------------------------------------------

  const laserPanel = panel("Frame & laser");
  const frameBtn = button({
    label: "Frame job",
    iconName: "frame",
    title: "Trace the job bounding box without firing",
    onClick: () => wrap(() => api.frame(), "Framing failed"),
  });
  const powerField = numberField({
    label: "Power %",
    value: powerPercent,
    min: 0,
    max: 100,
    onCommit: (value) => {
      powerPercent = value;
    },
  });
  const setPowerBtn = button({
    label: "Set power",
    onClick: () =>
      wrap(() => api.setPower(powerPercent / 100), "Set power failed"),
  });
  const focusBtn = button({
    label: "Focus power",
    onClick: () =>
      wrap(
        () => api.setFocusPower(powerPercent / 100),
        "Set focus power failed",
      ),
  });
  const fireBtn = el("button", "btn btn-danger fire-btn");
  fireBtn.type = "button";
  fireBtn.textContent = "TEST FIRE";
  fireBtn.title = "Hold to fire the laser at the set power";
  const fire = (on: boolean): void => {
    wrap(
      () => api.setPower(on ? powerPercent / 100 : 0),
      "Laser test failed",
    );
  };
  fireBtn.addEventListener("pointerdown", () => fire(true));
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) {
    fireBtn.addEventListener(ev, () => fire(false));
  }
  const laserRow = el("div", "btn-row");
  laserRow.append(setPowerBtn, focusBtn, fireBtn);
  laserPanel.body.append(frameBtn, powerField.root, laserRow);
  root.appendChild(laserPanel.root);

  // -- Reactive sync ---------------------------------------------------

  function sync(): void {
    const { machineState, viewport } = machineStore.getState();
    const connected = machineState?.connected ?? false;
    const jobActive = jobStore.getState().active;
    const movable = connected && !jobActive;

    rebuildDpad();
    for (const btn of dpadButtons) btn.disabled = !movable;
    homeAllBtn.disabled = !movable;
    for (const btn of axisHomeButtons) btn.disabled = !movable;
    setOriginBtn.disabled = !movable;
    gotoOriginBtn.disabled = !movable;
    syncWcsBtn.disabled = !connected;
    frameBtn.disabled = !connected || jobActive;
    wcsSelect.setDisabled(!connected);
    setPowerBtn.disabled = !connected;
    focusBtn.disabled = !connected;
    fireBtn.disabled = !connected;

    const wcsName =
      machineState?.active_wcs ?? viewport?.active_wcs.name ?? "G54";
    if (WCS_SLOTS.includes(wcsName)) {
      wcsSelect.setValue(wcsName);
    }
  }

  machineStore.subscribe(sync);
  jobStore.subscribe(sync);
  sync();

  return { root };
}
