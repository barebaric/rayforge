import { api } from "../stores/api";
import { pushToast } from "../stores/uiStore";
import type { LayerInfo } from "../api/types";
import { button } from "./ui/Button";
import { el } from "./ui/dom";

export interface StepPickerTarget {
  uid: string;
  name: string;
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Opens the shared "Add step" dialog listing the registered step
 * types for the given layer and posts the chosen one.
 */
export async function openStepPicker(
  layer: StepPickerTarget,
  refreshDoc: () => Promise<void>,
): Promise<void> {
  let types;
  try {
    types = await api.getStepTypes();
  } catch (err) {
    pushToast("error", `Loading step types failed: ${describe(err)}`);
    return;
  }
  const content = el("div", "step-picker");
  const note = el("div", "panel-note");
  note.textContent = `Add a step to layer “${layer.name}”`;
  content.appendChild(note);
  for (const entry of types) {
    const btn = button({
      label: entry.typelabel || entry.step_type,
      iconName: "zap",
      onClick: () => {
        closePicker();
        void addStep(layer, entry.step_type, refreshDoc);
      },
    });
    btn.classList.add("step-picker-item");
    content.appendChild(btn);
  }

  const backdrop = el("div", "modal-backdrop");
  const dialog = el("div", "dialog");
  const dialogHeader = el("header", "dialog-header");
  const title = el("span", "dialog-title");
  title.textContent = "Add step";
  const closeBtn = button({
    iconName: "x",
    variant: "icon",
    title: "Close",
    onClick: closePicker,
  });
  dialogHeader.append(title, closeBtn);
  const dialogBody = el("div", "dialog-body");
  dialogBody.appendChild(content);
  dialog.append(dialogHeader, dialogBody);
  backdrop.appendChild(dialog);
  backdrop.addEventListener("click", (ev) => {
    if (ev.target === backdrop) closePicker();
  });
  document.body.appendChild(backdrop);

  function closePicker(): void {
    backdrop.remove();
  }
}

export function openStepPickerForLayer(
  layer: LayerInfo,
  refreshDoc: () => Promise<void>,
): Promise<void> {
  return openStepPicker({ uid: layer.uid, name: layer.name }, refreshDoc);
}

async function addStep(
  layer: StepPickerTarget,
  stepType: string,
  refreshDoc: () => Promise<void>,
): Promise<void> {
  try {
    await api.postStep(layer.uid, { step_type: stepType });
    await refreshDoc();
  } catch (err) {
    pushToast("error", `Adding step failed: ${describe(err)}`);
  }
}
