import { docStore } from "../stores/docStore";
import { el } from "./ui/dom";
import { icon } from "./ui/icons";

export interface HistoryMenuRefs {
  root: HTMLElement;
}

/**
 * Small display-only dropdown listing the undo/redo stacks from
 * /doc/history. Labels refresh each time the popup opens.
 */
export function buildHistoryMenu(): HistoryMenuRefs {
  const root = el("div", "menu history-menu");
  const btn = el("button", "history-btn");
  btn.type = "button";
  btn.title = "History";
  btn.appendChild(icon("chevronDown", 15));

  const popup = el("div", "menu-popup history-popup");
  let open = false;

  const onDocPointerDown = (ev: PointerEvent): void => {
    if (!root.contains(ev.target as Node)) close();
  };
  const onDocKeydown = (ev: KeyboardEvent): void => {
    if (ev.key === "Escape") close();
  };

  function close(): void {
    popup.remove();
    btn.classList.remove("open");
    open = false;
    document.removeEventListener("pointerdown", onDocPointerDown, true);
    document.removeEventListener("keydown", onDocKeydown, true);
  }

  function openPopup(): void {
    popup.replaceChildren(...historyRows());
    root.appendChild(popup);
    btn.classList.add("open");
    open = true;
    document.addEventListener("pointerdown", onDocPointerDown, true);
    document.addEventListener("keydown", onDocKeydown, true);
  }
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    if (open) close();
    else openPopup();
  });

  root.appendChild(btn);
  return { root };
}

function historyRows(): HTMLElement[] {
  const { undoLabels, redoLabels } = docStore.getState();
  const rows: HTMLElement[] = [];
  if (undoLabels.length === 0 && redoLabels.length === 0) {
    rows.push(disabledRow("No history"));
    return rows;
  }
  rows.push(sectionRow("Redo"));
  const redos = [...redoLabels].reverse();
  if (redos.length === 0) rows.push(disabledRow("Nothing to redo"));
  for (const label of redos) rows.push(disabledRow(label));
  rows.push(sectionRow("Undo"));
  const undos = [...undoLabels].reverse();
  if (undos.length === 0) rows.push(disabledRow("Nothing to undo"));
  for (const label of undos) rows.push(disabledRow(label));
  return rows;
}

function disabledRow(label: string): HTMLElement {
  const row = el("button", "menu-item");
  row.type = "button";
  row.disabled = true;
  const span = el("span", "menu-item-label");
  span.textContent = label;
  row.appendChild(span);
  return row;
}

function sectionRow(label: string): HTMLElement {
  const row = el("div", "history-section");
  row.textContent = label;
  return row;
}
