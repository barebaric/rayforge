import { el } from "./ui/dom";
import type { LayoutOption } from "../api/types";

export type AutoLayoutDecision =
  | { kind: "none" }
  | { kind: "run"; layout: LayoutOption }
  | { kind: "picker"; layouts: LayoutOption[] };

/**
 * Decides what clicking "Auto layout" does for the given strategy
 * list: nothing, run the only strategy directly, or open the picker.
 */
export function decideAutoLayout(
  layouts: LayoutOption[],
): AutoLayoutDecision {
  if (layouts.length === 0) return { kind: "none" };
  if (layouts.length === 1) return { kind: "run", layout: layouts[0]! };
  return { kind: "picker", layouts };
}

export interface LayoutPickerRefs {
  root: HTMLElement;
  close(): void;
}

/**
 * Small popup listing the registered strategies, anchored below the
 * given element. Clicking an entry resolves via onPick; clicking
 * elsewhere or pressing Escape closes with null.
 */
export function openLayoutPicker(
  anchor: HTMLElement,
  layouts: LayoutOption[],
  onPick: (layout: LayoutOption | null) => void,
): LayoutPickerRefs {
  const root = el("div", "menu-popup layout-picker");
  for (const layout of layouts) {
    const row = el("button", "menu-item");
    row.type = "button";
    const label = el("span", "menu-item-label");
    label.textContent = layout.label;
    row.appendChild(label);
    row.addEventListener("click", () => {
      close();
      onPick(layout);
    });
    root.appendChild(row);
  }

  function close(): void {
    root.remove();
    document.removeEventListener("pointerdown", onOutside, true);
    document.removeEventListener("keydown", onKey, true);
  }

  function onOutside(ev: Event): void {
    if (root.contains(ev.target as Node)) return;
    close();
    onPick(null);
  }

  function onKey(ev: KeyboardEvent): void {
    if (ev.key !== "Escape") return;
    close();
    onPick(null);
  }

  document.addEventListener("pointerdown", onOutside, true);
  document.addEventListener("keydown", onKey, true);

  const rect = anchor.getBoundingClientRect();
  root.style.left = `${rect.left}px`;
  root.style.top = `${rect.bottom + 6}px`;
  document.body.appendChild(root);

  return { root, close };
}

/** Convenience wrapper resolving the picker to a promise. */
export function pickLayout(
  anchor: HTMLElement,
  layouts: LayoutOption[],
): Promise<LayoutOption | null> {
  return new Promise((resolve) => {
    openLayoutPicker(anchor, layouts, resolve);
  });
}
