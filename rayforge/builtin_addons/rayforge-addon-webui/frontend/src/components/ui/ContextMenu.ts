import { el } from "./dom";

export interface ContextMenuItem {
  label?: string;
  action?: () => void;
  disabled?: boolean;
  separator?: boolean;
  danger?: boolean;
}

let openPopup: HTMLElement | null = null;

function closeContextMenu(): void {
  openPopup?.remove();
  openPopup = null;
}

document.addEventListener("pointerdown", (ev) => {
  if (openPopup && !openPopup.contains(ev.target as Node)) {
    closeContextMenu();
  }
});

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") closeContextMenu();
});

/** Opens a floating action menu near the pointer (tree rows, kebab buttons). */
export function openContextMenu(
  x: number,
  y: number,
  items: ContextMenuItem[],
): void {
  closeContextMenu();
  const popup = el("div", "menu-popup context-popup");
  for (const item of items) {
    if (item.separator) {
      popup.appendChild(el("div", "menu-sep"));
      continue;
    }
    const row = el("button", "menu-item");
    row.type = "button";
    if (item.danger) row.classList.add("danger");
    const label = el("span", "menu-item-label");
    label.textContent = item.label ?? "";
    row.appendChild(label);
    if (item.disabled) {
      row.disabled = true;
    } else {
      row.addEventListener("click", () => {
        closeContextMenu();
        item.action?.();
      });
    }
    popup.appendChild(row);
  }
  popup.style.left = `${Math.max(4, Math.min(x, window.innerWidth - 230))}px`;
  popup.style.top = `${Math.max(4, Math.min(y, window.innerHeight - 40 - items.length * 26))}px`;
  document.body.appendChild(popup);
  openPopup = popup;
}
