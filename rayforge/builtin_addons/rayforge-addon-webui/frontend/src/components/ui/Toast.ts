import {
  dismissToast,
  uiStore,
  type Toast,
  type ToastLevel,
} from "../../stores/uiStore";
import { clearElement, el } from "./dom";
import { icon } from "./icons";

const AUTO_DISMISS_MS = 5000;

const LEVEL_ICON: Record<ToastLevel, "info" | "check" | "alert" | "x"> = {
  info: "info",
  success: "check",
  warning: "alert",
  error: "x",
};

export function toastContainer(): HTMLElement {
  const root = el("div", "toasts");
  let known = new Set<number>();

  uiStore.subscribe((state) => {
    const toasts = state.toasts;
    for (const toast of toasts) {
      if (!known.has(toast.id)) {
        window.setTimeout(() => dismissToast(toast.id), AUTO_DISMISS_MS);
      }
    }
    known = new Set(toasts.map((t) => t.id));
    render(root, toasts);
  });

  return root;
}

function render(root: HTMLElement, toasts: Toast[]): void {
  clearElement(root);
  for (const toast of toasts) {
    const item = el("div", `toast toast-${toast.level}`);
    item.appendChild(icon(LEVEL_ICON[toast.level] ?? "info", 14));
    const msg = el("span", "toast-msg");
    msg.textContent = toast.message;
    const close = el("button", "toast-close");
    close.type = "button";
    close.appendChild(icon("x", 12));
    close.addEventListener("click", () => dismissToast(toast.id));
    item.append(msg, close);
    root.appendChild(item);
  }
}
