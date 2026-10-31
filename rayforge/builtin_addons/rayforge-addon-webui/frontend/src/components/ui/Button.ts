import { el } from "./dom";
import { icon, type IconName } from "./icons";

export type ButtonVariant = "ghost" | "primary" | "danger" | "icon";

export interface ButtonOptions {
  label?: string;
  iconName?: IconName;
  iconSize?: number;
  variant?: ButtonVariant;
  title?: string;
  disabled?: boolean;
  active?: boolean;
  onClick?: (ev: MouseEvent) => void;
}

export function button(opts: ButtonOptions = {}): HTMLButtonElement {
  const btn = el("button", "btn");
  const variant = opts.variant ?? (opts.label ? "ghost" : "icon");
  btn.classList.add(`btn-${variant}`);
  if (opts.iconName) {
    btn.appendChild(icon(opts.iconName, opts.iconSize ?? 16));
  }
  if (opts.label) {
    const span = el("span", "btn-label");
    span.textContent = opts.label;
    btn.appendChild(span);
  }
  if (opts.title) btn.title = opts.title;
  if (opts.disabled) btn.disabled = true;
  if (opts.active) btn.classList.add("active");
  if (opts.onClick) btn.addEventListener("click", opts.onClick);
  return btn;
}
