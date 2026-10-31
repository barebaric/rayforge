import { el } from "./dom";

export interface ToggleHandle {
  root: HTMLElement;
  setChecked(checked: boolean): void;
}

export function toggle(
  checked: boolean,
  onChange?: (checked: boolean) => void,
): ToggleHandle {
  const btn = el("button", "toggle");
  btn.type = "button";
  btn.role = "switch";
  btn.setAttribute("aria-checked", String(checked));
  if (checked) btn.classList.add("on");
  const knob = el("span", "toggle-knob");
  btn.appendChild(knob);
  btn.addEventListener("click", () => {
    const next = !btn.classList.contains("on");
    btn.classList.toggle("on", next);
    btn.setAttribute("aria-checked", String(next));
    onChange?.(next);
  });
  return {
    root: btn,
    setChecked(checked: boolean) {
      btn.classList.toggle("on", checked);
      btn.setAttribute("aria-checked", String(checked));
    },
  };
}
