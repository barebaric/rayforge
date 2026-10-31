import { el } from "./dom";

export interface TabDef {
  id: string;
  label: string;
}

export interface TabsHandle {
  root: HTMLElement;
  setActive(id: string): void;
}

export function tabs(
  defs: TabDef[],
  activeId: string,
  onChange: (id: string) => void,
): TabsHandle {
  const root = el("div", "tabs");
  const buttons = new Map<string, HTMLButtonElement>();

  for (const def of defs) {
    const btn = el("button", "tab");
    btn.type = "button";
    btn.textContent = def.label;
    btn.dataset.tab = def.id;
    btn.addEventListener("click", () => onChange(def.id));
    buttons.set(def.id, btn);
    root.appendChild(btn);
  }

  function setActive(id: string): void {
    for (const [tabId, btn] of buttons) {
      btn.classList.toggle("active", tabId === id);
    }
  }
  setActive(activeId);

  return { root, setActive };
}
