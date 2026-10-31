import { el } from "./dom";
import { icon, type IconName } from "./icons";

export interface ToolbarItem {
  iconName: IconName;
  title: string;
  /** Tiny label rendered under the icon. */
  label?: string;
  active?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}

/** Grouped icon toolbar; groups are separated by hairline dividers. */
export function toolbar(groups: ToolbarItem[][]): HTMLElement {
  const root = el("div", "toolbar");
  groups.forEach((group, index) => {
    if (index > 0) root.appendChild(el("div", "toolbar-sep"));
    for (const item of group) {
      const btn = el("button", "toolbar-btn");
      btn.type = "button";
      btn.title = item.title;
      btn.appendChild(icon(item.iconName, 16));
      if (item.label) {
        const label = el("span", "tool-label");
        label.textContent = item.label;
        btn.appendChild(label);
      }
      if (item.active) btn.classList.add("active");
      if (item.disabled) btn.disabled = true;
      if (item.onClick) btn.addEventListener("click", item.onClick);
      root.appendChild(btn);
    }
  });
  return root;
}
