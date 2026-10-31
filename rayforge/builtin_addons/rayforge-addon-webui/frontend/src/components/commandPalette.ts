import { el } from "./ui/dom";
import { icon, type IconName } from "./ui/icons";

export interface CommandEntry {
  label: string;
  accelerator?: string;
  iconName?: IconName;
  run: () => void;
}

export interface CommandPaletteRefs {
  root: HTMLElement;
  input: HTMLInputElement;
}

/**
 * Minimal command palette anchored to the "Search commands" input in
 * the top bar: filters a static list and executes via the shared
 * shell actions. Enter runs the highlighted match, arrows move it,
 * Escape or clicking outside closes it.
 */
export function buildCommandPalette(
  entries: CommandEntry[],
): CommandPaletteRefs {
  const root = el("div", "command-palette");
  const input = el("input", "command-input") as HTMLInputElement;
  input.type = "text";
  input.spellcheck = false;
  input.placeholder = "Search commands";
  root.appendChild(icon("search", 13));
  root.appendChild(input);

  const popup = el("div", "menu-popup command-popup");
  let filtered: CommandEntry[] = [];
  let active = 0;

  function close(): void {
    popup.remove();
  }

  function isOpen(): boolean {
    return popup.parentElement === root;
  }

  function render(): void {
    const query = input.value.trim().toLowerCase();
    filtered = entries.filter((entry) =>
      entry.label.toLowerCase().includes(query),
    );
    active = 0;
    const rows = filtered.map((entry, index) => {
      const row = el("button", "menu-item");
      row.type = "button";
      if (index === active) row.classList.add("active");
      if (entry.iconName) row.appendChild(icon(entry.iconName, 14));
      const label = el("span", "menu-item-label");
      label.textContent = entry.label;
      row.appendChild(label);
      if (entry.accelerator) {
        const accel = el("span", "menu-item-accel");
        accel.textContent = entry.accelerator;
        row.appendChild(accel);
      }
      row.addEventListener("click", () => execute(entry));
      return row;
    });
    if (rows.length === 0) {
      const empty = el("div", "command-empty");
      empty.textContent = "No matching commands";
      popup.replaceChildren(empty);
      return;
    }
    popup.replaceChildren(...rows);
  }

  function execute(entry: CommandEntry): void {
    close();
    input.value = "";
    input.blur();
    entry.run();
  }

  function move(delta: number): void {
    if (filtered.length === 0) return;
    active = (active + delta + filtered.length) % filtered.length;
    const rows = Array.from(
      popup.querySelectorAll(".menu-item"),
    ) as HTMLElement[];
    rows.forEach((row, index) => row.classList.toggle("active", index === active));
    rows[active]?.scrollIntoView({ block: "nearest" });
  }

  input.addEventListener("focus", () => {
    render();
    root.appendChild(popup);
  });
  input.addEventListener("input", render);
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      move(1);
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      move(-1);
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      const entry = filtered[active];
      if (entry) execute(entry);
    } else if (ev.key === "Escape") {
      close();
      input.blur();
    }
  });

  document.addEventListener("pointerdown", (ev) => {
    if (isOpen() && !root.contains(ev.target as Node)) close();
  });

  return { root, input };
}
