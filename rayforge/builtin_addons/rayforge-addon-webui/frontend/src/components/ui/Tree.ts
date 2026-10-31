import { clearElement, el } from "./dom";
import { icon, type IconName } from "./icons";

export interface TreeRow {
  id: string;
  depth: number;
  label: string;
  icon?: IconName;
  colorChip?: string;
  hint?: string;
  expandable?: boolean;
  expanded?: boolean;
  selected?: boolean;
  visible?: boolean;
  active?: boolean;
  dimmed?: boolean;
  statusDot?: "on" | "off";
  renamable?: boolean;
  trailing?: HTMLElement[];
  onToggleExpanded?: () => void;
  onSelect?: (ev: MouseEvent) => void;
  onToggleVisible?: () => void;
  onChipClick?: (ev: MouseEvent) => void;
  onRename?: (name: string) => void;
  onContext?: (ev: MouseEvent) => void;
}

/** Renders a flat row list as an indented tree. Rows are re-rendered by the caller. */
export function renderTree(rows: TreeRow[]): HTMLElement {
  const root = el("div", "tree");
  for (const row of rows) {
    root.appendChild(renderRow(row));
  }
  if (rows.length === 0) {
    const empty = el("div", "tree-empty");
    empty.textContent = "Empty document";
    root.appendChild(empty);
  }
  return root;
}

function renderRow(row: TreeRow): HTMLElement {
  const line = el("div", "tree-row");
  line.style.paddingLeft = `${8 + row.depth * 14}px`;
  line.dataset.uid = row.id;
  if (row.selected) line.classList.add("selected");
  if (row.active) line.classList.add("active-row");
  if (row.dimmed) line.classList.add("dimmed");

  const expander = el("span", "tree-expander");
  if (row.expandable) {
    const btn = el("button", "tree-expander-btn");
    btn.type = "button";
    btn.appendChild(icon(row.expanded ? "chevronDown" : "chevronRight", 12));
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      row.onToggleExpanded?.();
    });
    expander.appendChild(btn);
  }
  line.appendChild(expander);

  if (row.visible !== undefined) {
    const vis = el("button", "tree-vis");
    vis.type = "button";
    vis.title = "Toggle visibility";
    vis.appendChild(icon(row.visible ? "eye" : "eyeOff", 13));
    vis.classList.toggle("off", !row.visible);
    vis.addEventListener("click", (ev) => {
      ev.stopPropagation();
      row.onToggleVisible?.();
    });
    line.appendChild(vis);
  }

  if (row.colorChip) {
    if (row.onChipClick) {
      const chipBtn = el("button", "tree-chip-btn");
      chipBtn.type = "button";
      chipBtn.title = "Change color";
      chipBtn.appendChild(renderChip(row.colorChip));
      chipBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        row.onChipClick?.(ev);
      });
      line.appendChild(chipBtn);
    } else {
      line.appendChild(renderChip(row.colorChip));
    }
  }

  if (row.icon) {
    line.appendChild(icon(row.icon, 13));
  }

  if (row.statusDot) {
    const dot = el("span", `tree-dot ${row.statusDot}`);
    line.appendChild(dot);
  }

  const label = el("span", "tree-label");
  label.textContent = row.label;
  line.appendChild(label);

  if (row.renamable && row.onRename) {
    label.addEventListener("dblclick", (ev) => {
      ev.stopPropagation();
      beginRename(label, row);
    });
  }

  if (row.hint) {
    const hint = el("span", "tree-hint");
    hint.textContent = row.hint;
    line.appendChild(hint);
  }

  for (const node of row.trailing ?? []) {
    const host = el("span", "tree-actions");
    host.appendChild(node);
    line.appendChild(host);
  }

  line.addEventListener("click", (ev) => row.onSelect?.(ev));
  if (row.onContext) {
    line.addEventListener("contextmenu", (ev) => {
      ev.preventDefault();
      row.onContext?.(ev);
    });
  }
  return line;
}

function renderChip(color: string): HTMLElement {
  const chip = el("span", "tree-chip");
  chip.style.background = color;
  return chip;
}

function beginRename(
  label: HTMLElement,
  row: TreeRow,
): void {
  const input = el("input", "tree-rename-input") as HTMLInputElement;
  input.type = "text";
  input.value = row.label;
  input.spellcheck = false;
  let cancelled = false;
  const commit = () => {
    if (cancelled) return;
    const name = input.value.trim();
    if (name && name !== row.label) row.onRename?.(name);
  };
  input.addEventListener("click", (ev) => ev.stopPropagation());
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      commit();
      cancelled = true;
      input.blur();
    } else if (ev.key === "Escape") {
      cancelled = true;
      input.blur();
    }
  });
  input.addEventListener("blur", commit);
  label.replaceChildren(input);
  input.focus();
  input.select();
}

export function replaceTreeContent(
  container: HTMLElement,
  rows: TreeRow[],
): void {
  clearElement(container);
  container.appendChild(renderTree(rows));
}
