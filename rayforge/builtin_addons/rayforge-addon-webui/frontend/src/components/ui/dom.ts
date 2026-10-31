export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  children: Array<Node | string> = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  for (const child of children) {
    node.append(child);
  }
  return node;
}

export function clearElement(node: HTMLElement): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function fmtMm(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "–";
  }
  return value.toFixed(digits);
}

/** Compact duration readout: "42.5s", "1m 05s", or "–" when unknown. */
export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
    return "–";
  }
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  const shown = rest >= 60 ? 59 : rest;
  return `${minutes}m ${String(shown).padStart(2, "0")}s`;
}
