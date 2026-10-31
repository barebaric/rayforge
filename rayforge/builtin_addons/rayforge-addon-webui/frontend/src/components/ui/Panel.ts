import { el } from "./dom";

export function panel(
  title: string,
  extra?: HTMLElement,
): { root: HTMLElement; header: HTMLElement; body: HTMLElement } {
  const root = el("section", "panel");
  const header = el("header", "panel-header");
  const heading = el("span", "panel-title");
  heading.textContent = title;
  header.appendChild(heading);
  if (extra) header.appendChild(extra);
  const body = el("div", "panel-body");
  root.append(header, body);
  return { root, header, body };
}
