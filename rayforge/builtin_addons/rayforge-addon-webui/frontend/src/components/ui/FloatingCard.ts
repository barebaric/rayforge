import { el } from "./dom";
import { icon, type IconName } from "./icons";

export interface FloatingCardRefs {
  root: HTMLElement;
  stub: HTMLElement;
}

/**
 * Floating translucent card shell: header with title + collapse
 * chevron. While collapsed, the card is hidden and a slim vertical
 * stub keeps a way to reopen it.
 */
export function floatingCard(
  side: "left" | "right",
  title: string,
  iconName: IconName,
  body: HTMLElement,
): FloatingCardRefs {
  const root = el("aside", `${side}-panel floating-panel`);

  const heading = el("span", "floating-card-title");
  heading.textContent = title;
  const collapseBtn = el("button", "btn-icon btn-ghost panel-collapse");
  collapseBtn.type = "button";
  collapseBtn.title = `Collapse ${title.toLowerCase()} panel`;
  collapseBtn.appendChild(
    icon(side === "left" ? "chevronLeft" : "chevronRight", 15),
  );

  const header = el("header", "floating-card-header");
  header.append(heading, collapseBtn);
  root.append(header, body);

  const stubLabel = el("span", "panel-stub-label");
  stubLabel.textContent = title;
  const stub = el("button", `panel-stub ${side}-stub`);
  stub.type = "button";
  stub.title = `Expand ${title.toLowerCase()} panel`;
  stub.append(icon(iconName, 15), stubLabel);

  collapseBtn.addEventListener("click", () => {
    root.hidden = true;
    stub.hidden = false;
  });
  stub.addEventListener("click", () => {
    stub.hidden = true;
    root.hidden = false;
  });
  stub.hidden = true;

  return { root, stub };
}
