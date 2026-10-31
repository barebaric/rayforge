const PATHS: Record<string, string> = {
  pointer: `<path d="M4 3l7.5 18 2.4-7.4L21 11z"/>`,
  move: `<path d="M12 2v20M2 12h20"/><path d="M9 5l3-3 3 3M9 19l3 3 3-3M5 9l-3 3 3 3M19 9l3 3-3 3"/>`,
  rotate: `<path d="M21 12a9 9 0 1 1-2.64-6.36L21 8"/><path d="M21 3v5h-5"/>`,
  scale: `<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>`,
  upload: `<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5M12 3v12"/>`,
  frame: `<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/>`,
  fit: `<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M16 21h3a2 2 0 0 0 2-2v-3M8 21H5a2 2 0 0 1-2-2v-3"/><circle cx="12" cy="12" r="1.6"/>`,
  viewTop: `<rect x="4" y="4" width="16" height="16" rx="2"/><circle cx="12" cy="12" r="1.6"/><path d="M12 4v5M12 15v5M4 12h5M15 12h5"/>`,
  viewFront: `<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 15h16"/>`,
  viewIso: `<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="M3.3 7l8.7 5 8.7-5M12 22V12"/>`,
  play: `<path d="M6 4l14 8-14 8z"/>`,
  pause: `<path d="M7 4h3v16H7zM14 4h3v16h-3z"/>`,
  stop: `<rect x="5" y="5" width="14" height="14" rx="2"/>`,
  stepForward: `<path d="M5 4l10 8-10 8z"/><path d="M19 5v14"/>`,
  stepBack: `<path d="M19 4L9 12l10 8z"/><path d="M5 5v14"/>`,
  undo: `<path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-15-6.7L3 13"/>`,
  redo: `<path d="M21 7v6h-6"/><path d="M3 17a9 9 0 0 1 15-6.7L21 13"/>`,
  filePlus: `<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M12 18v-6M9 15h6"/>`,
  save: `<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/>`,
  folder: `<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>`,
  eye: `<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>`,
  eyeOff: `<path d="M17.94 17.94A10.5 10.5 0 0 1 12 20c-7 0-11-8-11-8a19.8 19.8 0 0 1 5.06-5.94M9.9 4.24A9.7 9.7 0 0 1 12 4c7 0 11 8 11 8a19.8 19.8 0 0 1-3.22 4.31"/><path d="M1 1l22 22"/>`,
  chevronRight: `<path d="M9 18l6-6-6-6"/>`,
  chevronDown: `<path d="M6 9l6 6 6-6"/>`,
  chevronUp: `<path d="M18 15l-6-6-6 6"/>`,
  chevronLeft: `<path d="M15 18l-6-6 6-6"/>`,
  home: `<path d="M3 10.5L12 3l9 7.5"/><path d="M5 9.5V21h5v-6h4v6h5V9.5"/>`,
  crosshairSmall: `<circle cx="12" cy="12" r="5"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4"/>`,
  terminal: `<path d="M4 17l6-5-6-5"/><path d="M12 19h8"/>`,
  kebab: `<circle cx="12" cy="5" r="1.4"/><circle cx="12" cy="12" r="1.4"/><circle cx="12" cy="19" r="1.4"/>`,
  layers: `<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5M2 12l10 5 10-5"/>`,
  box: `<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="M3.3 7l8.7 5 8.7-5M12 22V12"/>`,
  zap: `<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>`,
  machine: `<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>`,
  link: `<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>`,
  sliders: `<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3"/><path d="M1 14h6M9 8h6M17 16h6"/>`,
  crosshair: `<circle cx="12" cy="12" r="9"/><path d="M23 12h-4M5 12H1M12 1v4M12 19v4"/>`,
  grid: `<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>`,
  x: `<path d="M18 6L6 18M6 6l12 12"/>`,
  check: `<path d="M20 6L9 17l-5-5"/>`,
  trash: `<path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M10 11v6M14 11v6"/>`,
  copy: `<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>`,
  alert: `<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/>`,
  info: `<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>`,
  plug: `<path d="M12 22v-5M9 8V2M15 8V2M18 8v3a6 6 0 0 1-12 0V8z"/>`,
  search: `<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>`,
  zoomIn: `<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M8 11h6M11 8v6"/>`,
  zoomOut: `<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M8 11h6"/>`,
  flipH: `<path d="M12 3v18"/><path d="M8 8L4 12l4 4"/><path d="M16 8l4 4-4 4"/>`,
  flipV: `<path d="M3 12h18"/><path d="M8 8l4-4 4 4"/><path d="M8 16l4 4 4-4"/>`,
  alignLeft: `<path d="M4 3v18"/><rect x="8" y="5" width="12" height="5" rx="1"/><rect x="8" y="14" width="8" height="5" rx="1"/>`,
  alignCenterH: `<path d="M12 3v18"/><rect x="4" y="5" width="16" height="5" rx="1"/><rect x="7" y="14" width="10" height="5" rx="1"/>`,
  alignRight: `<path d="M20 3v18"/><rect x="4" y="5" width="12" height="5" rx="1"/><rect x="8" y="14" width="8" height="5" rx="1"/>`,
  alignTop: `<path d="M3 4h18"/><rect x="5" y="8" width="5" height="12" rx="1"/><rect x="14" y="8" width="5" height="8" rx="1"/>`,
  alignCenterV: `<path d="M3 12h18"/><rect x="5" y="4" width="5" height="16" rx="1"/><rect x="14" y="7" width="5" height="10" rx="1"/>`,
  alignBottom: `<path d="M3 20h18"/><rect x="5" y="4" width="5" height="12" rx="1"/><rect x="14" y="8" width="5" height="8" rx="1"/>`,
  plugOff: `<path d="M12 22v-5M9 8V2M15 8V2M18 8v3c0 1-.2 1.9-.6 2.7"/><path d="M6.6 6.6A6 6 0 0 0 6 11v3"/><path d="M4 4l16 16"/>`,
  refresh: `<path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.5 9a9 9 0 0 1 14.9-3.4L23 10M1 14l4.6 4.4A9 9 0 0 0 20.5 15"/>`,
};

export type IconName = keyof typeof PATHS;

export function icon(name: IconName, size = 16): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.8");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.classList.add("icon", `icon-${name}`);
  svg.innerHTML = PATHS[name] ?? "";
  return svg;
}
