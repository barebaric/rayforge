import { api } from "../../stores/api";
import { button } from "../ui/Button";
import { panel } from "../ui/Panel";
import { el } from "../ui/dom";
import {
  CommandHistory,
  ConsoleLog,
  type ConsoleLine,
} from "./consoleHistory";

export interface ConsoleSectionRefs {
  root: HTMLElement;
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * The G-code console: a monospace log of sent commands plus an input
 * line with up/down command recall. The API does not stream responses,
 * so entries show the send result; machine state changes arrive via
 * the normal machine events.
 */
export function buildConsoleSection(): ConsoleSectionRefs {
  const root = el("div", "console-section");
  const p = panel("G-code console");
  const history = new CommandHistory();
  const log = new ConsoleLog();
  const logView = el("div", "console-log mono");

  const renderLog = (lines: ConsoleLine[]): void => {
    while (logView.firstChild) logView.removeChild(logView.firstChild);
    for (const line of lines) {
      const row = el("div", `console-line console-${line.kind}`);
      row.textContent = line.text;
      logView.appendChild(row);
    }
    logView.scrollTop = logView.scrollHeight;
  };

  log.append(
    "info",
    "Raw G-code is sent to the machine; responses are not streamed.",
  );
  renderLog(log.all());

  const inputRow = el("div", "console-input-row");
  const input = el("input", "console-input mono") as HTMLInputElement;
  input.type = "text";
  input.spellcheck = false;
  input.placeholder = "e.g. G0 X10 Y10";
  const send = (): void => {
    const gcode = input.value.trim();
    if (!gcode) return;
    input.value = "";
    history.push(gcode);
    renderLog(log.append("cmd", `> ${gcode}`));
    api
      .runRaw(gcode)
      .then(() => renderLog(log.append("ok", "sent")))
      .catch((err: unknown) =>
        renderLog(log.append("error", `error: ${describe(err)}`)),
      );
  };
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      send();
      history.reset();
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      const prev = history.older(input.value);
      if (prev !== null) input.value = prev;
    } else if (ev.key === "ArrowDown") {
      ev.preventDefault();
      const next = history.newer();
      if (next !== null) input.value = next;
    }
  });
  const sendBtn = button({
    iconName: "terminal",
    title: "Send G-code",
    onClick: () => {
      send();
      history.reset();
    },
  });
  inputRow.append(input, sendBtn);
  p.body.append(logView, inputRow);
  root.appendChild(p.root);

  return { root };
}
