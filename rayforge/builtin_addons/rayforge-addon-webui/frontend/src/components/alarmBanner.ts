import { api } from "../stores/api";
import { machineStore, setMachineState } from "../stores/machineStore";
import { pushToast } from "../stores/uiStore";
import { button } from "./ui/Button";
import { el } from "./ui/dom";
import { icon } from "./ui/icons";

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Red banner overlay for the viewport: shown while the machine state
 * reports an alarm (or a driver error). Offers "Clear alarm".
 */
export function buildAlarmBanner(): HTMLElement {
  const root = el("div", "alarm-banner");
  root.classList.add("hidden");

  root.appendChild(icon("alert", 16));
  const text = el("div", "alarm-text");
  const title = el("span", "alarm-title");
  const desc = el("span", "alarm-desc");
  text.append(title, desc);
  root.appendChild(text);

  const clearBtn = button({
    label: "Clear alarm",
    variant: "danger",
    onClick: () => {
      api
        .clearAlarm()
        .then(() => api.getMachineState())
        .then((state) => setMachineState(state))
        .catch((err: unknown) =>
          pushToast("error", `Clear alarm failed: ${describe(err)}`),
        );
    },
  });
  root.appendChild(clearBtn);

  machineStore.subscribe(({ machineState }) => {
    const error = machineState?.error ?? null;
    const alarmed = machineState?.status === "ALARM" || error !== null;
    root.classList.toggle("hidden", !alarmed);
    if (!alarmed) return;
    title.textContent = error?.title || "Machine alarm";
    desc.textContent =
      error?.description ||
      `The machine reported state "${machineState?.status ?? "ALARM"}".`;
  });

  return root;
}
