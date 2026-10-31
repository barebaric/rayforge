import { el } from "./dom";

export interface SliderHandle {
  root: HTMLElement;
  setValue(value: number): void;
}

export function slider(
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
  onInput?: (value: number) => void,
): SliderHandle {
  const root = el("label", "slider-field");
  const caption = el("span", "num-field-label");
  caption.textContent = label;
  const input = el("input", "slider-input") as HTMLInputElement;
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  const readout = el("span", "slider-readout");
  readout.textContent = String(value);
  input.addEventListener("input", () => {
    readout.textContent = input.value;
    onInput?.(Number.parseFloat(input.value));
  });
  root.append(caption, input, readout);
  return {
    root,
    setValue(value: number) {
      input.value = String(value);
      readout.textContent = String(value);
    },
  };
}
