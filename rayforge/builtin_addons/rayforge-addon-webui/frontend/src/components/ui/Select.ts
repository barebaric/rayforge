import { el } from "./dom";

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectHandle {
  root: HTMLElement;
  setValue(value: string): void;
  setDisabled(disabled: boolean): void;
}

export function select(
  label: string,
  options: SelectOption[],
  value: string,
  onChange?: (value: string) => void,
): SelectHandle {
  const root = el("label", "select-field");
  const caption = el("span", "num-field-label");
  caption.textContent = label;
  const sel = el("select", "select-input") as HTMLSelectElement;
  for (const opt of options) {
    const option = el("option") as HTMLOptionElement;
    option.value = opt.value;
    option.textContent = opt.label;
    sel.appendChild(option);
  }
  sel.value = value;
  if (onChange) {
    sel.addEventListener("change", () => onChange(sel.value));
  }
  root.append(caption, sel);
  return {
    root,
    setValue(v: string) {
      sel.value = v;
    },
    setDisabled(disabled: boolean) {
      sel.disabled = disabled;
    },
  };
}
