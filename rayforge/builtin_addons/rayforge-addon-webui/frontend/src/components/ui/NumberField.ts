import { el } from "./dom";

export interface NumberFieldHandle {
  root: HTMLElement;
  setValue(value: number): void;
}

export interface NumberFieldOptions {
  label: string;
  value: number;
  unit?: string;
  step?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  onCommit?: (value: number) => void;
}

export function numberField(opts: NumberFieldOptions): NumberFieldHandle {
  const root = el("label", "num-field");
  const caption = el("span", "num-field-label");
  caption.textContent = opts.label;
  const input = el("input", "num-field-input") as HTMLInputElement;
  input.type = "text";
  input.inputMode = "decimal";
  input.spellcheck = false;
  input.value = formatNumber(opts.value);
  if (opts.disabled) input.disabled = true;

  const commit = () => {
    const parsed = Number.parseFloat(input.value.replace(",", "."));
    if (Number.isNaN(parsed)) {
      input.value = formatNumber(opts.value);
      return;
    }
    let value = parsed;
    if (opts.min !== undefined) value = Math.max(opts.min, value);
    if (opts.max !== undefined) value = Math.min(opts.max, value);
    input.value = formatNumber(value);
    if (value !== opts.value) {
      opts.value = value;
      opts.onCommit?.(value);
    }
  };
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      commit();
      input.blur();
    } else if (ev.key === "Escape") {
      input.value = formatNumber(opts.value);
      input.blur();
    }
  });
  input.addEventListener("blur", commit);

  root.append(caption);
  const wrap = el("span", "num-field-wrap");
  wrap.appendChild(input);
  if (opts.unit) {
    const unit = el("span", "num-field-unit");
    unit.textContent = opts.unit;
    wrap.appendChild(unit);
  }
  root.appendChild(wrap);

  return {
    root,
    setValue(value: number) {
      if (document.activeElement !== input) {
        input.value = formatNumber(value);
      }
      opts.value = value;
    },
  };
}

function formatNumber(value: number): string {
  return String(Math.round(value * 100) / 100);
}
