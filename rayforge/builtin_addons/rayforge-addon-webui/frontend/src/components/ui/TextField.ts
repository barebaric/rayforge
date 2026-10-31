import { el } from "./dom";

export interface TextFieldHandle {
  root: HTMLElement;
  setValue(value: string): void;
}

export interface TextFieldOptions {
  label: string;
  value: string;
  placeholder?: string;
  onCommit?: (value: string) => void;
}

export function textField(opts: TextFieldOptions): TextFieldHandle {
  const root = el("label", "num-field");
  const caption = el("span", "num-field-label");
  caption.textContent = opts.label;
  const input = el("input", "num-field-input text-field-input") as HTMLInputElement;
  input.type = "text";
  input.value = opts.value;
  input.spellcheck = false;
  if (opts.placeholder) input.placeholder = opts.placeholder;

  const commit = () => {
    const value = input.value.trim();
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
      input.value = opts.value;
      input.blur();
    }
  });
  input.addEventListener("blur", commit);

  root.append(caption, input);
  return {
    root,
    setValue(value: string) {
      if (document.activeElement !== input) input.value = value;
      opts.value = value;
    },
  };
}
