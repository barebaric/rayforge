/**
 * Pure command-recall logic for the G-code console: input history with
 * up/down browsing that preserves the in-progress draft.
 */

export class CommandHistory {
  private readonly items: string[] = [];
  private readonly limit: number;
  private index: number | null = null;
  private draft = "";

  constructor(limit = 100) {
    this.limit = limit;
  }

  get size(): number {
    return this.items.length;
  }

  /** Records a sent command; consecutive duplicates are collapsed. */
  push(command: string): void {
    const trimmed = command.trim();
    if (!trimmed) return;
    if (this.items[this.items.length - 1] === trimmed) {
      this.index = null;
      this.draft = "";
      return;
    }
    this.items.push(trimmed);
    if (this.items.length > this.limit) {
      this.items.shift();
    }
    this.index = null;
    this.draft = "";
  }

  /**
   * Recalls the previous command (ArrowUp). Returns null when history
   * is exhausted; `current` is remembered as the draft on first use.
   */
  older(current: string): string | null {
    if (this.items.length === 0) return null;
    if (this.index === null) {
      this.draft = current;
      this.index = this.items.length;
    }
    if (this.index > 0) {
      this.index--;
    }
    return this.items[this.index] ?? null;
  }

  /**
   * Recalls the next command (ArrowDown). Returns null while at the
   * newest entry; restores the draft when browsing past the end.
   */
  newer(): string | null {
    if (this.index === null) return null;
    if (this.index < this.items.length - 1) {
      this.index++;
      return this.items[this.index] ?? null;
    }
    this.index = null;
    return this.draft;
  }

  /** Leaves browse mode after sending so the next ArrowUp starts fresh. */
  reset(): void {
    this.index = null;
    this.draft = "";
  }
}

export interface ConsoleLine {
  id: number;
  kind: "cmd" | "ok" | "error" | "info";
  text: string;
}

let lineSeq = 0;

/** Fixed-capacity console log buffer. */
export class ConsoleLog {
  private readonly lines: ConsoleLine[] = [];
  private readonly limit: number;

  constructor(limit = 400) {
    this.limit = limit;
  }

  all(): ConsoleLine[] {
    return [...this.lines];
  }

  append(kind: ConsoleLine["kind"], text: string): ConsoleLine[] {
    this.lines.push({ id: ++lineSeq, kind, text });
    if (this.lines.length > this.limit) {
      this.lines.splice(0, this.lines.length - this.limit);
    }
    return this.all();
  }
}
