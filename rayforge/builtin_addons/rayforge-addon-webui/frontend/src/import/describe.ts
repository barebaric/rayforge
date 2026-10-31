/** Formats an unknown error for toasts and status lines. */
export function describeError(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
