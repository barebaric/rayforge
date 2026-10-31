import type { WsEvent } from "./api/types";

export type WsStatus = "connecting" | "open" | "closed";

export interface WsHandlers {
  onEvent(ev: WsEvent): void;
  onStatus(status: WsStatus): void;
}

export interface EventConnection {
  close(): void;
}

const MAX_BACKOFF_MS = 10000;
const BASE_BACKOFF_MS = 500;

/**
 * Opens the event websocket and keeps it alive with exponential
 * backoff. `onReconnect` fires on every successful re-open after the
 * first connection so callers can refetch snapshots.
 */
export function connectWs(
  url: string,
  handlers: WsHandlers,
  onReconnect?: () => void,
): EventConnection {
  let ws: WebSocket | null = null;
  let attempt = 0;
  let closed = false;
  let everOpened = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  function scheduleReconnect(): void {
    if (closed) return;
    const delay = Math.min(
      MAX_BACKOFF_MS,
      BASE_BACKOFF_MS * 2 ** Math.min(attempt, 4),
    );
    attempt++;
    timer = setTimeout(open, delay);
  }

  function open(): void {
    if (closed) return;
    handlers.onStatus("connecting");
    try {
      ws = new WebSocket(url);
    } catch {
      scheduleReconnect();
      return;
    }
    ws.onopen = () => {
      const isReconnect = everOpened;
      everOpened = true;
      attempt = 0;
      handlers.onStatus("open");
      if (isReconnect) onReconnect?.();
    };
    ws.onmessage = (msg: MessageEvent) => {
      try {
        handlers.onEvent(JSON.parse(String(msg.data)) as WsEvent);
      } catch {
        return;
      }
    };
    ws.onerror = () => {
      ws?.close();
    };
    ws.onclose = () => {
      handlers.onStatus("closed");
      scheduleReconnect();
    };
  }

  open();
  return {
    close() {
      closed = true;
      if (timer !== undefined) clearTimeout(timer);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
      handlers.onStatus("closed");
    },
  };
}

export function dispatchWsEvent(
  ev: WsEvent,
  targets: {
    onDoc?(ev: Extract<WsEvent, { type: "doc" }>): void;
    onViewport?(data: Extract<WsEvent, { type: "viewport" }>["data"]): void;
    onMachine?(data: Extract<WsEvent, { type: "machine" }>["data"]): void;
    onJob?(data: Extract<WsEvent, { type: "job" }>["data"]): void;
    onNotify?(level: string, message: string): void;
  },
): void {
  switch (ev.type) {
    case "doc":
      targets.onDoc?.(ev);
      break;
    case "viewport":
      targets.onViewport?.(ev.data);
      break;
    case "machine":
      targets.onMachine?.(ev.data);
      break;
    case "job":
      targets.onJob?.(ev.data);
      break;
    case "notify":
      targets.onNotify?.(ev.level, ev.message);
      break;
    default:
      break;
  }
}
