import type { ServerMessage } from "./types";

export type ConnState = "connecting" | "open" | "closed";

/** WebSocket client with exponential auto-reconnect (1s -> 8s). */
export class SimSocket {
  private ws: WebSocket | null = null;
  private retry = 1000;
  private closedByUser = false;
  private reconnectTimer: number | undefined;
  state: ConnState = "connecting";

  constructor(
    private readonly url: string,
    private readonly onMessage: (msg: ServerMessage) => void,
    private readonly onState: (s: ConnState) => void,
  ) {}

  connect(): void {
    window.clearTimeout(this.reconnectTimer);
    this.reconnectTimer = undefined;
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return; // never open a duplicate socket
    }
    this.closedByUser = false;
    this.setState("connecting");
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 1000;
      this.setState("open");
    };
    ws.onmessage = (ev) => {
      try {
        this.onMessage(JSON.parse(ev.data as string) as ServerMessage);
      } catch (err) {
        console.warn("bad server message", err);
      }
    };
    ws.onclose = () => {
      if (this.ws !== ws) return; // a stale socket closing late
      this.setState("closed");
      if (this.closedByUser) return;
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = window.setTimeout(() => this.connect(), this.retry);
      this.retry = Math.min(this.retry * 2, 8000);
    };
    ws.onerror = () => ws.close();
  }

  send(cmd: Record<string, unknown>): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(cmd));
    return true;
  }

  close(): void {
    this.closedByUser = true;
    window.clearTimeout(this.reconnectTimer);
    this.reconnectTimer = undefined;
    this.ws?.close();
  }

  private setState(s: ConnState): void {
    this.state = s;
    this.onState(s);
  }
}
