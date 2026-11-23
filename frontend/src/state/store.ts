import type {
  ActorInfo,
  Area,
  Clock,
  Delta,
  PathInfo,
  Pos,
  RelationshipAxes,
  ServerMessage,
  TimelineEvent,
  WorldObject,
} from "./types";

export interface Bubble {
  text: string;
  kind: "speech" | "thought";
  until: number;
}

/** Client-side actor with tweening state on top of the authoritative server state. */
export interface ActorView extends ActorInfo {
  dispX: number;
  dispY: number;
  dispFacing: number;
  pathStart: Pos | null;
  bubble: Bubble | null;
}

type Listener = (reason: string) => void;

const MAX_TIMELINE = 300;

export class WorldStore {
  ready = false;
  /** True while the 🎞 replay viewer owns the store: live messages are not applied. */
  replaying = false;
  /** Replay is auto-playing fast enough that speech bubbles should be condensed. */
  replayCondensed = false;
  width = 32;
  height = 24;
  name = "";
  water: number[][] = [];
  areas: Area[] = [];
  clock: Clock = { tick: 0, day: 0, minute: 0, speed: 1, paused: true, weather: "sunny", merchant_present: false };
  clockReceivedAt = 0;
  simMinutesPerSecond = 2;
  objects = new Map<string, WorldObject>();
  actors = new Map<string, ActorView>();
  relationships: Record<string, Record<string, RelationshipAxes>> = {};
  timeline: TimelineEvent[] = [];
  /** Monotonic count of timeline entries ever received (changes even at the 300 cap). */
  timelineSeq = 0;
  private listeners = new Set<Listener>();

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(reason: string): void {
    for (const fn of this.listeners) fn(reason);
  }

  /** Sim tick predicted for "now" from the last clock message (capped extrapolation). */
  predictedTick(now: number): number {
    if (this.clock.paused) return this.clock.tick;
    const tps = this.clock.speed * this.simMinutesPerSecond;
    return this.clock.tick + Math.min(((now - this.clockReceivedAt) / 1000) * tps, 1.5);
  }

  apply(msg: ServerMessage, now = performance.now()): void {
    if (this.replaying) return; // the backend keeps running; we re-sync on exit
    if (msg.type === "snapshot") {
      this.width = msg.map.width;
      this.height = msg.map.height;
      this.name = msg.map.name;
      this.water = msg.map.water;
      this.areas = msg.map.areas;
      this.clock = msg.clock;
      this.clockReceivedAt = now;
      this.simMinutesPerSecond = msg.clock.sim_minutes_per_real_second ?? 2;
      this.objects = new Map(msg.objects.map((o) => [o.id, o]));
      const prev = this.actors;
      this.actors = new Map(
        msg.actors.map((a) => {
          const old = prev.get(a.actor_id);
          const view: ActorView = {
            ...a,
            dispX: old?.dispX ?? a.x,
            dispY: old?.dispY ?? a.y,
            dispFacing: old?.dispFacing ?? a.facing,
            pathStart: a.path ? { x: a.x, y: a.y } : null,
            bubble: old?.bubble ?? null,
          };
          return [a.actor_id, view];
        }),
      );
      this.relationships = msg.relationships;
      this.timeline = msg.events.slice(-MAX_TIMELINE);
      this.timelineSeq += 1;
      this.ready = true;
      this.emit("snapshot");
      return;
    }
    if (msg.type === "batch") {
      for (const d of msg.deltas) this.applyDelta(d, now);
      this.emit("batch");
    }
  }

  private applyDelta(d: Delta, now: number): void {
    switch (d.type) {
      case "clock": {
        const weatherChanged = d.weather !== this.clock.weather;
        this.clock = { ...this.clock, ...d };
        this.clockReceivedAt = now;
        if (weatherChanged) this.emit("weather");
        break;
      }
      case "path_move": {
        const a = this.actors.get(d.actor_id);
        if (!a) break;
        const path: PathInfo = { start_tick: d.start_tick, step_ticks: d.step_ticks, waypoints: d.waypoints };
        a.path = path;
        // server sends the tile the walk starts from; never trust a pre-batch position
        a.pathStart = d.start_pos ?? { x: a.x, y: a.y };
        if (d.start_pos) {
          a.x = d.start_pos.x;
          a.y = d.start_pos.y;
        }
        break;
      }
      case "agent_state": {
        const a = this.actors.get(d.actor_id);
        if (!a) break;
        const routineChanged = d.routine !== a.routine && d.routine;
        Object.assign(a, d);
        if (d.activity !== "walking") {
          a.path = null; // interrupted / arrived: stop tweening along the old path
          a.pathStart = null;
        }
        if (routineChanged && !a.in_conversation && (!a.bubble || a.bubble.kind === "thought")) {
          a.bubble = { text: d.routine as string, kind: "thought", until: now + 2600 };
        }
        break;
      }
      case "object_change": {
        const o = this.objects.get(d.object_id);
        if (o) {
          o.state = d.state;
          o.version = d.version;
          if (d.data) o.data = d.data;
        } else {
          this.needsResync = true;
        }
        break;
      }
      case "conversation": {
        if (d.speaker_id && d.utterance) {
          const a = this.actors.get(d.speaker_id);
          if (a) a.bubble = { text: d.utterance, kind: "speech", until: now + 4200 };
        }
        break;
      }
      case "event": {
        this.timeline.push(d);
        this.timelineSeq += 1;
        if (this.timeline.length > MAX_TIMELINE) this.timeline.splice(0, this.timeline.length - MAX_TIMELINE);
        this.emit("event");
        break;
      }
      case "god_event":
        break;
    }
  }

  /** Merge a villager's outgoing and incoming edges (from the inspector payload). */
  updateRelationships(
    id: string,
    outgoing: Record<string, RelationshipAxes>,
    incoming: Record<string, RelationshipAxes>,
  ): void {
    this.relationships[id] = { ...(this.relationships[id] ?? {}), ...outgoing };
    for (const [other, rel] of Object.entries(incoming)) {
      this.relationships[other] = { ...(this.relationships[other] ?? {}), [id]: rel };
    }
  }

  /** Set when a delta references an object the client has never seen (e.g. a God spawn). */
  needsResync = false;

  name_of(id: string | null | undefined): string {
    if (!id) return "?";
    return this.actors.get(id)?.name ?? id;
  }
}
