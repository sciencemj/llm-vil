export interface Pos {
  x: number;
  y: number;
}

export interface WorldObject {
  id: string;
  kind: string;
  pos: Pos;
  state: string;
  owner: string | null;
  name: string;
  version: number;
  data: Record<string, unknown>;
}

export interface PathInfo {
  start_tick: number;
  step_ticks: number;
  waypoints: Pos[];
}

export interface AgentState {
  actor_id: string;
  x: number;
  y: number;
  facing: number;
  activity: string;
  hunger: number;
  stamina: number;
  mood: number;
  routine: string | null;
  in_conversation: boolean;
}

export interface ActorInfo extends AgentState {
  name: string;
  occupation: string;
  home_area: string;
  path: PathInfo | null;
}

export interface Clock {
  tick: number;
  day: number;
  minute: number;
  speed: number;
  paused: boolean;
  weather: string;
  merchant_present: boolean;
  ticks_per_day?: number;
  sim_minutes_per_real_second?: number;
}

export interface Area {
  id: string;
  name: string;
  rect: [number, number, number, number];
}

export interface RelationshipAxes {
  target_id: string;
  familiarity: number;
  affection: number;
  trust: number;
  respect: number;
  summary: string;
}

export interface TimelineEvent {
  type: "event";
  tick: number;
  kind: string;
  actor_id: string | null;
  target_id: string | null;
  pos: Pos | null;
  data: Record<string, unknown>;
  importance: number;
}

export interface Snapshot {
  type: "snapshot";
  map: { width: number; height: number; name: string; water: number[][]; areas: Area[] };
  clock: Clock;
  objects: WorldObject[];
  actors: ActorInfo[];
  relationships: Record<string, Record<string, RelationshipAxes>>;
  events: TimelineEvent[];
  metrics: Record<string, number>;
}

export type Delta =
  | ({ type: "clock" } & Clock)
  | {
      type: "path_move";
      actor_id: string;
      start_tick: number;
      step_ticks: number;
      waypoints: Pos[];
      facing: number;
      start_pos?: Pos;
    }
  | { type: "object_change"; object_id: string; state: string; version: number; data?: Record<string, unknown> }
  | {
      type: "conversation";
      session_id: string;
      participants: string[];
      state: string;
      speaker_id: string | null;
      utterance: string | null;
      emotion: string | null;
    }
  | { type: "god_event"; kind: string; data: Record<string, unknown> }
  | ({ type: "agent_state" } & AgentState)
  | TimelineEvent;

export type ServerMessage =
  | Snapshot
  | { type: "batch"; tick: number; deltas: Delta[] }
  | { type: "ack"; cmd?: string; ok: boolean; error?: string };

export interface AgentDetail {
  id: string;
  name: string;
  age: number;
  occupation: string;
  traits: string[];
  persona: string;
  activity: string;
  needs: { hunger: number; stamina: number; mood: number };
  inventory: Record<string, number>;
  routine: { routine: string; target: string; item: string; reason: string } | null;
  plan: { start_minute: number; end_minute: number; routine: string; note: string }[];
  recent_memories: { id: string; tick: number; kind: string; text: string; importance: number }[];
  reflections: { id: string; tick: number; text: string }[];
  goals: string[];
  goal_details?: { text: string; evidence_count: number; tick: number }[];
  rumors?: { id: string; tick: number; text: string; source_id: string | null; source: string; confidence: number; about: string[] }[];
  semantic_facts: string[];
  relationships: Record<string, RelationshipAxes>;
  incoming: Record<string, RelationshipAxes>;
  world_notes: string[];
  in_conversation: boolean;
}
