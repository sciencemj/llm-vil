import { el, toast } from "./util";

interface ProfileSummary {
  name: string;
  provider: string | null;
  base_url: string | null;
  fast_model: string | null;
  smart_model: string | null;
  structured: string | null;
  timeout_s: number | null;
  api_key_env: string | null;
  needs_key: boolean;
  has_api_key: boolean;
  key_hint: string | null;
  embeddings: string | null;
}

interface ConfigSummary {
  active_profile: string | null;
  active_embed_profile: string | null;
  default_profile: string;
  default_embed_profile: string | null;
  profiles: ProfileSummary[];
  embed_profiles: { name: string; provider: string; model: string | null; dim: number | null }[];
  providers: string[];
  structured_modes: string[];
}

interface ProbeResult {
  ok: boolean;
  latency_ms: number | null;
  provider: string | null;
  model: string | null;
  error: string | null;
}

const NEW = "__new__";
const PROVIDER_KO: Record<string, string> = {
  demo: "demo (오프라인 규칙)",
  openai_compat: "OpenAI 호환 (OpenAI·Ollama·vLLM·OpenRouter…)",
  anthropic: "Anthropic",
  gemini: "Gemini",
};
const DEFAULT_ENV: Record<string, string> = {
  openai_compat: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
  gemini: "GEMINI_API_KEY",
};

function field(label: string, control: HTMLElement, hint = ""): HTMLElement {
  return el(
    "label",
    { class: "field" },
    el("span", { class: "flabel" }, label),
    control,
    hint ? el("span", { class: "hint" }, hint) : null,
  );
}

const STRUCTURED_BY_PROVIDER: Record<string, string[]> = {
  demo: [],
  openai_compat: ["json_schema", "json_object", "prompt_only"],
  anthropic: ["tool_call"],
  gemini: ["json_schema"],
};

async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let data: (T & { detail?: unknown }) | null = null;
  try {
    data = text ? (JSON.parse(text) as T & { detail?: unknown }) : null;
  } catch {
    data = null; // e.g. a proxy's HTML 502 page
  }
  if (!res.ok || data === null) {
    const detail = data && typeof data.detail === "string" ? data.detail : null;
    throw new Error(detail ?? `서버 오류 (HTTP ${res.status})`);
  }
  return data;
}

/**
 * Provider & API-key settings. The API key input is write-only: the server only ever
 * returns `has_api_key` and a masked hint, and the field is cleared after every request.
 */
export class LLMSettings {
  private cfg: ConfigSummary | null = null;
  private profileSel = el("select", { "aria-label": "프로필" });
  private newName = el("input", { type: "text", placeholder: "새 프로필 이름 (예: my_vllm)", maxlength: "40" });
  private provider = el("select", { "aria-label": "공급자" });
  private baseUrl = el("input", { type: "text", placeholder: "http://127.0.0.1:8000/v1" });
  private fastModel = el("input", { type: "text", placeholder: "빠른 모델 id" });
  private smartModel = el("input", { type: "text", placeholder: "똑똑한 모델 id (비우면 빠른 모델)" });
  private structured = el("select", { "aria-label": "구조화 모드" });
  private timeout = el("input", { type: "number", min: "1", max: "600", step: "1" });
  private keyEnv = el("input", { type: "text", placeholder: "MY_PROVIDER_API_KEY" });
  private apiKey = el("input", { type: "password", autocomplete: "off", placeholder: "새 키 입력" });
  private keyBadge = el("span", { class: "badge" });
  private embed = el("select", { "aria-label": "임베더" });
  private setDefault = el("input", { type: "checkbox" });
  private result = el("div", { class: "result" });
  private active = el("div", { class: "persona" });
  private busy = false;
  onApplied: () => void = () => {};

  constructor(root: HTMLElement) {
    this.profileSel.addEventListener("change", () => this.fillForm());
    this.provider.addEventListener("change", () => this.onProviderChange());
    this.keyEnv.addEventListener("input", () => this.syncVisibility());
    const test = el("button", { class: "btn" }, "🔌 연결 테스트");
    const save = el("button", { class: "btn" }, "💾 저장");
    const apply = el("button", { class: "btn on" }, "⚡ 저장 & 즉시 적용");
    test.addEventListener("click", () => void this.run("test"));
    save.addEventListener("click", () => void this.run("save"));
    apply.addEventListener("click", () => void this.run("apply"));
    root.append(
      this.active,
      field("프로필", el("div", { class: "row" }, this.profileSel, this.newName)),
      field("공급자", this.provider),
      field("Base URL", this.baseUrl, "OpenAI 호환 엔드포인트 주소 (Ollama: http://localhost:11434/v1)"),
      field("빠른 모델 (fast)", this.fastModel),
      field("똑똑한 모델 (smart)", this.smartModel),
      field("구조화 모드", this.structured),
      field("타임아웃 (초)", this.timeout),
      field("키 환경변수", this.keyEnv, "backend/.env 에 저장될 변수 이름"),
      field("API 키", el("div", { class: "row" }, this.apiKey, this.keyBadge), "새 키 입력 시에만 갱신 (비워두면 유지)"),
      field("임베더", this.embed),
      el("label", { class: "row" }, this.setDefault, el("span", {}, "기본 프로필로 지정")),
      el("div", { class: "row" }, test, save, apply),
      this.result,
    );
  }

  async load(select?: string): Promise<void> {
    try {
      this.cfg = await api<ConfigSummary>("/api/llm/config");
    } catch (err) {
      this.result.textContent = `설정을 불러오지 못했어요: ${(err as Error).message}`;
      return;
    }
    const c = this.cfg;
    const keep = select ?? this.profileSel.value;
    this.profileSel.replaceChildren(
      ...c.profiles.map((p) =>
        el("option", { value: p.name }, `${p.name}${p.name === c.active_profile ? " (실행 중)" : ""}${p.name === c.default_profile ? " ★" : ""}`),
      ),
      el("option", { value: NEW }, "＋ 새 프로필…"),
    );
    this.profileSel.value = keep && (keep === NEW || c.profiles.some((p) => p.name === keep)) ? keep : c.active_profile ?? c.default_profile;
    this.provider.replaceChildren(...c.providers.map((p) => el("option", { value: p }, PROVIDER_KO[p] ?? p)));
    this.embed.replaceChildren(
      el("option", { value: "" }, `(기본: ${c.default_embed_profile ?? "local"})`),
      ...c.embed_profiles.map((e) => el("option", { value: e.name }, `${e.name}${e.model ? ` · ${e.model}` : ""}`)),
    );
    this.active.textContent = `실행 중: ${c.active_profile ?? "?"} · 임베더 ${c.active_embed_profile ?? "?"}`;
    this.fillForm();
  }

  private current(): ProfileSummary | undefined {
    return this.cfg?.profiles.find((p) => p.name === this.profileSel.value);
  }

  private fillForm(): void {
    const isNew = this.profileSel.value === NEW;
    this.newName.classList.toggle("hidden", !isNew);
    this.result.className = "result";
    this.result.textContent = ""; // a previous profile's test result no longer applies
    const p = this.current();
    this.provider.value = p?.provider ?? "openai_compat";
    this.fillStructuredOptions();
    this.baseUrl.value = p?.base_url ?? "";
    this.fastModel.value = p?.fast_model ?? "";
    this.smartModel.value = p?.smart_model && p.smart_model !== p.fast_model ? p.smart_model : "";
    this.structured.value = p?.structured && p.structured !== "native" ? p.structured : "";
    if (this.structured.selectedIndex < 0) this.structured.value = "";
    this.timeout.value = p?.timeout_s != null ? String(p.timeout_s) : "";
    this.keyEnv.value = p?.api_key_env ?? "";
    this.embed.value = p?.embeddings ?? "";
    this.setDefault.checked = !!p && p.name === this.cfg?.default_profile;
    this.apiKey.value = "";
    this.syncVisibility();
  }

  /** Only modes the provider's adapter implements (openai_compat never gets tool_call). */
  private fillStructuredOptions(): void {
    const keep = this.structured.value;
    const modes = STRUCTURED_BY_PROVIDER[this.provider.value] ?? [];
    this.structured.replaceChildren(
      el("option", { value: "" }, "(공급자 기본값)"),
      ...modes.map((s) => el("option", { value: s }, s)),
    );
    this.structured.value = modes.includes(keep) ? keep : "";
  }

  /** User changed the provider: suggest a default key variable only for a brand-new profile. */
  private onProviderChange(): void {
    this.fillStructuredOptions();
    const prov = this.provider.value;
    if (this.profileSel.value === NEW && !this.keyEnv.value && DEFAULT_ENV[prov]) {
      this.keyEnv.value = DEFAULT_ENV[prov];
    }
    this.syncVisibility();
  }

  private syncVisibility(): void {
    const prov = this.provider.value;
    const p = this.current();
    const remote = prov !== "demo";
    for (const node of [this.baseUrl, this.fastModel, this.smartModel, this.structured, this.timeout, this.keyEnv, this.apiKey]) {
      node.closest(".field")?.classList.toggle("hidden", !remote);
    }
    // keyless profiles (e.g. local Ollama, api_key_env: null) keep an empty env field
    const keyless = !this.keyEnv.value.trim() && !(p?.needs_key ?? false);
    if (p?.has_api_key) {
      this.keyBadge.className = "badge st-ok";
      this.keyBadge.textContent = `설정됨 ✓ ${p.key_hint ?? ""}`;
    } else if (keyless) {
      this.keyBadge.className = "badge";
      this.keyBadge.textContent = "키 불필요";
    } else {
      this.keyBadge.className = "badge st-fallback";
      this.keyBadge.textContent = "미설정";
    }
  }

  private profileName(): string {
    return this.profileSel.value === NEW ? this.newName.value.trim() : this.profileSel.value;
  }

  private fields(): Record<string, unknown> {
    const prov = this.provider.value;
    const f: Record<string, unknown> = { provider: prov };
    if (this.embed.value) f.embeddings = this.embed.value;
    if (prov === "demo") return f;
    const set = (k: string, v: string) => {
      if (v.trim()) f[k] = v.trim();
    };
    set("base_url", this.baseUrl.value);
    set("fast_model", this.fastModel.value);
    set("smart_model", this.smartModel.value);
    set("structured", this.structured.value);
    set("timeout_s", this.timeout.value);
    f.api_key_env = this.keyEnv.value.trim() || null; // explicit null clears the variable
    return f;
  }

  private async run(kind: "test" | "save" | "apply"): Promise<void> {
    if (this.busy) return;
    const profile = this.profileName();
    if (!profile) {
      this.result.textContent = "프로필 이름을 입력해 주세요";
      return;
    }
    this.busy = true;
    this.result.className = "result";
    this.result.textContent = kind === "test" ? "연결 확인 중…" : "저장 중…";
    const apiKey = this.apiKey.value || undefined;
    try {
      if (kind === "test") {
        const r = await api<ProbeResult>("/api/llm/test", { profile, fields: this.fields(), api_key: apiKey });
        this.result.className = `result ${r.ok ? "ok" : "fail"}`;
        this.result.textContent = r.ok
          ? `✓ 연결 성공 · ${r.provider} ${r.model ?? ""} · ${r.latency_ms?.toFixed(0)}ms`
          : `✗ 실패 · ${r.error ?? "알 수 없는 오류"}`;
        return;
      }
      await api("/api/llm/config", { profile, fields: this.fields(), api_key: apiKey, set_default: this.setDefault.checked });
      this.apiKey.value = "";
      if (kind === "apply") {
        await api("/api/llm/switch", { profile, embed_profile: this.embed.value || null });
        toast(`⚡ ${profile} 프로필로 전환했어요`);
        this.onApplied();
      } else {
        toast(`💾 ${profile} 저장됨`);
      }
      await this.load(profile); // keep the (possibly new) profile selected
      this.result.className = "result ok";
      this.result.textContent = kind === "apply" ? `✓ 저장 후 ${profile} 적용됨` : `✓ ${profile} 저장됨`;
    } catch (err) {
      this.result.className = "result fail";
      this.result.textContent = `✗ ${(err as Error).message}`;
    } finally {
      this.apiKey.value = "";
      this.busy = false;
    }
  }
}
