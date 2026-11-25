"""``llm-vil`` command line: provider configuration and the live server.

    llm-vil config show
    llm-vil config set --profile my_vllm --provider openai_compat \\
        --base-url http://127.0.0.1:8000/v1 --fast-model qwen2.5-7b --api-key-env VLLM_KEY \\
        --prompt-key --embed-profile local --set-default
    llm-vil config test --profile my_vllm
    llm-vil serve --llm-profile my_vllm --embed-profile local

Keys go to ``backend/.env`` (mode 600) and are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path
from typing import Any

from .llm.config_store import (
    PROVIDERS,
    STRUCTURED,
    ConfigError,
    ConfigPaths,
    config_summary,
    probe_profile,
    save_profile_and_key,
)
from .run_options import add_run_arguments, run_options

DEFAULT_SAVE_PATH = Path(__file__).resolve().parents[2] / "saves" / "village.sqlite"


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = "  ".join("{:<" + str(w) + "}" for w in widths)
    out = [line.format(*header), line.format(*["-" * w for w in widths])]
    out += [line.format(*[str(c) for c in r]) for r in rows]
    return "\n".join(out)


def cmd_show(args: argparse.Namespace, paths: ConfigPaths) -> int:
    s = config_summary(None, None, paths)  # ConfigError handled in main()
    rows = []
    for p in s["profiles"]:
        key = "-" if not p["needs_key"] and not p["api_key_env"] else (
            f"✓ {p['key_hint']}" if p["has_api_key"] else "✗")
        rows.append([
            ("* " if p["name"] == s["default_profile"] else "  ") + p["name"],
            p["provider"] or "?", p["base_url"] or "-", p["fast_model"] or "-",
            p["smart_model"] or "-", p["structured"] or "-",
            "-" if not p["api_key_env"] else f"{p['api_key_env']} {key}",
            p["embeddings"] or s["default_embed_profile"] or "-",
        ])
    print(_table(rows, ["profile", "provider", "base_url", "fast_model", "smart_model",
                        "structured", "api key", "embedder"]))
    print(f"\n* default profile: {s['default_profile']}   "
          f"default embedder: {s['default_embed_profile']}   "
          f"local overrides: {'yes' if s['local_overrides'] else 'no'}")
    return 0


def cmd_set(args: argparse.Namespace, paths: ConfigPaths) -> int:
    fields: dict[str, Any] = {}
    for attr, key in (("provider", "provider"), ("base_url", "base_url"),
                      ("fast_model", "fast_model"), ("smart_model", "smart_model"),
                      ("structured", "structured"), ("timeout_s", "timeout_s"),
                      ("api_key_env", "api_key_env"), ("embed_profile", "embeddings")):
        value = getattr(args, attr)
        if value is not None:
            fields[key] = value
    key = args.api_key
    if args.prompt_key:
        key = getpass.getpass("API key (input hidden): ")
    try:  # validates profile, env name and key before writing anything
        save_profile_and_key(args.profile, fields, key, set_default=args.set_default,
                             paths=paths)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"saved profile '{args.profile}' to {paths.local_yaml.name}"
          + (f"; API key stored in {paths.env_file.name} (mode 600)" if key else "")
          + ("; set as default" if args.set_default else ""))
    return 0


def cmd_test(args: argparse.Namespace, paths: ConfigPaths) -> int:
    profile = args.profile or config_summary(None, None, paths)["default_profile"]
    res = asyncio.run(probe_profile(profile, paths=paths))
    if res["ok"]:
        print(f"✓ {profile}: {res['provider']} {res['model']} OK in {res['latency_ms']} ms")
        return 0
    print(f"✗ {profile}: {res['error']}", file=sys.stderr)
    return 1


def cmd_serve(args: argparse.Namespace, paths: ConfigPaths) -> int:
    import uvicorn

    from .server.app import create_app

    save_path = None if args.memory_only else (args.save_path or str(DEFAULT_SAVE_PATH))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"saving to {save_path} (use --memory-only for an ephemeral run)")
    options = run_options(args)
    if options["paused"]:
        print("starting paused: press ▶ 재생 in the app to run (or pass --autostart)")
    app = create_app(speed=args.speed, store_path=save_path, **options,
                     fresh=args.fresh,
                     llm_profile=args.llm_profile, embed_profile=args.embed_profile,
                     llm_config_path=str(paths.base_yaml), env_file=str(paths.env_file))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llm-vil", description="llm-vil village tools")
    ap.add_argument("--config", help="path to llm.yaml (llm.local.yaml is read beside it)")
    ap.add_argument("--env-file", help="path to the .env holding API keys")
    sub = ap.add_subparsers(dest="command", required=True)

    cfg = sub.add_parser("config", help="show / edit / test LLM provider profiles")
    csub = cfg.add_subparsers(dest="action", required=True)
    csub.add_parser("show", help="table of profiles, models and key status")
    st = csub.add_parser("set", help="create or update a profile (writes llm.local.yaml)")
    st.add_argument("--profile", default="openai_compat")
    st.add_argument("--provider", choices=PROVIDERS)
    st.add_argument("--base-url")
    st.add_argument("--fast-model")
    st.add_argument("--smart-model")
    st.add_argument("--structured", choices=[s for s in STRUCTURED if s != "native"])
    st.add_argument("--timeout-s", type=float)
    st.add_argument("--api-key-env", help="env var name that holds the key")
    keys = st.add_mutually_exclusive_group()
    keys.add_argument("--api-key", help="key value (prefer --prompt-key: avoids shell history)")
    keys.add_argument("--prompt-key", action="store_true", help="ask for the key (hidden)")
    st.add_argument("--embed-profile", help="embedder for this profile: local | demo | ...")
    st.add_argument("--set-default", action="store_true")
    ts = csub.add_parser("test", help="one tiny JSON call against a profile")
    ts.add_argument("--profile")

    sv = sub.add_parser("serve", help="run the live village server")
    sv.add_argument("--llm-profile", default=os.environ.get("LLM_VIL_PROFILE"))
    sv.add_argument("--embed-profile", default=os.environ.get("LLM_VIL_EMBED_PROFILE"))
    sv.add_argument("--host", default=os.environ.get("LLM_VIL_HOST", "127.0.0.1"))
    sv.add_argument("--port", type=int, default=int(os.environ.get("LLM_VIL_PORT", "8780")))
    sv.add_argument("--speed", type=float, default=1.0)
    add_run_arguments(sv)
    sv.add_argument("--save-path", default=os.environ.get("LLM_VIL_SAVE"),
                    help=f"SQLite save file (default: {DEFAULT_SAVE_PATH})")
    sv.add_argument("--memory-only", action="store_true",
                    help="no save file: checkpoints disabled, nothing persisted")
    sv.add_argument("--fresh", action="store_true",
                    help="start a brand-new village (wipes the save's checkpoints and log); "
                         "by default the latest checkpoint is resumed")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ConfigPaths.from_args(args.config, args.env_file)
    try:
        if args.command == "serve":
            return cmd_serve(args, paths)
        return {"show": cmd_show, "set": cmd_set, "test": cmd_test}[args.action](args, paths)
    except ConfigError as exc:  # e.g. broken llm.local.yaml: clean message, no traceback
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
