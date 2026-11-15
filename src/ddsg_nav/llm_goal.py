"""LLM fallback for goal parsing.

The LLM only *picks room ids* from the graph; it never plans or flies. Its answer is
checked against the graph before it is used, so a made-up room is an error, not a route.

Any OpenAI-compatible chat API works (DeepSeek by default). Configure with environment
variables or a `.env` file in the working directory (see `.env.example`):

    DDSG_LLM_API_KEY    required; without it the LLM is off and only the rules run
    DDSG_LLM_BASE_URL   default https://api.deepseek.com
    DDSG_LLM_MODEL      default deepseek-chat

Only the standard library is used (no extra dependency).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .graph.model import SceneGraph

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class LLMError(RuntimeError):
    """The LLM could not be reached or gave an unusable answer."""


class ChatFn(Protocol):
    def __call__(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_s: float = 20.0


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def config_from_env(dotenv: Path | None = Path(".env")) -> LLMConfig | None:
    """LLMConfig from the environment (a `.env` file fills in missing values), or None
    if no API key is set."""
    file_vals = _read_dotenv(dotenv) if dotenv else {}

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name) or file_vals.get(name) or default

    key = get("DDSG_LLM_API_KEY")
    if not key:
        return None
    return LLMConfig(
        api_key=key,
        base_url=get("DDSG_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        model=get("DDSG_LLM_MODEL", DEFAULT_MODEL),
    )


def openai_compatible_chat(cfg: LLMConfig) -> ChatFn:
    """A ChatFn that calls `{base_url}/chat/completions` and returns the reply text."""

    def chat(system: str, user: str) -> str:
        body = {
            "model": cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{cfg.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"LLM API returned HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"LLM API unreachable: {e}") from e
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected LLM API response: {str(data)[:300]}") from e

    return chat


def llm_from_env() -> ChatFn | None:
    """The configured LLM, or None when no API key is set."""
    cfg = config_from_env()
    return openai_compatible_chat(cfg) if cfg else None


SYSTEM_PROMPT = """You turn a drone navigation command into a start room and a goal room.
The user message gives the known rooms as JSON and then the command.
Rules:
- Use only room ids from the list. Never invent a room.
- "start" is where the drone departs. If the command does not say, use null
  (the drone's current room is used then).
- "goal" is where the drone should go. If no listed room fits, use null.
- Use names, aliases and neighbors to resolve phrases like "the room next to X".
Reply with a single JSON object only:
{"start": "<room id or null>", "goal": "<room id or null>", "reason": "<short>"}"""


def describe_rooms(graph: SceneGraph) -> list[dict]:
    rg = graph.room_graph()
    return [
        {
            "id": r.id,
            "name": r.name,
            "aliases": list(r.aliases),
            "neighbors": sorted(rg.neighbors(r.id)),
        }
        for r in graph.rooms.values()
    ]


def build_user_prompt(text: str, graph: SceneGraph, current_room: str | None) -> str:
    rooms = json.dumps(describe_rooms(graph), ensure_ascii=False)
    cur = current_room if current_room is not None else "unknown"
    return f"Rooms: {rooms}\nCurrent room: {cur}\nCommand: {text}"


def _extract_json(reply: str) -> dict:
    try:
        obj = json.loads(reply)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", reply, re.DOTALL)  # tolerate ```json fences or chatter
        if not m:
            raise LLMError(f"LLM reply is not JSON: {reply[:200]!r}")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM reply is not JSON: {reply[:200]!r}") from e
    if not isinstance(obj, dict):
        raise LLMError(f"LLM reply is not a JSON object: {reply[:200]!r}")
    return obj


def ask_llm(
    text: str, graph: SceneGraph, chat: ChatFn, current_room: str | None = None
) -> tuple[str | None, str | None]:
    """Ask the LLM for (start, goal) room ids. Ids are checked against the graph; a
    missing value is returned as None. Raises LLMError for unusable answers."""
    obj = _extract_json(chat(SYSTEM_PROMPT, build_user_prompt(text, graph, current_room)))

    def room(key: str) -> str | None:
        v = obj.get(key)
        if v is None or (isinstance(v, str) and v.strip().lower() in {"", "null", "none"}):
            return None
        if not isinstance(v, str):
            raise LLMError(f"LLM gave a non-string {key}: {v!r}")
        try:
            return graph.find_room(v.strip()).id
        except KeyError:
            raise LLMError(f"LLM picked a room that is not in the graph: {v!r}") from None

    return room("start"), room("goal")


__all__: list[str] = [
    "ChatFn", "LLMConfig", "LLMError", "ask_llm", "config_from_env", "llm_from_env",
    "openai_compatible_chat",
]
