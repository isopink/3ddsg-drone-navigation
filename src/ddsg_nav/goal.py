"""Goal parsing: "go from room A to room C" -> start/goal room ids.

Rules first (room ids/names only), with an optional LLM fallback (see llm_goal.py).
Supported rule forms (case-insensitive for names):
    "go from A to C", "from Room A to Room C", "A -> C", "A→C",
    "A방에서 C방으로", "A방부터 C방까지", "C방으로 가줘" (needs current_room)

With `llm=` the LLM is asked when the rules fail, or when they only guessed (no
from/to, 에서/으로 or arrow marker), e.g. "회의실 옆 방으로 가줘". The LLM can only pick
rooms that exist in the graph. If it fails, a confident-enough rule result is still used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .graph.model import SceneGraph
from .llm_goal import ChatFn, LLMError, ask_llm


class GoalError(ValueError):
    """The text could not be turned into a start/goal pair."""


@dataclass(frozen=True)
class Goal:
    start: str  # room id
    goal: str  # room id
    source: str = "rules"  # "rules" or "llm"


@dataclass(frozen=True)
class _Mention:
    room_id: str
    start: int
    end: int


def _find_mentions(text: str, graph: SceneGraph) -> list[_Mention]:
    found: list[_Mention] = []
    for room in graph.rooms.values():
        for token in {room.id, room.name, *room.aliases}:
            flags = 0 if len(token) == 1 else re.IGNORECASE  # single letters: case-sensitive
            pattern = rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"
            for m in re.finditer(pattern, text, flags):
                found.append(_Mention(room.id, m.start(), m.end()))
    # drop mentions contained in a longer one ("A" inside "Room A")
    found.sort(key=lambda m: (m.start, -(m.end - m.start)))
    kept: list[_Mention] = []
    for m in found:
        if kept and m.start < kept[-1].end:
            continue
        kept.append(m)
    return kept


def _parse_rules(
    text: str, graph: SceneGraph, current_room: str | None
) -> tuple[str | None, str | None, bool]:
    """(start, goal, explicit). `explicit` is False when only the fallbacks were used."""
    mentions = _find_mentions(text, graph)
    if not mentions:
        raise GoalError(f"no known room in {text!r}")

    start: str | None = None
    goal: str | None = None

    # English: "from X to Y" (also "to Y from X")
    f = re.search(r"\bfrom\b", text, re.IGNORECASE)
    t = re.search(r"\bto\b", text, re.IGNORECASE)
    if f and t:
        after_f = [m for m in mentions if m.start >= f.end()]
        after_t = [m for m in mentions if m.start >= t.end()]
        if f.start() < t.start():
            after_f = [m for m in after_f if m.start < t.start()]
        elif f.start() > t.start():
            after_t = [m for m in after_t if m.start < f.start()]
        start = after_f[0].room_id if after_f else None
        goal = after_t[0].room_id if after_t else None

    # Korean markers right after the room mention
    if start is None or goal is None:
        for m in mentions:
            rest = text[m.end :]
            if re.match(r"\s*(?:호실|호|실|방)?\s*(에서|부터)", rest):
                start = start or m.room_id
            elif re.match(r"\s*(?:호실|호|실|방)?\s*(으로|로|까지)", rest):
                goal = goal or m.room_id

    # arrows
    if start is None or goal is None:
        arrow = re.search(r"->|=>|→|>", text)
        if arrow:
            before = [m for m in mentions if m.end <= arrow.start()]
            after = [m for m in mentions if m.start >= arrow.end()]
            if before and after:
                start, goal = before[-1].room_id, after[0].room_id

    explicit = goal is not None

    # fallbacks
    if start is None and goal is None:
        if len(mentions) >= 2:
            start, goal = mentions[0].room_id, mentions[1].room_id
        elif current_room is not None:
            start, goal = graph.find_room(current_room).id, mentions[0].room_id
    elif start is None and current_room is not None:
        start = graph.find_room(current_room).id
    elif goal is None:
        raise GoalError(f"no goal room in {text!r}")
    return start, goal, explicit


def _finish(start: str | None, goal: str | None, text: str, source: str) -> Goal:
    if start is None or goal is None:
        raise GoalError(f"could not find both start and goal in {text!r} (pass current_room?)")
    if start == goal:
        raise GoalError(f"start and goal are the same room ({start})")
    return Goal(start=start, goal=goal, source=source)


def parse_goal(
    text: str,
    graph: SceneGraph,
    current_room: str | None = None,
    llm: ChatFn | None = None,
) -> Goal:
    """Turn a command into start/goal room ids. `llm` (e.g. llm_goal.llm_from_env())
    is an optional fallback; without it only the rules run."""
    rule_goal: Goal | None = None
    rule_error: GoalError | None = None
    try:
        start, goal, explicit = _parse_rules(text, graph, current_room)
        rule_goal = _finish(start, goal, text, "rules")
        if explicit or llm is None:
            return rule_goal
    except GoalError as e:
        if llm is None:
            raise
        rule_error = e

    try:
        start, goal = ask_llm(text, graph, llm, current_room)
    except LLMError as e:
        if rule_goal is not None:
            return rule_goal  # the rules' guess beats nothing
        raise GoalError(f"{rule_error}; LLM fallback failed: {e}") from e
    if start is None and current_room is not None:
        start = graph.find_room(current_room).id
    return _finish(start, goal, text, "llm")
