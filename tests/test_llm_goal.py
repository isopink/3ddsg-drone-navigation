"""LLM fallback for goal parsing. No network: the LLM is a fake ChatFn."""
import json

import pytest

from ddsg_nav.goal import GoalError, parse_goal
from ddsg_nav.graph.fake import build_fake_graph
from ddsg_nav.llm_goal import (
    LLMError,
    _read_dotenv,
    ask_llm,
    build_user_prompt,
    config_from_env,
)


@pytest.fixture(scope="module")
def graph():
    return build_fake_graph()


class FakeLLM:
    def __init__(self, reply="", error: Exception | None = None):
        self.reply, self.error, self.calls = reply, error, []

    def __call__(self, system, user):
        self.calls.append(user)
        if self.error:
            raise self.error
        return self.reply


def reply(start, goal):
    return json.dumps({"start": start, "goal": goal, "reason": "test"})


# --- when the LLM is (not) asked ---------------------------------------------------


@pytest.mark.parametrize("text", ["go from A to C", "A방에서 C방으로", "A -> C", "C방으로 가줘"])
def test_explicit_rules_do_not_call_llm(graph, text):
    llm = FakeLLM(reply("B", "B"))
    g = parse_goal(text, graph, current_room="A", llm=llm)
    assert (g.start, g.goal, g.source) == ("A", "C", "rules")
    assert llm.calls == []


def test_llm_used_when_rules_find_no_room(graph):
    llm = FakeLLM(reply("A", "C"))
    g = parse_goal("에이방에서 씨방으로 가줘", graph, llm=llm)
    assert (g.start, g.goal, g.source) == ("A", "C", "llm")
    assert len(llm.calls) == 1


def test_llm_used_when_rules_only_guessed(graph):
    # "A 옆 방으로": rules would guess goal=A; the LLM resolves "next to A"
    llm = FakeLLM(reply(None, "B"))
    g = parse_goal("A 옆 방으로 가줘", graph, current_room="C", llm=llm)
    assert (g.start, g.goal, g.source) == ("C", "B", "llm")


def test_without_llm_rules_behave_as_before(graph):
    with pytest.raises(GoalError):
        parse_goal("에이방에서 씨방으로", graph)


# --- the LLM answer is checked, never trusted ----------------------------------------


def test_invented_room_is_rejected(graph):
    with pytest.raises(GoalError, match="not in the graph"):
        parse_goal("에이방에서 디방으로", graph, llm=FakeLLM(reply("A", "Z")))


def test_same_room_is_rejected(graph):
    with pytest.raises(GoalError, match="same room"):
        parse_goal("여기서 여기로", graph, llm=FakeLLM(reply("A", "A")))


def test_missing_goal_is_rejected(graph):
    with pytest.raises(GoalError):
        parse_goal("어디든 가", graph, current_room="A", llm=FakeLLM(reply(None, None)))


def test_llm_failure_falls_back_to_rule_guess(graph):
    llm = FakeLLM(error=LLMError("offline"))
    g = parse_goal("A 옆 방으로 가줘", graph, current_room="C", llm=llm)
    assert (g.start, g.goal, g.source) == ("C", "A", "rules")


def test_llm_failure_without_rule_result_raises(graph):
    with pytest.raises(GoalError, match="LLM fallback failed"):
        parse_goal("에이에서 씨로", graph, llm=FakeLLM(error=LLMError("offline")))


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"start": "A", "goal": "C"}\n```',
        'Sure: {"start": "Room A", "goal": "c"}',  # names and case resolve to ids
    ],
)
def test_reply_parsing_is_tolerant(graph, raw):
    assert ask_llm("x", graph, FakeLLM(raw)) == ("A", "C")


@pytest.mark.parametrize("raw", ["no json here", '["A", "C"]', '{"start": 1, "goal": "C"}'])
def test_bad_replies_raise(graph, raw):
    with pytest.raises(LLMError):
        ask_llm("x", graph, FakeLLM(raw))


def test_prompt_lists_rooms_and_neighbors(graph):
    p = build_user_prompt("go to C", graph, "A")
    rooms = json.loads(p.split("\n")[0].removeprefix("Rooms: "))
    by_id = {r["id"]: r for r in rooms}
    assert set(by_id) == set(graph.rooms)
    assert "B" in by_id["A"]["neighbors"]
    assert "Current room: A" in p


# --- configuration ------------------------------------------------------------------


def test_no_key_means_no_llm(tmp_path, monkeypatch):
    monkeypatch.delenv("DDSG_LLM_API_KEY", raising=False)
    assert config_from_env(tmp_path / ".env") is None


def test_dotenv_and_env_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nDDSG_LLM_API_KEY="from-file"\nDDSG_LLM_MODEL=m1\n', encoding="utf-8")
    assert _read_dotenv(env)["DDSG_LLM_API_KEY"] == "from-file"
    monkeypatch.delenv("DDSG_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DDSG_LLM_MODEL", "m2")
    cfg = config_from_env(env)
    assert (cfg.api_key, cfg.model, cfg.base_url) == ("from-file", "m2", "https://api.deepseek.com")
