"""Goal parsing with real-style room numbers (401호, 302호)."""
import pytest

from ddsg_nav.goal import GoalError, parse_goal
from ddsg_nav.graph.fake import build_fake_graph
from ddsg_nav.graph.model import Room


def _numbered_graph(name_fmt: str):
    g = build_fake_graph()
    for old, new in (("A", "401"), ("B", "303"), ("C", "302")):
        r = g.rooms.pop(old)
        g.rooms[new] = Room(new, name_fmt.format(new), r.bounds)
    return g


# the stored room name must not matter: "401호", "Room 401" and "401" all work
@pytest.fixture(scope="module", params=["{}호", "Room {}", "{}"])
def numbered(request):
    return _numbered_graph(request.param)


@pytest.mark.parametrize(
    "text",
    [
        "401호에서 302호로 가줘",
        "401호에서 302호로",
        "401에서 302로 가줘",
        "401호부터 302호까지",
        "401호 -> 302호",
        "401호→302호",
        "302호로 가줘 401호에서",  # reversed order
        "302호로 401호에서 가줘",
        "401호에서 출발해서 302호로 이동해줘",
        "go from 401 to 302",
        "from room 401 to room 302",
    ],
)
def test_numbered_rooms(numbered, text):
    goal = parse_goal(text, numbered)
    assert (goal.start, goal.goal) == ("401", "302"), text


def test_numbered_goal_only_uses_current_room(numbered):
    goal = parse_goal("302호로 가줘", numbered, current_room="401")
    assert (goal.start, goal.goal) == ("401", "302")


def test_numbered_goal_only_without_current_room_fails(numbered):
    with pytest.raises(GoalError):
        parse_goal("302호로 가줘", numbered)


def test_numbered_unknown_room(numbered):
    with pytest.raises(GoalError):
        parse_goal("999호에서 888호로 가줘", numbered)


def test_numbers_inside_longer_numbers_do_not_match(numbered):
    # 4011호 is not 401호
    with pytest.raises(GoalError):
        parse_goal("4011호에서 3020호로", numbered)
