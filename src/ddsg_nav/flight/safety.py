"""Pre-flight checks. Pure functions (no cflib), so they can be unit-tested.

Lessons from real flights: a locked/crashed drone silently ignores take-off,
a frozen or blocked height sensor makes the motors saturate, and flying in the
dark makes the optical-flow position estimate diverge.
"""
from __future__ import annotations

from collections.abc import Sequence

# Bit order of the firmware's `supervisor.info` log variable.
SUPERVISOR_BITS = [
    "canBeArmed",
    "isArmed",
    "autoArm",
    "canFly",
    "isFlying",
    "isTumbled",
    "isLocked",
    "isCrashed",
    "hlcIsFlying",
    "hlcIsTrajFinished",
]

MIN_BATTERY_V = 3.7
MAX_GROUND_HEIGHT_MM = 80  # on the ground the Flow deck reads about 20-30 mm


class SafetyError(RuntimeError):
    """A pre-flight check failed. Do not take off."""


def decode_supervisor(value: int) -> dict[str, bool]:
    return {name: bool(value >> i & 1) for i, name in enumerate(SUPERVISOR_BITS)}


def check_supervisor(flags: dict[str, bool]) -> None:
    bad = [k for k in ("isLocked", "isCrashed", "isTumbled") if flags.get(k)]
    if bad:
        raise SafetyError(
            f"drone state {bad}: power-cycle the Crazyflie on a flat surface and try again"
        )


def check_battery(vbat: float, minimum: float = MIN_BATTERY_V) -> None:
    if vbat < minimum:
        raise SafetyError(f"battery {vbat:.2f} V is below {minimum} V: charge first")


def check_ground_height(values_mm: Sequence[int], max_mm: int = MAX_GROUND_HEIGHT_MM) -> None:
    """Height sensor sanity: low on the ground and not dead.

    A dead sensor reads 0 every time (seen after a failed deck start) or an error value such
    as 65535 (caught by `max_mm`). A steady small value is NOT treated as frozen: very close
    to the floor the sensor often repeats the same reading (e.g. 4 mm ten times) while it is
    healthy, and that falsely blocked take-off.
    """
    if not values_mm:
        raise SafetyError("no height sensor readings")
    if max(values_mm) > max_mm:
        raise SafetyError(
            f"height sensor reads {max(values_mm)} mm on the ground (expected < {max_mm}): "
            "sensor blocked or faulty"
        )
    if all(v == 0 for v in values_mm):
        raise SafetyError(
            "height sensor reads 0 mm every time: the Flow deck sensor did not start. "
            "Power-cycle the drone (battery out) and check the deck pins"
        )
