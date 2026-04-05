"""Integration hook for CoGames."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["TribalVillagePufferPolicy"]


def __getattr__(name: str) -> Any:
    if name != "TribalVillagePufferPolicy":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    policy = import_module("tribal_village_env.cogames.policy").TribalVillagePufferPolicy
    globals()[name] = policy
    return policy
