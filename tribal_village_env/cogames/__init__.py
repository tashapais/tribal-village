"""Integration hook for CoGames."""

from .policy import TribalVillagePufferPolicy  # noqa: F401 - triggers policy registration side effects

__all__ = ["TribalVillagePufferPolicy"]
