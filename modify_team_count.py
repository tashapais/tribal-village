#!/usr/bin/env python3
"""Modify Nim source files to set a different team count for recompilation."""
import os
import re
import sys

# All 8 castle unique units in order
ALL_CASTLE_UNITS = [
    "UnitSamurai",        # Team 0
    "UnitLongbowman",     # Team 1
    "UnitCataphract",     # Team 2
    "UnitWoadRaider",     # Team 3
    "UnitTeutonicKnight", # Team 4
    "UnitHuskarl",        # Team 5
    "UnitMameluke",       # Team 6
    "UnitJanissary",      # Team 7
]


def modify_types(types_path: str, num_teams: int) -> None:
    with open(types_path, "r") as f:
        content = f.read()

    # 1. Replace MapRoomObjectsTeams constant
    content = re.sub(
        r"MapRoomObjectsTeams\* = \d+",
        f"MapRoomObjectsTeams* = {num_teams}",
        content,
    )

    # 2. Generate new TeamMasks array
    lines = []
    for i in range(num_teams):
        bit = 1 << i
        lines.append(f"    0b{bit:08b}'u8,  # Team {i}")
    lines.append("    0b00000000'u8   # Goblins/invalid (no team affiliation)")
    masks_body = "\n".join(lines)
    new_masks = (
        f"TeamMasks*: array[MapRoomObjectsTeams + 1, TeamMask] = [\n"
        f"{masks_body}\n"
        f"  ]"
    )

    content = re.sub(
        r"TeamMasks\*: array\[MapRoomObjectsTeams \+ 1, TeamMask\] = \[.*?\]",
        new_masks,
        content,
        flags=re.DOTALL,
    )

    # 3. Replace AllTeamsMask
    all_mask = (1 << num_teams) - 1
    content = re.sub(
        r"AllTeamsMask\*: TeamMask = 0b[01]+'u8",
        f"AllTeamsMask*: TeamMask = 0b{all_mask:08b}'u8",
        content,
    )

    with open(types_path, "w") as f:
        f.write(content)

    print(f"Modified {types_path} for {num_teams} teams")


def modify_registry(registry_path: str, num_teams: int) -> None:
    with open(registry_path, "r") as f:
        content = f.read()

    # Rebuild CastleUniqueUnits array with only the first num_teams entries
    units = ALL_CASTLE_UNITS[:num_teams]
    lines = []
    for i, unit in enumerate(units):
        comma = "," if i < len(units) - 1 else ""
        lines.append(f"  {unit}{comma}        # Team {i}")
    units_body = "\n".join(lines)
    new_array = (
        f"const CastleUniqueUnits*: array[MapRoomObjectsTeams, AgentUnitClass] = [\n"
        f"{units_body}\n"
        f"]"
    )

    content = re.sub(
        r"const CastleUniqueUnits\*: array\[MapRoomObjectsTeams, AgentUnitClass\] = \[.*?\]",
        new_array,
        content,
        flags=re.DOTALL,
    )

    with open(registry_path, "w") as f:
        f.write(content)

    print(f"Modified {registry_path} for {num_teams} teams")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <types.nim path> <num_teams>")
        sys.exit(1)
    types_path = sys.argv[1]
    num_teams = int(sys.argv[2])

    modify_types(types_path, num_teams)

    # Also modify registry.nim in the same directory
    src_dir = os.path.dirname(types_path)
    registry_path = os.path.join(src_dir, "registry.nim")
    if os.path.exists(registry_path):
        modify_registry(registry_path, num_teams)
