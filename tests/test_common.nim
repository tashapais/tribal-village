## Common test harness setup code shared across behavior test files.
## Consolidates duplicated helpers for:
## - Game simulation (running steps with AI controller)
## - Resource and stockpile utilities
## - Role and agent counting
## - Building utilities

import std/strformat
import environment
import agent_control
import types
import items
import terrain
import test_utils

export test_utils
export environment
export agent_control
export types
export items
export terrain

# Re-export key functions from test_utils for convenience
export makeEmptyEnv, addAgentAt, addBuilding, addBuildings, addAltar, addResource
export setStockpile, stepNoop, stepAction, newTestController, dirIndex, decodeAction

# ============================================================================
# Game Simulation Helpers
# ============================================================================

proc runGameSteps*(env: Environment, steps: int) =
  ## Run the game for N steps using the global AI controller.
  for i in 0 ..< steps:
    let actions = getActions(env)
    env.step(addr actions)

proc initBrutalAI*(seed: int = 42) =
  ## Initialize global AI controller with Brutal difficulty for all teams.
  ## Use Brutal to ensure deterministic test behavior without random NOOP actions.
  initGlobalController(BuiltinAI, seed = seed)
  for teamId in 0 ..< MapRoomObjectsTeams:
    globalController.aiController.setDifficulty(teamId, DiffBrutal)

proc setupGameWithAI*(seed: int = 42): Environment =
  ## Create a new environment and initialize AI controller with Brutal difficulty.
  result = newEnvironment()
  initBrutalAI(seed)

proc setupGameWithAI*(config: EnvironmentConfig, seed: int = 42): Environment =
  ## Create a new environment with custom config and initialize AI controller.
  result = newEnvironment(config, seed)
  initBrutalAI(seed xor 0x12345678)  # XOR to break correlation

# ============================================================================
# Stockpile and Resource Helpers
# ============================================================================

proc printStockpileSummary*(env: Environment, teamId: int, label: string) =
  ## Print a formatted summary of a team's stockpile resources.
  let food = env.stockpileCount(teamId, ResourceFood)
  let wood = env.stockpileCount(teamId, ResourceWood)
  let gold = env.stockpileCount(teamId, ResourceGold)
  let stone = env.stockpileCount(teamId, ResourceStone)
  echo fmt"  [{label}] Team {teamId}: food={food} wood={wood} gold={gold} stone={stone}"

proc getTotalStockpile*(env: Environment, teamId: int): int =
  ## Get total resources in a team's stockpile.
  env.stockpileCount(teamId, ResourceFood) +
    env.stockpileCount(teamId, ResourceWood) +
    env.stockpileCount(teamId, ResourceGold) +
    env.stockpileCount(teamId, ResourceStone)

proc getTotalStockpileAllTeams*(env: Environment): int =
  ## Get total resources across all teams.
  for teamId in 0 ..< MapRoomObjectsTeams:
    result += getTotalStockpile(env, teamId)

proc setTeamResources*(env: Environment, teamId: int,
                       food: int = 0, wood: int = 0,
                       gold: int = 0, stone: int = 0) =
  ## Set a team's stockpile resources.
  setStockpile(env, teamId, ResourceFood, food)
  setStockpile(env, teamId, ResourceWood, wood)
  setStockpile(env, teamId, ResourceGold, gold)
  setStockpile(env, teamId, ResourceStone, stone)

proc setAllTeamsResources*(env: Environment,
                           food: int = 0, wood: int = 0,
                           gold: int = 0, stone: int = 0) =
  ## Set the same stockpile resources for all teams.
  for teamId in 0 ..< MapRoomObjectsTeams:
    setTeamResources(env, teamId, food, wood, gold, stone)

proc giveTeamPlentyOfResources*(env: Environment, teamId: int, amount: int = 500) =
  ## Give a team plenty of each resource type.
  setTeamResources(env, teamId, amount, amount, amount, amount)

proc giveAllTeamsPlentyOfResources*(env: Environment, amount: int = 500) =
  ## Give all teams plenty of each resource type.
  setAllTeamsResources(env, amount, amount, amount, amount)

# ============================================================================
# Role and Agent Counting Helpers
# ============================================================================

proc countRolesByTeam*(teamId: int): tuple[gatherers, builders, fighters: int] =
  ## Count agents by role for a given team.
  let controller = globalController.aiController
  let startIdx = teamId * MapAgentsPerTeam
  let endIdx = min(startIdx + MapAgentsPerTeam, MapAgents)
  for agentId in startIdx ..< endIdx:
    if controller.isAgentInitialized(agentId):
      let role = controller.getAgentRole(agentId)
      case role
      of Gatherer: inc result.gatherers
      of Builder: inc result.builders
      of Fighter: inc result.fighters
      of Scripted: discard

proc printRoleSummary*(teamId: int, label: string) =
  ## Print a formatted summary of agent roles for a team.
  let roles = countRolesByTeam(teamId)
  echo fmt"  [{label}] Team {teamId}: gatherers={roles.gatherers} builders={roles.builders} fighters={roles.fighters}"

proc countAliveUnits*(env: Environment, teamId: int): int =
  ## Count agents that are currently alive for a team.
  for agent in env.agents:
    if getTeamId(agent) == teamId and isAgentAlive(env, agent):
      inc result

proc countDeadUnits*(env: Environment, teamId: int): int =
  ## Count agents that were once alive but are now dead (killed).
  let startIdx = teamId * MapAgentsPerTeam
  let endIdx = min(startIdx + MapAgentsPerTeam, env.agents.len)
  for i in startIdx ..< endIdx:
    let agent = env.agents[i]
    if not agent.isNil and env.terminated[i] != 0.0 and agent.hp <= 0:
      inc result

proc countActiveAgents*(env: Environment): int =
  ## Count all agents that are currently alive across all teams.
  for agent in env.agents:
    if not agent.isNil and agent.hp > 0:
      inc result

proc countAgentsPerTeam*(env: Environment): array[MapRoomObjectsTeams, int] =
  ## Count alive agents per team.
  for agent in env.agents:
    if not agent.isNil and agent.hp > 0:
      let teamId = getTeamId(agent)
      if teamId >= 0 and teamId < MapRoomObjectsTeams:
        inc result[teamId]

# ============================================================================
# Building Helpers
# ============================================================================

const BuildingKinds* = {Altar, TownCenter, House, Barracks, ArcheryRange, Stable,
                        Blacksmith, Market, Monastery, University, Castle, Wonder,
                        SiegeWorkshop, MangonelWorkshop, TrebuchetWorkshop,
                        Dock, Outpost, GuardTower, Wall, Door, Mill, Granary,
                        LumberCamp, Quarry, MiningCamp, WeavingLoom, ClayOven,
                        Lantern, Temple}

proc countBuildings*(env: Environment, teamId: int): int =
  ## Count standing buildings owned by a team.
  for kind in BuildingKinds:
    for thing in env.thingsByKind[kind]:
      if not thing.isNil and thing.teamId == teamId and thing.hp > 0:
        inc result

proc countAllBuildings*(env: Environment): int =
  ## Count all standing buildings across all teams.
  for kind in BuildingKinds:
    for thing in env.thingsByKind[kind]:
      if not thing.isNil and thing.hp > 0:
        inc result

proc countBuildingsPerTeam*(env: Environment): array[MapRoomObjectsTeams, int] =
  ## Count buildings per team.
  for kind in BuildingKinds:
    for thing in env.thingsByKind[kind]:
      if not thing.isNil and thing.hp > 0:
        let teamId = thing.teamId
        if teamId >= 0 and teamId < MapRoomObjectsTeams:
          inc result[teamId]

proc countDamagedBuildings*(env: Environment, teamId: int): int =
  ## Count buildings owned by team that have HP < maxHp.
  for thing in env.things:
    if thing.isNil:
      continue
    let isRepairable = isBuildingKind(thing.kind) or thing.kind in {Wall, Door}
    if not isRepairable:
      continue
    if thing.teamId != teamId:
      continue
    if thing.maxHp > 0 and thing.hp < thing.maxHp:
      inc result

proc damageBuilding*(thing: Thing, damageAmount: int) =
  ## Apply damage to a building, reducing its HP (but not destroying it).
  if thing.maxHp > 0:
    thing.hp = max(1, thing.hp - damageAmount)

proc findBuildingToTest*(env: Environment, teamId: int, preferNonWall: bool = true): Thing =
  ## Find a building suitable for testing (preferring non-wall if requested).
  if preferNonWall:
    for thing in env.things:
      if thing.isNil:
        continue
      if thing.teamId == teamId and thing.maxHp > 0 and thing.hp == thing.maxHp:
        if isBuildingKind(thing.kind) and thing.kind != Wall and thing.kind != Door:
          return thing
  # Fallback to wall
  for thing in env.things:
    if thing.isNil:
      continue
    if thing.teamId == teamId and thing.maxHp > 0 and thing.hp == thing.maxHp:
      if thing.kind == Wall:
        return thing
  return nil

# ============================================================================
# Combat and HP Helpers
# ============================================================================

proc getTotalHp*(env: Environment): int =
  ## Get total HP across all agents.
  for agent in env.agents:
    if not agent.isNil and agent.hp > 0:
      result += agent.hp

proc getTotalHpPerTeam*(env: Environment): array[MapRoomObjectsTeams, int] =
  ## Get total HP per team.
  for agent in env.agents:
    if not agent.isNil and agent.hp > 0:
      let teamId = getTeamId(agent)
      if teamId >= 0 and teamId < MapRoomObjectsTeams:
        result[teamId] += agent.hp

# ============================================================================
# Action Tracking Helpers
# ============================================================================

proc countNonNoopActions*(actions: array[MapAgents, uint16]): int =
  ## Count non-NOOP actions in an action array.
  for i in 0 ..< MapAgents:
    if actions[i] != 0:
      inc result

proc runAndCountActions*(env: Environment, steps: int): tuple[actions, noops: int] =
  ## Run game steps and count actions vs NOOPs.
  for step in 0 ..< steps:
    let actions = getActions(env)
    for i in 0 ..< MapAgents:
      if actions[i] != 0:
        inc result.actions
      else:
        inc result.noops
    env.step(addr actions)

# ============================================================================
# Test Constants
# ============================================================================

const
  DefaultTestSeed* = 42
  ShortSimSteps* = 100
  MediumSimSteps* = 200
  LongSimSteps* = 300
  VeryLongSimSteps* = 500
