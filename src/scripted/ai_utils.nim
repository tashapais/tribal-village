## Shared utility functions for the AI system.
## Consolidates duplicate logic from ai_core.nim, fighter.nim, and other scripted modules.
## Import this module for common stance checks, entity lookups, and counting helpers.

import vmath
import ../types
import ../environment, ../common_types

export types, environment, common_types

const
  DefensiveRetaliationWindow* = 30  ## Steps after being attacked that defensive stance allows retaliation
  MaxNavalPerTeam* = 5  ## Cap naval training per team (used by builder + fighter)

# ---------------------------------------------------------------------------
# Stance Behavior Checks
# ---------------------------------------------------------------------------
# Consolidated stance checking logic. Different behaviors allow different stances:
#   - AutoAttack: Aggressive, StandGround (and Defensive if recently attacked)
#   - Chase: Aggressive (and Defensive if recently attacked)
#   - MovementToAttack: Aggressive (and Defensive if recently attacked)
# ---------------------------------------------------------------------------

type
  StanceBehavior* = enum
    ## Behaviors that can be allowed/denied based on agent stance.
    BehaviorAutoAttack     ## Auto-attacking enemies in range
    BehaviorChase          ## Chasing enemies beyond current position
    BehaviorMovementToAttack ## Moving toward enemies to engage

proc stanceAllows*(env: Environment, agent: Thing, behavior: StanceBehavior): bool =
  ## Check if agent's stance allows a specific behavior.
  ## Uses DefensiveRetaliationWindow for defensive stance retaliation checks.
  ##
  ## Behavior rules:
  ##   AutoAttack: Aggressive=yes, StandGround=yes, Defensive=retaliation, NoAttack=no
  ##   Chase: Aggressive=yes, StandGround=no, Defensive=retaliation, NoAttack=no
  ##   MovementToAttack: Aggressive=yes, StandGround=no, Defensive=retaliation, NoAttack=no
  case behavior
  of BehaviorAutoAttack:
    case agent.stance
    of StanceAggressive, StanceStandGround: true
    of StanceDefensive:
      agent.lastAttackedStep > 0 and
        (env.currentStep - agent.lastAttackedStep) <= DefensiveRetaliationWindow
    of StanceNoAttack: false
  of BehaviorChase, BehaviorMovementToAttack:
    case agent.stance
    of StanceAggressive: true
    of StanceDefensive:
      agent.lastAttackedStep > 0 and
        (env.currentStep - agent.lastAttackedStep) <= DefensiveRetaliationWindow
    of StanceStandGround, StanceNoAttack: false

# ---------------------------------------------------------------------------
# Generic Spatial Enemy Search
# ---------------------------------------------------------------------------
# Extracts the common pattern for finding nearest enemy agents using spatial index.
# Used by fighter.nim's findNearestMeleeEnemyUncached, findNearestSiegeEnemyUncached, etc.
# ---------------------------------------------------------------------------

type
  AgentUnitClassFilter* = enum
    ## Filter types for enemy searches
    FilterExclude  ## Exclude the specified unit classes
    FilterInclude  ## Only include the specified unit classes

proc findNearestEnemyOfClass*(env: Environment, agent: Thing, radius: int,
                               classes: set[AgentUnitClass], filterType: AgentUnitClassFilter): Thing =
  ## Find the nearest enemy agent within radius, filtered by unit class.
  ##
  ## Parameters:
  ##   env: The environment
  ##   agent: The searching agent
  ##   radius: Maximum Chebyshev distance to search
  ##   classes: Set of unit classes to filter
  ##   filterType: FilterExclude to skip these classes, FilterInclude to only match these
  ##
  ## Returns:
  ##   Nearest matching enemy, or nil if none found
  let teamMask = getTeamMask(agent)
  var bestEnemy: Thing = nil
  var bestDist = int.high

  let (cx, cy) = cellCoords(agent.pos)
  let clampedMax = min(radius, max(SpatialCellsX, SpatialCellsY) * SpatialCellSize)
  let cellRadius = distToCellRadius16(clampedMax)

  for ddx in -cellRadius .. cellRadius:
    for ddy in -cellRadius .. cellRadius:
      let nx = cx + ddx
      let ny = cy + ddy
      if nx < 0 or nx >= SpatialCellsX or ny < 0 or ny >= SpatialCellsY:
        continue
      for other in env.spatialIndex.kindCells[Agent][nx][ny]:
        if other.isNil or other.agentId == agent.agentId:
          continue
        if not isAgentAlive(env, other):
          continue
        # Bitwise team check: same team = skip
        if (getTeamMask(other) and teamMask) != 0:
          continue
        # Apply class filter
        case filterType
        of FilterExclude:
          if other.unitClass in classes:
            continue
        of FilterInclude:
          if other.unitClass notin classes:
            continue
        let dist = int(chebyshevDist(agent.pos, other.pos))
        if dist > radius:
          continue
        if dist < bestDist:
          bestDist = dist
          bestEnemy = other

  bestEnemy

# ---------------------------------------------------------------------------
# Team Unit Counting
# ---------------------------------------------------------------------------
# Generic helper for counting agents by criteria, used by teamSiegeCount, teamNavalCount, etc.
# ---------------------------------------------------------------------------

proc countTeamAgentsByClass*(env: Environment, teamId: int, classes: set[AgentUnitClass]): int =
  ## Count alive agents on a team matching any of the given unit classes.
  for id in 0 ..< MapAgents:
    if id < env.agents.len and env.terminated[id] == 0.0:
      let agent = env.agents[id]
      if getTeamId(agent) == teamId and agent.unitClass in classes:
        inc result

proc countTeamNavalAgents*(env: Environment, teamId: int): int =
  ## Count alive naval units for a team.
  for id in 0 ..< MapAgents:
    if id < env.agents.len and env.terminated[id] == 0.0:
      let agent = env.agents[id]
      if getTeamId(agent) == teamId and agent.isWaterUnit:
        inc result
