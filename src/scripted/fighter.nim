import std/tables

import ai_build_helpers
export ai_build_helpers

import ai_utils
export ai_utils

import options
export options

import coordination
export coordination

import ../formations
export formations

const
  DividerInvSqrt2 = 0.70710677'f32
  # Order matters: agents start at (agentId mod len), so position determines which
  # building type each agent tries first. Spread rare buildings (Castle, Monastery,
  # Dock) across the rotation to ensure diverse unit production.
  FighterTrainKinds = [Castle, Barracks, Monastery, ArcheryRange, Dock, Stable, MangonelWorkshop, SiegeWorkshop, TrebuchetWorkshop]
  FighterSiegeKinds = {MangonelWorkshop, SiegeWorkshop, TrebuchetWorkshop}
  FighterNavalKinds = {Dock}
  MaxSiegePerTeam = 3  # Cap siege training to prevent resource drain

# Per-step cache for isThreateningAlly results to avoid redundant spatial scans
# Key: (enemyAgentId * MapRoomObjectsTeams + teamId), Value: isThreatening
var threateningCacheStep: int = -1
var threateningCache: Table[int, bool]

# Per-step caches for expensive AI lookups
# These avoid redundant scans when canStart/shouldTerminate/act all call the same lookup
# Uses PerAgentCache[T] from ai_core.nim to eliminate boilerplate
var meleeEnemyCache: PerAgentCache[Thing]
var siegeEnemyCache: PerAgentCache[Thing]
var friendlyMonkCache: PerAgentCache[Thing]
var combatAllyCache: PerAgentCache[Thing]
var scoutEnemyCache: PerAgentCache[Thing]
var seesEnemyStructureCache: PerAgentCache[bool]
var allyNearbyCache: PerAgentCache[bool]

const
  SiegeUnitClasses = {UnitBatteringRam, UnitMangonel, UnitTrebuchet, UnitScorpion}

proc teamSiegeCount(env: Environment, teamId: int): int =
  ## Count alive siege units for a team (BatteringRam, Mangonel, Trebuchet, Scorpion).
  ## Uses consolidated countTeamAgentsByClass from ai_utils.
  countTeamAgentsByClass(env, teamId, SiegeUnitClasses)

proc teamSiegeAtCap(env: Environment, teamId: int): bool =
  ## Returns true if team has reached siege training cap.
  teamSiegeCount(env, teamId) >= MaxSiegePerTeam

proc teamNavalCount(env: Environment, teamId: int): int =
  ## Count alive naval units for a team.
  ## Uses consolidated countTeamNavalAgents from ai_utils.
  countTeamNavalAgents(env, teamId)

proc teamNavalAtCap(env: Environment, teamId: int): bool =
  ## Returns true if team has reached naval training cap.
  teamNavalCount(env, teamId) >= MaxNavalPerTeam

proc stanceAllowsChase*(env: Environment, agent: Thing): bool =
  ## Returns true if the agent's stance allows chasing enemies.
  ## Delegates to ai_utils.stanceAllows for consolidated stance logic.
  stanceAllows(env, agent, BehaviorChase)

proc stanceAllowsMovementToAttack*(env: Environment, agent: Thing): bool =
  ## Returns true if the agent's stance allows moving to attack.
  ## Delegates to ai_utils.stanceAllows for consolidated stance logic.
  stanceAllows(env, agent, BehaviorMovementToAttack)

proc fighterIsEnclosed(env: Environment, agent: Thing): bool =
  for _, d in Directions8:
    let np = agent.pos + d
    if canEnterForMove(env, agent, agent.pos, np):
      return false
  true

proc isThreateningAlly(env: Environment, enemy: Thing, teamId: int): bool =
  ## Check if an enemy is close enough to any ally to be considered a threat.
  ## Uses spatial index instead of scanning all agents.
  ## Optimized: per-step cache to avoid redundant scans when multiple fighters
  ## evaluate the same enemy. Also uses bitwise team mask comparison for O(1) team checks.

  # Invalidate cache if step changed
  if threateningCacheStep != env.currentStep:
    threateningCacheStep = env.currentStep
    threateningCache.clear()

  # Check cache first
  let cacheKey = enemy.agentId * MapRoomObjectsTeams + teamId
  if cacheKey in threateningCache:
    return threateningCache[cacheKey]

  # Compute and cache result
  let (cx, cy) = cellCoords(enemy.pos)
  let clampedMax = min(AllyThreatRadius, max(SpatialCellsX, SpatialCellsY) * SpatialCellSize)
  let cellRadius = distToCellRadius16(clampedMax)
  let teamMask = getTeamMask(teamId)  # Pre-compute for bitwise checks
  for ddx in -cellRadius .. cellRadius:
    for ddy in -cellRadius .. cellRadius:
      let nx = cx + ddx
      let ny = cy + ddy
      if nx < 0 or nx >= SpatialCellsX or ny < 0 or ny >= SpatialCellsY:
        continue
      for other in env.spatialIndex.kindCells[Agent][nx][ny]:
        if other.isNil or not isAgentAlive(env, other):
          continue
        # Bitwise team check: (otherMask and teamMask) != 0 means same team
        if (getTeamMask(other) and teamMask) == 0:
          continue
        if int(chebyshevDist(enemy.pos, other.pos)) <= AllyThreatRadius:
          threateningCache[cacheKey] = true
          return true
  threateningCache[cacheKey] = false
  false

proc cachedIsThreateningAlly(controller: Controller, env: Environment, enemy: Thing, teamId: int): bool =
  ## Check if an enemy is threatening an ally, with per-step caching.
  ## Cache avoids redundant spatial scans when multiple fighters evaluate the same enemy.
  let agentId = enemy.agentId
  if agentId < 0 or agentId >= MapAgents:
    return isThreateningAlly(env, enemy, teamId)

  # Invalidate cache if step changed
  if controller.allyThreatCacheStep[teamId] != env.currentStep:
    controller.allyThreatCacheStep[teamId] = env.currentStep
    # Reset cache entries for this team (mark all as uncached)
    for i in 0 ..< MapAgents:
      controller.allyThreatCache[teamId][i] = -1'i8

  # Check cache
  let cached = controller.allyThreatCache[teamId][agentId]
  if cached >= 0:
    return cached == 1

  # Compute and cache
  let isThreat = isThreateningAlly(env, enemy, teamId)
  controller.allyThreatCache[teamId][agentId] = if isThreat: 1'i8 else: 0'i8
  isThreat

proc scoreEnemy(controller: Controller, env: Environment, agent: Thing, enemy: Thing, teamId: int): float =
  ## Score an enemy for target selection. Higher score = better target.
  ## Considers: distance, HP ratio, threat to allies, class counters, and unit value.
  var score = 0.0
  let dist = int(chebyshevDist(agent.pos, enemy.pos))

  # Base score from distance (closer is better, max ~20 points for adjacent)
  score += float(20 - min(dist, 20))

  # Bonus for low HP enemies (easier to finish off) - up to 15 points
  let hpRatio = if enemy.maxHp > 0: float(enemy.hp) / float(enemy.maxHp) else: 1.0
  if hpRatio <= LowHpThreshold:
    score += 15.0  # High priority for very low HP targets
  elif hpRatio <= 0.5:
    score += 10.0  # Medium priority for half-HP targets
  elif hpRatio <= 0.75:
    score += 5.0   # Small bonus for wounded targets

  # Bonus for enemies threatening allies - up to 20 points (uses per-step cache)
  if cachedIsThreateningAlly(controller, env, enemy, teamId):
    score += 20.0

  # Class counter bonus - prioritize targets we deal bonus damage to (up to 12 points)
  # This encourages smart unit matchups (infantry hunting cavalry, etc.)
  let counterBonus = BonusDamageByClass[agent.unitClass][enemy.unitClass]
  if counterBonus > 0:
    score += float(counterBonus) * 6.0  # +6 per point of counter damage

  # Siege unit priority - high-value targets that threaten structures (up to 15 points)
  if enemy.unitClass in {UnitBatteringRam, UnitMangonel, UnitTrebuchet}:
    score += 15.0  # Prioritize siege units as they're dangerous

  # Unit value consideration - prioritize killing expensive/powerful units (up to 10 points)
  # Based on max HP as a rough proxy for unit importance
  let valueScore = float(min(enemy.maxHp, 15)) * 0.67
  score += valueScore

  score

proc fighterFindNearbyEnemy*(controller: Controller, env: Environment, agent: Thing,
                            state: var AgentState): Thing =
  ## Find the best enemy target using smart target selection with periodic re-evaluation.
  ## Prioritizes: enemies threatening allies > low HP enemies > closest enemies.
  ## On lower difficulties (advancedTargetingEnabled=false), simply picks the closest enemy.
  ##
  ## Optimized: scans grid tiles within enemyRadius instead of all agents.
  let enemyRadius = ObservationRadius.int32 * 2
  let teamId = getTeamId(agent)
  let diffConfig = controller.getDifficulty(teamId)
  let useAdvancedTargeting = diffConfig.advancedTargetingEnabled

  # Check if we should use cached target or re-evaluate
  # Re-evaluate every TargetSwapInterval ticks or if cache is stale
  let shouldReevaluate = (env.currentStep - state.fighterEnemyStep) >= TargetSwapInterval

  if not shouldReevaluate and state.fighterEnemyStep >= 0 and
      state.fighterEnemyAgentId >= 0 and state.fighterEnemyAgentId < MapAgents:
    let cached = env.agents[state.fighterEnemyAgentId]
    if cached.agentId != agent.agentId and
        isAgentAlive(env, cached) and
        not sameTeam(agent, cached) and
        int(chebyshevDist(agent.pos, cached.pos)) <= enemyRadius.int:
      return cached

  # Re-evaluate: find the best target using spatial index
  var bestScore = float.low
  var bestDist = int.high
  var bestEnemyId = -1

  # Use spatial index to only check agents in nearby cells
  let (cx, cy) = cellCoords(agent.pos)
  let clampedMax = min(enemyRadius.int, max(SpatialCellsX, SpatialCellsY) * SpatialCellSize)
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
        if sameTeam(agent, other):
          continue
        let dist = int(chebyshevDist(agent.pos, other.pos))
        if dist > enemyRadius.int:
          continue

        if useAdvancedTargeting:
          let score = scoreEnemy(controller, env, agent, other, teamId)
          if score > bestScore:
            bestScore = score
            bestEnemyId = other.agentId
        else:
          if dist < bestDist:
            bestDist = dist
            bestEnemyId = other.agentId

  state.fighterEnemyStep = env.currentStep
  state.fighterEnemyAgentId = bestEnemyId
  if bestEnemyId >= 0 and bestEnemyId < env.agents.len:
    return env.agents[bestEnemyId]

proc fighterSeesEnemyStructureUncached(env: Environment, agent: Thing): bool =
  ## Internal: actual search logic for enemy structures in vision.
  ## Optimized: uses spatial index for O(cells) instead of O(all buildings) iteration.
  let teamId = getTeamId(agent)
  let radius = ObservationRadius.int
  # Use spatial query to find nearest enemy building within observation radius
  # Note: findNearestEnemyBuildingSpatial checks all TeamBuildingKinds which includes
  # the relevant attackable structures (Wall, Outpost, GuardTower, Castle, TownCenter, Monastery)
  let enemy = findNearestEnemyBuildingSpatial(env, agent.pos, teamId, radius)
  not enemy.isNil

proc fighterSeesEnemyStructure(env: Environment, agent: Thing): bool =
  ## Check if agent can see an enemy structure within observation radius.
  ## Cached per-step per-agent to avoid redundant scans in canStart/shouldTerminate/act.
  seesEnemyStructureCache.getWithAgent(env, agent, fighterSeesEnemyStructureUncached)

proc canStartFighterMonk(controller: Controller, env: Environment, agent: Thing,
                         agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitMonk:
    return false
  # Activate when carrying a relic (need to deposit) or relics exist on map
  agent.inventoryRelic > 0 or env.thingsByKind[Relic].len > 0

proc shouldTerminateFighterMonk(controller: Controller, env: Environment, agent: Thing,
                                agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitMonk:
    return true
  # Terminate when not carrying and no relics left to collect
  agent.inventoryRelic == 0 and env.thingsByKind[Relic].len == 0

proc findNearestRelicGlobal(env: Environment, pos: IVec2): Thing =
  ## Find nearest relic anywhere on the map using thingsByKind.
  ## O(num_relics) scan — fine since there are typically <20 relics.
  result = nil
  var minDist = int.high
  for relic in env.thingsByKind[Relic]:
    if relic.isNil or not isValidPos(relic.pos):
      continue
    let dist = int(chebyshevDist(pos, relic.pos))
    if dist < minDist:
      minDist = dist
      result = relic

proc optFighterMonk(controller: Controller, env: Environment, agent: Thing,
                    agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  # Priority 1: If carrying a relic, deposit it in the nearest monastery
  if agent.inventoryRelic > 0:
    let monastery = env.findNearestFriendlyThingSpiral(state, teamId, Monastery)
    if not isNil(monastery):
      return actOrMove(controller, env, agent, agentId, state, monastery.pos, 3'u16)
    # No monastery found via spiral — try global search
    var bestMonastery: Thing = nil
    var bestDist = int.high
    for m in env.thingsByKind[Monastery]:
      if m.isNil or not isValidPos(m.pos):
        continue
      if getTeamId(m) != teamId:
        continue
      let dist = int(chebyshevDist(agent.pos, m.pos))
      if dist < bestDist:
        bestDist = dist
        bestMonastery = m
    if not isNil(bestMonastery):
      return actOrMove(controller, env, agent, agentId, state, bestMonastery.pos, 3'u16)
    # Still no monastery — return home to stay safe
    if agent.homeAltar.x >= 0:
      return controller.moveTo(env, agent, agentId, state, agent.homeAltar)
    return 0'u16

  # Priority 2: Find and collect nearest relic (global search)
  let relic = findNearestRelicGlobal(env, agent.pos)
  if not isNil(relic):
    return actOrMove(controller, env, agent, agentId, state, relic.pos, 3'u16)

  # No relics to collect — stay near monastery for safety (0 attack, fragile)
  let monastery = env.findNearestFriendlyThingSpiral(state, teamId, Monastery)
  if not isNil(monastery):
    let dist = chebyshevDist(agent.pos, monastery.pos)
    if dist > 8:
      return actOrMove(controller, env, agent, agentId, state, monastery.pos, 3'u16)

  0'u16

proc canStartFighterBreakout(controller: Controller, env: Environment, agent: Thing,
                             agentId: int, state: var AgentState): bool =
  fighterIsEnclosed(env, agent)

proc shouldTerminateFighterBreakout(controller: Controller, env: Environment, agent: Thing,
                                    agentId: int, state: var AgentState): bool =
  not fighterIsEnclosed(env, agent)

proc optFighterBreakout(controller: Controller, env: Environment, agent: Thing,
                        agentId: int, state: var AgentState): uint16 =
  for dirIdx in 0 .. 7:
    let targetPos = agent.pos + Directions8[dirIdx]
    if not isValidPos(targetPos):
      continue
    if env.hasDoor(targetPos):
      return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, dirIdx.uint8))
    let blocker = env.getThing(targetPos)
    if not isNil(blocker) and blocker.kind in {Wall, Skeleton, Spawner, Tumor}:
      return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, dirIdx.uint8))
  0'u16


proc findNearestFriendlyMonkUncached(env: Environment, agent: Thing): Thing =
  ## Internal: actual search logic for nearest friendly monk.
  ## Optimized: uses bitwise team mask comparison for O(1) team checks.
  let teamMask = getTeamMask(agent)  # Pre-compute for bitwise checks
  var bestMonk: Thing = nil
  var bestDist = int.high
  let (cx, cy) = cellCoords(agent.pos)
  let clampedMax = min(HealerSeekRadius, max(SpatialCellsX, SpatialCellsY) * SpatialCellSize)
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
        # Bitwise team check: (otherMask and teamMask) == 0 means different team
        if (getTeamMask(other) and teamMask) == 0 or other.unitClass != UnitMonk:
          continue
        let dist = int(chebyshevDist(agent.pos, other.pos))
        if dist <= HealerSeekRadius and dist < bestDist:
          bestDist = dist
          bestMonk = other
  bestMonk

proc findNearestFriendlyMonk(env: Environment, agent: Thing): Thing =
  ## Find the nearest friendly monk to seek healing from using spatial index.
  ## Cached per-step per-agent to avoid redundant scans in canStart/shouldTerminate/act.
  friendlyMonkCache.getWithAgent(env, agent, findNearestFriendlyMonkUncached)

proc findNearestCombatAllyUncached(env: Environment, agent: Thing): Thing =
  ## Internal: actual search logic for nearest friendly combat unit (for retreat).
  ## Prioritizes healthy allies (HP > 50%) that are not too close.
  ## Combat units: ManAtArms, Knight, Archer (ranged support counts too).
  ## Optimized: uses bitwise team mask comparison for O(1) team checks.
  let teamMask = getTeamMask(agent)  # Pre-compute for bitwise checks
  var bestAlly: Thing = nil
  var bestDist = int.high
  let (cx, cy) = cellCoords(agent.pos)
  let clampedMax = min(RetreatAllySeekRadius, max(SpatialCellsX, SpatialCellsY) * SpatialCellSize)
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
        # Bitwise team check: (otherMask and teamMask) == 0 means different team
        if (getTeamMask(other) and teamMask) == 0:
          continue
        # Only consider combat units that can help in a fight
        if other.unitClass notin {UnitManAtArms, UnitLongSwordsman, UnitChampion,
                                  UnitKnight, UnitCavalier, UnitPaladin, UnitArcher,
                                  UnitCrossbowman, UnitArbalester, UnitScout, UnitLightCavalry,
                                  UnitHussar, UnitCamel, UnitHeavyCamel, UnitImperialCamel}:
          continue
        # Prefer healthy allies (HP > 50%) - don't retreat to wounded allies
        if other.hp * 2 < other.maxHp:
          continue
        let dist = int(chebyshevDist(agent.pos, other.pos))
        # Must be within search radius but not too close (already grouped)
        if dist > RetreatAllySeekRadius or dist < RetreatAllyMinDist:
          continue
        if dist < bestDist:
          bestDist = dist
          bestAlly = other
  bestAlly

proc findNearestCombatAlly(env: Environment, agent: Thing): Thing =
  ## Find the nearest friendly combat unit to retreat toward using spatial index.
  ## Cached per-step per-agent to avoid redundant scans in canStart/shouldTerminate/act.
  combatAllyCache.getWithAgent(env, agent, findNearestCombatAllyUncached)

proc countNearbyAllies(env: Environment, agent: Thing, radius: int): int =
  ## Count allied agents within radius Chebyshev distance.
  ## Excludes the agent itself from the count.
  countAlliesInRangeSpatial(env, agent.pos, getTeamId(agent), radius, agent.agentId)

proc countNearbyEnemies(env: Environment, agent: Thing, radius: int): int =
  ## Count enemy agents within radius Chebyshev distance.
  countEnemiesInRangeSpatial(env, agent.pos, getTeamId(agent), radius)

proc hasAllyNearbyUncached(env: Environment, agent: Thing): bool =
  ## Check if any ally (other than self) is within 4 tiles.
  ## Uncached version - use hasAllyNearby for cached lookups.
  hasAllyInRangeSpatial(env, agent.pos, getTeamId(agent), 4, agent.agentId)

proc hasAllyNearby(env: Environment, agent: Thing): bool =
  ## Check if any ally (other than self) is within 4 tiles.
  ## Cached per-step per-agent to avoid redundant scans in canStart/shouldTerminate.
  allyNearbyCache.getWithAgent(env, agent, hasAllyNearbyUncached)

proc shouldWaitForAllies*(env: Environment, agent: Thing): bool =
  ## Return true if agent should wait for nearby allies before engaging.
  ## Delays engagement when outnumbered but more allies are approaching.
  ## This enables coordinated attacks rather than piecemeal engagements.
  let nearbyAllies = countNearbyAllies(env, agent, radius=5)
  let nearbyEnemies = countNearbyEnemies(env, agent, radius=7)
  # Wait if outnumbered and allies are coming
  result = nearbyEnemies > nearbyAllies + 1 and
           countNearbyAllies(env, agent, radius=10) > nearbyAllies

proc canStartFighterSeekHealer(controller: Controller, env: Environment, agent: Thing,
                               agentId: int, state: var AgentState): bool =
  ## Seek healer when low HP and no bread available.
  ## This is more targeted than generic retreat - actively seeks monk healing.
  if agent.hp * 3 > agent.maxHp:  # Only when HP <= 33%
    return false
  if agent.inventoryBread > 0:  # Can self-heal with bread instead
    return false
  not isNil(findNearestFriendlyMonk(env, agent))

proc shouldTerminateFighterSeekHealer(controller: Controller, env: Environment, agent: Thing,
                                      agentId: int, state: var AgentState): bool =
  ## Stop seeking healer when HP recovered or no monk available
  if agent.hp * 3 > agent.maxHp:  # HP recovered above threshold
    return true
  if agent.inventoryBread > 0:  # Got bread, can self-heal
    return true
  isNil(findNearestFriendlyMonk(env, agent))  # No monk to seek

proc optFighterSeekHealer(controller: Controller, env: Environment, agent: Thing,
                          agentId: int, state: var AgentState): uint16 =
  ## Move toward the nearest friendly monk to benefit from their healing aura.
  let monk = findNearestFriendlyMonk(env, agent)
  if isNil(monk):
    return 0'u16
  let dist = int(chebyshevDist(agent.pos, monk.pos))
  # Already within monk's healing aura - stay put and wait for healing
  if dist <= MonkHealRadius:
    return 0'u16
  # Move toward the monk
  controller.moveTo(env, agent, agentId, state, monk.pos)

proc canStartFighterRetreat(controller: Controller, env: Environment, agent: Thing,
                            agentId: int, state: var AgentState): bool =
  agent.hp * 3 <= agent.maxHp

proc shouldTerminateFighterRetreat(controller: Controller, env: Environment, agent: Thing,
                                   agentId: int, state: var AgentState): bool =
  agent.hp * 3 > agent.maxHp

proc optFighterRetreat(controller: Controller, env: Environment, agent: Thing,
                       agentId: int, state: var AgentState): uint16 =
  ## Retreat when HP is low. Prioritizes retreating toward allied combat units
  ## for mutual defense, falling back to defensive buildings if no allies nearby.
  if agent.hp * 3 > agent.maxHp:
    return 0'u16
  let teamId = getTeamId(agent)
  let basePos = agent.getBasePos()
  state.basePosition = basePos

  # First priority: retreat toward nearby allied combat units
  # Grouping up with allies provides mutual defense and increases survivability
  let ally = findNearestCombatAlly(env, agent)
  if not isNil(ally):
    return controller.moveTo(env, agent, agentId, state, ally.pos)

  # Fallback: retreat to defensive buildings
  var safePos = basePos
  for kind in [Outpost, Barracks, TownCenter, Monastery]:
    let safe = env.findNearestFriendlyThingSpiral(state, teamId, kind)
    if not isNil(safe):
      safePos = safe.pos
      break
  controller.moveTo(env, agent, agentId, state, safePos)

proc canStartFighterDividerDefense(controller: Controller, env: Environment, agent: Thing,
                                   agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitVillager:
    return false
  let enemy = fighterFindNearbyEnemy(controller, env, agent, state)
  not isNil(enemy)

proc shouldTerminateFighterDividerDefense(controller: Controller, env: Environment, agent: Thing,
                                          agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitVillager:
    return true
  let enemy = fighterFindNearbyEnemy(controller, env, agent, state)
  isNil(enemy)

proc optFighterDividerDefense(controller: Controller, env: Environment, agent: Thing,
                              agentId: int, state: var AgentState): uint16 =
  let enemy = fighterFindNearbyEnemy(controller, env, agent, state)
  if isNil(enemy):
    return 0'u16
  let teamId = getTeamId(agent)
  let basePos = agent.getBasePos()
  state.basePosition = basePos
  # Request defense from builders via coordination system
  requestDefenseFromBuilder(env, agent, enemy.pos)

  var enemyBase: Thing = nil
  var bestAltarDist = int.high
  for altar in env.thingsByKind[Altar]:
    if altar.teamId == teamId:
      continue
    let dist = abs(altar.pos.x - basePos.x) + abs(altar.pos.y - basePos.y)
    if dist < bestAltarDist:
      bestAltarDist = dist
      enemyBase = altar
  let enemyPos = if not isNil(enemyBase): enemyBase.pos else: enemy.pos

  let dx = float32(enemyPos.x - basePos.x)
  let dy = float32(enemyPos.y - basePos.y)
  var lineDir = ivec2(1, 0)
  var bestScore = abs(dx * float32(lineDir.x) + dy * float32(lineDir.y))
  let candidates = [
    (ivec2(1, 0), 1.0'f32),
    (ivec2(0, 1), 1.0'f32),
    (ivec2(1, 1), DividerInvSqrt2),
    (ivec2(1, -1), DividerInvSqrt2)
  ]
  for entry in candidates:
    let dot = abs(dx * float32(entry[0].x) + dy * float32(entry[0].y))
    let score = dot * entry[1]
    if score < bestScore:
      bestScore = score
      lineDir = entry[0]

  let midPos = ivec2(
    (basePos.x + enemyPos.x) div 2,
    (basePos.y + enemyPos.y) div 2
  )

  var n1 = ivec2(0, 0)
  var n2 = ivec2(0, 0)
  if lineDir.x != 0 and lineDir.y == 0:
    n1 = ivec2(0, 1)
    n2 = ivec2(0, -1)
  elif lineDir.x == 0 and lineDir.y != 0:
    n1 = ivec2(1, 0)
    n2 = ivec2(-1, 0)
  elif lineDir.x == 1 and lineDir.y == 1:
    n1 = ivec2(1, -1)
    n2 = ivec2(-1, 1)
  else:
    n1 = ivec2(1, 1)
    n2 = ivec2(-1, -1)
  let toBase = basePos - midPos
  let normal = if toBase.x * n1.x + toBase.y * n1.y >= 0: n1 else: n2

  let dist = max(abs(enemyPos.x - basePos.x), abs(enemyPos.y - basePos.y))
  let halfLen = max(DividerHalfLengthMin, min(DividerHalfLengthMax, dist div 2))

  var bestDoor = ivec2(-1, -1)
  var bestDoorDist = int.high
  var bestOutpost = ivec2(-1, -1)
  var bestOutpostDist = int.high
  var bestWall = ivec2(-1, -1)
  var bestWallDist = int.high

  for offset in -halfLen .. halfLen:
    let pos = midPos + ivec2(lineDir.x * offset, lineDir.y * offset)
    if not isValidPos(pos):
      continue
    let posTerrain = env.terrain[pos.x][pos.y]
    if posTerrain == TerrainRoad or isRampTerrain(posTerrain):
      continue
    let distToAgent = int(chebyshevDist(agent.pos, pos))
    let raw = (offset + DividerDoorOffset) mod DividerDoorSpacing
    let normalized = if raw < 0: raw + DividerDoorSpacing else: raw
    let isDoorSlot = normalized == 0
    if isDoorSlot:
      if env.hasDoor(pos):
        let outpostPos = pos + normal
        let outpostTerrain = env.terrain[outpostPos.x][outpostPos.y]
        if isValidPos(outpostPos) and outpostTerrain != TerrainRoad and
            not isRampTerrain(outpostTerrain) and env.canPlace(outpostPos):
          let outDist = int(chebyshevDist(agent.pos, outpostPos))
          if outDist < bestOutpostDist:
            bestOutpostDist = outDist
            bestOutpost = outpostPos
      else:
        if env.canPlace(pos):
          if distToAgent < bestDoorDist:
            bestDoorDist = distToAgent
            bestDoor = pos
    else:
      if env.canPlace(pos):
        if distToAgent < bestWallDist:
          bestWallDist = distToAgent
          bestWall = pos

  var targetKind = Wall
  if bestDoor.x >= 0:
    targetKind = Door
  elif bestOutpost.x >= 0:
    targetKind = Outpost
  let targetPos = (if targetKind == Door: bestDoor elif targetKind == Outpost: bestOutpost else: bestWall)
  if targetPos.x >= 0:
    case targetKind
    of Door:
      if not env.canAffordBuild(agent, thingItem("Door")):
        let (didDrop, actDrop) = controller.dropoffCarrying(
          env, agent, agentId, state,
          allowWood = true,
          allowStone = true,
          allowGold = true
        )
        if didDrop: return actDrop
        let (didWood, actWood) = controller.ensureWood(env, agent, agentId, state)
        if didWood: return actWood
      let (didDoor, doorAct) = goToAdjacentAndBuild(
        controller, env, agent, agentId, state, targetPos, BuildIndexDoor
      )
      if didDoor: return doorAct
    of Outpost:
      if not env.canAffordBuild(agent, thingItem("Outpost")):
        let (didDrop, actDrop) = controller.dropoffCarrying(
          env, agent, agentId, state,
          allowWood = true,
          allowStone = true,
          allowGold = true
        )
        if didDrop: return actDrop
        let (didWood, actWood) = controller.ensureWood(env, agent, agentId, state)
        if didWood: return actWood
      let idx = buildIndexFor(Outpost)
      if idx >= 0:
        let (didOutpost, outpostAct) = goToAdjacentAndBuild(
          controller, env, agent, agentId, state, targetPos, idx
        )
        if didOutpost: return outpostAct
    else:
      if not env.canAffordBuild(agent, thingItem("Wall")):
        let (didDrop, actDrop) = controller.dropoffCarrying(
          env, agent, agentId, state,
          allowWood = true,
          allowStone = true,
          allowGold = true
        )
        if didDrop: return actDrop
        let (didWood, actWood) = controller.ensureWood(env, agent, agentId, state)
        if didWood: return actWood
      let (didWall, wallAct) = goToAdjacentAndBuild(
        controller, env, agent, agentId, state, targetPos, BuildIndexWall
      )
      if didWall: return wallAct
    return controller.moveTo(env, agent, agentId, state, enemy.pos)
  0'u16

proc canStartFighterLanterns(controller: Controller, env: Environment, agent: Thing,
                             agentId: int, state: var AgentState): bool =
  ## Only start lantern work if agent has lanterns or is a villager (can craft them)
  agent.inventoryLantern > 0 or agent.unitClass == UnitVillager

proc shouldTerminateFighterLanterns(controller: Controller, env: Environment, agent: Thing,
                                    agentId: int, state: var AgentState): bool =
  ## Terminate when agent has no lanterns and isn't a villager (can't craft more)
  agent.inventoryLantern == 0 and agent.unitClass != UnitVillager

# Building kinds that need lanterns - shared between cache refresh and optFighterLanterns
const LanternBuildingKinds* = [Outpost, GuardTower, TownCenter, House, Barracks, ArcheryRange,
  Stable, SiegeWorkshop, MangonelWorkshop, Blacksmith, Market, Dock, Monastery,
  University, Castle, Granary, LumberCamp, Quarry, MiningCamp, Mill, WeavingLoom,
  ClayOven, Altar, Wall]

proc refreshUnlitBuildingCache(controller: Controller, env: Environment, teamId: int) =
  ## Refresh the per-team unlit building cache if stale.
  ## Called once per step per team, caches all unlit building positions.
  ## Requires env.tempLanternSpacing to be populated with team lanterns first.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return
  if controller.unlitBuildingCacheStep[teamId] == env.currentStep:
    return  # Cache is fresh for this team this step
  controller.unlitBuildingCacheStep[teamId] = env.currentStep
  controller.unlitBuildingCounts[teamId] = 0

  # Helper to check if position has a lantern nearby (uses pre-populated tempLanternSpacing)
  proc hasLanternNear(env: Environment, pos: IVec2): bool =
    for lantern in env.tempLanternSpacing:
      if chebyshevDist(lantern.pos, pos) <= 3:
        return true
    return false

  # Collect all unlit buildings for this team
  for kind in LanternBuildingKinds:
    for thing in env.thingsByKind[kind]:
      if thing.isNil or thing.teamId != teamId:
        continue
      if hasLanternNear(env, thing.pos):
        continue
      # Found an unlit building - add to cache
      if controller.unlitBuildingCounts[teamId] < MaxUnlitBuildingsPerTeam:
        controller.unlitBuildingPositions[teamId][controller.unlitBuildingCounts[teamId]] = thing.pos
        controller.unlitBuildingCounts[teamId] += 1

proc optFighterLanterns(controller: Controller, env: Environment, agent: Thing,
                        agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  let basePos = agent.getBasePos()
  state.basePosition = basePos
  var target = ivec2(-1, -1)
  var unlit: Thing = nil
  var bestUnlitDist = int.high
  # Pre-collect healthy team lantern positions once to avoid repeated spatial queries.
  # Each hasTeamLanternNear call allocates a seq and runs a spatial query; this replaces
  # O(buildings + candidates) spatial queries with one O(lanterns) scan + simple distance checks.
  var teamLanternCount = 0
  var teamLanternFarthest = 0
  env.tempLanternSpacing.setLen(0)
  for thing in env.thingsByKind[Lantern]:
    if thing.isNil or not thing.lanternHealthy or thing.teamId != teamId:
      continue
    env.tempLanternSpacing.add(thing)
    inc teamLanternCount
    let dist = int(chebyshevDist(basePos, thing.pos))
    if dist > teamLanternFarthest:
      teamLanternFarthest = dist

  # Inline lantern proximity check using pre-collected positions
  template hasLanternNearCached(checkPos: IVec2): bool =
    var found = false
    for lantern in env.tempLanternSpacing:
      if chebyshevDist(lantern.pos, checkPos) <= 3:
        found = true
        break
    found

  # Use per-team-per-step cache of unlit building positions instead of iterating all buildings.
  # This reduces O(buildings * lanterns) per agent to O(cached_unlit * lanterns) for verification.
  refreshUnlitBuildingCache(controller, env, teamId)

  # Find closest unlit building from cache (verify still unlit - lantern may have been placed this step)
  if teamId >= 0 and teamId < MapRoomObjectsTeams:
    for i in 0 ..< controller.unlitBuildingCounts[teamId]:
      let pos = controller.unlitBuildingPositions[teamId][i]
      # Verify position still has a building (may have been destroyed)
      let thing = env.getThing(pos)
      if thing.isNil or thing.teamId != teamId:
        continue
      # Verify still unlit (lantern may have been placed since cache was built)
      if hasLanternNearCached(pos):
        continue
      let dist = abs(pos.x - agent.pos.x).int + abs(pos.y - agent.pos.y).int
      if dist < bestUnlitDist:
        bestUnlitDist = dist
        unlit = thing

  if not isNil(unlit):
    var bestPos = ivec2(-1, -1)
    var bestDist = int.high
    for dx in -2 .. 2:
      for dy in -2 .. 2:
        if abs(dx) + abs(dy) > 2:
          continue
        let cand = unlit.pos + ivec2(dx.int32, dy.int32)
        if not isLanternPlacementValid(env, cand):
          continue
        if hasLanternNearCached(cand):
          continue
        let dist = abs(cand.x - agent.pos.x).int + abs(cand.y - agent.pos.y).int
        if dist < bestDist:
          bestDist = dist
          bestPos = cand
    if bestPos.x >= 0:
      target = bestPos
  if target.x < 0:
    let desiredRadius = max(ObservationRadius + 1, max(3, teamLanternFarthest + 2 + teamLanternCount div 6))
    for _ in 0 ..< 18:
      let candidate = getNextSpiralPoint(state)
      if chebyshevDist(candidate, basePos) < desiredRadius:
        continue
      if not isLanternPlacementValid(env, candidate):
        continue
      if hasLanternNearCached(candidate):
        continue
      target = candidate
      break

  if target.x >= 0:
    if agent.inventoryLantern > 0:
      return actOrMove(controller, env, agent, agentId, state, target, 6'u16)

    if controller.getBuildingCount(env, teamId, WeavingLoom) == 0 and agent.unitClass == UnitVillager:
      if chebyshevDist(agent.pos, basePos) > 2'i32:
        let avoidDir = (if state.blockedMoveSteps > 0: state.blockedMoveDir else: -1)
        let dir = getMoveTowards(env, agent, agent.pos, basePos, controller.rng, avoidDir)
        if dir >= 0:
          return saveStateAndReturn(controller, agentId, state, encodeAction(1'u16, dir.uint8))
        # Fall through to try building if can't move
      let (didBuild, buildAct) = controller.tryBuildIfMissing(env, agent, agentId, state, teamId, WeavingLoom)
      if didBuild: return buildAct

    let hasLanternInput = agent.inventoryWheat > 0 or agent.inventoryWood > 0
    let loom = env.findNearestFriendlyThingSpiral(state, teamId, WeavingLoom)
    if hasLanternInput:
      if not isNil(loom):
        return actOrMove(controller, env, agent, agentId, state, loom.pos, 3'u16)
      return controller.moveNextSearch(env, agent, agentId, state)

    let food = env.stockpileCount(teamId, ResourceFood)
    let wood = env.stockpileCount(teamId, ResourceWood)
    if wood <= food:
      let (didWood, actWood) = controller.ensureWood(env, agent, agentId, state)
      if didWood: return actWood
      return controller.moveNextSearch(env, agent, agentId, state)

    for kind in [Wheat, Stubble]:
      let wheat = env.findNearestThingSpiral(state, kind)
      if isNil(wheat):
        continue
      return actOrMove(controller, env, agent, agentId, state, wheat.pos, 3'u16)
    let (didWood, actWood) = controller.ensureWood(env, agent, agentId, state)
    if didWood: return actWood
    return controller.moveNextSearch(env, agent, agentId, state)

  0'u16

proc canStartFighterDropoffFood(controller: Controller, env: Environment, agent: Thing,
                                agentId: int, state: var AgentState): bool =
  for key, count in agent.inventory.pairs:
    if count > 0 and isFoodItem(key):
      return true
  false

proc shouldTerminateFighterDropoffFood(controller: Controller, env: Environment, agent: Thing,
                                       agentId: int, state: var AgentState): bool =
  # Terminate when no longer carrying food
  for key, count in agent.inventory.pairs:
    if count > 0 and isFoodItem(key):
      return false
  true

proc optFighterDropoffFood(controller: Controller, env: Environment, agent: Thing,
                           agentId: int, state: var AgentState): uint16 =
  let (didFoodDrop, foodDropAct) =
    controller.dropoffCarrying(env, agent, agentId, state, allowFood = true)
  if didFoodDrop: return foodDropAct
  0'u16

proc fighterShouldSkipKind(kind: ThingKind, siegeAtCap, navalAtCap: bool): bool {.inline.} =
  (siegeAtCap and kind in FighterSiegeKinds) or
  (navalAtCap and kind in FighterNavalKinds)

proc fighterHasReadyTrainQueue(env: Environment, teamId: int, siegeAtCap, navalAtCap: bool): bool =
  ## Check if any friendly training building has a ready queue entry.
  ## A villager can convert immediately at such a building (pre-paid).
  for kind in FighterTrainKinds:
    if fighterShouldSkipKind(kind, siegeAtCap, navalAtCap):
      continue
    for building in env.thingsByKind[kind]:
      if building.teamId == teamId and building.productionQueueHasReady():
        return true
  false

proc canStartFighterTrain(controller: Controller, env: Environment, agent: Thing,
                          agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitVillager:
    return false
  let teamId = getTeamId(agent)
  let siegeAtCap = teamSiegeAtCap(env, teamId)
  let navalAtCap = teamNavalAtCap(env, teamId)
  # Check for ready queue entries first (free conversion, already paid)
  if fighterHasReadyTrainQueue(env, teamId, siegeAtCap, navalAtCap):
    return true
  for kind in FighterTrainKinds:
    if fighterShouldSkipKind(kind, siegeAtCap, navalAtCap):
      continue
    if controller.getBuildingCount(env, teamId, kind) == 0:
      continue
    if not env.canSpendStockpile(teamId, buildingTrainCosts(kind)):
      continue
    return true
  false

proc shouldTerminateFighterTrain(controller: Controller, env: Environment, agent: Thing,
                                 agentId: int, state: var AgentState): bool =
  ## Terminate when no longer a villager (was trained) or can't afford/collect any training
  if agent.unitClass != UnitVillager:
    return true
  let teamId = getTeamId(agent)
  let siegeAtCap = teamSiegeAtCap(env, teamId)
  let navalAtCap = teamNavalAtCap(env, teamId)
  # Don't terminate if there's a ready queue entry to collect
  if fighterHasReadyTrainQueue(env, teamId, siegeAtCap, navalAtCap):
    return false
  for kind in FighterTrainKinds:
    if fighterShouldSkipKind(kind, siegeAtCap, navalAtCap):
      continue
    if controller.getBuildingCount(env, teamId, kind) == 0:
      continue
    if env.canSpendStockpile(teamId, buildingTrainCosts(kind)):
      return false  # Can still train, don't terminate
  true  # No training options available

proc optFighterTrain(controller: Controller, env: Environment, agent: Thing,
                     agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  let siegeAtCap = teamSiegeAtCap(env, teamId)
  let navalAtCap = teamNavalAtCap(env, teamId)
  # First: go to any building with a ready queue entry (free conversion)
  for kind in FighterTrainKinds:
    if fighterShouldSkipKind(kind, siegeAtCap, navalAtCap):
      continue
    for building in env.thingsByKind[kind]:
      if building.teamId == teamId and building.productionQueueHasReady():
        return actOrMove(controller, env, agent, agentId, state, building.pos, 3'u16)
  # Second: queue new training and go to building (rotate starting type for diversity)
  let startIdx = agentId mod FighterTrainKinds.len
  for offset in 0 ..< FighterTrainKinds.len:
    let kind = FighterTrainKinds[(startIdx + offset) mod FighterTrainKinds.len]
    if fighterShouldSkipKind(kind, siegeAtCap, navalAtCap):
      continue
    if controller.getBuildingCount(env, teamId, kind) == 0:
      continue
    if not env.canSpendStockpile(teamId, buildingTrainCosts(kind)):
      continue
    let building = env.findNearestFriendlyThingSpiral(state, teamId, kind)
    if isNil(building):
      continue
    # Batch-queue additional units if resources allow and queue has room
    if building.productionQueue.entries.len < ProductionQueueMaxSize:
      discard env.tryBatchQueueTrain(building, teamId, BatchTrainSmall)
    return actOrMove(controller, env, agent, agentId, state, building.pos, 3'u16)
  0'u16

proc canStartFighterBecomeSiege(controller: Controller, env: Environment, agent: Thing,
                                agentId: int, state: var AgentState): bool =
  ## True siege conversion: combat units (infantry/cavalry lines) can convert to siege
  ## when they see enemy structures and a SiegeWorkshop is available.
  if agent.unitClass notin {UnitManAtArms, UnitLongSwordsman, UnitChampion,
                            UnitKnight, UnitCavalier, UnitPaladin}:
    return false
  if not fighterSeesEnemyStructure(env, agent):
    return false
  let teamId = getTeamId(agent)
  if controller.getBuildingCount(env, teamId, SiegeWorkshop) == 0:
    return false
  if not env.canSpendStockpile(teamId, buildingTrainCosts(SiegeWorkshop)):
    return false
  true

proc shouldTerminateFighterBecomeSiege(controller: Controller, env: Environment, agent: Thing,
                                       agentId: int, state: var AgentState): bool =
  ## Terminate when unit class changes (became siege) or conditions no longer met
  if agent.unitClass notin {UnitManAtArms, UnitLongSwordsman, UnitChampion,
                            UnitKnight, UnitCavalier, UnitPaladin}:
    return true
  if not fighterSeesEnemyStructure(env, agent):
    return true
  let teamId = getTeamId(agent)
  if controller.getBuildingCount(env, teamId, SiegeWorkshop) == 0:
    return true
  if not env.canSpendStockpile(teamId, buildingTrainCosts(SiegeWorkshop)):
    return true
  false

proc optFighterBecomeSiege(controller: Controller, env: Environment, agent: Thing,
                           agentId: int, state: var AgentState): uint16 =
  ## Move to SiegeWorkshop and interact to convert to battering ram
  let teamId = getTeamId(agent)
  let building = env.findNearestFriendlyThingSpiral(state, teamId, SiegeWorkshop)
  if isNil(building) or building.cooldown != 0:
    return 0'u16
  actOrMove(controller, env, agent, agentId, state, building.pos, 3'u16)

proc canStartFighterMaintainGear(controller: Controller, env: Environment, agent: Thing,
                                 agentId: int, state: var AgentState): bool =
  if agent.inventoryArmor < ArmorPoints:
    return true
  agent.unitClass in {UnitManAtArms, UnitLongSwordsman, UnitChampion} and agent.inventorySpear == 0

proc shouldTerminateFighterMaintainGear(controller: Controller, env: Environment, agent: Thing,
                                        agentId: int, state: var AgentState): bool =
  # Terminate when fully geared (armor at max, and spear if infantry line)
  if agent.inventoryArmor < ArmorPoints:
    return false
  if agent.unitClass in {UnitManAtArms, UnitLongSwordsman, UnitChampion} and agent.inventorySpear == 0:
    return false
  true

proc optFighterMaintainGear(controller: Controller, env: Environment, agent: Thing,
                            agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  if agent.inventoryArmor < ArmorPoints:
    let (didSmith, actSmith) = controller.moveToNearestSmith(env, agent, agentId, state, teamId)
    if didSmith: return actSmith
    return 0'u16

  if agent.unitClass in {UnitManAtArms, UnitLongSwordsman, UnitChampion} and agent.inventorySpear == 0:
    if agent.inventoryWood == 0:
      let (didWood, actWood) = controller.ensureWood(env, agent, agentId, state)
      if didWood: return actWood
    let (didSmith, actSmith) = controller.moveToNearestSmith(env, agent, agentId, state, teamId)
    if didSmith: return actSmith
  0'u16

const
  # Unit classes excluded from melee enemy search (ranged + special)
  NonMeleeClasses = RangedUnitClasses + {UnitMonk, UnitBoat, UnitTradeCog}

proc findNearestMeleeEnemyUncached(env: Environment, agent: Thing): Thing =
  ## Internal: actual search logic for nearest melee enemy.
  ## Uses consolidated findNearestEnemyOfClass from ai_utils.
  let r = KiteTriggerDistance + 2
  findNearestEnemyOfClass(env, agent, r, NonMeleeClasses, FilterExclude)

proc findNearestMeleeEnemy(env: Environment, agent: Thing): Thing =
  ## Find the nearest enemy agent that is a melee unit (not archer, mangonel, or monk)
  ## Cached per-step per-agent to avoid redundant scans in canStart/shouldTerminate/act.
  meleeEnemyCache.getWithAgent(env, agent, findNearestMeleeEnemyUncached)

proc isSiegeThreateningStructure(env: Environment, siege: Thing, teamId: int): bool =
  ## Check if enemy siege unit is close to any friendly structures
  ## Optimized: uses thingsByKind to only check attackable structure types
  ## instead of iterating all env.things (O(k) where k = attackable structures, not O(n)).
  let radius = SiegeNearStructureRadius
  # Only check building kinds that are in AttackableStructures
  for kind in [Wall, Door, Outpost, GuardTower, Castle, TownCenter, Monastery, Wonder]:
    for thing in env.thingsByKind[kind]:
      if thing.isNil or thing.teamId != teamId:
        continue
      if int(chebyshevDist(siege.pos, thing.pos)) <= radius:
        return true
  false

proc findNearestSiegeEnemyUncached(env: Environment, agent: Thing, prioritizeThreatening: bool = true): Thing =
  ## Internal: actual search logic for nearest siege enemy.
  ## Optimized: uses spatial index cells instead of grid scan.
  let teamId = getTeamId(agent)
  let teamMask = getTeamMask(teamId)  # Pre-compute for bitwise checks
  var bestEnemy: Thing = nil
  var bestDist = int.high
  var bestThreatening = false

  let r = AntiSiegeDetectionRadius
  # Use spatial index cells instead of grid scan
  let (cx, cy) = cellCoords(agent.pos)
  let clampedMax = min(r, max(SpatialCellsX, SpatialCellsY) * SpatialCellSize)
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
        # Bitwise team check: (otherMask and teamMask) != 0 means same team (skip)
        if (getTeamMask(other) and teamMask) != 0:
          continue
        if other.unitClass notin {UnitBatteringRam, UnitMangonel, UnitTrebuchet}:
          continue
        let dist = int(chebyshevDist(agent.pos, other.pos))
        if dist > r:
          continue

        let threatening = prioritizeThreatening and isSiegeThreateningStructure(env, other, teamId)

        if threatening and not bestThreatening:
          bestThreatening = true
          bestDist = dist
          bestEnemy = other
        elif threatening == bestThreatening and dist < bestDist:
          bestDist = dist
          bestEnemy = other
  bestEnemy

proc findNearestSiegeEnemyPrioritized(env: Environment, agent: Thing): Thing =
  ## Wrapper that calls uncached with prioritizeThreatening=true for caching.
  findNearestSiegeEnemyUncached(env, agent, true)

proc findNearestSiegeEnemy(env: Environment, agent: Thing, prioritizeThreatening: bool = true): Thing =
  ## Find the nearest enemy siege unit (BatteringRam or Mangonel)
  ## Cached per-step per-agent to avoid redundant scans in canStart/shouldTerminate/act.
  ## Note: cache only applies when prioritizeThreatening=true (default).
  if not prioritizeThreatening:
    return findNearestSiegeEnemyUncached(env, agent, prioritizeThreatening)
  siegeEnemyCache.getWithAgent(env, agent, findNearestSiegeEnemyPrioritized)

proc canStartFighterAntiSiege(controller: Controller, env: Environment, agent: Thing,
                              agentId: int, state: var AgentState): bool =
  ## Anti-siege triggers when there's an enemy siege unit nearby
  ## Requires stance that allows chasing
  if not stanceAllowsChase(env, agent):
    return false
  not isNil(findNearestSiegeEnemy(env, agent))

proc shouldTerminateFighterAntiSiege(controller: Controller, env: Environment, agent: Thing,
                                     agentId: int, state: var AgentState): bool =
  ## Terminate when no more siege units nearby
  isNil(findNearestSiegeEnemy(env, agent))

proc optFighterAntiSiege(controller: Controller, env: Environment, agent: Thing,
                         agentId: int, state: var AgentState): uint16 =
  ## Move toward and attack enemy siege units
  let siege = findNearestSiegeEnemy(env, agent)
  if isNil(siege):
    return 0'u16
  actOrMove(controller, env, agent, agentId, state, siege.pos, 2'u16)

const
  # Ranged units that should kite (move away from melee threats while attacking)
  # Excludes siege units (Scorpion, Mangonel, Trebuchet) which should stand and fire
  KitingRangedUnits = {
    UnitArcher, UnitCrossbowman, UnitArbalester,
    UnitSkirmisher, UnitEliteSkirmisher,
    UnitCavalryArcher, UnitHeavyCavalryArcher,
    UnitHandCannoneer
  }

proc canStartFighterKite(controller: Controller, env: Environment, agent: Thing,
                         agentId: int, state: var AgentState): bool =
  ## Kiting triggers for ranged units when a melee enemy is within trigger distance
  ## StandGround stance disables kiting (no movement allowed)
  ## Excludes siege units (Scorpion) which should stand and fire rather than kite
  if agent.unitClass notin KitingRangedUnits:
    return false
  if not stanceAllowsMovementToAttack(env, agent):
    return false
  let meleeEnemy = findNearestMeleeEnemy(env, agent)
  if isNil(meleeEnemy):
    return false
  let dist = int(chebyshevDist(agent.pos, meleeEnemy.pos))
  dist <= KiteTriggerDistance

proc shouldTerminateFighterKite(controller: Controller, env: Environment, agent: Thing,
                                agentId: int, state: var AgentState): bool =
  ## Terminate when no melee enemy within trigger distance
  if agent.unitClass notin KitingRangedUnits:
    return true
  let meleeEnemy = findNearestMeleeEnemy(env, agent)
  if isNil(meleeEnemy):
    return true
  let dist = int(chebyshevDist(agent.pos, meleeEnemy.pos))
  dist > KiteTriggerDistance

proc optFighterKite(controller: Controller, env: Environment, agent: Thing,
                    agentId: int, state: var AgentState): uint16 =
  ## Move away from the nearest melee enemy while staying within attack range
  let meleeEnemy = findNearestMeleeEnemy(env, agent)
  if isNil(meleeEnemy):
    return 0'u16

  let dist = int(chebyshevDist(agent.pos, meleeEnemy.pos))
  # If already at safe distance, no need to kite
  if dist > KiteTriggerDistance:
    return 0'u16

  # Calculate direction away from enemy
  let dx = agent.pos.x - meleeEnemy.pos.x
  let dy = agent.pos.y - meleeEnemy.pos.y
  let awayDir = ivec2(signi(dx), signi(dy))

  # Try to move in the direction away from enemy
  # Check multiple directions, preferring directly away, then diagonals
  var candidates: array[3, IVec2]
  var numCandidates = 0
  # Primary direction: directly away
  if awayDir.x != 0 or awayDir.y != 0:
    candidates[numCandidates] = awayDir; inc numCandidates
  # Secondary: perpendicular directions (allows strafing)
  if awayDir.x != 0 and awayDir.y != 0:
    # Diagonal away - try the two perpendicular diagonals
    candidates[numCandidates] = ivec2(awayDir.x, 0); inc numCandidates
    candidates[numCandidates] = ivec2(0, awayDir.y); inc numCandidates
  elif awayDir.x != 0:
    # Moving horizontally - can strafe vertically
    candidates[numCandidates] = ivec2(awayDir.x, 1); inc numCandidates
    candidates[numCandidates] = ivec2(awayDir.x, -1); inc numCandidates
  elif awayDir.y != 0:
    # Moving vertically - can strafe horizontally
    candidates[numCandidates] = ivec2(1, awayDir.y); inc numCandidates
    candidates[numCandidates] = ivec2(-1, awayDir.y); inc numCandidates

  # Try each candidate direction
  for i in 0 ..< numCandidates:
    let dir = candidates[i]
    let targetPos = agent.pos + dir
    if not isValidPos(targetPos):
      continue
    if not canEnterForMove(env, agent, agent.pos, targetPos):
      continue
    # Check that we maintain attack range (stay within ArcherBaseRange of any enemy)
    # For now, just move away - the attack opportunity check will handle attacking
    let dirIdx = vecToOrientation(dir)
    return saveStateAndReturn(controller, agentId, state, encodeAction(1'u16, dirIdx.uint8))

  # If can't move directly away, try any direction that increases distance
  for dirIdx in 0 .. 7:
    let dir = Directions8[dirIdx]
    let targetPos = agent.pos + dir
    if not isValidPos(targetPos):
      continue
    if not canEnterForMove(env, agent, agent.pos, targetPos):
      continue
    let newDist = int(chebyshevDist(targetPos, meleeEnemy.pos))
    if newDist > dist:
      return saveStateAndReturn(controller, agentId, state, encodeAction(1'u16, dirIdx.uint8))

  # Can't kite, return 0 to let other options handle it
  0'u16

proc canStartFighterHuntPredators(controller: Controller, env: Environment, agent: Thing,
                                  agentId: int, state: var AgentState): bool =
  ## Hunting predators requires chasing them - check stance
  if not stanceAllowsChase(env, agent):
    return false
  agent.hp * 2 >= agent.maxHp and not isNil(findNearestPredator(env, agent.pos))

proc shouldTerminateFighterHuntPredators(controller: Controller, env: Environment, agent: Thing,
                                         agentId: int, state: var AgentState): bool =
  # Terminate when HP drops below threshold or no predator nearby
  agent.hp * 2 < agent.maxHp or isNil(findNearestPredator(env, agent.pos))

proc optFighterHuntPredators(controller: Controller, env: Environment, agent: Thing,
                             agentId: int, state: var AgentState): uint16 =
  let target = findNearestPredator(env, agent.pos)
  if isNil(target):
    return 0'u16
  actOrMove(controller, env, agent, agentId, state, target.pos, 2'u16)

proc canStartFighterClearGoblins(controller: Controller, env: Environment, agent: Thing,
                                 agentId: int, state: var AgentState): bool =
  ## Clearing goblin structures requires chasing - check stance
  if not stanceAllowsChase(env, agent):
    return false
  agent.hp * 2 >= agent.maxHp and not isNil(findNearestGoblinStructure(env, agent.pos))

proc shouldTerminateFighterClearGoblins(controller: Controller, env: Environment, agent: Thing,
                                        agentId: int, state: var AgentState): bool =
  # Terminate when HP drops below threshold or no goblin structure nearby
  agent.hp * 2 < agent.maxHp or isNil(findNearestGoblinStructure(env, agent.pos))

proc optFighterClearGoblins(controller: Controller, env: Environment, agent: Thing,
                            agentId: int, state: var AgentState): uint16 =
  let target = findNearestGoblinStructure(env, agent.pos)
  if isNil(target):
    return 0'u16
  actOrMove(controller, env, agent, agentId, state, target.pos, 2'u16)

# Escort behavior: respond to protection requests from coordination system
proc canStartFighterEscort(controller: Controller, env: Environment, agent: Thing,
                           agentId: int, state: var AgentState): bool =
  ## Check if there's a nearby protection request to respond to
  if not stanceAllowsChase(env, agent):
    return false
  # Only combat units can escort
  if agent.unitClass notin {UnitManAtArms, UnitLongSwordsman, UnitChampion,
                            UnitKnight, UnitCavalier, UnitPaladin, UnitScout, UnitArcher,
                            UnitCrossbowman, UnitArbalester, UnitLightCavalry, UnitHussar,
                            UnitCamel, UnitHeavyCamel, UnitImperialCamel}:
    return false
  let (should, _) = fighterShouldEscort(env, agent)
  should

proc shouldTerminateFighterEscort(controller: Controller, env: Environment, agent: Thing,
                                  agentId: int, state: var AgentState): bool =
  ## Terminate when no more protection requests or target reached
  let (should, _) = fighterShouldEscort(env, agent)
  not should

proc optFighterEscort(controller: Controller, env: Environment, agent: Thing,
                      agentId: int, state: var AgentState): uint16 =
  ## Move toward the unit requesting protection and engage any enemies along the way
  let (should, targetPos) = fighterShouldEscort(env, agent)
  if not should:
    return 0'u16

  # First check for attack opportunity - engage enemies
  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  # Check for nearby enemies and engage them
  let enemy = fighterFindNearbyEnemy(controller, env, agent, state)
  if not isNil(enemy):
    return actOrMove(controller, env, agent, agentId, state, enemy.pos, 2'u16)

  # Move toward the protected unit
  let dist = int(chebyshevDist(agent.pos, targetPos))
  if dist <= EscortRadius:
    # Already close enough - stay nearby but allow other behaviors
    return 0'u16
  controller.moveTo(env, agent, agentId, state, targetPos)

proc canStartFighterAggressive(controller: Controller, env: Environment, agent: Thing,
                               agentId: int, state: var AgentState): bool =
  ## Aggressive hunting requires chasing - check stance
  if not stanceAllowsChase(env, agent):
    return false
  if agent.hp * 2 >= agent.maxHp:
    return true
  hasAllyNearby(env, agent)

proc shouldTerminateFighterAggressive(controller: Controller, env: Environment, agent: Thing,
                                      agentId: int, state: var AgentState): bool =
  # Terminate when HP drops low and no allies nearby for support
  if agent.hp * 2 >= agent.maxHp:
    return false
  not hasAllyNearby(env, agent)

proc optFighterAggressive(controller: Controller, env: Environment, agent: Thing,
                          agentId: int, state: var AgentState): uint16 =
  for kind in [Tumor, Spawner]:
    let target = env.findNearestThingSpiral(state, kind)
    if not isNil(target):
      return actOrMove(controller, env, agent, agentId, state, target.pos, 2'u16)
  let (didHunt, actHunt) = controller.ensureHuntFood(env, agent, agentId, state)
  if didHunt: return actHunt
  0'u16

# Attack-Move: Move to destination, attacking any enemies encountered along the way
# Like AoE2's attack-move: path to destination, engage enemies in range, resume after combat

proc canStartFighterAttackMove*(controller: Controller, env: Environment, agent: Thing,
                                agentId: int, state: var AgentState): bool =
  ## Attack-move is active when the agent has a valid attack-move destination set.
  ## Requires stance that allows movement to attack.
  if not stanceAllowsMovementToAttack(env, agent):
    return false
  state.attackMoveTarget.x >= 0

proc shouldTerminateFighterAttackMove*(controller: Controller, env: Environment, agent: Thing,
                                       agentId: int, state: var AgentState): bool =
  ## Terminate when destination is reached or attack-move is cancelled.
  if state.attackMoveTarget.x < 0:
    return true
  # Reached destination (within 1 tile)
  chebyshevDist(agent.pos, state.attackMoveTarget) <= 1'i32

proc optFighterAttackMove*(controller: Controller, env: Environment, agent: Thing,
                           agentId: int, state: var AgentState): uint16 =
  ## Attack-move behavior: move toward destination, but engage enemies along the way.
  ## After defeating an enemy, resume path to destination.
  if state.attackMoveTarget.x < 0:
    return 0'u16

  # Check if we've reached the destination
  if chebyshevDist(agent.pos, state.attackMoveTarget) <= 1'i32:
    # Clear the attack-move target - we've arrived
    state.attackMoveTarget = ivec2(-1, -1)
    return 0'u16

  # Look for enemies within detection radius
  let enemy = fighterFindNearbyEnemy(controller, env, agent, state)
  if not isNil(enemy):
    let enemyDist = int(chebyshevDist(agent.pos, enemy.pos))
    if enemyDist <= AttackMoveDetectionRadius:
      # Enemy found - engage!
      return actOrMove(controller, env, agent, agentId, state, enemy.pos, 2'u16)

  # No enemy nearby - continue moving toward destination
  controller.moveTo(env, agent, agentId, state, state.attackMoveTarget)

proc setAttackMoveTarget*(controller: Controller, agentId: int, target: IVec2) =
  ## Set an attack-move target for a specific agent.
  ## The agent will move toward the target while engaging enemies along the way.
  if agentId >= 0 and agentId < MapAgents:
    controller.agents[agentId].attackMoveTarget = target

proc clearAttackMoveTarget*(controller: Controller, agentId: int) =
  ## Clear the attack-move target for a specific agent.
  if agentId >= 0 and agentId < MapAgents:
    controller.agents[agentId].attackMoveTarget = ivec2(-1, -1)

proc getAttackMoveTarget*(controller: Controller, agentId: int): IVec2 =
  ## Get the current attack-move target for an agent.
  if agentId >= 0 and agentId < MapAgents:
    return controller.agents[agentId].attackMoveTarget
  ivec2(-1, -1)

# Battering Ram AI: Simple forward movement with attack-on-block behavior
# 1. Move forward in current orientation
# 2. If blocked, attack blocking target
# 3. If target destroyed, resume moving forward

proc canStartBatteringRamAdvance(controller: Controller, env: Environment, agent: Thing,
                                  agentId: int, state: var AgentState): bool =
  agent.unitClass == UnitBatteringRam

proc shouldTerminateBatteringRamAdvance(controller: Controller, env: Environment, agent: Thing,
                                         agentId: int, state: var AgentState): bool =
  # Never terminates - battering ram always uses this behavior
  agent.unitClass != UnitBatteringRam

proc optBatteringRamAdvance(controller: Controller, env: Environment, agent: Thing,
                            agentId: int, state: var AgentState): uint16 =
  ## Simple battering ram AI: move forward, attack blockers
  let delta = OrientationDeltas[agent.orientation.int]
  let forwardPos = agent.pos + delta

  # Check if there's something blocking forward movement
  let blocking = env.getThing(forwardPos)
  if not isNil(blocking):
    # Attack the blocking thing (verb 2 = attack)
    return actOrMove(controller, env, agent, agentId, state, forwardPos, 2'u16)

  # Check for blocking agent
  if not isValidPos(forwardPos):
    return actOrMove(controller, env, agent, agentId, state, forwardPos, 2'u16)
  let blockingAgent = env.grid[forwardPos.x][forwardPos.y]
  if not isNil(blockingAgent) and blockingAgent.agentId != agent.agentId:
    return actOrMove(controller, env, agent, agentId, state, forwardPos, 2'u16)

  # Check terrain passability
  if not canEnterForMove(env, agent, agent.pos, forwardPos):
    # Something blocks us (wall, terrain) - try to attack forward
    return actOrMove(controller, env, agent, agentId, state, forwardPos, 2'u16)

  # Path is clear - move forward (verb 1 = move)
  let dirIdx = agent.orientation.int
  return saveStateAndReturn(controller, agentId, state, encodeAction(1'u16, dirIdx.uint8))

# Formation movement: maintain position within control group formation

proc canStartFighterFormation(controller: Controller, env: Environment, agent: Thing,
                              agentId: int, state: var AgentState): bool =
  ## Formation movement activates when the agent is in a control group with an active formation
  ## and is not at its assigned slot position.
  let groupIdx = findAgentControlGroup(agentId)
  if groupIdx < 0:
    return false
  if not isFormationActive(groupIdx):
    return false
  let groupSize = aliveGroupSize(groupIdx, env)
  if groupSize < 2:
    return false
  let myIndex = agentIndexInGroup(groupIdx, agentId, env)
  if myIndex < 0:
    return false
  let center = calcGroupCenter(groupIdx, env)
  if center.x < 0:
    return false
  let targetPos = getFormationTargetForAgent(groupIdx, myIndex, center, groupSize)
  if targetPos.x < 0:
    return false
  int(chebyshevDist(agent.pos, targetPos)) > FormationArrivalThreshold

proc shouldTerminateFighterFormation(controller: Controller, env: Environment, agent: Thing,
                                     agentId: int, state: var AgentState): bool =
  ## Terminate when agent reaches its formation slot or formation is deactivated.
  let groupIdx = findAgentControlGroup(agentId)
  if groupIdx < 0 or not isFormationActive(groupIdx):
    return true
  let groupSize = aliveGroupSize(groupIdx, env)
  if groupSize < 2:
    return true
  let myIndex = agentIndexInGroup(groupIdx, agentId, env)
  if myIndex < 0:
    return true
  let center = calcGroupCenter(groupIdx, env)
  if center.x < 0:
    return true
  let targetPos = getFormationTargetForAgent(groupIdx, myIndex, center, groupSize)
  if targetPos.x < 0:
    return true
  int(chebyshevDist(agent.pos, targetPos)) <= FormationArrivalThreshold

proc optFighterFormation(controller: Controller, env: Environment, agent: Thing,
                         agentId: int, state: var AgentState): uint16 =
  ## Move toward formation slot position. Attacks enemies encountered along the way.
  let groupIdx = findAgentControlGroup(agentId)
  if groupIdx < 0 or not isFormationActive(groupIdx):
    return 0'u16

  let groupSize = aliveGroupSize(groupIdx, env)
  let myIndex = agentIndexInGroup(groupIdx, agentId, env)
  if myIndex < 0:
    return 0'u16

  let center = calcGroupCenter(groupIdx, env)
  if center.x < 0:
    return 0'u16

  let targetPos = getFormationTargetForAgent(groupIdx, myIndex, center, groupSize)
  if targetPos.x < 0:
    return 0'u16

  # Already at slot
  if int(chebyshevDist(agent.pos, targetPos)) <= FormationArrivalThreshold:
    return 0'u16

  # Check for attack opportunity while moving to slot
  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  # Move toward formation slot
  controller.moveTo(env, agent, agentId, state, targetPos)

# Patrol behavior - walk between waypoints and attack enemies encountered
const PatrolArrivalThreshold = 2  # Distance at which we consider waypoint "reached"

proc canStartFighterPatrol(controller: Controller, env: Environment, agent: Thing,
                           agentId: int, state: var AgentState): bool =
  ## Patrol activates when patrol mode is enabled for this agent.
  ## Supports both legacy 2-point patrol and multi-waypoint patrol.
  if not state.patrolActive:
    return false
  # Multi-waypoint patrol: need at least 2 waypoints
  if state.patrolWaypointCount >= 2:
    return true
  # Legacy 2-point patrol: need both points set
  state.patrolPoint1.x >= 0 and state.patrolPoint2.x >= 0

proc shouldTerminateFighterPatrol(controller: Controller, env: Environment, agent: Thing,
                                  agentId: int, state: var AgentState): bool =
  ## Patrol terminates when patrol mode is disabled
  not state.patrolActive

proc optFighterPatrol(controller: Controller, env: Environment, agent: Thing,
                      agentId: int, state: var AgentState): uint16 =
  ## Patrol between waypoints, attacking any enemies encountered.
  ## Uses AoE2-style patrol: walk to waypoint, attack nearby enemies, continue patrol.
  ## Supports both legacy 2-point patrol and multi-waypoint patrol (2-8 points).

  # First check for attack opportunity - attack takes priority during patrol
  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  # Check for nearby enemies and chase them if stance allows
  if stanceAllowsChase(env, agent):
    let enemy = fighterFindNearbyEnemy(controller, env, agent, state)
    if not isNil(enemy):
      # Move toward enemy to engage
      return controller.moveTo(env, agent, agentId, state, enemy.pos)

  # Multi-waypoint patrol mode
  if state.patrolWaypointCount >= 2:
    let target = state.patrolWaypoints[state.patrolCurrentWaypoint]
    let distToTarget = int(chebyshevDist(agent.pos, target))
    if distToTarget <= PatrolArrivalThreshold:
      # Advance to next waypoint (wraps to first after last)
      state.patrolCurrentWaypoint = (state.patrolCurrentWaypoint + 1) mod state.patrolWaypointCount
      let newTarget = state.patrolWaypoints[state.patrolCurrentWaypoint]
      return controller.moveTo(env, agent, agentId, state, newTarget)
    return controller.moveTo(env, agent, agentId, state, target)

  # Legacy 2-point patrol mode
  let target = if state.patrolToSecondPoint: state.patrolPoint2 else: state.patrolPoint1

  # Check if we've reached the current waypoint
  let distToTarget = int(chebyshevDist(agent.pos, target))
  if distToTarget <= PatrolArrivalThreshold:
    # Switch direction
    state.patrolToSecondPoint = not state.patrolToSecondPoint
    # Get the new target after switching
    let newTarget = if state.patrolToSecondPoint: state.patrolPoint2 else: state.patrolPoint1
    return controller.moveTo(env, agent, agentId, state, newTarget)

  # Move toward current waypoint
  controller.moveTo(env, agent, agentId, state, target)

# Scout behavior - reconnaissance with visibility tracking and enemy detection
# Scouts explore outward from base, flee when enemies spotted, and report threats

proc scoutFindNearbyEnemyUncached(env: Environment, agent: Thing): Thing =
  ## Internal: actual search logic for scout nearby enemy.
  findNearbyEnemyForFlee(env, agent, ScoutFleeRadius)

proc scoutFindNearbyEnemy(env: Environment, agent: Thing): Thing =
  ## Find nearest enemy agent within scout detection radius using spatial index.
  ## Cached per-step per-agent to avoid redundant scans in canStart/shouldTerminate/act.
  scoutEnemyCache.getWithAgent(env, agent, scoutFindNearbyEnemyUncached)

proc canStartScoutFlee(controller: Controller, env: Environment, agent: Thing,
                       agentId: int, state: var AgentState): bool =
  ## Scout flee triggers when scout mode is active and enemies are nearby.
  ## Scout-line units are light reconnaissance units - survival is priority.
  if agent.unitClass notin {UnitScout, UnitLightCavalry, UnitHussar}:
    return false
  if not state.scoutActive:
    return false
  let enemy = scoutFindNearbyEnemy(env, agent)
  if not isNil(enemy):
    # Record enemy sighting and report to threat map
    controller.recordScoutEnemySighting(agentId, env.currentStep.int32)
    return true
  # Also flee if recently saw enemy (recovery period)
  let stepsSinceEnemy = env.currentStep.int32 - state.scoutLastEnemySeenStep
  stepsSinceEnemy < ScoutFleeRecoverySteps

proc shouldTerminateScoutFlee(controller: Controller, env: Environment, agent: Thing,
                              agentId: int, state: var AgentState): bool =
  ## Stop fleeing when no enemies nearby and recovery period passed.
  let enemy = scoutFindNearbyEnemy(env, agent)
  if not isNil(enemy):
    return false  # Still enemies nearby - keep fleeing
  let stepsSinceEnemy = env.currentStep.int32 - state.scoutLastEnemySeenStep
  stepsSinceEnemy >= ScoutFleeRecoverySteps

proc optScoutFlee(controller: Controller, env: Environment, agent: Thing,
                  agentId: int, state: var AgentState): uint16 =
  ## Flee away from enemies toward base. Scouts prioritize survival over combat.
  ## Reports enemy positions to the team's shared threat map.
  let teamId = getTeamId(agent)
  let basePos = agent.getBasePos()
  state.basePosition = basePos

  # Find enemies and report to threat map
  let enemy = scoutFindNearbyEnemy(env, agent)
  if not isNil(enemy):
    # Report threat to team (high priority sighting from scout)
    controller.reportThreat(teamId, enemy.pos, 2, env.currentStep.int32,
                            agentId = enemy.agentId.int32, isStructure = false)
    controller.recordScoutEnemySighting(agentId, env.currentStep.int32)

  # Flee toward safe positions (altar, outpost, town center)
  var safePos = basePos
  for kind in [Altar, Outpost, TownCenter]:
    let safe = env.findNearestFriendlyThingSpiral(state, teamId, kind)
    if not isNil(safe):
      safePos = safe.pos
      break

  controller.moveTo(env, agent, agentId, state, safePos)

proc canStartScoutExplore(controller: Controller, env: Environment, agent: Thing,
                          agentId: int, state: var AgentState): bool =
  ## Scout exploration activates when scout mode is active and agent is a scout-line unit.
  agent.unitClass in {UnitScout, UnitLightCavalry, UnitHussar} and state.scoutActive

proc shouldTerminateScoutExplore(controller: Controller, env: Environment, agent: Thing,
                                 agentId: int, state: var AgentState): bool =
  ## Terminate exploration when scout mode is disabled.
  not state.scoutActive

proc optScoutExplore(controller: Controller, env: Environment, agent: Thing,
                     agentId: int, state: var AgentState): uint16 =
  ## Explore outward from base in a systematic sector-rotating pattern.
  ## Prioritizes unexplored areas (fog of war) and avoids known threats.
  ## Reports enemies to threat map. Scouts rotate through 4 quadrants to
  ## ensure even map coverage rather than purely following the spiral.
  let teamId = getTeamId(agent)
  let basePos = agent.getBasePos()
  state.basePosition = basePos

  # Update threat map and revealed map from scout's extended vision
  controller.updateThreatMapFromVision(env, agent, env.currentStep.int32)

  # Initialize explore radius if needed
  if state.scoutExploreRadius <= 0:
    state.scoutExploreRadius = ScoutVisionRange.int32  # Start with scout's vision range

  # Find a direction to explore that prioritizes unexplored tiles
  # Combine spiral search with sector rotation for systematic coverage
  var bestTarget = ivec2(-1, -1)
  var bestScore = int.low

  # Pre-check for threats (optimization: skip threat lookups when no threats known)
  let hasThreats = controller.hasKnownThreats(teamId, env.currentStep.int32)

  # Sector-based bias: rotate through quadrants (NE, SE, SW, NW) for even coverage.
  # Each scout uses agentId to offset which sector it starts in, spreading coverage
  # across multiple scouts. Sector rotates every ScoutSectorRotationSteps steps.
  let sectorIdx = ((env.currentStep.int div ScoutSectorRotationSteps) + agentId) mod 4
  let sectorDx: array[4, int32] = [1'i32, 1'i32, -1'i32, -1'i32]  # NE, SE, SW, NW
  let sectorDy: array[4, int32] = [-1'i32, 1'i32, 1'i32, -1'i32]

  # Try multiple candidate positions around the exploration frontier
  for _ in 0 ..< 16:  # Check more candidates for better exploration coverage
    let candidate = getNextSpiralPoint(state)
    let distFromBase = int(chebyshevDist(candidate, basePos))

    # Skip if too close to base (already explored) or too far
    if distFromBase < state.scoutExploreRadius.int - 5:
      continue
    if distFromBase > state.scoutExploreRadius.int + 20:
      continue

    # Check for threats near this position (skip if no threats known)
    let threatStrength = if hasThreats:
      controller.getTotalThreatStrength(teamId, candidate, 8, env.currentStep.int32)
    else:
      0'i32

    # Score: prefer unexplored positions at the frontier with fewer threats
    var score = 100 - abs(distFromBase - state.scoutExploreRadius.int) * 2
    score -= threatStrength.int * 20  # Heavily penalize threat areas

    # Bonus for unexplored tiles (fog of war clearing)
    if not env.isRevealed(teamId, candidate):
      score += 50  # Strong preference for unexplored areas

    # Sector bias: bonus for candidates in the current rotation sector
    let relX = candidate.x - basePos.x
    let relY = candidate.y - basePos.y
    if (relX >= 0) == (sectorDx[sectorIdx] >= 0) and
       (relY >= 0) == (sectorDy[sectorIdx] >= 0):
      score += 25  # Moderate sector bonus to steer without overriding fog priority

    # Sample only 4 cardinal directions + center to check for unexplored tiles nearby
    # (Optimization: O(5) instead of O(49) per candidate)
    var nearbyUnexplored = 0
    if not env.isRevealed(teamId, candidate):
      inc nearbyUnexplored
    for d in CardinalOffsets:
      let nearby = candidate + d * 3
      if isValidPos(nearby) and not env.isRevealed(teamId, nearby):
        inc nearbyUnexplored
    score += nearbyUnexplored * 10  # Bonus for areas with more unexplored tiles nearby

    if score > bestScore:
      bestScore = score
      bestTarget = candidate
      # Early-exit: good-enough candidate found (unexplored + decent position)
      # Threshold 140 = unexplored(+50) + good distance(~90+) with minimal threats
      if bestScore >= 140:
        break

  # If no good target found, use the spiral position directly
  if bestTarget.x < 0:
    bestTarget = getNextSpiralPoint(state)

  # Gradually expand exploration radius
  let distFromBase = int(chebyshevDist(agent.pos, basePos))
  if distFromBase >= state.scoutExploreRadius.int:
    state.scoutExploreRadius += ScoutExploreGrowth.int32

  # Move toward exploration target
  controller.moveTo(env, agent, agentId, state, bestTarget)

# Hold Position: Stay at current location, attack enemies in range but don't chase
proc canStartFighterHoldPosition(controller: Controller, env: Environment, agent: Thing,
                                 agentId: int, state: var AgentState): bool =
  ## Hold position activates when explicitly enabled via API.
  state.holdPositionActive and state.holdPositionTarget.x >= 0

proc shouldTerminateFighterHoldPosition(controller: Controller, env: Environment, agent: Thing,
                                        agentId: int, state: var AgentState): bool =
  ## Hold position terminates when disabled.
  not state.holdPositionActive

proc optFighterHoldPosition(controller: Controller, env: Environment, agent: Thing,
                            agentId: int, state: var AgentState): uint16 =
  ## Hold position: stay at the designated location, attack enemies in range,
  ## and engage enemies within HoldPositionEngageRadius but return afterward.
  ## Unlike StandGround, can move to attack nearby enemies.
  if not state.holdPositionActive or state.holdPositionTarget.x < 0:
    return 0'u16

  # Check for attack opportunity (melee or ranged in place)
  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  # If too far from hold position, prioritize returning
  let distFromHold = int(chebyshevDist(agent.pos, state.holdPositionTarget))
  if distFromHold > HoldPositionReturnRadius:
    return controller.moveTo(env, agent, agentId, state, state.holdPositionTarget)

  # Look for enemies within engage radius of the hold position
  let enemy = fighterFindNearbyEnemy(controller, env, agent, state)
  if not isNil(enemy):
    let enemyDistFromHold = int(chebyshevDist(enemy.pos, state.holdPositionTarget))
    if enemyDistFromHold <= HoldPositionEngageRadius:
      # Enemy is within engage radius of hold position - move to attack
      return actOrMove(controller, env, agent, agentId, state, enemy.pos, 2'u16)

  # Stay put
  0'u16

# Follow: Follow another agent, maintaining proximity

proc canStartFighterFollow(controller: Controller, env: Environment, agent: Thing,
                           agentId: int, state: var AgentState): bool =
  ## Follow activates when follow mode is enabled and target is valid and alive.
  if not state.followActive or state.followTargetAgentId < 0:
    return false
  if state.followTargetAgentId >= env.agents.len:
    return false
  let target = env.agents[state.followTargetAgentId]
  isAgentAlive(env, target)

proc shouldTerminateFighterFollow(controller: Controller, env: Environment, agent: Thing,
                                  agentId: int, state: var AgentState): bool =
  ## Follow terminates when disabled or target dies.
  if not state.followActive or state.followTargetAgentId < 0:
    return true
  if state.followTargetAgentId >= env.agents.len:
    return true
  let target = env.agents[state.followTargetAgentId]
  not isAgentAlive(env, target)

proc optFighterFollow(controller: Controller, env: Environment, agent: Thing,
                      agentId: int, state: var AgentState): uint16 =
  ## Follow: stay close to the target agent, attack enemies along the way.
  ## If target dies, follow is automatically terminated.
  if not state.followActive or state.followTargetAgentId < 0:
    return 0'u16
  if state.followTargetAgentId >= env.agents.len:
    state.followActive = false
    return 0'u16
  let target = env.agents[state.followTargetAgentId]
  if not isAgentAlive(env, target):
    state.followActive = false
    state.followTargetAgentId = -1
    return 0'u16

  # Check for attack opportunity
  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  # Check distance to target
  let dist = int(chebyshevDist(agent.pos, target.pos))
  if dist > FollowProximityRadius:
    # Too far - move toward target
    return controller.moveTo(env, agent, agentId, state, target.pos)

  # Within range - stay put
  0'u16

# Guard: Stay near a target (agent or position), attack enemies within range, return after combat

proc canStartFighterGuard(controller: Controller, env: Environment, agent: Thing,
                          agentId: int, state: var AgentState): bool =
  ## Guard activates when guard mode is enabled and target is valid.
  if not state.guardActive:
    return false
  # Combat units only
  if agent.unitClass notin {UnitManAtArms, UnitLongSwordsman, UnitChampion,
                            UnitKnight, UnitCavalier, UnitPaladin, UnitScout, UnitArcher,
                            UnitCrossbowman, UnitArbalester, UnitLightCavalry, UnitHussar,
                            UnitCamel, UnitHeavyCamel, UnitImperialCamel}:
    return false
  # If guarding an agent, check if it's alive
  if state.guardTargetAgentId >= 0:
    if state.guardTargetAgentId >= env.agents.len:
      return false
    let target = env.agents[state.guardTargetAgentId]
    return isAgentAlive(env, target)
  # If guarding a position, just need valid position
  state.guardTargetPos.x >= 0

proc shouldTerminateFighterGuard(controller: Controller, env: Environment, agent: Thing,
                                 agentId: int, state: var AgentState): bool =
  ## Guard terminates when disabled or target agent dies.
  if not state.guardActive:
    return true
  # If guarding an agent, terminate if it dies
  if state.guardTargetAgentId >= 0:
    if state.guardTargetAgentId >= env.agents.len:
      return true
    let target = env.agents[state.guardTargetAgentId]
    if not isAgentAlive(env, target):
      return true
  # If guarding a position, never auto-terminate (only when disabled)
  false

proc optFighterGuard(controller: Controller, env: Environment, agent: Thing,
                     agentId: int, state: var AgentState): uint16 =
  ## Guard: stay within GuardRadius of target, attack enemies within range,
  ## return to guard position after combat.
  if not state.guardActive:
    return 0'u16

  # Determine guard center position
  var guardCenter: IVec2
  if state.guardTargetAgentId >= 0:
    # Guarding an agent - use their position
    if state.guardTargetAgentId >= env.agents.len:
      state.guardActive = false
      return 0'u16
    let target = env.agents[state.guardTargetAgentId]
    if not isAgentAlive(env, target):
      state.guardActive = false
      state.guardTargetAgentId = -1
      return 0'u16
    guardCenter = target.pos
  else:
    # Guarding a position
    if state.guardTargetPos.x < 0:
      state.guardActive = false
      return 0'u16
    guardCenter = state.guardTargetPos

  # First check for immediate attack opportunity
  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  # Check for enemies within GuardRadius of the guard center and engage
  let enemy = fighterFindNearbyEnemy(controller, env, agent, state)
  if not isNil(enemy):
    let enemyDistToCenter = int(chebyshevDist(enemy.pos, guardCenter))
    if enemyDistToCenter <= GuardRadius:
      # Enemy is within guard radius of center - engage
      return actOrMove(controller, env, agent, agentId, state, enemy.pos, 2'u16)

  # Check our distance from guard center
  let dist = int(chebyshevDist(agent.pos, guardCenter))
  if dist > GuardRadius:
    # Too far from guard position - return to center
    return controller.moveTo(env, agent, agentId, state, guardCenter)

  # Within range and no enemies - stay put
  0'u16

# ============================================================================
# Naval Unit AI Behaviors
# ============================================================================

const NavalUnitClasses* = {UnitGalley, UnitFireShip, UnitFishingShip,
                           UnitTransportShip, UnitDemoShip, UnitCannonGalleon}

proc findNearestEnemyShip(env: Environment, agent: Thing, radius: int): Thing =
  ## Find nearest enemy water unit within radius using spatial index.
  var best: Thing = nil
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
        if sameTeam(agent, other):
          continue
        if not other.isWaterUnit:
          continue
        let dist = int(chebyshevDist(agent.pos, other.pos))
        if dist <= radius and dist < bestDist:
          bestDist = dist
          best = other
  best

proc findNearestEnemyOnWater(env: Environment, agent: Thing, radius: int): Thing =
  ## Find nearest enemy (ship or land unit near water) within radius.
  var best: Thing = nil
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
        if sameTeam(agent, other):
          continue
        let dist = int(chebyshevDist(agent.pos, other.pos))
        if dist <= radius and dist < bestDist:
          bestDist = dist
          best = other
  best

proc findNearestFriendlyDock(env: Environment, agent: Thing): Thing =
  ## Find nearest friendly dock.
  let teamId = getTeamId(agent)
  env.findNearestFriendlyThingSpatial(agent.pos, teamId, Dock, int.high)

# FishingShip: Gather fish resources
proc canStartFishingShipFish(controller: Controller, env: Environment, agent: Thing,
                             agentId: int, state: var AgentState): bool =
  agent.unitClass == UnitFishingShip and env.thingsByKind[Fish].len > 0

proc shouldTerminateFishingShipFish(controller: Controller, env: Environment, agent: Thing,
                                    agentId: int, state: var AgentState): bool =
  agent.unitClass != UnitFishingShip or env.thingsByKind[Fish].len == 0

proc optFishingShipFish(controller: Controller, env: Environment, agent: Thing,
                        agentId: int, state: var AgentState): uint16 =
  ## Fishing ship gathers fish from water tiles, returns to dock to deposit.
  # If carrying fish, return to dock to deposit
  if getInv(agent, ItemFish) > 0:
    let dock = findNearestFriendlyDock(env, agent)
    if not isNil(dock):
      return actOrMove(controller, env, agent, agentId, state, dock.pos, 3'u16)
    # No dock - just hold fish for now
    return 0'u16

  # Find and gather fish
  let fish = env.findNearestThingSpiral(state, Fish)
  if isNil(fish):
    return 0'u16
  actOrMove(controller, env, agent, agentId, state, fish.pos, 3'u16)

# Galley: Ranged combat ship
proc canStartGalleyAttack(controller: Controller, env: Environment, agent: Thing,
                          agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitGalley:
    return false
  let enemy = findNearestEnemyOnWater(env, agent, GalleyBaseRange * 3)
  not isNil(enemy)

proc shouldTerminateGalleyAttack(controller: Controller, env: Environment, agent: Thing,
                                 agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitGalley:
    return true
  let enemy = findNearestEnemyOnWater(env, agent, GalleyBaseRange * 3)
  isNil(enemy)

proc optGalleyAttack(controller: Controller, env: Environment, agent: Thing,
                     agentId: int, state: var AgentState): uint16 =
  ## Galley attacks enemies at range, prioritizing other ships.
  # First check for immediate attack opportunity
  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  # Prioritize enemy ships
  let enemyShip = findNearestEnemyShip(env, agent, GalleyBaseRange * 3)
  if not isNil(enemyShip):
    return actOrMove(controller, env, agent, agentId, state, enemyShip.pos, 2'u16)

  # Fall back to any enemy
  let enemy = findNearestEnemyOnWater(env, agent, GalleyBaseRange * 3)
  if not isNil(enemy):
    return actOrMove(controller, env, agent, agentId, state, enemy.pos, 2'u16)
  0'u16

# FireShip: Anti-ship specialist
proc canStartFireShipAttack(controller: Controller, env: Environment, agent: Thing,
                            agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitFireShip:
    return false
  # Fire ships prioritize enemy water units
  let enemyShip = findNearestEnemyShip(env, agent, ObservationRadius.int * 2)
  not isNil(enemyShip)

proc shouldTerminateFireShipAttack(controller: Controller, env: Environment, agent: Thing,
                                   agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitFireShip:
    return true
  let enemyShip = findNearestEnemyShip(env, agent, ObservationRadius.int * 2)
  isNil(enemyShip)

proc optFireShipAttack(controller: Controller, env: Environment, agent: Thing,
                       agentId: int, state: var AgentState): uint16 =
  ## Fire ship aggressively pursues and attacks enemy water units.
  ## Gets bonus damage vs water units.
  # Check for immediate attack opportunity
  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  # Chase enemy ships
  let enemyShip = findNearestEnemyShip(env, agent, ObservationRadius.int * 2)
  if not isNil(enemyShip):
    return actOrMove(controller, env, agent, agentId, state, enemyShip.pos, 2'u16)
  0'u16

# DemoShip: Kamikaze attack ship
proc canStartDemoShipKamikaze(controller: Controller, env: Environment, agent: Thing,
                              agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitDemoShip:
    return false
  # Demo ships look for high-value targets (ships or buildings near water)
  let enemy = findNearestEnemyOnWater(env, agent, ObservationRadius.int * 3)
  not isNil(enemy)

proc shouldTerminateDemoShipKamikaze(controller: Controller, env: Environment, agent: Thing,
                                     agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitDemoShip:
    return true
  let enemy = findNearestEnemyOnWater(env, agent, ObservationRadius.int * 3)
  isNil(enemy)

proc optDemoShipKamikaze(controller: Controller, env: Environment, agent: Thing,
                         agentId: int, state: var AgentState): uint16 =
  ## Demo ship moves toward enemy and attacks (self-destructs on hit).
  ## Prioritizes: enemy ships > docks > other coastal targets.
  # Check for immediate attack opportunity (this is the kamikaze strike)
  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  # Prioritize enemy ships
  let enemyShip = findNearestEnemyShip(env, agent, ObservationRadius.int * 3)
  if not isNil(enemyShip):
    return controller.moveTo(env, agent, agentId, state, enemyShip.pos)

  # Check for enemy docks
  let teamId = getTeamId(agent)
  for dock in env.thingsByKind[Dock]:
    if dock.teamId != teamId and dock.teamId >= 0:
      return controller.moveTo(env, agent, agentId, state, dock.pos)

  # Fall back to any enemy
  let enemy = findNearestEnemyOnWater(env, agent, ObservationRadius.int * 3)
  if not isNil(enemy):
    return controller.moveTo(env, agent, agentId, state, enemy.pos)
  0'u16

# CannonGalleon: Long-range siege ship
proc canStartCannonGalleonSiege(controller: Controller, env: Environment, agent: Thing,
                                agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitCannonGalleon:
    return false
  # Cannon galleons look for buildings or enemy units
  let teamId = getTeamId(agent)
  let enemyBuilding = findNearestEnemyBuildingSpatial(env, agent.pos, teamId, CannonGalleonBaseRange * 3)
  if not isNil(enemyBuilding):
    return true
  let enemy = findNearestEnemyOnWater(env, agent, CannonGalleonBaseRange * 3)
  not isNil(enemy)

proc shouldTerminateCannonGalleonSiege(controller: Controller, env: Environment, agent: Thing,
                                       agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitCannonGalleon:
    return true
  let teamId = getTeamId(agent)
  let enemyBuilding = findNearestEnemyBuildingSpatial(env, agent.pos, teamId, CannonGalleonBaseRange * 3)
  if not isNil(enemyBuilding):
    return false
  let enemy = findNearestEnemyOnWater(env, agent, CannonGalleonBaseRange * 3)
  isNil(enemy)

proc optCannonGalleonSiege(controller: Controller, env: Environment, agent: Thing,
                           agentId: int, state: var AgentState): uint16 =
  ## Cannon galleon attacks buildings and units at long range.
  ## Prioritizes: buildings > ships > other units.
  # Check for immediate attack opportunity
  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  let teamId = getTeamId(agent)

  # Prioritize enemy buildings (siege role)
  let enemyBuilding = findNearestEnemyBuildingSpatial(env, agent.pos, teamId, CannonGalleonBaseRange * 3)
  if not isNil(enemyBuilding):
    return actOrMove(controller, env, agent, agentId, state, enemyBuilding.pos, 2'u16)

  # Then enemy ships
  let enemyShip = findNearestEnemyShip(env, agent, CannonGalleonBaseRange * 3)
  if not isNil(enemyShip):
    return actOrMove(controller, env, agent, agentId, state, enemyShip.pos, 2'u16)

  # Fall back to any enemy
  let enemy = findNearestEnemyOnWater(env, agent, CannonGalleonBaseRange * 3)
  if not isNil(enemy):
    return actOrMove(controller, env, agent, agentId, state, enemy.pos, 2'u16)
  0'u16

# TransportShip: Unit transport with docking behavior
proc canStartTransportShipDock(controller: Controller, env: Environment, agent: Thing,
                               agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitTransportShip:
    return false
  # Transport ships move toward friendly docks when carrying units or when idle
  let dock = findNearestFriendlyDock(env, agent)
  not isNil(dock)

proc shouldTerminateTransportShipDock(controller: Controller, env: Environment, agent: Thing,
                                      agentId: int, state: var AgentState): bool =
  if agent.unitClass != UnitTransportShip:
    return true
  let dock = findNearestFriendlyDock(env, agent)
  isNil(dock)

proc optTransportShipDock(controller: Controller, env: Environment, agent: Thing,
                          agentId: int, state: var AgentState): uint16 =
  ## Transport ship patrols near friendly docks to pick up/drop off units.
  let dock = findNearestFriendlyDock(env, agent)
  if isNil(dock):
    return 0'u16

  let dist = int(chebyshevDist(agent.pos, dock.pos))
  # Stay within 3 tiles of dock
  if dist <= 3:
    return 0'u16  # Already close enough
  controller.moveTo(env, agent, agentId, state, dock.pos)

let FighterOptions* = [
  OptionDef(
    name: "BatteringRamAdvance",
    canStart: canStartBatteringRamAdvance,
    shouldTerminate: shouldTerminateBatteringRamAdvance,
    act: optBatteringRamAdvance,
    interruptible: false  # Battering ram AI is not interruptible - it just advances and attacks
  ),
  # Naval unit behaviors
  OptionDef(
    name: "DemoShipKamikaze",
    canStart: canStartDemoShipKamikaze,
    shouldTerminate: shouldTerminateDemoShipKamikaze,
    act: optDemoShipKamikaze,
    interruptible: false  # Demo ship kamikaze is not interruptible - commit to attack
  ),
  OptionDef(
    name: "FishingShipFish",
    canStart: canStartFishingShipFish,
    shouldTerminate: shouldTerminateFishingShipFish,
    act: optFishingShipFish,
    interruptible: true
  ),
  OptionDef(
    name: "GalleyAttack",
    canStart: canStartGalleyAttack,
    shouldTerminate: shouldTerminateGalleyAttack,
    act: optGalleyAttack,
    interruptible: true
  ),
  OptionDef(
    name: "FireShipAttack",
    canStart: canStartFireShipAttack,
    shouldTerminate: shouldTerminateFireShipAttack,
    act: optFireShipAttack,
    interruptible: true
  ),
  OptionDef(
    name: "CannonGalleonSiege",
    canStart: canStartCannonGalleonSiege,
    shouldTerminate: shouldTerminateCannonGalleonSiege,
    act: optCannonGalleonSiege,
    interruptible: true
  ),
  OptionDef(
    name: "TransportShipDock",
    canStart: canStartTransportShipDock,
    shouldTerminate: shouldTerminateTransportShipDock,
    act: optTransportShipDock,
    interruptible: true
  ),
  OptionDef(
    name: "FighterBreakout",
    canStart: canStartFighterBreakout,
    shouldTerminate: shouldTerminateFighterBreakout,
    act: optFighterBreakout,
    interruptible: true
  ),
  OptionDef(
    name: "FighterRetreat",
    canStart: canStartFighterRetreat,
    shouldTerminate: shouldTerminateFighterRetreat,
    act: optFighterRetreat,
    interruptible: true
  ),
  OptionDef(
    name: "ScoutFlee",
    canStart: canStartScoutFlee,
    shouldTerminate: shouldTerminateScoutFlee,
    act: optScoutFlee,
    interruptible: false  # Scout flee is not interruptible - survival is priority
  ),
  EmergencyHealOption,
  OptionDef(
    name: "FighterSeekHealer",
    canStart: canStartFighterSeekHealer,
    shouldTerminate: shouldTerminateFighterSeekHealer,
    act: optFighterSeekHealer,
    interruptible: true
  ),
  OptionDef(
    name: "FighterMonk",
    canStart: canStartFighterMonk,
    shouldTerminate: shouldTerminateFighterMonk,
    act: optFighterMonk,
    interruptible: true
  ),
  OptionDef(
    name: "FighterPatrol",
    canStart: canStartFighterPatrol,
    shouldTerminate: shouldTerminateFighterPatrol,
    act: optFighterPatrol,
    interruptible: true
  ),
  OptionDef(
    name: "FighterHoldPosition",
    canStart: canStartFighterHoldPosition,
    shouldTerminate: shouldTerminateFighterHoldPosition,
    act: optFighterHoldPosition,
    interruptible: true
  ),
  OptionDef(
    name: "FighterFollow",
    canStart: canStartFighterFollow,
    shouldTerminate: shouldTerminateFighterFollow,
    act: optFighterFollow,
    interruptible: true
  ),
  OptionDef(
    name: "FighterGuard",
    canStart: canStartFighterGuard,
    shouldTerminate: shouldTerminateFighterGuard,
    act: optFighterGuard,
    interruptible: true
  ),
  OptionDef(
    name: "FighterTrain",
    canStart: canStartFighterTrain,
    shouldTerminate: shouldTerminateFighterTrain,
    act: optFighterTrain,
    interruptible: true
  ),
  OptionDef(
    name: "FighterDividerDefense",
    canStart: canStartFighterDividerDefense,
    shouldTerminate: shouldTerminateFighterDividerDefense,
    act: optFighterDividerDefense,
    interruptible: true
  ),
  OptionDef(
    name: "FighterLanterns",
    canStart: canStartFighterLanterns,
    shouldTerminate: shouldTerminateFighterLanterns,
    act: optFighterLanterns,
    interruptible: true
  ),
  OptionDef(
    name: "FighterDropoffFood",
    canStart: canStartFighterDropoffFood,
    shouldTerminate: shouldTerminateFighterDropoffFood,
    act: optFighterDropoffFood,
    interruptible: true
  ),
  OptionDef(
    name: "FighterBecomeSiege",
    canStart: canStartFighterBecomeSiege,
    shouldTerminate: shouldTerminateFighterBecomeSiege,
    act: optFighterBecomeSiege,
    interruptible: true
  ),
  OptionDef(
    name: "FighterMaintainGear",
    canStart: canStartFighterMaintainGear,
    shouldTerminate: shouldTerminateFighterMaintainGear,
    act: optFighterMaintainGear,
    interruptible: true
  ),
  OptionDef(
    name: "FighterKite",
    canStart: canStartFighterKite,
    shouldTerminate: shouldTerminateFighterKite,
    act: optFighterKite,
    interruptible: true
  ),
  OptionDef(
    name: "FighterAntiSiege",
    canStart: canStartFighterAntiSiege,
    shouldTerminate: shouldTerminateFighterAntiSiege,
    act: optFighterAntiSiege,
    interruptible: true
  ),
  OptionDef(
    name: "FighterEscort",
    canStart: canStartFighterEscort,
    shouldTerminate: shouldTerminateFighterEscort,
    act: optFighterEscort,
    interruptible: true
  ),
  OptionDef(
    name: "FighterHuntPredators",
    canStart: canStartFighterHuntPredators,
    shouldTerminate: shouldTerminateFighterHuntPredators,
    act: optFighterHuntPredators,
    interruptible: true
  ),
  OptionDef(
    name: "FighterClearGoblins",
    canStart: canStartFighterClearGoblins,
    shouldTerminate: shouldTerminateFighterClearGoblins,
    act: optFighterClearGoblins,
    interruptible: true
  ),
  AntiTumorPatrolOption,
  SmeltGoldOption,
  CraftBreadOption,
  StoreValuablesOption,
  OptionDef(
    name: "FighterAggressive",
    canStart: canStartFighterAggressive,
    shouldTerminate: shouldTerminateFighterAggressive,
    act: optFighterAggressive,
    interruptible: true
  ),
  OptionDef(
    name: "FighterAttackMove",
    canStart: canStartFighterAttackMove,
    shouldTerminate: shouldTerminateFighterAttackMove,
    act: optFighterAttackMove,
    interruptible: true
  ),
  OptionDef(
    name: "FighterFormation",
    canStart: canStartFighterFormation,
    shouldTerminate: shouldTerminateFighterFormation,
    act: optFighterFormation,
    interruptible: true
  ),
  OptionDef(
    name: "ScoutExplore",
    canStart: canStartScoutExplore,
    shouldTerminate: shouldTerminateScoutExplore,
    act: optScoutExplore,
    interruptible: true  # Can be interrupted by higher priority behaviors
  ),
  FallbackSearchOption
]
