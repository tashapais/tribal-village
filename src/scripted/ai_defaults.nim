## AI defaults: role catalog management, decision-making, and controller update loop.

import std/tables
import ai_build_helpers
export ai_build_helpers

import ai_audit
export ai_audit

import economy
export economy

import evolution
export evolution

import settlement
export settlement

import ../replay_analyzer
export replay_analyzer

const
  EvolutionEnabled = defined(enableEvolution)
  ReplayAnalysisEnabled = defined(enableReplayAnalysis)
  ScriptedRoleHistoryPath = "data/role_history.json"
  ScriptedScoreStep = 5000
  ScriptedGeneratedRoleCount = 16
  ScriptedRoleExplorationChance = 0.08
  ScriptedRoleMutationChance = 0.25
  ScriptedTempleAssignEnabled = true

when ReplayAnalysisEnabled:
  const ScriptedReplayDir = "data/replays"

type
  ScriptedRoleState = object
    initialized: bool
    catalog: RoleCatalog
    roleOptionsCache: seq[seq[OptionDef]]
    roleOptionsCached: seq[bool]
    roleAssignments: array[MapAgents, int]
    roleIsScripted: array[MapAgents, bool]
    pendingHybridRoles: array[MapAgents, int]
    coreRoleIds: array[AgentRole, int]
    lastEpisodeStep: int
    scoredAtStep: bool
    evolutionConfig: EvolutionConfig
    rolePool: seq[int]

var scriptedState: ScriptedRoleState

proc resetScriptedAssignments(state: var ScriptedRoleState) =
  for i in 0 ..< MapAgents:
    state.roleAssignments[i] = -1
    state.roleIsScripted[i] = false
    state.pendingHybridRoles[i] = -1

proc ensureRoleCache(state: var ScriptedRoleState) =
  if state.roleOptionsCache.len < state.catalog.roles.len:
    let needed = state.catalog.roles.len - state.roleOptionsCache.len
    for _ in 0 ..< needed:
      state.roleOptionsCache.add @[]
      state.roleOptionsCached.add false

proc buildCoreRole(catalog: var RoleCatalog, name: string,
                   options: openArray[OptionDef],
                   kind: AgentRole): int =
  let existing = findRoleId(catalog, name)
  if existing >= 0:
    catalog.roles[existing].kind = kind
    catalog.roles[existing].origin = "core"
    return existing
  var ids: seq[int] = @[]
  for opt in options:
    let id = findBehaviorId(catalog, opt.name)
    if id >= 0:
      ids.add id
  let tier = RoleTier(behaviorIds: ids, selection: TierFixed)
  let role = newRoleDef(catalog, name, @[tier], "core", kind)
  registerRole(catalog, role)

proc rebuildRolePool(state: var ScriptedRoleState) =
  state.rolePool.setLen(0)
  for role in state.catalog.roles:
    if role.origin != "core":
      state.rolePool.add role.id
  if state.rolePool.len == 0:
    for roleId in state.coreRoleIds:
      if roleId >= 0:
        state.rolePool.add roleId

proc generateRandomRole(state: var ScriptedRoleState, rng: var Rand,
                        origin: string): int =
  var role = sampleRole(state.catalog, rng, state.evolutionConfig)
  if randChance(rng, ScriptedRoleMutationChance):
    role = mutateRole(state.catalog, rng, role, state.evolutionConfig.mutationRate)
  role.origin = origin
  let id = registerRole(state.catalog, role)
  ensureRoleCache(state)
  if origin != "core":
    state.rolePool.add id
  id

proc initScriptedState(controller: Controller) =
  if scriptedState.initialized:
    return
  scriptedState.lastEpisodeStep = -1
  scriptedState.scoredAtStep = false
  resetScriptedAssignments(scriptedState)
  scriptedState.evolutionConfig = defaultEvolutionConfig()
  scriptedState.catalog = initRoleCatalog()
  scriptedState.catalog.seedDefaultBehaviorCatalog()
  if EvolutionEnabled:
    scriptedState.catalog.loadRoleHistory(ScriptedRoleHistoryPath)
  scriptedState.coreRoleIds = [
    buildCoreRole(scriptedState.catalog, "GathererCore", GathererOptions, Gatherer),
    buildCoreRole(scriptedState.catalog, "BuilderCore", BuilderOptions, Builder),
    buildCoreRole(scriptedState.catalog, "FighterCore", FighterOptions, Fighter),
    -1
  ]
  ensureRoleCache(scriptedState)
  if EvolutionEnabled:
    var nonCore = 0
    for role in scriptedState.catalog.roles:
      if role.origin != "core":
        inc nonCore
    while nonCore < ScriptedGeneratedRoleCount:
      discard generateRandomRole(scriptedState, controller.rng, "sampled")
      inc nonCore
  rebuildRolePool(scriptedState)
  resetScriptedAssignments(scriptedState)
  scriptedState.initialized = true

proc setAgentRole(agentId: int, state: var AgentState, roleId: int) =
  state.roleId = roleId
  scriptedState.roleAssignments[agentId] = roleId
  if roleId >= 0 and roleId < scriptedState.catalog.roles.len:
    state.role = scriptedState.catalog.roles[roleId].kind
    scriptedState.roleIsScripted[agentId] = scriptedState.catalog.roles[roleId].origin != "core"
  else:
    scriptedState.roleIsScripted[agentId] = false
  state.activeOptionId = -1
  state.activeOptionTicks = 0

proc assignScriptedRole(controller: Controller, agentId: int,
                        state: var AgentState) =
  initScriptedState(controller)
  if ScriptedTempleAssignEnabled and scriptedState.pendingHybridRoles[agentId] >= 0:
    let roleId = scriptedState.pendingHybridRoles[agentId]
    scriptedState.pendingHybridRoles[agentId] = -1
    setAgentRole(agentId, state, roleId)
    return
  var roleId = -1
  if EvolutionEnabled:
    if randChance(controller.rng, ScriptedRoleExplorationChance):
      roleId = generateRandomRole(scriptedState, controller.rng, "explore")
    else:
      roleId = pickRoleIdWeighted(scriptedState.catalog, controller.rng, scriptedState.rolePool)
  if roleId < 0:
    roleId = scriptedState.coreRoleIds[Gatherer]
  setAgentRole(agentId, state, roleId)

proc roleOptionsFor(roleId: int, rng: var Rand): seq[OptionDef] =
  if roleId < 0 or roleId >= scriptedState.catalog.roles.len:
    return @[]
  ensureRoleCache(scriptedState)
  if not scriptedState.roleOptionsCached[roleId]:
    scriptedState.roleOptionsCache[roleId] =
      materializeRoleOptions(scriptedState.catalog, scriptedState.catalog.roles[roleId], rng)
    scriptedState.roleOptionsCached[roleId] = true
  scriptedState.roleOptionsCache[roleId]

proc roleIdForAgent(controller: Controller, agentId: int): int

proc applyScriptedScoring(controller: Controller, env: Environment) =
  let score = env.scoreTerritory()
  let total = max(1, score.scoredTiles)
  var teamScores: array[MapRoomObjectsTeams, float32]
  for teamId in 0 ..< MapRoomObjectsTeams:
    teamScores[teamId] = float32(score.teamTiles[teamId]) / float32(total)
  var roleTeamCounts: Table[(int, int), int]
  for agent in env.agents:
    if not isAgentAlive(env, agent):
      continue
    let roleId = roleIdForAgent(controller, agent.agentId)
    if roleId < 0:
      continue
    let teamId = getTeamId(agent)
    if teamId < 0 or teamId >= MapRoomObjectsTeams:
      continue
    let key = (roleId, teamId)
    roleTeamCounts[key] = roleTeamCounts.getOrDefault(key, 0) + 1
  for key, count in roleTeamCounts.pairs:
    let roleId = key[0]
    let teamId = key[1]
    if roleId < 0 or roleId >= scriptedState.catalog.roles.len:
      continue
    if teamId < 0 or teamId >= MapRoomObjectsTeams:
      continue
    let sampleTeamScore = teamScores[teamId]
    let weight = min(4, count)
    recordRoleScore(scriptedState.catalog.roles[roleId], sampleTeamScore, sampleTeamScore >= 0.5, weight = weight)
    lockRoleNameIfFit(scriptedState.catalog.roles[roleId], scriptedState.evolutionConfig.lockFitnessThreshold)
    for tier in scriptedState.catalog.roles[roleId].tiers:
      for behaviorId in tier.behaviorIds:
        if behaviorId >= 0 and behaviorId < scriptedState.catalog.behaviors.len:
          recordBehaviorScore(scriptedState.catalog.behaviors[behaviorId], sampleTeamScore, weight = weight)
          inc scriptedState.catalog.behaviors[behaviorId].uses
  # Apply replay analysis feedback if enabled
  when ReplayAnalysisEnabled:
    let replayDir = getEnv("TV_REPLAY_DIR", ScriptedReplayDir)
    if replayDir.len > 0:
      let analyses = analyzeReplayBatch(replayDir)
      if analyses.len > 0:
        applyBatchFeedback(scriptedState.catalog, analyses)

  scriptedState.catalog.saveRoleHistory(ScriptedRoleHistoryPath)

proc roleIdForAgent(controller: Controller, agentId: int): int =
  if controller.agentsInitialized[agentId]:
    let stateRoleId = controller.agents[agentId].roleId
    if stateRoleId >= 0 and stateRoleId < scriptedState.catalog.roles.len:
      return stateRoleId
  let assigned = scriptedState.roleAssignments[agentId]
  if assigned >= 0 and assigned < scriptedState.catalog.roles.len:
    return assigned
  let stateRole = controller.agents[agentId].role
  let coreId = scriptedState.coreRoleIds[stateRole]
  if coreId >= 0:
    return coreId
  scriptedState.coreRoleIds[Gatherer]

proc injectBehavior(role: var RoleDef, rng: var Rand, catalog: RoleCatalog) =
  if role.tiers.len == 0 or catalog.behaviors.len == 0:
    return
  let newId = randIntExclusive(rng, 0, catalog.behaviors.len)
  for id in role.tiers[0].behaviorIds:
    if id == newId:
      return
  role.tiers[0].behaviorIds.add newId

proc processTempleHybridRequests(controller: Controller, env: Environment) =
  if env.templeHybridRequests.len == 0:
    return
  for req in env.templeHybridRequests:
    if req.childId < 0 or req.childId >= MapAgents:
      continue
    let roleAId = roleIdForAgent(controller, req.parentA)
    let roleBId = roleIdForAgent(controller, req.parentB)
    if roleAId < 0 or roleBId < 0:
      continue
    let roleA = scriptedState.catalog.roles[roleAId]
    let roleB = scriptedState.catalog.roles[roleBId]
    var hybrid = recombineRoles(scriptedState.catalog, controller.rng, roleA, roleB)
    if randChance(controller.rng, ScriptedRoleMutationChance):
      hybrid = mutateRole(scriptedState.catalog, controller.rng, hybrid, scriptedState.evolutionConfig.mutationRate)
    if randChance(controller.rng, 0.35):
      injectBehavior(hybrid, controller.rng, scriptedState.catalog)
    hybrid.origin = "temple"
    let newRoleId = registerRole(scriptedState.catalog, hybrid)
    ensureRoleCache(scriptedState)
    scriptedState.rolePool.add newRoleId
    scriptedState.pendingHybridRoles[req.childId] = newRoleId
    if ScriptedTempleAssignEnabled:
      controller.agentsInitialized[req.childId] = false
  env.templeHybridRequests.setLen(0)

const GoblinAvoidRadius = 6

proc tryPrioritizeHearts(controller: Controller, env: Environment, agent: Thing,
                         agentId: int, state: var AgentState): tuple[did: bool, action: uint16] =
  let teamId = getTeamId(agent)
  var altarPos = ivec2(-1, -1)
  var altarHearts = 0
  if agent.homeAltar.x >= 0:
    let homeAltar = env.getThing(agent.homeAltar)
    if not isNil(homeAltar) and homeAltar.kind == Altar and homeAltar.teamId == teamId:
      altarPos = homeAltar.pos
      altarHearts = homeAltar.hearts
  if altarPos.x < 0:
    # Use spatial query instead of O(n) altar scan
    let nearestAltar = findNearestFriendlyThingSpatial(env, agent.pos, teamId, Altar, 1000)
    if not nearestAltar.isNil:
      altarPos = nearestAltar.pos
      altarHearts = nearestAltar.hearts
  if altarPos.x < 0 or altarHearts >= 10:
    return (false, 0'u16)

  if agent.inventoryBar > 0:
    if isAdjacent(agent.pos, altarPos):
      return (true, controller.useAt(env, agent, agentId, state, altarPos))
    return (true, controller.moveTo(env, agent, agentId, state, altarPos))

  if agent.inventoryGold > 0:
    let (didKnown, actKnown) = controller.tryMoveToKnownResource(
      env, agent, agentId, state, state.closestMagmaPos, {Magma}, 3'u16)
    if didKnown: return (true, actKnown)
    let magmaGlobal = findNearestThing(env, agent.pos, Magma, maxDist = int.high)
    if not isNil(magmaGlobal):
      updateClosestSeen(state, state.basePosition, magmaGlobal.pos, state.closestMagmaPos)
      if isAdjacent(agent.pos, magmaGlobal.pos):
        return (true, controller.useAt(env, agent, agentId, state, magmaGlobal.pos))
      return (true, controller.moveTo(env, agent, agentId, state, magmaGlobal.pos))
    return (true, controller.moveNextSearch(env, agent, agentId, state))

  if agent.unitClass == UnitVillager:
    let (didGold, actGold) = controller.ensureGold(env, agent, agentId, state)
    if didGold: return (true, actGold)

  (false, 0'u16)

proc decideRoleFromCatalog(controller: Controller, env: Environment, agent: Thing,
                           agentId: int, state: var AgentState): uint16 =
  if state.role == Gatherer:
    updateGathererTask(controller, env, agent, state)
  var roleId = state.roleId
  if roleId < 0 or roleId >= scriptedState.catalog.roles.len:
    roleId = roleIdForAgent(controller, agentId)
  # Dynamic defense priority: Builders use threat-aware option ordering
  if state.role == Builder and isBuilderUnderThreat(env, agent):
    return runOptions(controller, env, agent, agentId, state, BuilderOptionsThreat)
  let options = roleOptionsFor(roleId, controller.rng)
  if options.len == 0:
    return 0'u16
  return runOptions(controller, env, agent, agentId, state, options)

proc decideAction*(controller: Controller, env: Environment, agentId: int): uint16 =
  let agent = env.agents[agentId]

  # Skip inactive agents
  if not isAgentAlive(env, agent):
    setAuditBranch(BranchInactive)
    return encodeAction(0'u16, 0'u16)

  initScriptedState(controller)

  # Initialize agent role based on game phase
  # Early: economy-heavy (3G/2B/1F), Mid: balanced (2G/2B/2F), Late: military (2G/1B/3F)
  if not controller.agentsInitialized[agentId]:
    let slot = agentId mod MapAgentsPerTeam
    let gameProgress = if env.config.maxSteps > 0:
      env.currentStep.float / env.config.maxSteps.float
    else:
      0.0
    # Determine gatherer/builder/fighter slot counts based on phase
    let (nGatherers, nBuilders) = if gameProgress < EarlyGameThreshold:
      (EarlyGameGatherers, EarlyGameBuilders)
    elif gameProgress < LateGameThreshold:
      (MidGameGatherers, MidGameBuilders)
    else:
      (LateGameGatherers, LateGameBuilders)
    let slotMod = slot mod 6
    var role =
      if slotMod < nGatherers: Gatherer
      elif slotMod < nGatherers + nBuilders: Builder
      else: Fighter

    # Preserve any patrol state that was set before initialization
    let existingState = controller.agents[agentId]
    var initState = AgentState(
      role: role,
      roleId: -1,
      activeOptionId: -1,
      fighterEnemyAgentId: -1,
      fighterEnemyStep: -1,
      spiralClockwise: (agentId mod 2) == 0,
      basePosition: agent.pos,
      lastSearchPosition: agent.pos,
      lastPosition: agent.pos,
      escapeDirection: ivec2(0, -1),
      blockedMoveDir: -1,
      cachedWaterPos: ivec2(-1, -1),
      buildTarget: ivec2(-1, -1),
      buildStand: ivec2(-1, -1),
      buildIndex: -1,
      plannedTarget: ivec2(-1, -1),
      pathBlockedTarget: ivec2(-1, -1),
      # Preserve patrol and attack-move state
      patrolPoint1: existingState.patrolPoint1,
      patrolPoint2: existingState.patrolPoint2,
      patrolToSecondPoint: existingState.patrolToSecondPoint,
      patrolActive: existingState.patrolActive,
      # Preserve attack-move target (-1,-1 = inactive); normalize (0,<=0) to (-1,-1)
      attackMoveTarget: if existingState.attackMoveTarget.x == 0 and existingState.attackMoveTarget.y <= 0:
                          ivec2(-1, -1)
                        else:
                          existingState.attackMoveTarget
    )
    clearCachedPositions(initState)
    if ScriptedTempleAssignEnabled and scriptedState.pendingHybridRoles[agentId] >= 0:
      let pending = scriptedState.pendingHybridRoles[agentId]
      scriptedState.pendingHybridRoles[agentId] = -1
      setAgentRole(agentId, initState, pending)
    elif role == Scripted:
      assignScriptedRole(controller, agentId, initState)
    else:
      var roleId = scriptedState.coreRoleIds[role]
      if roleId < 0:
        roleId = scriptedState.coreRoleIds[Gatherer]
      setAgentRole(agentId, initState, roleId)
    controller.agents[agentId] = initState
    controller.agentsInitialized[agentId] = true

    # Fighters need Defensive stance to engage enemies (villagers default to NoAttack)
    if role == Fighter and agent.stance == StanceNoAttack:
      agent.stance = StanceDefensive

  var state = controller.agents[agentId]

  # Apply any pending stance modification
  if state.stanceModified:
    agent.stance = state.pendingStance
    state.stanceModified = false
    controller.agents[agentId] = state

  # Get team info and difficulty settings
  let currentStep = env.currentStep.int32
  let teamId = getTeamId(agent)
  let diffConfig = controller.getDifficulty(teamId)

  # Decision delay based on difficulty - simulates "thinking time"
  # Lower difficulty = more delays, making AI slower to react
  if controller.shouldApplyDecisionDelay(teamId):
    setAuditBranch(BranchDecisionDelay)
    return saveStateAndReturn(controller, agentId, state, encodeAction(0'u16, 0'u16))

  # Handle stopped state - agent remains idle until new command or threshold expires
  if state.stoppedActive:
    # Initialize deferred expiry if using sentinel (-1)
    if state.stoppedUntilStep < 0:
      state.stoppedUntilStep = currentStep + StopIdleSteps
      controller.agents[agentId] = state
    if currentStep >= state.stoppedUntilStep:
      # Threshold expired, resume normal behavior
      state.stoppedActive = false
      state.stoppedUntilStep = 0
      controller.agents[agentId] = state
    else:
      # Still stopped - return NOOP
      setAuditBranch(BranchStopped)
      return saveStateAndReturn(controller, agentId, state, encodeAction(0'u16, 0'u16))

  # Update shared threat map with what this agent can see
  # Only if threat response is enabled for this difficulty level
  # Staggered: only update 1/5 of agents per step to reduce overhead (5x speedup)
  # Decay also staggered to every ThreatMapStaggerInterval steps for additional speedup
  if diffConfig.threatResponseEnabled and teamId >= 0 and teamId < MapRoomObjectsTeams:
    if currentStep mod ThreatMapStaggerInterval == 0 and
        controller.threatMaps[teamId].lastUpdateStep != currentStep:
      controller.decayThreats(teamId, currentStep)
    if agent.agentId mod ThreatMapStaggerInterval == currentStep mod ThreatMapStaggerInterval:
      controller.updateThreatMapFromVision(env, agent, currentStep)

  # AI auto-trigger Town Bell when enemy army detected near base
  # Staggered: check once per team per TownBellAutoCheckInterval steps
  if teamId >= 0 and teamId < MapRoomObjectsTeams and
      diffConfig.threatResponseEnabled and
      currentStep - controller.townBellAutoCheckStep[teamId] >= TownBellAutoCheckInterval:
    controller.townBellAutoCheckStep[teamId] = currentStep
    # Count enemies near any of the team's TownCenters
    var enemyCount = 0
    block countEnemies:
      for tc in env.thingsByKind[TownCenter]:
        if tc.teamId != teamId:
          continue
        let (cx, cy) = cellCoords(tc.pos)
        let cellRadius = distToCellRadius16(TownBellAutoTriggerRadius)
        for ddx in -cellRadius .. cellRadius:
          for ddy in -cellRadius .. cellRadius:
            let nx = cx + ddx
            let ny = cy + ddy
            if nx < 0 or nx >= SpatialCellsX or ny < 0 or ny >= SpatialCellsY:
              continue
            for other in env.spatialIndex.kindCells[Agent][nx][ny]:
              if other.isNil or not isAgentAlive(env, other):
                continue
              if getTeamId(other) != teamId and getTeamId(other) >= 0:
                let dist = int(chebyshevDist(tc.pos, other.pos))
                if dist <= TownBellAutoTriggerRadius:
                  inc enemyCount
                  if enemyCount >= TownBellAutoTriggerCount:
                    break countEnemies
    if enemyCount >= TownBellAutoTriggerCount and not env.townBellActive[teamId]:
      env.townBellActive[teamId] = true
    elif enemyCount == 0 and env.townBellActive[teamId]:
      # Auto-deactivate when threat passes: ungarrison all team buildings
      env.townBellActive[teamId] = false
      for kind in [TownCenter, Castle, GuardTower, House]:
        for building in env.thingsByKind[kind]:
          if building.teamId == teamId and building.garrisonedUnits.len > 0:
            discard env.ungarrisonAllUnits(building)

  # Auto-enable scout mode for scout-line units (Scout, LightCavalry, Hussar)
  # Scouts are trained at Stables and should automatically enter scouting behavior
  if agent.unitClass in {UnitScout, UnitLightCavalry, UnitHussar} and not state.scoutActive:
    state.scoutActive = true
    state.scoutExploreRadius = ObservationRadius.int32 + 5
    state.scoutLastEnemySeenStep = -100  # Long ago

  # Auto-reassign monks to Fighter role for monk-specific behavior (relic collection,
  # healing, conversion). Monks trained from Gatherer/Builder-role villagers would
  # otherwise lack FighterMonk behavior entirely.
  if agent.unitClass == UnitMonk and state.role != Fighter:
    let roleId = scriptedState.coreRoleIds[Fighter]
    setAgentRole(agentId, state, roleId)

  if agent.unitClass == UnitGoblin:
    # Count relics held by goblins using thingsByKind[Agent] filtered for goblins
    # This is still O(agents_in_nearby_cells) but avoids scanning ALL 1000 agents
    var totalRelicsHeld = 0
    for other in env.thingsByKind[Agent]:
      if other.unitClass == UnitGoblin and isAgentAlive(env, other):
        totalRelicsHeld += other.inventoryRelic
    if totalRelicsHeld >= MapRoomObjectsRelics and env.thingsByKind[Relic].len == 0:
      setAuditBranch(BranchGoblinRelic)
      return saveStateAndReturn(controller, agentId, state, encodeAction(0'u16, 0'u16))

    # Use spatial index to find nearest non-goblin threat instead of scanning all agents
    var nearestThreat: Thing = nil
    var threatDist = int.high
    block findThreat:
      let searchRadius = GoblinAvoidRadius + 5
      let (cx, cy) = cellCoords(agent.pos)
      let clampedMax = min(searchRadius, max(SpatialCellsX, SpatialCellsY) * SpatialCellSize)
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
            if not isAgentAlive(env, other) or other.unitClass == UnitGoblin:
              continue
            let dist = int(chebyshevDist(agent.pos, other.pos))
            if dist < threatDist:
              threatDist = dist
              nearestThreat = other

    if not isNil(nearestThreat) and threatDist <= GoblinAvoidRadius:
      let dx = signi(agent.pos.x - nearestThreat.pos.x)
      let dy = signi(agent.pos.y - nearestThreat.pos.y)
      let awayTarget = clampToPlayable(agent.pos + ivec2(dx * 6, dy * 6))
      setAuditBranch(BranchGoblinAvoid)
      return controller.moveTo(env, agent, agentId, state, awayTarget)

    let relic = env.findNearestThingSpiral(state, Relic)
    if not isNil(relic):
      setAuditBranch(BranchGoblinSearch)
      return actOrMove(controller, env, agent, agentId, state, relic.pos, 3'u16)

    setAuditBranch(BranchGoblinSearch)
    return controller.moveNextSearch(env, agent, agentId, state)

  # --- Simple bail-out to avoid getting stuck/oscillation ---
  # Update recent positions history (ring buffer size 12)
  state.recentPositions[state.recentPosIndex] = agent.pos
  state.recentPosIndex = (state.recentPosIndex + 1) mod 12
  if state.recentPosCount < 12:
    inc state.recentPosCount

  proc recentAt(offset: int): IVec2 =
    let idx = (state.recentPosIndex - 1 - offset + 12 * 12) mod 12
    state.recentPositions[idx]

  if state.blockedMoveSteps > 0:
    dec state.blockedMoveSteps
    if state.blockedMoveSteps <= 0:
      state.blockedMoveDir = -1

  if state.lastActionVerb == 1 and state.recentPosCount >= 2:
    if recentAt(1) == agent.pos and state.lastActionArg >= 0 and state.lastActionArg <= 7:
      state.blockedMoveDir = state.lastActionArg
      state.blockedMoveSteps = 4
      state.plannedPath.setLen(0)
      state.pathBlockedTarget = ivec2(-1, -1)

  # Enter escape mode if stuck in 1-3 tiles for 10+ steps
  let stuckWindow = if state.role == Builder: 6 else: 10
  if not state.escapeMode and state.recentPosCount >= stuckWindow:
    var uniqueCount = 0
    var unique: array[3, IVec2]
    for i in 0 ..< stuckWindow:
      let p = recentAt(i)
      var seen = false
      for j in 0 ..< uniqueCount:
        if unique[j] == p:
          seen = true
          break
      if not seen:
        if uniqueCount < 3:
          unique[uniqueCount] = p
          inc uniqueCount
        else:
          uniqueCount = 4
          break
    if uniqueCount <= 3:
      state.plannedTarget = ivec2(-1, -1)
      state.plannedPath.setLen(0)
      state.plannedPathIndex = 0
      state.pathBlockedTarget = ivec2(-1, -1)
      clearCachedPositions(state)
      state.escapeMode = true
      state.escapeStepsRemaining = 10
      state.recentPosCount = 0
      state.recentPosIndex = 0
      # Choose an escape direction: prefer any empty cardinal, shuffled
      var dirs = CardinalOffsets
      for i in countdown(dirs.len - 1, 1):
        let j = randIntInclusive(controller.rng, 0, i)
        let tmp = dirs[i]
        dirs[i] = dirs[j]
        dirs[j] = tmp
      var chosen = ivec2(0, -1)
      for d in dirs:
        if isPassable(env, agent, agent.pos + d):
          chosen = d
          break
      state.escapeDirection = chosen

  # If in escape mode, try to move in escape direction for a few steps
  if state.escapeMode and state.escapeStepsRemaining > 0:
    let tryDirs = [state.escapeDirection,
                   ivec2(state.escapeDirection.y, -state.escapeDirection.x),  # perpendicular 1
                   ivec2(-state.escapeDirection.y, state.escapeDirection.x),  # perpendicular 2
                   ivec2(-state.escapeDirection.x, -state.escapeDirection.y)] # opposite
    for d in tryDirs:
      let np = agent.pos + d
      if isPassable(env, agent, np):
        dec state.escapeStepsRemaining
        if state.escapeStepsRemaining <= 0:
          state.escapeMode = false
        state.lastPosition = agent.pos
        setAuditBranch(BranchEscape)
        return saveStateAndReturn(controller, agentId, state, encodeAction(1'u16, vecToOrientation(d).uint8))
    # If all blocked, drop out of escape for this tick
    state.escapeMode = false

  # From here on, ensure lastPosition is updated this tick regardless of branch
  state.lastPosition = agent.pos
  # Anchor spiral search around home altar when possible (common base-centric search)
  if agent.homeAltar.x >= 0:
    state.basePosition = agent.homeAltar
  else:
    state.basePosition = agent.pos

  let attackDir = findAttackOpportunity(env, agent)
  if attackDir >= 0:
    setAuditBranch(BranchAttackOpportunity)
    return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, attackDir.uint8))

  # Patrol behavior - applies to all roles when patrol is active
  if state.patrolActive and state.patrolPoint1.x >= 0 and state.patrolPoint2.x >= 0:
    # Patrol always attacks enemies (AoE2-style: patrol overrides stance for engagement)
    let patrolAttackDir = findAttackOpportunity(env, agent, ignoreStance = true)
    if patrolAttackDir >= 0:
      setAuditBranch(BranchPatrolChase)
      return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, patrolAttackDir.uint8))
    # Chase nearby enemies (patrol overrides stance for chasing too)
    let enemy = fighterFindNearbyEnemy(controller, env, agent, state)
    if not isNil(enemy):
      setAuditBranch(BranchPatrolChase)
      return controller.moveTo(env, agent, agentId, state, enemy.pos)

    # Determine current patrol target
    let target = if state.patrolToSecondPoint: state.patrolPoint2 else: state.patrolPoint1

    # Check if we've reached the current waypoint (within threshold of 2 tiles)
    let distToTarget = int(chebyshevDist(agent.pos, target))
    if distToTarget <= 2:
      # Switch direction
      state.patrolToSecondPoint = not state.patrolToSecondPoint
      # Get the new target after switching
      let newTarget = if state.patrolToSecondPoint: state.patrolPoint2 else: state.patrolPoint1
      setAuditBranch(BranchPatrolMove)
      return controller.moveTo(env, agent, agentId, state, newTarget)

    # Move toward current waypoint
    setAuditBranch(BranchPatrolMove)
    return controller.moveTo(env, agent, agentId, state, target)

  # Rally point behavior - newly trained units move toward their rally destination
  # Units wait briefly at rally point for others to arrive, creating natural grouping
  if agent.rallyTarget.x >= 0:
    if chebyshevDist(agent.pos, agent.rallyTarget) <= 2'i32:
      # At rally point - wait for nearby allies to group up before clearing
      let teamId = getTeamId(agent)
      let nearbyAllies = countAlliesInRangeSpatial(env, agent.pos, teamId, 4, agent.agentId)
      let stepsAtRally = env.currentStep - state.rallyArrivalStep
      if state.rallyArrivalStep <= 0:
        state.rallyArrivalStep = env.currentStep
      # Clear rally after grouping period or if enough allies nearby
      if stepsAtRally >= RallyWaitSteps or nearbyAllies >= RallyMinGroupSize:
        agent.rallyTarget = ivec2(-1, -1)
        state.rallyArrivalStep = 0
      else:
        # Hold position at rally point while waiting
        setAuditBranch(BranchRallyPoint)
        return saveStateAndReturn(controller, agentId, state, 0'u16)
    else:
      state.rallyArrivalStep = 0
      setAuditBranch(BranchRallyPoint)
      return controller.moveTo(env, agent, agentId, state, agent.rallyTarget)

  # Attack-move behavior - applies to all roles when attack-move target is set
  if state.attackMoveTarget.x >= 0:
    # Check if we've reached the destination (within 1 tile)
    if chebyshevDist(agent.pos, state.attackMoveTarget) <= 1'i32:
      # Clear the attack-move target - we've arrived
      state.attackMoveTarget = ivec2(-1, -1)
      controller.agents[agentId].attackMoveTarget = ivec2(-1, -1)
      # Check for queued commands (shift-queue) and execute next if present
      if state.commandQueueCount > 0:
        controller.executeQueuedCommand(agentId, agent.pos)
        # Re-read state to get the command that was just set
        state = controller.agents[agentId]
        # Continue processing with updated state
    else:
      # Attack-move always engages enemies (overrides stance, like patrol)
      let amAttackDir = findAttackOpportunity(env, agent, ignoreStance = true)
      if amAttackDir >= 0:
        setAuditBranch(BranchAttackMoveEngage)
        return saveStateAndReturn(controller, agentId, state, encodeAction(2'u16, amAttackDir.uint8))
      # Check for nearby enemies to chase toward
      let enemy = fighterFindNearbyEnemy(controller, env, agent, state)
      if not isNil(enemy):
        let enemyDist = int(chebyshevDist(agent.pos, enemy.pos))
        if enemyDist <= 8:  # Attack-move detection radius
          # When outnumbered, wait for allies to approach before engaging
          if shouldWaitForAllies(env, agent):
            setAuditBranch(BranchAttackMoveAdvance)
            return saveStateAndReturn(controller, agentId, state, 0'u16)
          # Enemy found - engage!
          setAuditBranch(BranchAttackMoveEngage)
          return actOrMove(controller, env, agent, agentId, state, enemy.pos, 2'u16)
      # No enemy nearby - continue moving toward destination
      setAuditBranch(BranchAttackMoveAdvance)
      return controller.moveTo(env, agent, agentId, state, state.attackMoveTarget)

  # Settler migration: settlers move to new town site, ignoring normal role behaviors.
  # This check is high priority - settlers skip gather/build/fight but keep threat response.
  if agent.isSettler and agent.settlerTarget.x >= 0 and not agent.settlerArrived:
    let settlerAction = optSettlerMigrate(controller, env, agent, agentId, state)
    if settlerAction != 0'u16:
      setAuditBranch(BranchSettlerMigrate)
      return saveStateAndReturn(controller, agentId, state, settlerAction)
    # If optSettlerMigrate returned 0 (arrived or aborted), fall through to normal behavior

  # Global: prioritize getting hearts to 10 via gold -> magma -> altar (gatherers only).
  if state.role == Gatherer:
    let (didHearts, heartsAct) = tryPrioritizeHearts(controller, env, agent, agentId, state)
    if didHearts:
      setAuditBranch(BranchHearts)
      return heartsAct

  # Global: keep population cap ahead of current population (gatherers only).
  if state.role == Gatherer and agent.unitClass == UnitVillager:
    let teamId = getTeamId(agent)
    if needsPopCapHouse(controller, env, teamId):
      let houseKey = thingItem("House")
      let costs = buildCostsForKey(houseKey)
      var requiredWood = 0
      if costs.len > 0:
        for cost in costs:
          if stockpileResourceForItem(cost.key) == ResourceWood:
            requiredWood += cost.count
      if requiredWood > 0 and
          env.stockpileCount(teamId, ResourceWood) + agent.inventoryWood < requiredWood:
        let (didWood, actWood) = controller.ensureWood(env, agent, agentId, state)
        if didWood:
          setAuditBranch(BranchPopCapWood)
          return actWood
      if env.canAffordBuild(agent, houseKey):
        let (didHouse, houseAct) =
          tryBuildHouseForPopCap(controller, env, agent, agentId, state, teamId, state.basePosition)
        if didHouse:
          setAuditBranch(BranchPopCapBuild)
          return houseAct

  # Role-based decision making (unified priority lists)
  setAuditBranch(BranchRoleCatalog)
  let action = decideRoleFromCatalog(controller, env, agent, agentId, state)
  return saveStateAndReturn(controller, agentId, state, action)

proc isAgentReassignable(state: AgentState, agent: Thing): bool =
  ## Check if an agent is idle enough to be reassigned to a different role.
  ## Agents in active combat, patrol, attack-move, etc. should not be disrupted.
  if state.patrolActive: return false
  if state.attackMoveTarget.x >= 0: return false
  if state.stoppedActive: return false
  if state.scoutActive: return false
  if state.holdPositionActive: return false
  if state.followActive: return false
  if state.guardActive: return false
  if state.fighterEnemyAgentId >= 0: return false
  if agent.isSettler and agent.settlerTarget.x >= 0: return false
  # Don't reassign non-villager units (military units keep their role)
  if agent.unitClass != UnitVillager: return false
  true

proc reassignRolesForPhase(controller: Controller, env: Environment) =
  ## Periodically re-evaluate role assignments based on game phase.
  ## Reassigns idle agents from over-represented roles to under-represented ones.
  let gameProgress = if env.config.maxSteps > 0:
    env.currentStep.float / env.config.maxSteps.float
  else:
    0.0
  let (targetGatherers, targetBuilders) = if gameProgress < EarlyGameThreshold:
    (EarlyGameGatherers, EarlyGameBuilders)
  elif gameProgress < LateGameThreshold:
    (MidGameGatherers, MidGameBuilders)
  else:
    (LateGameGatherers, LateGameBuilders)
  for teamId in 0 ..< MapRoomObjectsTeams:
    # Count current role distribution for this team
    var counts: array[AgentRole, int]
    var reassignable: array[AgentRole, seq[int]]
    for role in AgentRole:
      reassignable[role] = @[]

    let teamStart = teamId * MapAgentsPerTeam
    let teamEnd = teamStart + MapAgentsPerTeam
    for agentId in teamStart ..< teamEnd:
      if not controller.agentsInitialized[agentId]: continue
      let agent = env.agents[agentId]
      if not isAgentAlive(env, agent): continue
      let state = controller.agents[agentId]
      let role = state.role
      counts[role] += 1
      if isAgentReassignable(state, agent):
        reassignable[role].add agentId

    # Compute target counts proportional to alive agents (scale from 6-slot ratio)
    let totalAlive = counts[Gatherer] + counts[Builder] + counts[Fighter]
    if totalAlive < 2: continue  # Too few agents to reassign

    let wantGatherers = (totalAlive * targetGatherers + 3) div 6
    let wantBuilders = (totalAlive * targetBuilders + 3) div 6
    let wantFighters = totalAlive - wantGatherers - wantBuilders

    # Identify surplus and deficit roles
    var surplus: array[AgentRole, int]
    surplus[Gatherer] = counts[Gatherer] - wantGatherers
    surplus[Builder] = counts[Builder] - wantBuilders
    surplus[Fighter] = counts[Fighter] - wantFighters

    # Reassign from roles with surplus to roles with deficit
    for targetRole in [Fighter, Builder, Gatherer]:
      if surplus[targetRole] >= 0: continue  # No deficit
      var needed = -surplus[targetRole]
      for sourceRole in [Gatherer, Builder, Fighter]:
        if needed <= 0: break
        if surplus[sourceRole] <= 0: continue  # No surplus
        let available = min(needed, min(surplus[sourceRole], reassignable[sourceRole].len))
        for i in 0 ..< available:
          let agentId = reassignable[sourceRole][reassignable[sourceRole].len - 1 - i]
          let roleId = scriptedState.coreRoleIds[targetRole]
          if roleId < 0: continue
          setAgentRole(agentId, controller.agents[agentId], roleId)
          # Set stance for new fighters
          if targetRole == Fighter:
            let agent = env.agents[agentId]
            if agent.stance == StanceNoAttack:
              agent.stance = StanceDefensive
        surplus[sourceRole] -= available
        surplus[targetRole] += available
        needed -= available

# Compatibility function for updateController
proc updateController*(controller: Controller, env: Environment) =
  initScriptedState(controller)
  # Clean up expired coordination requests and resource reservations
  clearExpiredRequests(env.currentStep)
  clearExpiredReservations(env)
  # Update economy tracking for all teams
  for teamId in 0 ..< MapRoomObjectsTeams:
    updateEconomy(controller, env, teamId)
  if scriptedState.lastEpisodeStep >= 0 and env.currentStep < scriptedState.lastEpisodeStep:
    for i in 0 ..< MapAgents:
      controller.agentsInitialized[i] = false
    controller.buildingCountsStep = -1
    resetScriptedAssignments(scriptedState)
    scriptedState.scoredAtStep = false
    scriptedState.lastEpisodeStep = -1
    # Clear shared threat maps and town split cooldowns on episode reset
    for teamId in 0 ..< MapRoomObjectsTeams:
      controller.clearThreatMap(teamId)
      controller.townSplitLastStep[teamId] = 0
    # Reset economy state on episode reset
    resetEconomy()
    # Clear resource reservations on episode reset
    for teamId in 0 ..< MapRoomObjectsTeams:
      teamReservations[teamId] = ReservationState()
  if EvolutionEnabled:
    if not scriptedState.scoredAtStep and env.currentStep >= ScriptedScoreStep:
      applyScriptedScoring(controller, env)
      scriptedState.scoredAtStep = true
  if ScriptedTempleAssignEnabled:
    processTempleHybridRequests(controller, env)
  # Periodically evaluate tribute between allied teams
  if env.currentStep mod TributeCheckInterval == 0:
    for teamId in 0 ..< MapRoomObjectsTeams:
      evaluateTribute(env, teamId)
  # Periodically re-evaluate role assignments based on game phase
  if env.currentStep > 0 and env.currentStep mod RoleReassignInterval == 0:
    reassignRolesForPhase(controller, env)
  # Update adaptive difficulty for teams that have it enabled
  controller.updateAdaptiveDifficulty(env)
  # Check for town splits (AI settlement expansion)
  controller.checkAndTriggerTownSplit(env)
  # Check if settler groups have arrived and should found new towns
  controller.checkSettlerArrivals(env)
  scriptedState.lastEpisodeStep = env.currentStep
