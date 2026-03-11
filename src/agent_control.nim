## Unified Action Interface for Agent Control
## Supports both external neural network control and built-in AI control
## Controller type is specified when creating the environment

import std/os, std/strutils

import scripted/ai_defaults
export ai_defaults

import formations
export formations

when defined(stepTiming):
  import std/monotimes
  import std/algorithm

  # AI decision timing state
  let aiTimingEnabled = getEnv("TV_AI_TIMING", "0") != "0"
  let aiTimingInterval = block:
    let raw = getEnv("TV_AI_TIMING_INTERVAL", "100")
    try: parseInt(raw) except: 100
  let aiTimingTopN = block:
    let raw = getEnv("TV_AI_TIMING_TOP_N", "10")
    try: parseInt(raw) except: 10

  var aiTimingCumTotal: float64 = 0.0
  var aiTimingCumMax: float64 = 0.0
  var aiTimingStepCount: int = 0
  var aiTimingAgentCum: array[MapAgents, float64]
  var aiTimingAgentMax: array[MapAgents, float64]
  var aiTimingAgentCount: array[MapAgents, int]

  proc aiMsBetween(a, b: MonoTime): float64 =
    (b.ticks - a.ticks).float64 / 1_000_000.0

  proc resetAiTimingCounters() =
    aiTimingCumTotal = 0.0
    aiTimingCumMax = 0.0
    aiTimingStepCount = 0
    for i in 0 ..< MapAgents:
      aiTimingAgentCum[i] = 0.0
      aiTimingAgentMax[i] = 0.0
      aiTimingAgentCount[i] = 0

  proc printAiTimingReport(currentStep: int) =
    if aiTimingStepCount == 0:
      return
    let n = aiTimingStepCount.float64

    # Collect agents with timing data and sort by cumulative time
    type AgentTimingEntry = tuple[agentId: int, cumMs: float64, maxMs: float64, count: int]
    var entries: seq[AgentTimingEntry] = @[]
    for i in 0 ..< MapAgents:
      if aiTimingAgentCount[i] > 0:
        entries.add((agentId: i, cumMs: aiTimingAgentCum[i], maxMs: aiTimingAgentMax[i], count: aiTimingAgentCount[i]))

    # Sort by cumulative time descending
    entries.sort(proc(a, b: AgentTimingEntry): int =
      if a.cumMs > b.cumMs: -1
      elif a.cumMs < b.cumMs: 1
      else: 0
    )

    echo ""
    echo "=== AI Decision Timing Report (steps ", currentStep - aiTimingStepCount + 1, "-", currentStep, ", n=", aiTimingStepCount, ") ==="
    echo "Total AI decision time: avg=", formatFloat(aiTimingCumTotal / n, ffDecimal, 4), "ms, max=", formatFloat(aiTimingCumMax, ffDecimal, 4), "ms"
    echo ""
    echo "Top ", aiTimingTopN, " slowest agents (by cumulative time):"
    echo align("Agent", 8), " | ", align("Avg ms", 10), " | ", align("Max ms", 10), " | ", align("Decisions", 10)
    echo repeat("-", 8), "-+-", repeat("-", 10), "-+-", repeat("-", 10), "-+-", repeat("-", 10)

    let showCount = min(aiTimingTopN, entries.len)
    for i in 0 ..< showCount:
      let e = entries[i]
      let avgMs = e.cumMs / e.count.float64
      echo align($e.agentId, 8), " | ",
           align(formatFloat(avgMs, ffDecimal, 4), 10), " | ",
           align(formatFloat(e.maxMs, ffDecimal, 4), 10), " | ",
           align($e.count, 10)
    echo ""
    resetAiTimingCounters()

const
  ActionsFile = "actions.tmp"

# Helper template to reduce nil-check boilerplate for AI controller access
template withBuiltinAI(body: untyped) =
  ## Execute body only if globalController has a BuiltinAI controller.
  ## Used to guard access to aiController methods. Also matches HybridAI.
  if not isNil(globalController) and globalController.controllerType in {BuiltinAI, HybridAI}:
    body

type
  ControllerType* = enum
    BuiltinAI,      # Use built-in Nim AI controller
    ExternalNN,     # Use external neural network (Python)
    HybridAI        # BuiltinAI runs, but Python can override non-NOOP actions

  AgentController* = ref object
    controllerType*: ControllerType
    # Built-in AI controller (when using BuiltinAI)
    aiController*: Controller
    # External action callback (when using ExternalNN)
    externalActionCallback*: proc(): array[MapAgents, uint16]

# Global agent controller instance
var globalController*: AgentController

proc initGlobalController*(controllerType: ControllerType, seed: int = int(nowSeconds() * 1000)) =
  ## Initialize the global controller with specified type
  initAuditLog()
  case controllerType:
  of BuiltinAI:
    globalController = AgentController(
      controllerType: BuiltinAI,
      aiController: newController(seed),
      externalActionCallback: nil
    )
  of ExternalNN:
    # External callback will be set later via setExternalActionCallback
    globalController = AgentController(
      controllerType: ExternalNN,
      aiController: nil,
      externalActionCallback: nil
    )
    # Start automatic play mode for external controller
    play = true
  of HybridAI:
    # BuiltinAI drives behavior, but external actions can override non-NOOP
    globalController = AgentController(
      controllerType: HybridAI,
      aiController: newController(seed),
      externalActionCallback: nil
    )
    play = true

proc setExternalActionCallback*(callback: proc(): array[MapAgents, uint16]) =
  ## Set the external action callback for neural network control
  if not isNil(globalController) and globalController.controllerType in {ExternalNN, HybridAI}:
    globalController.externalActionCallback = callback

proc getActions*(env: Environment): array[MapAgents, uint16] =
  ## Get actions for all agents using the configured controller
  case globalController.controllerType
  of BuiltinAI:
    var actions: array[MapAgents, uint16]
    let controller = globalController.aiController

    when defined(stepTiming):
      var tLoopStart, tAgentStart, tAgentEnd: MonoTime
      var tLoopTotalMs: float64 = 0.0
      if aiTimingEnabled:
        tLoopStart = getMonoTime()

    for i in 0 ..< env.agents.len:
      when defined(stepTiming):
        if aiTimingEnabled:
          tAgentStart = getMonoTime()

      setAuditBranch(BranchInactive)
      actions[i] = controller.decideAction(env, i)

      when defined(stepTiming):
        if aiTimingEnabled:
          tAgentEnd = getMonoTime()
          let agentMs = aiMsBetween(tAgentStart, tAgentEnd)
          aiTimingAgentCum[i] += agentMs
          if agentMs > aiTimingAgentMax[i]:
            aiTimingAgentMax[i] = agentMs
          inc aiTimingAgentCount[i]

      when defined(aiAudit):
        let agent = env.agents[i]
        let teamId = if not agent.isNil: getTeamId(agent) else: -1
        let role = if controller.agentsInitialized[i]: controller.agents[i].role else: Gatherer
        recordAuditDecision(i, teamId, role, actions[i])

    when defined(stepTiming):
      if aiTimingEnabled:
        let tLoopEnd = getMonoTime()
        tLoopTotalMs = aiMsBetween(tLoopStart, tLoopEnd)
        aiTimingCumTotal += tLoopTotalMs
        if tLoopTotalMs > aiTimingCumMax:
          aiTimingCumMax = tLoopTotalMs
        inc aiTimingStepCount
        if aiTimingStepCount >= aiTimingInterval:
          printAiTimingReport(env.currentStep.int)

    controller.updateController(env)
    printAuditSummary(env.currentStep.int)
    return actions
  of ExternalNN:
    if not isNil(globalController.externalActionCallback):
      return globalController.externalActionCallback()

    if fileExists(ActionsFile):
      try:
        let lines = readFile(ActionsFile).replace("\r", "").replace("\n\n", "\n").split("\n")
        if lines.len >= MapAgents:
          var fileActions: array[MapAgents, uint16]
          for i in 0 ..< MapAgents:
            let parts = lines[i].split(',')
            if parts.len >= 2:
              fileActions[i] = encodeAction(parseInt(parts[0]).uint16, parseInt(parts[1]).uint16)
            elif parts.len == 1 and parts[0].len > 0:
              fileActions[i] = parseInt(parts[0]).uint16

          discard tryRemoveFile(ActionsFile)

          return fileActions
      except CatchableError:
        discard

    echo "❌ FATAL ERROR: ExternalNN controller configured but no callback or actions file found!"
    echo "Python environment must call setExternalActionCallback() or provide " & ActionsFile & "!"
    raise newException(ValueError, "ExternalNN controller has no actions - Python communication failed!")
  of HybridAI:
    # BuiltinAI drives all behavior, but external actions override non-NOOP
    var actions: array[MapAgents, uint16]
    let controller = globalController.aiController

    # Get builtin AI actions first
    for i in 0 ..< env.agents.len:
      setAuditBranch(BranchInactive)
      actions[i] = controller.decideAction(env, i)

    controller.updateController(env)
    printAuditSummary(env.currentStep.int)
    return actions

# Attack-Move API
# These functions allow external code to set attack-move targets for agents.
# Attack-move: unit moves toward destination, attacking any enemies encountered along the way.

proc setAgentAttackMoveTarget*(agentId: int, target: IVec2) =
  ## Set an attack-move target for an agent.
  ## The agent will move toward the target while engaging enemies along the way.
  ## Requires BuiltinAI controller. Clears any stopped state.
  withBuiltinAI:
    globalController.aiController.clearAgentStop(agentId)
    globalController.aiController.setAttackMoveTarget(agentId, target)

proc setAgentAttackMoveTargetXY*(agentId: int, x, y: int32) =
  ## Set an attack-move target for an agent using x,y coordinates.
  setAgentAttackMoveTarget(agentId, ivec2(x, y))

proc clearAgentAttackMoveTarget*(agentId: int) =
  ## Clear the attack-move target for an agent, stopping attack-move behavior.
  withBuiltinAI:
    globalController.aiController.clearAttackMoveTarget(agentId)

proc getAgentAttackMoveTarget*(agentId: int): IVec2 =
  ## Get the current attack-move target for an agent.
  ## Returns (-1, -1) if no attack-move is active.
  withBuiltinAI:
    return globalController.aiController.getAttackMoveTarget(agentId)
  ivec2(-1, -1)

proc isAgentAttackMoveActive*(agentId: int): bool =
  ## Check if an agent currently has an active attack-move target.
  let target = getAgentAttackMoveTarget(agentId)
  target.x >= 0

# Patrol API
# These functions allow external code to set patrol behavior for agents.
# Patrol: unit walks back and forth between two waypoints, attacking enemies encountered.

proc setAgentPatrol*(agentId: int, point1, point2: IVec2) =
  ## Set patrol waypoints for an agent. Enables patrol mode.
  ## The agent will walk between the two points, attacking any enemies encountered.
  ## Requires BuiltinAI controller. Clears any stopped state.
  withBuiltinAI:
    globalController.aiController.clearAgentStop(agentId)
    globalController.aiController.setPatrol(agentId, point1, point2)

proc setAgentPatrolXY*(agentId: int, x1, y1, x2, y2: int32) =
  ## Set patrol waypoints for an agent using x,y coordinates.
  setAgentPatrol(agentId, ivec2(x1, y1), ivec2(x2, y2))

proc clearAgentPatrol*(agentId: int) =
  ## Clear the patrol for an agent, disabling patrol mode.
  withBuiltinAI:
    globalController.aiController.clearPatrol(agentId)

proc getAgentPatrolTarget*(agentId: int): IVec2 =
  ## Get the current patrol target waypoint for an agent.
  ## Returns (-1, -1) if no patrol is active.
  withBuiltinAI:
    return globalController.aiController.getPatrolTarget(agentId)
  ivec2(-1, -1)

proc isAgentPatrolActive*(agentId: int): bool =
  ## Check if an agent currently has patrol mode active.
  withBuiltinAI:
    return globalController.aiController.isPatrolActive(agentId)
  false

# Multi-waypoint Patrol API
# These functions allow external code to set custom patrol routes with 2-8 waypoints.
# Agent cycles through waypoints in order, wrapping to first after reaching last.

proc setAgentMultiWaypointPatrol*(agentId: int, waypoints: seq[IVec2]) =
  ## Set a multi-waypoint patrol route for an agent.
  ## Accepts 2-8 waypoints. Agent cycles through in order, wrapping after last.
  ## Requires BuiltinAI controller. Clears any stopped state.
  withBuiltinAI:
    globalController.aiController.clearAgentStop(agentId)
    globalController.aiController.setMultiWaypointPatrol(agentId, waypoints)

proc setAgentMultiWaypointPatrolXY*(agentId: int, coords: seq[tuple[x, y: int32]]) =
  ## Set a multi-waypoint patrol route using x,y coordinate tuples.
  var waypoints: seq[IVec2] = @[]
  for coord in coords:
    waypoints.add(ivec2(coord.x, coord.y))
  setAgentMultiWaypointPatrol(agentId, waypoints)

proc getAgentPatrolWaypointCount*(agentId: int): int =
  ## Get the number of waypoints in a multi-waypoint patrol route.
  ## Returns 0 if using legacy 2-point patrol or no patrol active.
  withBuiltinAI:
    return globalController.aiController.getPatrolWaypointCount(agentId)
  0

proc getAgentPatrolCurrentWaypointIndex*(agentId: int): int =
  ## Get the current waypoint index in multi-waypoint patrol (0-based).
  withBuiltinAI:
    return globalController.aiController.getPatrolCurrentWaypointIndex(agentId)
  0

# Stance API
# These functions allow external code to set combat stance for agents.
# Deferred versions (no env required) apply the stance on next decideAction.

proc setAgentStance*(agentId: int, stance: AgentStance) =
  ## Set the combat stance for an agent (deferred).
  ## The stance will be applied on the next decideAction call.
  ## Requires BuiltinAI controller.
  withBuiltinAI:
    globalController.aiController.setAgentStanceDeferred(agentId, stance)

proc getAgentStance*(agentId: int): AgentStance =
  ## Get the pending combat stance for an agent.
  ## Returns the pending stance if one has been set via setAgentStance(agentId, stance).
  ## Returns StanceDefensive if no stance has been set or if not using BuiltinAI.
  ## Note: To get the agent's current actual stance, use getAgentStance(env, agentId).
  withBuiltinAI:
    return globalController.aiController.getAgentPendingStance(agentId)
  StanceDefensive

proc clearAgentStance*(agentId: int) =
  ## Clear any pending stance modification for an agent.
  ## Requires BuiltinAI controller.
  withBuiltinAI:
    globalController.aiController.clearAgentStanceModified(agentId)

# Environment-based stance API (for direct modification when env is available)

proc setAgentStance*(env: Environment, agentId: int, stance: AgentStance) =
  ## Set the combat stance for an agent (immediate, requires env).
  if agentId >= 0 and agentId < env.agents.len:
    let agent = env.agents[agentId]
    if isAgentAlive(env, agent):
      agent.stance = stance

proc getAgentStance*(env: Environment, agentId: int): AgentStance =
  ## Get the current combat stance for an agent (requires env).
  if agentId >= 0 and agentId < env.agents.len:
    let agent = env.agents[agentId]
    if isAgentAlive(env, agent):
      return agent.stance
  StanceDefensive

# Garrison API
# These functions allow external code to garrison/ungarrison units.

proc garrisonAgentInBuilding*(env: Environment, agentId: int, buildingX, buildingY: int32): bool =
  ## Garrison an agent into the building at the given position.
  ## Returns true if successful.
  if agentId < 0 or agentId >= env.agents.len:
    return false
  let agent = env.agents[agentId]
  if not isAgentAlive(env, agent):
    return false
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return false
  garrisonUnitInBuilding(env, agent, thing)

proc ungarrisonAllFromBuilding*(env: Environment, buildingX, buildingY: int32): int32 =
  ## Ungarrison all units from the building at the given position.
  ## Returns the number of units ungarrisoned.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return 0
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return 0
  let units = ungarrisonAllUnits(env, thing)
  units.len.int32

proc getGarrisonCount*(env: Environment, buildingX, buildingY: int32): int32 =
  ## Get the number of units garrisoned in the building at the given position.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return 0
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return 0
  thing.garrisonedUnits.len.int32

proc isAgentGarrisoned*(env: Environment, agentId: int): bool =
  ## Check if an agent is currently garrisoned inside a building.
  ## Note: Garrisoned agents are NOT on the grid (pos is -1,-1), so we check
  ## terminated status directly rather than isAgentAlive (which requires grid presence).
  if agentId < 0 or agentId >= env.agents.len:
    return false
  if env.terminated[agentId] != 0.0:
    return false
  let agent = env.agents[agentId]
  agent.isGarrisoned

# Production Queue API
# These functions allow external code to queue/cancel unit training at buildings.

proc queueUnitTraining*(env: Environment, buildingX, buildingY: int32, teamId: int32): bool =
  ## Queue a unit for training at the building at the given position.
  ## The unit type and cost are determined by the building type.
  ## Returns true if successfully queued.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return false
  if not buildingHasTrain(thing.kind):
    return false
  let unitClass = buildingTrainUnit(thing.kind, teamId)
  let costs = buildingTrainCosts(thing.kind)
  queueTrainUnit(env, thing, teamId, unitClass, costs)

proc cancelLastQueuedUnit*(env: Environment, buildingX, buildingY: int32): bool =
  ## Cancel the last unit in the production queue at the given building.
  ## Returns true if a unit was cancelled.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return false
  cancelLastQueued(env, thing)

proc cancelQueuedUnitAtIndex*(env: Environment, buildingX, buildingY: int32, index: int32): bool =
  ## Cancel a specific unit in the production queue at the given index.
  ## Returns true if a unit was cancelled.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return false
  cancelQueueEntry(env, thing, index.int)

proc getProductionQueueSize*(env: Environment, buildingX, buildingY: int32): int32 =
  ## Get the number of units in the production queue at the given building.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return 0
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return 0
  thing.productionQueue.entries.len.int32

proc getProductionQueueEntryProgress*(env: Environment, buildingX, buildingY: int32, index: int32): int32 =
  ## Get the remaining steps for a production queue entry.
  ## Returns -1 if invalid.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return -1
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return -1
  if index < 0 or index >= thing.productionQueue.entries.len.int32:
    return -1
  thing.productionQueue.entries[index].remainingSteps.int32

proc canBuildingTrainUnit*(env: Environment, buildingX, buildingY: int32, unitClass: int32, teamId: int32): bool =
  ## Check if a building can train the specified unit class.
  ## Returns true if the building supports training this unit type.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return false
  if not buildingHasTrain(thing.kind):
    return false
  if unitClass < 0 or unitClass > ord(AgentUnitClass.high):
    return false
  let requestedClass = AgentUnitClass(unitClass)
  # Check if unit class is disabled for this team
  if teamId >= 0 and teamId < MapRoomObjectsTeams:
    if requestedClass in env.teamModifiers[teamId].disabledUnits:
      return false
  let defaultClass = buildingTrainUnit(thing.kind, teamId.int)
  # Building can only train its default unit class for the given team
  requestedClass == defaultClass

proc queueUnitTrainingWithClass*(env: Environment, buildingX, buildingY: int32, teamId: int32, unitClass: int32): bool =
  ## Queue a specific unit class for training at the building.
  ## Validates that the building can produce the requested unit class.
  ## Returns true if successfully queued.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return false
  if not buildingHasTrain(thing.kind):
    return false
  if unitClass < 0 or unitClass > ord(AgentUnitClass.high):
    return false
  let requestedClass = AgentUnitClass(unitClass)
  # Check if unit class is disabled for this team
  if teamId >= 0 and teamId < MapRoomObjectsTeams:
    if requestedClass in env.teamModifiers[teamId].disabledUnits:
      return false
  let defaultClass = buildingTrainUnit(thing.kind, teamId.int)
  # Validate the building can train this unit class
  if requestedClass != defaultClass:
    return false
  var costs = buildingTrainCosts(thing.kind)
  # Apply per-unit cost multiplier
  if teamId >= 0 and teamId < MapRoomObjectsTeams:
    let mult = env.teamModifiers[teamId].trainCostMultiplier[requestedClass]
    if mult != 0.0'f32 and mult != 1.0'f32:
      for i in 0 ..< costs.len:
        costs[i] = (res: costs[i].res, count: max(1, int(float32(costs[i].count) * mult + 0.5)))
  queueTrainUnit(env, thing, teamId.int, requestedClass, costs)

proc cancelAllTrainingQueue*(env: Environment, buildingX, buildingY: int32): int32 =
  ## Cancel all units in the production queue at the given building.
  ## Returns the number of units cancelled (resources are refunded for each).
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return 0
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return 0
  var cancelled: int32 = 0
  while thing.productionQueue.entries.len > 0:
    if cancelLastQueued(env, thing):
      inc cancelled
    else:
      break
  cancelled

proc getProductionQueueEntryUnitClass*(env: Environment, buildingX, buildingY: int32, index: int32): int32 =
  ## Get the unit class for a production queue entry.
  ## Returns -1 if invalid, otherwise the AgentUnitClass enum ordinal.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return -1
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return -1
  if index < 0 or index >= thing.productionQueue.entries.len.int32:
    return -1
  ord(thing.productionQueue.entries[index].unitClass).int32

proc getProductionQueueEntryTotalSteps*(env: Environment, buildingX, buildingY: int32, index: int32): int32 =
  ## Get the total training steps for a production queue entry.
  ## Returns -1 if invalid.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return -1
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return -1
  if index < 0 or index >= thing.productionQueue.entries.len.int32:
    return -1
  thing.productionQueue.entries[index].totalSteps.int32

proc isProductionQueueReady*(env: Environment, buildingX, buildingY: int32): bool =
  ## Check if the building has a queue entry ready for unit conversion.
  ## Returns true if the front entry has completed training.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return false
  productionQueueHasReady(thing)

# Research API
# These functions allow external code to research technologies at buildings.

proc researchBlacksmithUpgrade*(env: Environment, agentId: int, buildingX, buildingY: int32): bool =
  ## Research the next blacksmith upgrade at the given building.
  ## The agent must be a villager at the building.
  if agentId < 0 or agentId >= env.agents.len:
    return false
  let agent = env.agents[agentId]
  if not isAgentAlive(env, agent):
    return false
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or thing.kind != Blacksmith:
    return false
  tryResearchBlacksmithUpgrade(env, agent, thing)

proc researchUniversityTech*(env: Environment, agentId: int, buildingX, buildingY: int32): bool =
  ## Research the next university technology at the given building.
  if agentId < 0 or agentId >= env.agents.len:
    return false
  let agent = env.agents[agentId]
  if not isAgentAlive(env, agent):
    return false
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or thing.kind != University:
    return false
  tryResearchUniversityTech(env, agent, thing)

proc researchCastleTech*(env: Environment, agentId: int, buildingX, buildingY: int32): bool =
  ## Research the next castle unique technology at the given building.
  if agentId < 0 or agentId >= env.agents.len:
    return false
  let agent = env.agents[agentId]
  if not isAgentAlive(env, agent):
    return false
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or thing.kind != Castle:
    return false
  tryResearchCastleTech(env, agent, thing)

proc researchUnitUpgrade*(env: Environment, agentId: int, buildingX, buildingY: int32): bool =
  ## Research the next unit upgrade at the given building.
  if agentId < 0 or agentId >= env.agents.len:
    return false
  let agent = env.agents[agentId]
  if not isAgentAlive(env, agent):
    return false
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing):
    return false
  tryResearchUnitUpgrade(env, agent, thing)

proc hasBlacksmithUpgrade*(env: Environment, teamId: int, upgradeType: int32): int32 =
  ## Get the current level of a blacksmith upgrade for a team.
  ## upgradeType: 0=MeleeAttack, 1=ArcherAttack, 2=InfantryArmor, 3=CavalryArmor, 4=ArcherArmor
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return 0
  if upgradeType < 0 or upgradeType > ord(BlacksmithUpgradeType.high).int32:
    return 0
  env.teamBlacksmithUpgrades[teamId].levels[BlacksmithUpgradeType(upgradeType)].int32

proc hasUniversityTechResearched*(env: Environment, teamId: int, techType: int32): bool =
  ## Check if a university tech has been researched for a team.
  ## techType: 0=Ballistics, 1=MurderHoles, 2=Masonry, etc.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false
  if techType < 0 or techType > ord(UniversityTechType.high).int32:
    return false
  hasUniversityTech(env, teamId, UniversityTechType(techType))

proc hasCastleTechResearched*(env: Environment, teamId: int, techType: int32): bool =
  ## Check if a castle tech has been researched for a team.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false
  if techType < 0 or techType > ord(CastleTechType.high).int32:
    return false
  hasCastleTech(env, teamId, CastleTechType(techType))

proc hasUnitUpgradeResearched*(env: Environment, teamId: int, upgradeType: int32): bool =
  ## Check if a unit upgrade has been researched for a team.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false
  if upgradeType < 0 or upgradeType > ord(UnitUpgradeType.high).int32:
    return false
  hasUnitUpgrade(env, teamId, UnitUpgradeType(upgradeType))

# Extended Research Control API
# These functions allow external code to start/query research at buildings by position.
# Research categories: 0=Blacksmith, 1=University, 2=Castle, 3=UnitUpgrade

const
  ResearchCategoryBlacksmith* = 0'i32
  ResearchCategoryUniversity* = 1'i32
  ResearchCategoryCastle* = 2'i32
  ResearchCategoryUnitUpgrade* = 3'i32

proc startResearchAtBuilding*(env: Environment, buildingX, buildingY: int32,
                               researchCategory: int32, researchType: int32): bool =
  ## Start research at a building by position (no villager required).
  ## researchCategory: 0=Blacksmith, 1=University, 2=Castle, 3=UnitUpgrade
  ## researchType: specific tech/upgrade index within category
  ## Returns true if research was successfully started.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return false

  case researchCategory
  of ResearchCategoryBlacksmith:
    if thing.kind != Blacksmith:
      return false
    if researchType < 0 or researchType > ord(BlacksmithUpgradeType.high).int32:
      return false
    uiResearchBlacksmithUpgrade(env, thing, BlacksmithUpgradeType(researchType))
  of ResearchCategoryUniversity:
    if thing.kind != University:
      return false
    if researchType < 0 or researchType > ord(UniversityTechType.high).int32:
      return false
    uiResearchUniversityTech(env, thing, UniversityTechType(researchType))
  of ResearchCategoryCastle:
    if thing.kind != Castle:
      return false
    if researchType < 0 or researchType > 1:  # 0=Castle Age, 1=Imperial Age
      return false
    uiResearchCastleTech(env, thing, researchType.int)
  of ResearchCategoryUnitUpgrade:
    # Unit upgrades require matching building type
    if researchType < 0 or researchType > ord(UnitUpgradeType.high).int32:
      return false
    let upgradeType = UnitUpgradeType(researchType)
    if upgradeBuilding(upgradeType) != thing.kind:
      return false
    # Use the UI version to research without villager
    let teamId = thing.teamId
    if teamId < 0 or teamId >= MapRoomObjectsTeams:
      return false
    if env.teamUnitUpgrades[teamId].researched[upgradeType]:
      return false
    let prereq = upgradePrerequisite(upgradeType)
    if prereq != upgradeType and not env.teamUnitUpgrades[teamId].researched[prereq]:
      return false
    let costs = upgradeCosts(upgradeType)
    if not env.spendStockpile(teamId, costs):
      return false
    env.teamUnitUpgrades[teamId].researched[upgradeType] = true
    env.upgradeExistingUnits(teamId, upgradeSourceUnit(upgradeType), upgradeTargetUnit(upgradeType))
    thing.cooldown = 8
    true
  else:
    false

proc isResearchInProgress*(env: Environment, buildingX, buildingY: int32): bool =
  ## Check if a building has research in progress (is on cooldown).
  ## Returns true if the building exists and has cooldown > 0.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return false
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return false
  thing.cooldown > 0

proc getResearchCooldown*(env: Environment, buildingX, buildingY: int32): int32 =
  ## Get the remaining research cooldown for a building.
  ## Returns 0 if no research is in progress, -1 if invalid building.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return -1
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return -1
  thing.cooldown.int32

proc getAvailableResearchCount*(env: Environment, buildingX, buildingY: int32): int32 =
  ## Get the number of available (unresearched, affordable) research options at a building.
  ## Returns 0 if building is invalid or has no research options.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return 0
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return 0

  let teamId = thing.teamId
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return 0

  var count: int32 = 0
  case thing.kind
  of Blacksmith:
    for upgradeType in BlacksmithUpgradeType:
      let level = env.teamBlacksmithUpgrades[teamId].levels[upgradeType]
      if level < BlacksmithUpgradeMaxLevel:
        inc count
  of University:
    for techType in UniversityTechType:
      if not env.teamUniversityTechs[teamId].researched[techType]:
        inc count
  of Castle:
    let (castleAge, imperialAge) = castleTechsForTeam(teamId)
    if not env.teamCastleTechs[teamId].researched[castleAge]:
      inc count
    elif not env.teamCastleTechs[teamId].researched[imperialAge]:
      inc count
  of Barracks, Stable, ArcheryRange:
    for upgradeType in UnitUpgradeType:
      if upgradeBuilding(upgradeType) != thing.kind:
        continue
      if env.teamUnitUpgrades[teamId].researched[upgradeType]:
        continue
      let prereq = upgradePrerequisite(upgradeType)
      if prereq != upgradeType and not env.teamUnitUpgrades[teamId].researched[prereq]:
        continue
      inc count
  else:
    discard

  count

proc getAvailableResearchAtIndex*(env: Environment, buildingX, buildingY: int32,
                                   index: int32): tuple[category: int32, researchType: int32] =
  ## Get the category and type of the nth available research at a building.
  ## Returns (-1, -1) if index is out of range or building is invalid.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return (-1'i32, -1'i32)
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return (-1'i32, -1'i32)

  let teamId = thing.teamId
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return (-1'i32, -1'i32)

  var currentIndex: int32 = 0
  case thing.kind
  of Blacksmith:
    for upgradeType in BlacksmithUpgradeType:
      let level = env.teamBlacksmithUpgrades[teamId].levels[upgradeType]
      if level < BlacksmithUpgradeMaxLevel:
        if currentIndex == index:
          return (ResearchCategoryBlacksmith, ord(upgradeType).int32)
        inc currentIndex
  of University:
    for techType in UniversityTechType:
      if not env.teamUniversityTechs[teamId].researched[techType]:
        if currentIndex == index:
          return (ResearchCategoryUniversity, ord(techType).int32)
        inc currentIndex
  of Castle:
    let (castleAge, imperialAge) = castleTechsForTeam(teamId)
    if not env.teamCastleTechs[teamId].researched[castleAge]:
      if currentIndex == index:
        return (ResearchCategoryCastle, 0'i32)
      inc currentIndex
    if env.teamCastleTechs[teamId].researched[castleAge] and
       not env.teamCastleTechs[teamId].researched[imperialAge]:
      if currentIndex == index:
        return (ResearchCategoryCastle, 1'i32)
      inc currentIndex
  of Barracks, Stable, ArcheryRange:
    for upgradeType in UnitUpgradeType:
      if upgradeBuilding(upgradeType) != thing.kind:
        continue
      if env.teamUnitUpgrades[teamId].researched[upgradeType]:
        continue
      let prereq = upgradePrerequisite(upgradeType)
      if prereq != upgradeType and not env.teamUnitUpgrades[teamId].researched[prereq]:
        continue
      if currentIndex == index:
        return (ResearchCategoryUnitUpgrade, ord(upgradeType).int32)
      inc currentIndex
  else:
    discard

  (-1'i32, -1'i32)

proc canAffordResearch*(env: Environment, teamId: int32, researchCategory: int32,
                         researchType: int32): bool =
  ## Check if a team can afford a specific research.
  ## Does not check if the research is available (use getAvailableResearchAtIndex for that).
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false

  case researchCategory
  of ResearchCategoryBlacksmith:
    if researchType < 0 or researchType > ord(BlacksmithUpgradeType.high).int32:
      return false
    let upgradeType = BlacksmithUpgradeType(researchType)
    let level = env.teamBlacksmithUpgrades[teamId].levels[upgradeType]
    if level >= BlacksmithUpgradeMaxLevel:
      return false
    let costMultiplier = level + 1
    let foodCost = BlacksmithUpgradeFoodCost * costMultiplier
    let goldCost = BlacksmithUpgradeGoldCost * costMultiplier
    env.canSpendStockpile(teamId, [(res: ResourceFood, count: foodCost), (res: ResourceGold, count: goldCost)])
  of ResearchCategoryUniversity:
    if researchType < 0 or researchType > ord(UniversityTechType.high).int32:
      return false
    let techIndex = researchType + 1
    let foodCost = UniversityTechFoodCost * techIndex.int
    let goldCost = UniversityTechGoldCost * techIndex.int
    let woodCost = UniversityTechWoodCost * techIndex.int
    env.canSpendStockpile(teamId, [(res: ResourceFood, count: foodCost), (res: ResourceGold, count: goldCost), (res: ResourceWood, count: woodCost)])
  of ResearchCategoryCastle:
    if researchType < 0 or researchType > 1:
      return false
    let isImperial = researchType == 1
    let foodCost = if isImperial: CastleTechImperialFoodCost else: CastleTechFoodCost
    let goldCost = if isImperial: CastleTechImperialGoldCost else: CastleTechGoldCost
    env.canSpendStockpile(teamId, [(res: ResourceFood, count: foodCost), (res: ResourceGold, count: goldCost)])
  of ResearchCategoryUnitUpgrade:
    if researchType < 0 or researchType > ord(UnitUpgradeType.high).int32:
      return false
    let upgradeType = UnitUpgradeType(researchType)
    let costs = upgradeCosts(upgradeType)
    env.canSpendStockpile(teamId, costs)
  else:
    false

proc getBuildingResearchCategory*(env: Environment, buildingX, buildingY: int32): int32 =
  ## Get the research category for a building (what type of research it supports).
  ## Returns: 0=Blacksmith, 1=University, 2=Castle, 3=UnitUpgrade, -1=none
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return -1
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return -1

  case thing.kind
  of Blacksmith: ResearchCategoryBlacksmith
  of University: ResearchCategoryUniversity
  of Castle: ResearchCategoryCastle
  of Barracks, Stable, ArcheryRange: ResearchCategoryUnitUpgrade
  else: -1

# Scout Mode API
# These functions allow external code to enable/disable scout mode for agents.

proc setAgentScoutMode*(agentId: int, active: bool) =
  ## Enable or disable scout mode for an agent.
  ## Requires BuiltinAI controller. Clears any stopped state when enabling.
  withBuiltinAI:
    if active:
      globalController.aiController.clearAgentStop(agentId)
    globalController.aiController.setScoutMode(agentId, active)

proc isAgentScoutModeActive*(agentId: int): bool =
  ## Check if an agent has scout mode active.
  withBuiltinAI:
    return globalController.aiController.isScoutModeActive(agentId)
  false

proc getAgentScoutExploreRadius*(agentId: int): int32 =
  ## Get the current scout exploration radius for an agent.
  withBuiltinAI:
    return globalController.aiController.getScoutExploreRadius(agentId)
  0

# Rally Point API
# These functions allow external code to set rally points on buildings.

proc setBuildingRallyPoint*(env: Environment, buildingX, buildingY: int32, rallyX, rallyY: int32) =
  ## Set the rally point for a building.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return
  setRallyPoint(thing, ivec2(rallyX, rallyY))

proc clearBuildingRallyPoint*(env: Environment, buildingX, buildingY: int32) =
  ## Clear the rally point for a building.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return
  clearRallyPoint(thing)

proc getBuildingRallyPoint*(env: Environment, buildingX, buildingY: int32): IVec2 =
  ## Get the rally point for a building. Returns (-1, -1) if no rally point is set.
  let pos = ivec2(buildingX, buildingY)
  if not isValidPos(pos):
    return ivec2(-1, -1)
  let thing = env.grid[pos.x][pos.y]
  if isNil(thing) or not isBuildingKind(thing.kind):
    return ivec2(-1, -1)
  if hasRallyPoint(thing):
    return thing.rallyPoint
  ivec2(-1, -1)

# Hold Position API
# These functions allow external code to set hold-position behavior for agents.
# Hold Position: agent stays at a location, attacks enemies in range but won't chase.

proc setAgentHoldPosition*(agentId: int, pos: IVec2) =
  ## Set hold position for an agent. The agent stays at the given position,
  ## attacks enemies in range, but won't chase.
  ## Requires BuiltinAI controller. Clears any stopped state.
  withBuiltinAI:
    globalController.aiController.clearAgentStop(agentId)
    globalController.aiController.setHoldPosition(agentId, pos)

proc setAgentHoldPositionXY*(agentId: int, x, y: int32) =
  ## Set hold position using x,y coordinates.
  setAgentHoldPosition(agentId, ivec2(x, y))

proc clearAgentHoldPosition*(agentId: int) =
  ## Clear hold position for an agent.
  withBuiltinAI:
    globalController.aiController.clearHoldPosition(agentId)

proc getAgentHoldPosition*(agentId: int): IVec2 =
  ## Get the hold position target. Returns (-1, -1) if not active.
  withBuiltinAI:
    return globalController.aiController.getHoldPosition(agentId)
  ivec2(-1, -1)

proc isAgentHoldPositionActive*(agentId: int): bool =
  ## Check if hold position is active for an agent.
  withBuiltinAI:
    return globalController.aiController.isHoldPositionActive(agentId)
  false

# Follow API
# These functions allow external code to set follow behavior for agents.
# Follow: agent follows a target agent, maintaining proximity.

proc setAgentFollowTarget*(agentId: int, targetAgentId: int) =
  ## Set an agent to follow another agent.
  ## Requires BuiltinAI controller. Clears any stopped state.
  withBuiltinAI:
    globalController.aiController.clearAgentStop(agentId)
    globalController.aiController.setFollowTarget(agentId, targetAgentId)

proc clearAgentFollowTarget*(agentId: int) =
  ## Clear follow target for an agent.
  withBuiltinAI:
    globalController.aiController.clearFollowTarget(agentId)

proc getAgentFollowTargetId*(agentId: int): int =
  ## Get the follow target agent ID. Returns -1 if not active.
  withBuiltinAI:
    return globalController.aiController.getFollowTargetId(agentId)
  -1

proc isAgentFollowActive*(agentId: int): bool =
  ## Check if follow mode is active for an agent.
  withBuiltinAI:
    return globalController.aiController.isFollowActive(agentId)
  false

# Guard API
# These functions allow external code to set guard behavior for agents.
# Guard: agent guards a target (agent or position), stays within radius, attacks enemies.

proc setAgentGuard*(agentId: int, targetAgentId: int) =
  ## Set an agent to guard another agent.
  ## The guarding agent stays within GuardRadius (5 tiles) of the target,
  ## attacks any enemies that enter range, and returns to guard position after combat.
  ## Requires BuiltinAI controller. Clears any stopped state.
  withBuiltinAI:
    globalController.aiController.clearAgentStop(agentId)
    globalController.aiController.setGuardTarget(agentId, targetAgentId)

proc setAgentGuardPosition*(agentId: int, pos: IVec2) =
  ## Set an agent to guard a specific position.
  ## The guarding agent stays within GuardRadius (5 tiles) of the position,
  ## attacks any enemies that enter range, and returns to guard position after combat.
  ## Requires BuiltinAI controller. Clears any stopped state.
  withBuiltinAI:
    globalController.aiController.clearAgentStop(agentId)
    globalController.aiController.setGuardPosition(agentId, pos)

proc setAgentGuardPositionXY*(agentId: int, x, y: int32) =
  ## Set guard position using x,y coordinates.
  setAgentGuardPosition(agentId, ivec2(x, y))

proc clearAgentGuard*(agentId: int) =
  ## Clear guard mode for an agent.
  withBuiltinAI:
    globalController.aiController.clearGuard(agentId)

proc getAgentGuardTargetId*(agentId: int): int =
  ## Get the guard target agent ID. Returns -1 if guarding a position or not active.
  withBuiltinAI:
    return globalController.aiController.getGuardTargetId(agentId)
  -1

proc getAgentGuardPosition*(agentId: int): IVec2 =
  ## Get the guard target position. Returns (-1, -1) if guarding an agent or not active.
  withBuiltinAI:
    return globalController.aiController.getGuardPosition(agentId)
  ivec2(-1, -1)

proc isAgentGuarding*(agentId: int): bool =
  ## Check if guard mode is active for an agent.
  withBuiltinAI:
    return globalController.aiController.isGuardActive(agentId)
  false

# Stop Command API
# Stops an agent completely: clears all orders, path, and active option.
# Agent remains idle until given a new command or idle threshold expires.

proc stopAgent*(agentId: int) =
  ## Stop an agent completely: clears all orders, path, and active option.
  ## Agent will remain idle until given a new command or until StopIdleSteps passes.
  withBuiltinAI:
    globalController.aiController.stopAgentDeferred(agentId)

proc clearAgentStop*(agentId: int) =
  ## Clear the stopped state for an agent, allowing normal behavior to resume.
  ## Called automatically when issuing new movement commands.
  withBuiltinAI:
    globalController.aiController.clearAgentStop(agentId)

proc isAgentStopped*(agentId: int): bool =
  ## Check if an agent is currently in stopped state.
  withBuiltinAI:
    return globalController.aiController.isAgentStopped(agentId)
  false

proc getAgentStoppedUntilStep*(agentId: int): int32 =
  ## Get the step at which the stopped state will expire.
  ## Returns 0 if not stopped or -1 if deferred (not yet initialized).
  withBuiltinAI:
    return globalController.aiController.getAgentStoppedUntilStep(agentId)
  0

# Command Queue API
# Shift-queue functionality for AoE2-style command queueing.
# When shift+clicking, commands are added to a queue and executed in order.

proc clearAgentCommandQueue*(agentId: int) =
  ## Clear all queued commands for an agent.
  withBuiltinAI:
    globalController.aiController.clearCommandQueue(agentId)

proc getAgentCommandQueueCount*(agentId: int): int =
  ## Get the number of commands in an agent's queue.
  withBuiltinAI:
    return globalController.aiController.getCommandQueueCount(agentId)
  0

proc hasAgentQueuedCommands*(agentId: int): bool =
  ## Check if an agent has commands in the queue.
  withBuiltinAI:
    return globalController.aiController.hasQueuedCommands(agentId)
  false

proc queueAgentAttackMove*(agentId: int, target: IVec2) =
  ## Queue an attack-move command for shift-queue.
  withBuiltinAI:
    globalController.aiController.queueAttackMove(agentId, target)

proc queueAgentAttackMoveXY*(agentId: int, x, y: int32) =
  ## Queue an attack-move command using x,y coordinates.
  queueAgentAttackMove(agentId, ivec2(x, y))

proc queueAgentPatrol*(agentId: int, target: IVec2) =
  ## Queue a patrol command for shift-queue.
  ## Patrol will go from the agent's position when the command executes to target.
  withBuiltinAI:
    globalController.aiController.queuePatrol(agentId, target)

proc queueAgentFollow*(agentId: int, targetAgentId: int) =
  ## Queue a follow command for shift-queue.
  withBuiltinAI:
    globalController.aiController.queueFollow(agentId, targetAgentId)

proc queueAgentGuard*(agentId: int, targetAgentId: int) =
  ## Queue a guard-agent command for shift-queue.
  withBuiltinAI:
    globalController.aiController.queueGuardAgent(agentId, targetAgentId)

proc queueAgentGuardPosition*(agentId: int, target: IVec2) =
  ## Queue a guard-position command for shift-queue.
  withBuiltinAI:
    globalController.aiController.queueGuardPosition(agentId, target)

proc queueAgentHoldPosition*(agentId: int, target: IVec2) =
  ## Queue a hold-position command for shift-queue.
  withBuiltinAI:
    globalController.aiController.queueHoldPosition(agentId, target)

proc executeNextQueuedCommand*(agentId: int, agentPos: IVec2) =
  ## Execute the next command in the queue (if any).
  ## Call this when the current command completes.
  withBuiltinAI:
    globalController.aiController.executeQueuedCommand(agentId, agentPos)

# Formation API
# Formation system for coordinated group movement (Line, Box formations).
# Formations are per-control-group, not per-agent.
# formations is imported at module level and re-exported

proc setControlGroupFormation*(groupIndex: int, formationType: int32) =
  ## Set formation type for a control group.
  ## formationType: 0=None, 1=Line, 2=Box, 3=Wedge, 4=Scatter
  if formationType >= 0 and formationType <= ord(FormationType.high):
    setFormation(groupIndex, FormationType(formationType))

proc getControlGroupFormation*(groupIndex: int): int32 =
  ## Get formation type for a control group.
  ## Returns: 0=None, 1=Line, 2=Box, 3=Wedge, 4=Scatter
  ord(getFormation(groupIndex)).int32

proc clearControlGroupFormation*(groupIndex: int) =
  ## Clear formation for a control group, returning units to free movement.
  clearFormation(groupIndex)

proc setControlGroupFormationRotation*(groupIndex: int, rotation: int32) =
  ## Set formation rotation (0-7 for 8 compass directions).
  setFormationRotation(groupIndex, rotation.int)

proc getControlGroupFormationRotation*(groupIndex: int): int32 =
  ## Get formation rotation for a control group.
  getFormationRotation(groupIndex).int32

# Agent-ID based Formation API
# Convenience functions that work directly with agent IDs rather than control group indices.
# These create/use control groups internally.

proc findAvailableControlGroup*(): int =
  ## Find an empty control group index, or return the last one (9) as fallback.
  ## Used by setFormationForAgents to allocate a group.
  for i in 0 ..< ControlGroupCount:
    if controlGroups[i].len == 0:
      return i
  # All groups in use - use the last one
  ControlGroupCount - 1

proc setFormationForAgents*(env: Environment, agentIds: seq[int], formationType: FormationType) =
  ## Set formation for a group of agents by their IDs.
  ## This creates a control group containing the specified agents and sets the formation.
  ## If no empty control group is available, uses control group 9 (overwriting it).
  if agentIds.len == 0:
    return
  let groupIndex = findAvailableControlGroup()
  # Assign agents to the control group
  controlGroups[groupIndex] = @[]
  for agentId in agentIds:
    if agentId >= 0 and agentId < env.agents.len:
      let agent = env.agents[agentId]
      if isAgentAlive(env, agent):
        controlGroups[groupIndex].add(agent)
  # Set the formation
  setFormation(groupIndex, formationType)

proc setFormationForAgentsWithRotation*(env: Environment, agentIds: seq[int],
                                         formationType: FormationType, rotation: int) =
  ## Set formation and rotation for a group of agents by their IDs.
  if agentIds.len == 0:
    return
  let groupIndex = findAvailableControlGroup()
  controlGroups[groupIndex] = @[]
  for agentId in agentIds:
    if agentId >= 0 and agentId < env.agents.len:
      let agent = env.agents[agentId]
      if isAgentAlive(env, agent):
        controlGroups[groupIndex].add(agent)
  setFormation(groupIndex, formationType)
  setFormationRotation(groupIndex, rotation)

proc clearFormationForAgents*(agentIds: seq[int]) =
  ## Clear formation for the specified agents.
  ## Finds which control group(s) contain the agents and clears their formations.
  for agentId in agentIds:
    let groupIdx = findAgentControlGroup(agentId)
    if groupIdx >= 0:
      clearFormation(groupIdx)

# Selection API
# Programmatic interface for the selection system (bridges GUI selection and control APIs).

proc selectUnits*(env: Environment, agentIds: seq[int]) =
  ## Replace current selection with the specified agents.
  selection = @[]
  for agentId in agentIds:
    if agentId >= 0 and agentId < env.agents.len:
      let agent = env.agents[agentId]
      if isAgentAlive(env, agent):
        selection.add(agent)

proc addToSelection*(env: Environment, agentId: int) =
  ## Add a single agent to the current selection (if alive and not already selected).
  if agentId >= 0 and agentId < env.agents.len:
    let agent = env.agents[agentId]
    if isAgentAlive(env, agent):
      for s in selection:
        if s.agentId == agentId:
          return
      selection.add(agent)

proc removeFromSelection*(agentId: int) =
  ## Remove a single agent from the current selection.
  for i in countdown(selection.len - 1, 0):
    if selection[i].agentId == agentId:
      selection.delete(i)
      return

proc clearSelection*() =
  ## Clear the current selection.
  selection = @[]

proc getSelectionCount*(): int =
  ## Get the number of currently selected units.
  selection.len

proc getSelectedAgentId*(index: int): int =
  ## Get the agent ID of a selected unit by index. Returns -1 if invalid index.
  if index >= 0 and index < selection.len:
    selection[index].agentId
  else:
    -1

proc createControlGroup*(env: Environment, groupIndex: int, agentIds: seq[int]) =
  ## Assign agents to a control group (0-9).
  if groupIndex < 0 or groupIndex >= ControlGroupCount:
    return
  controlGroups[groupIndex] = @[]
  for agentId in agentIds:
    if agentId >= 0 and agentId < env.agents.len:
      let agent = env.agents[agentId]
      if isAgentAlive(env, agent):
        controlGroups[groupIndex].add(agent)

proc recallControlGroup*(env: Environment, groupIndex: int) =
  ## Recall a control group into the current selection.
  if groupIndex < 0 or groupIndex >= ControlGroupCount:
    return
  # Filter out dead units
  var alive: seq[Thing] = @[]
  for thing in controlGroups[groupIndex]:
    if isAgentAlive(env, thing):
      alive.add(thing)
  controlGroups[groupIndex] = alive
  selection = alive

proc getControlGroupCount*(groupIndex: int): int =
  ## Get the number of units in a control group. Returns 0 if invalid index.
  if groupIndex >= 0 and groupIndex < ControlGroupCount:
    controlGroups[groupIndex].len
  else:
    0

proc getControlGroupAgentId*(groupIndex: int, index: int): int =
  ## Get the agent ID at a position in a control group. Returns -1 if invalid.
  if groupIndex >= 0 and groupIndex < ControlGroupCount and
     index >= 0 and index < controlGroups[groupIndex].len:
    controlGroups[groupIndex][index].agentId
  else:
    -1

proc issueCommandToSelection*(env: Environment, commandType: int32, targetX, targetY: int32) =
  ## Issue a command to all selected units.
  ## commandType: 0=attack-move, 1=patrol (from current pos to target), 2=stop,
  ##              3=hold position (at current pos), 4=follow (targetX=targetAgentId),
  ##              5=guard agent (targetX=targetAgentId), 6=guard position (target pos)
  let target = ivec2(targetX, targetY)
  for thing in selection:
    if isAgentAlive(env, thing):
      let agentId = thing.agentId
      case commandType
      of 0: # Attack-move
        setAgentAttackMoveTarget(agentId, target)
      of 1: # Patrol from current position to target
        setAgentPatrol(agentId, thing.pos, target)
      of 2: # Stop
        stopAgent(agentId)
      of 3: # Hold position at current location
        setAgentHoldPosition(agentId, thing.pos)
      of 4: # Follow (targetX = target agent ID)
        setAgentFollowTarget(agentId, targetX.int)
      of 5: # Guard agent (targetX = target agent ID)
        setAgentGuard(agentId, targetX.int)
      of 6: # Guard position
        setAgentGuardPosition(agentId, target)
      else:
        discard

# Economy Priority Override API
# These functions allow external code to override gatherer resource priorities.
# Individual overrides take precedence over team-level focus.
# Both override automatic task selection based on flow rates and bottlenecks.

proc setGathererPriority*(agentId: int, resource: StockpileResource) =
  ## Set an individual gatherer to prioritize collecting a specific resource.
  ## Overrides automatic task selection for this gatherer.
  ## resource: ResourceFood, ResourceWood, ResourceGold, or ResourceStone
  withBuiltinAI:
    globalController.aiController.setGathererPriority(agentId, resource)

proc setGathererPriorityInt*(agentId: int, resource: int32) =
  ## Set gatherer priority using integer resource index.
  ## resource: 0=Food, 1=Wood, 2=Gold, 3=Stone
  if resource < 0 or resource > ord(StockpileResource.high).int32:
    return
  setGathererPriority(agentId, StockpileResource(resource))

proc clearGathererPriority*(agentId: int) =
  ## Clear the individual gatherer priority override.
  ## Returns the gatherer to automatic task selection.
  withBuiltinAI:
    globalController.aiController.clearGathererPriority(agentId)

proc getGathererPriority*(agentId: int): StockpileResource =
  ## Get the current gatherer priority for an agent.
  ## Returns ResourceNone if no priority is set.
  withBuiltinAI:
    return globalController.aiController.getGathererPriority(agentId)
  ResourceNone

proc getGathererPriorityInt*(agentId: int): int32 =
  ## Get the current gatherer priority as an integer.
  ## Returns -1 if no priority is set, otherwise 0=Food, 1=Wood, 2=Gold, 3=Stone
  let resource = getGathererPriority(agentId)
  if resource == ResourceNone:
    return -1
  ord(resource).int32

proc isGathererPriorityActive*(agentId: int): bool =
  ## Check if an individual gatherer priority is active.
  withBuiltinAI:
    return globalController.aiController.isGathererPriorityActive(agentId)
  false

proc setTeamEconomyFocus*(teamId: int, resource: StockpileResource) =
  ## Set a team-level economy focus to bias all gatherers toward a resource.
  ## This affects all gatherers on the team that don't have individual overrides.
  ## resource: ResourceFood, ResourceWood, ResourceGold, or ResourceStone
  withBuiltinAI:
    globalController.aiController.setTeamEconomyFocus(teamId, resource)

proc setTeamEconomyFocusInt*(teamId: int, resource: int32) =
  ## Set team economy focus using integer resource index.
  ## resource: 0=Food, 1=Wood, 2=Gold, 3=Stone
  if resource < 0 or resource > ord(StockpileResource.high).int32:
    return
  setTeamEconomyFocus(teamId, StockpileResource(resource))

proc clearTeamEconomyFocus*(teamId: int) =
  ## Clear the team-level economy focus.
  ## Returns all gatherers to automatic task selection.
  withBuiltinAI:
    globalController.aiController.clearTeamEconomyFocus(teamId)

proc getTeamEconomyFocus*(teamId: int): StockpileResource =
  ## Get the current team economy focus.
  ## Returns ResourceNone if no focus is set.
  withBuiltinAI:
    return globalController.aiController.getTeamEconomyFocus(teamId)
  ResourceNone

proc getTeamEconomyFocusInt*(teamId: int): int32 =
  ## Get the current team economy focus as an integer.
  ## Returns -1 if no focus is set, otherwise 0=Food, 1=Wood, 2=Gold, 3=Stone
  let resource = getTeamEconomyFocus(teamId)
  if resource == ResourceNone:
    return -1
  ord(resource).int32

proc isTeamEconomyFocusActive*(teamId: int): bool =
  ## Check if a team economy focus is active.
  withBuiltinAI:
    return globalController.aiController.isTeamEconomyFocusActive(teamId)
  false
