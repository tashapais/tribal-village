## Economy Management and Worker Allocation System
## Tracks resource flow and detects bottlenecks for AI decision-making

import ai_core
export ai_core

const
  # Resource flow tracking window (in steps)
  EconomyTrackingWindow* = 60  # Track over ~1 minute of game time
  # Thresholds for bottleneck detection
  MinGatherersRatio* = 0.3    # At least 30% gatherers
  MaxGatherersRatio* = 0.7    # At most 70% gatherers
  MinBuildersRatio* = 0.1     # At least 10% builders
  MaxBuildersRatio* = 0.4     # At most 40% builders
  MinFightersRatio* = 0.1     # At least 10% fighters when under threat
  # Critical resource thresholds
  CriticalFoodLevel* = 3      # Food below this is critical
  CriticalWoodLevel* = 5      # Wood below this is critical

type
  ResourceSnapshot* = object
    food*: int
    wood*: int
    stone*: int
    gold*: int
    step*: int

  ResourceFlowRate* = object
    foodPerStep*: float
    woodPerStep*: float
    stonePerStep*: float
    goldPerStep*: float

  WorkerCounts* = object
    gatherers*: int
    builders*: int
    fighters*: int
    total*: int

  BottleneckKind* = enum
    NoBottleneck
    TooManyGatherers
    TooFewGatherers
    TooManyBuilders
    TooFewBuilders
    TooFewFighters
    FoodCritical
    WoodCritical
    StoneCritical

  EconomyState* = object
    # Circular buffer of resource snapshots
    snapshots*: array[EconomyTrackingWindow, ResourceSnapshot]
    snapshotIndex*: int
    snapshotCount*: int
    # Cached flow rates (updated periodically)
    flowRate*: ResourceFlowRate
    # Current bottleneck
    currentBottleneck*: BottleneckKind

# Team-indexed economy state (global storage)
var teamEconomy*: array[MapRoomObjectsTeams, EconomyState]

# Per-step cache for worker counts to avoid O(n) env.agents scan
var workerCountCacheStep: int = -1
var workerCountCache: array[MapRoomObjectsTeams, tuple[counts: WorkerCounts, hasEnemy: bool]]

proc recordSnapshot*(teamId: int, env: Environment) =
  ## Record current stockpile levels for resource flow tracking
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return

  template st: untyped = teamEconomy[teamId]
  let idx = st.snapshotIndex
  st.snapshots[idx] = ResourceSnapshot(
    food: env.stockpileCount(teamId, ResourceFood),
    wood: env.stockpileCount(teamId, ResourceWood),
    stone: env.stockpileCount(teamId, ResourceStone),
    gold: env.stockpileCount(teamId, ResourceGold),
    step: env.currentStep
  )
  st.snapshotIndex = (idx + 1) mod EconomyTrackingWindow
  if st.snapshotCount < EconomyTrackingWindow:
    inc st.snapshotCount

proc calculateFlowRate*(teamId: int): ResourceFlowRate =
  ## Calculate resource flow rates from recent snapshots
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return ResourceFlowRate()

  let state = teamEconomy[teamId]
  if state.snapshotCount < 2:
    return ResourceFlowRate()

  # Find oldest and newest snapshots
  let newestIdx = (state.snapshotIndex - 1 + EconomyTrackingWindow) mod EconomyTrackingWindow
  let oldestIdx = if state.snapshotCount >= EconomyTrackingWindow:
    state.snapshotIndex
  else:
    0

  let newest = state.snapshots[newestIdx]
  let oldest = state.snapshots[oldestIdx]
  let stepDiff = newest.step - oldest.step

  if stepDiff <= 0:
    return ResourceFlowRate()

  let divisor = float(stepDiff)
  result.foodPerStep = float(newest.food - oldest.food) / divisor
  result.woodPerStep = float(newest.wood - oldest.wood) / divisor
  result.stonePerStep = float(newest.stone - oldest.stone) / divisor
  result.goldPerStep = float(newest.gold - oldest.gold) / divisor

proc updateFlowRate*(teamId: int) =
  ## Update cached flow rate
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return
  teamEconomy[teamId].flowRate = calculateFlowRate(teamId)

proc getFlowRate*(teamId: int): ResourceFlowRate =
  ## Get current resource flow rate
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return ResourceFlowRate()
  teamEconomy[teamId].flowRate

proc countWorkersAndEnemies*(controller: Controller, env: Environment, teamId: int): tuple[counts: WorkerCounts, hasEnemy: bool] =
  ## Count agents by role for a team and detect enemy presence in a single pass.
  ## Cached per-step to avoid O(n) env.agents scan - all teams computed together on first call.
  if workerCountCacheStep != env.currentStep:
    # Cache miss - recompute for all teams in single pass
    workerCountCacheStep = env.currentStep
    for t in 0 ..< MapRoomObjectsTeams:
      workerCountCache[t] = (WorkerCounts(), false)
    for agent in env.agents:
      if not isAgentAlive(env, agent):
        continue
      let agentTeam = getTeamId(agent)
      if agentTeam < 0 or agentTeam >= MapRoomObjectsTeams:
        continue
      # Mark all other teams as having enemy presence
      for t in 0 ..< MapRoomObjectsTeams:
        if t != agentTeam:
          workerCountCache[t].hasEnemy = true
      inc workerCountCache[agentTeam].counts.total
      let agentId = agent.agentId
      if agentId < 0 or agentId >= MapAgents or not controller.agentsInitialized[agentId]:
        # Default to gatherer if not initialized
        inc workerCountCache[agentTeam].counts.gatherers
        continue
      case controller.agents[agentId].role
      of Gatherer:
        inc workerCountCache[agentTeam].counts.gatherers
      of Builder:
        inc workerCountCache[agentTeam].counts.builders
      of Fighter:
        inc workerCountCache[agentTeam].counts.fighters
      of Scripted:
        # Count scripted as gatherers for ratio purposes
        inc workerCountCache[agentTeam].counts.gatherers
  if teamId >= 0 and teamId < MapRoomObjectsTeams:
    return workerCountCache[teamId]
  (WorkerCounts(), false)

proc countWorkers*(controller: Controller, env: Environment, teamId: int): WorkerCounts =
  ## Count agents by role for a team using controller's agent state
  countWorkersAndEnemies(controller, env, teamId).counts

proc detectBottleneck*(controller: Controller, env: Environment, teamId: int): BottleneckKind =
  ## Detect economic bottlenecks for a team
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return NoBottleneck

  # Phase-aware critical thresholds: early game is more sensitive to food/wood shortages
  let gameProgress = if env.config.maxSteps > 0:
    env.currentStep.float / env.config.maxSteps.float
  else:
    0.5
  let critFood = if gameProgress < EarlyGameThreshold: CriticalFoodLevel + 2 else: CriticalFoodLevel
  let critWood = if gameProgress < EarlyGameThreshold: CriticalWoodLevel + 3 else: CriticalWoodLevel

  # Check critical resource levels first (cheap lookups before expensive iteration)
  if env.stockpileCount(teamId, ResourceFood) < critFood:
    return FoodCritical
  if env.stockpileCount(teamId, ResourceWood) < critWood:
    return WoodCritical

  # Single pass: count workers and detect enemies simultaneously
  let (counts, hasEnemy) = countWorkersAndEnemies(controller, env, teamId)
  if counts.total == 0:
    return NoBottleneck

  let total = float(counts.total)
  let fighterRatio = float(counts.fighters) / total

  if hasEnemy and fighterRatio < MinFightersRatio:
    return TooFewFighters

  let gathererRatio = float(counts.gatherers) / total
  if gathererRatio > MaxGatherersRatio:
    return TooManyGatherers
  if gathererRatio < MinGatherersRatio:
    return TooFewGatherers

  let builderRatio = float(counts.builders) / total
  if builderRatio > MaxBuildersRatio:
    return TooManyBuilders
  if builderRatio < MinBuildersRatio:
    return TooFewBuilders

  NoBottleneck

proc updateBottleneck*(controller: Controller, env: Environment, teamId: int) =
  ## Update current bottleneck state
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return
  teamEconomy[teamId].currentBottleneck = detectBottleneck(controller, env, teamId)

proc getCurrentBottleneck*(teamId: int): BottleneckKind =
  ## Get current bottleneck for team
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return NoBottleneck
  teamEconomy[teamId].currentBottleneck

proc updateEconomy*(controller: Controller, env: Environment, teamId: int) =
  ## Main update function - call once per step
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return

  let step = env.currentStep

  # Record snapshot every few steps to avoid excessive memory
  if step mod 5 == 0:
    recordSnapshot(teamId, env)

  # Update flow rate periodically
  if step mod 10 == 0:
    updateFlowRate(teamId)

  # Update bottleneck detection periodically (iterates all agents)
  if step mod 3 == 0:
    updateBottleneck(controller, env, teamId)

proc resetEconomy*() =
  ## Reset all economy state (call on environment reset)
  zeroMem(addr teamEconomy, sizeof(teamEconomy))
  workerCountCacheStep = -1

# ============================================================================
# Tribute AI - Transfer surplus resources to allied teams in deficit
# ============================================================================

const
  TributeCheckInterval* = 50     ## Steps between tribute checks
  TributeSurplusThreshold* = 8   ## Must have at least this much of a resource to consider tributing
  TributeDeficitThreshold* = 3   ## Ally must have less than this to receive tribute
  TributeAmountFraction* = 0.25  ## Send 25% of surplus above threshold

proc evaluateTribute*(env: Environment, teamId: int) =
  ## Evaluate whether this team should tribute resources to another team.
  ## AI tributes when it has surplus and another team is in deficit.
  ## Called periodically from economy update.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return

  # Check each resource type (except ResourceNone and ResourceWater)
  for res in [ResourceFood, ResourceWood, ResourceGold, ResourceStone]:
    let ownStock = env.stockpileCount(teamId, res)
    if ownStock <= TributeSurplusThreshold:
      continue  # Not enough surplus to tribute

    # Find a team in deficit for this resource
    var bestTarget = -1
    var lowestStock = TributeDeficitThreshold
    for otherTeam in 0 ..< MapRoomObjectsTeams:
      if otherTeam == teamId:
        continue
      # Tribute to any team in deficit (allies preferred if alliances exist)
      let otherStock = env.stockpileCount(otherTeam, res)
      if otherStock < lowestStock:
        lowestStock = otherStock
        bestTarget = otherTeam

    if bestTarget < 0:
      continue  # No team in deficit

    # Calculate tribute amount: fraction of surplus above threshold
    let surplus = ownStock - TributeSurplusThreshold
    let tributeAmount = max(TributeMinAmount, int(float(surplus) * TributeAmountFraction))

    # Execute the tribute
    discard env.tributeResources(teamId, bestTarget, res, tributeAmount)
