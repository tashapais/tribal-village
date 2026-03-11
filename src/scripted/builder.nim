import ai_build_helpers
export ai_build_helpers

import options
export options

import coordination
export coordination

# Use shared optionGuard template from ai_types
template builderGuard(canName, termName: untyped, body: untyped) {.dirty.} =
  optionGuard(canName, termName, body)

const
  CoreInfrastructureKinds = [Granary, LumberCamp, Quarry, MiningCamp]
  TechBuildingKinds = [
    Barracks, Blacksmith, ArcheryRange, Market,
    WeavingLoom, ClayOven, Stable, Monastery,
    Outpost, University, SiegeWorkshop, MangonelWorkshop, TrebuchetWorkshop,
    Castle, Wonder
  ]
  DefenseRequestBuildingKinds = [Barracks, Outpost]
  CampThresholds: array[4, tuple[kind: ThingKind, nearbyKinds: set[ThingKind], minCount: int]] = [
    (kind: LumberCamp, nearbyKinds: {Tree}, minCount: 6),
    (kind: MiningCamp, nearbyKinds: {Gold}, minCount: 3),
    (kind: Quarry, nearbyKinds: {Stone, Stalagmite}, minCount: 6),
    (kind: Granary, nearbyKinds: {Wheat, Stubble, Bush, Fish}, minCount: 6)
  ]
  # Radius to search for resource clusters that need a drop-off
  StrategicDropoffSearchRadius = 30
  # Minimum resources in a cluster to warrant a strategic drop-off
  StrategicDropoffMinResources = 5
  # Don't build strategic drop-offs if an existing one is within this distance
  StrategicDropoffMinSpacing = 6
  BuilderThreatRadius* = 15
  BuilderFleeRadius* = 8
  BuilderFleeRadiusConst = BuilderFleeRadius

proc getTotalBuildingCount(controller: Controller, env: Environment, teamId: int): int =
  ## Count total buildings for a team using the public getBuildingCount API.
  for kind in ThingKind:
    if isBuildingKind(kind):
      result += controller.getBuildingCount(env, teamId, kind)

proc calculateWallRingRadius(controller: Controller, env: Environment, teamId: int,
                             altarPos: IVec2): int =
  ## Calculate adaptive wall radius based on per-settlement building count.
  ## Starts at WallRingBaseRadius and grows by 1 for every WallRingBuildingsPerRadius buildings.
  let totalBuildings =
    if altarPos.x >= 0:
      getTotalBuildingCountNear(env, teamId, altarPos)
    else:
      getTotalBuildingCount(controller, env, teamId)
  let extraRadius = totalBuildings div WallRingBuildingsPerRadius
  result = min(WallRingMaxRadius, WallRingBaseRadius + extraRadius)

proc isBuilderUnderThreat*(env: Environment, agent: Thing): bool =
  ## Check if the builder's home area is under threat from enemies.
  let teamId = getTeamId(agent)
  let basePos = if agent.homeAltar.x >= 0: agent.homeAltar else: agent.pos
  let nearestEnemy = findNearestEnemyAgentSpatial(env, basePos, teamId, BuilderThreatRadius)
  if not nearestEnemy.isNil:
    return true
  not findNearestEnemyBuildingSpatial(env, basePos, teamId, BuilderThreatRadius).isNil

builderGuard(canStartBuilderFlee, shouldTerminateBuilderFlee):
  not isNil(findNearbyEnemyForFlee(env, agent, BuilderFleeRadiusConst))

proc optBuilderFlee(controller: Controller, env: Environment, agent: Thing,
                    agentId: int, state: var AgentState): uint16 =
  ## Flee toward home altar when enemies are nearby.
  ## This causes builders to abandon construction when threatened.
  let enemy = findNearbyEnemyForFlee(env, agent, BuilderFleeRadiusConst)
  if isNil(enemy):
    return 0'u16
  # Move toward home altar for safety
  fleeToBase(controller, env, agent, agentId, state)

proc refreshDamagedBuildingCache*(controller: Controller, env: Environment) =
  ## Refresh the per-team damaged building cache if stale.
  ## Called once per step, caches all damaged building positions by team.
  if controller.damagedBuildingCacheStep == env.currentStep:
    return  # Cache is fresh
  controller.damagedBuildingCacheStep = env.currentStep
  # Clear counts
  for t in 0 ..< MapRoomObjectsTeams:
    controller.damagedBuildingCounts[t] = 0
  # Optimized: iterate only building kinds via thingsByKind instead of all env.things
  # TeamBuildingKinds already includes Wall and Door
  for bKind in TeamBuildingKinds:
    for thing in env.thingsByKind[bKind]:
      if thing.teamId < 0 or thing.teamId >= MapRoomObjectsTeams:
        continue
      if thing.maxHp <= 0 or thing.hp >= thing.maxHp:
        continue  # Not damaged or doesn't have hp
      let t = thing.teamId
      if controller.damagedBuildingCounts[t] < MaxDamagedBuildingsPerTeam:
        controller.damagedBuildingPositions[t][controller.damagedBuildingCounts[t]] = thing.pos
        controller.damagedBuildingCounts[t] += 1

proc findDamagedBuilding*(controller: Controller, env: Environment, agent: Thing): Thing =
  ## Find nearest damaged friendly building that needs repair.
  ## Returns nil if no damaged building found.
  ## Uses per-step cache to avoid redundant O(n) scans of env.things.
  let teamId = getTeamId(agent)
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return nil
  # Ensure cache is fresh
  refreshDamagedBuildingCache(controller, env)
  # Find nearest from cached positions
  var best: Thing = nil
  var bestDist = int.high
  for i in 0 ..< controller.damagedBuildingCounts[teamId]:
    let pos = controller.damagedBuildingPositions[teamId][i]
    let thing = env.getThing(pos)
    if thing.isNil:
      # Also check background grid for doors
      let bgThing = env.getBackgroundThing(pos)
      if bgThing.isNil:
        continue
      if bgThing.maxHp <= 0 or bgThing.hp >= bgThing.maxHp:
        continue  # No longer damaged
      let dist = int(chebyshevDist(pos, agent.pos))
      if dist < bestDist:
        bestDist = dist
        best = bgThing
    else:
      # Verify still damaged (may have been repaired since cache was built)
      if thing.maxHp <= 0 or thing.hp >= thing.maxHp:
        continue
      let dist = int(chebyshevDist(pos, agent.pos))
      if dist < bestDist:
        bestDist = dist
        best = thing
  best

builderGuard(canStartBuilderRepair, shouldTerminateBuilderRepair):
  not isNil(findDamagedBuilding(controller, env, agent))

proc optBuilderRepair(controller: Controller, env: Environment, agent: Thing,
                      agentId: int, state: var AgentState): uint16 =
  ## Move to and repair a damaged friendly building.
  let building = findDamagedBuilding(controller, env, agent)
  if isNil(building):
    return 0'u16
  actOrMove(controller, env, agent, agentId, state, building.pos, 3'u16)

proc anyMissingBuilding(controller: Controller, env: Environment, teamId: int,
                        kinds: openArray[ThingKind]): bool =
  for kind in kinds:
    if controller.getBuildingCount(env, teamId, kind) == 0:
      return true
  false

proc buildFirstMissing(controller: Controller, env: Environment, agent: Thing,
                       agentId: int, state: var AgentState, teamId: int,
                       kinds: openArray[ThingKind]): uint16 =
  for kind in kinds:
    let (did, act) = controller.tryBuildIfMissing(env, agent, agentId, state, teamId, kind)
    if did: return act
  0'u16

builderGuard(canStartBuilderPlantOnFertile, shouldTerminateBuilderPlantOnFertile):
  agent.inventoryWheat > 0 or agent.inventoryWood > 0

proc hasCarryingResources(agent: Thing): bool =
  for key, count in agent.inventory.pairs:
    if count > 0 and (isFoodItem(key) or isStockpileResourceKey(key)):
      return true
  false

builderGuard(canStartBuilderDropoffCarrying, shouldTerminateBuilderDropoffCarrying):
  hasCarryingResources(agent)

proc optBuilderDropoffCarrying(controller: Controller, env: Environment, agent: Thing,
                               agentId: int, state: var AgentState): uint16 =
  let (didDrop, dropAct) = controller.dropoffCarrying(
    env, agent, agentId, state,
    allowFood = true,
    allowWood = true,
    allowStone = true,
    allowGold = true
  )
  if didDrop: return dropAct
  0'u16

builderGuard(canStartBuilderPopCap, shouldTerminateBuilderPopCap):
  needsPopCapHouse(controller, env, getTeamId(agent))

proc optBuilderPopCap(controller: Controller, env: Environment, agent: Thing,
                      agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  let basePos = if agent.homeAltar.x >= 0: agent.homeAltar else: agent.pos
  state.basePosition = basePos
  let (didHouse, houseAct) =
    tryBuildHouseForPopCap(controller, env, agent, agentId, state, teamId, basePos)
  if didHouse: return houseAct
  0'u16

builderGuard(canStartBuilderCoreInfrastructure, shouldTerminateBuilderCoreInfrastructure):
  let altarPos = agent.homeAltar
  if altarPos.x >= 0:
    anyMissingBuildingNear(env, getTeamId(agent), CoreInfrastructureKinds, altarPos)
  else:
    anyMissingBuilding(controller, env, getTeamId(agent), CoreInfrastructureKinds)

proc optBuilderCoreInfrastructure(controller: Controller, env: Environment, agent: Thing,
                                  agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  let altarPos = agent.homeAltar
  if altarPos.x >= 0:
    # Per-settlement: only build infrastructure missing near this agent's home altar
    for kind in CoreInfrastructureKinds:
      let (did, act) = controller.tryBuildForSettlement(env, agent, agentId, state, teamId, kind, altarPos)
      if did: return act
    0'u16
  else:
    buildFirstMissing(controller, env, agent, agentId, state, teamId, CoreInfrastructureKinds)

proc millResourceCount(env: Environment, pos: IVec2): int =
  countNearbyThings(env, pos, 4, {Wheat, Stubble}) + countNearbyTerrain(env, pos, 4, {Fertile})

proc canStartBuilderMillNearResource(controller: Controller, env: Environment, agent: Thing,
                                     agentId: int, state: var AgentState): bool =
  if agent.homeAltar.x >= 0 and
      max(abs(agent.pos.x - agent.homeAltar.x), abs(agent.pos.y - agent.homeAltar.y)) <= 10:
    return false
  let teamId = getTeamId(agent)
  if millResourceCount(env, agent.pos) < 8:
    return false
  nearestFriendlyBuildingDistance(env, teamId, [Mill, Granary, TownCenter], agent.pos) > 5

proc shouldTerminateBuilderMillNearResource(controller: Controller, env: Environment, agent: Thing,
                                            agentId: int, state: var AgentState): bool =
  not canStartBuilderMillNearResource(controller, env, agent, agentId, state)

proc optBuilderMillNearResource(controller: Controller, env: Environment, agent: Thing,
                                agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  let (didMill, actMill) = controller.tryBuildNearResource(
    env, agent, agentId, state, teamId, Mill, millResourceCount(env, agent.pos),
    8, [Mill, Granary, TownCenter], 5)
  if didMill: return actMill
  0'u16

proc canStartBuilderPlantIfMills(controller: Controller, env: Environment, agent: Thing,
                                 agentId: int, state: var AgentState): bool =
  (agent.inventoryWheat > 0 or agent.inventoryWood > 0) and
    controller.getBuildingCount(env, getTeamId(agent), Mill) >= 2

proc shouldTerminateBuilderPlantIfMills(controller: Controller, env: Environment, agent: Thing,
                                        agentId: int, state: var AgentState): bool =
  agent.inventoryWheat <= 0 and agent.inventoryWood <= 0

proc optBuilderPlantIfMills(controller: Controller, env: Environment, agent: Thing,
                            agentId: int, state: var AgentState): uint16 =
  let (didPlant, actPlant) = controller.tryPlantOnFertile(env, agent, agentId, state)
  if didPlant: return actPlant
  0'u16

proc campResourceCount(env: Environment, pos: IVec2, entry: tuple[kind: ThingKind, nearbyKinds: set[ThingKind], minCount: int]): int =
  ## Count nearby resources for a camp threshold entry.
  ## For Granary, also counts Fertile terrain tiles since food grows there.
  result = countNearbyThings(env, pos, 4, entry.nearbyKinds)
  if entry.kind == Granary:
    result += countNearbyTerrain(env, pos, 4, {Fertile})

proc canStartBuilderCampThreshold(controller: Controller, env: Environment, agent: Thing,
                                  agentId: int, state: var AgentState): bool =
  let teamId = getTeamId(agent)
  for entry in CampThresholds:
    let nearbyCount = campResourceCount(env, agent.pos, entry)
    if nearbyCount < entry.minCount:
      continue
    let dist = nearestFriendlyBuildingDistance(env, teamId, [entry.kind], agent.pos)
    if dist > 3:
      return true
  false

proc shouldTerminateBuilderCampThreshold(controller: Controller, env: Environment, agent: Thing,
                                         agentId: int, state: var AgentState): bool =
  ## Terminate when camp built nearby or conditions no longer met
  not canStartBuilderCampThreshold(controller, env, agent, agentId, state)

proc optBuilderCampThreshold(controller: Controller, env: Environment, agent: Thing,
                             agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  for entry in CampThresholds:
    let nearbyCount = campResourceCount(env, agent.pos, entry)
    let (did, act) = controller.tryBuildCampThreshold(
      env, agent, agentId, state, teamId, entry.kind,
      nearbyCount, entry.minCount,
      [entry.kind]
    )
    if did: return act
  0'u16

proc findStrategicDropoffTarget(env: Environment, agent: Thing, teamId: int): tuple[pos: IVec2, kind: ThingKind, found: bool] =
  ## Scan for resource clusters that are far from existing drop-offs.
  ## Samples positions on a grid within StrategicDropoffSearchRadius to find
  ## high-density resource areas lacking a nearby drop-off building.
  ## Returns the best cluster center and the kind of drop-off to build.
  result = (pos: ivec2(-1, -1), kind: LumberCamp, found: false)
  let basePos = if agent.homeAltar.x >= 0: agent.homeAltar else: agent.pos
  var bestScore = 0
  const gridStep = 4  # Sample every 4 tiles for efficiency
  let minX = max(0, basePos.x - StrategicDropoffSearchRadius)
  let maxX = min(MapWidth - 1, basePos.x + StrategicDropoffSearchRadius)
  let minY = max(0, basePos.y - StrategicDropoffSearchRadius)
  let maxY = min(MapHeight - 1, basePos.y + StrategicDropoffSearchRadius)
  for entry in CampThresholds:
    var x = minX
    while x <= maxX:
      var y = minY
      while y <= maxY:
        let samplePos = ivec2(x.int32, y.int32)
        let resCount = campResourceCount(env, samplePos, entry)
        if resCount >= StrategicDropoffMinResources:
          # Check that no existing drop-off is nearby
          let dropoffDist = nearestFriendlyBuildingDistance(env, teamId, [entry.kind], samplePos)
          if dropoffDist > StrategicDropoffMinSpacing:
            # Score: resource density, preferring clusters farther from existing drop-offs
            let score = resCount + min(dropoffDist, 20)
            if score > bestScore:
              bestScore = score
              result = (pos: samplePos, kind: entry.kind, found: true)
        y += gridStep
      x += gridStep

var strategicDropoffCache: PerAgentCache[tuple[pos: IVec2, kind: ThingKind, found: bool]]

proc canStartBuilderStrategicDropoff(controller: Controller, env: Environment, agent: Thing,
                                     agentId: int, state: var AgentState): bool =
  ## Check if there's a resource cluster that needs a strategic drop-off.
  ## Only activates when the team already has basic infrastructure.
  let teamId = getTeamId(agent)
  # Need at least a Granary or LumberCamp before doing strategic placement
  if controller.getBuildingCount(env, teamId, Granary) == 0 and
     controller.getBuildingCount(env, teamId, LumberCamp) == 0:
    return false
  # Use per-agent cache to avoid expensive grid scan on every canStart check
  let cached = strategicDropoffCache.getWithAgent(env, agent,
    proc(env: Environment, agent: Thing): tuple[pos: IVec2, kind: ThingKind, found: bool] =
      findStrategicDropoffTarget(env, agent, getTeamId(agent)))
  cached.found

proc shouldTerminateBuilderStrategicDropoff(controller: Controller, env: Environment, agent: Thing,
                                            agentId: int, state: var AgentState): bool =
  not canStartBuilderStrategicDropoff(controller, env, agent, agentId, state)

proc optBuilderStrategicDropoff(controller: Controller, env: Environment, agent: Thing,
                                agentId: int, state: var AgentState): uint16 =
  ## Move to a resource cluster and build a drop-off building there.
  let teamId = getTeamId(agent)
  let cached = strategicDropoffCache.getWithAgent(env, agent,
    proc(env: Environment, agent: Thing): tuple[pos: IVec2, kind: ThingKind, found: bool] =
      findStrategicDropoffTarget(env, agent, getTeamId(agent)))
  if not cached.found:
    return 0'u16
  # If we're already near the cluster, try to build
  let distToCluster = int(chebyshevDist(agent.pos, cached.pos))
  if distToCluster <= 4:
    for entry in CampThresholds:
      if entry.kind == cached.kind:
        let resCount = campResourceCount(env, agent.pos, entry)
        let (did, act) = controller.tryBuildCampThreshold(
          env, agent, agentId, state, teamId, entry.kind,
          resCount, 1,  # Lower threshold since we already validated the cluster
          [entry.kind],
          minSpacing = StrategicDropoffMinSpacing
        )
        if did: return act
        break
  # Move toward the cluster
  controller.moveTo(env, agent, agentId, state, cached.pos)

let BuilderStrategicDropoffOption = OptionDef(
  name: "BuilderStrategicDropoff", canStart: canStartBuilderStrategicDropoff,
  shouldTerminate: shouldTerminateBuilderStrategicDropoff, act: optBuilderStrategicDropoff,
  interruptible: true)

builderGuard(canStartBuilderTechBuildings, shouldTerminateBuilderTechBuildings):
  anyMissingBuilding(controller, env, getTeamId(agent), TechBuildingKinds)

proc optBuilderTechBuildings(controller: Controller, env: Environment, agent: Thing,
                             agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  buildFirstMissing(controller, env, agent, agentId, state, teamId, TechBuildingKinds)

proc canStartBuilderWallRing(controller: Controller, env: Environment, agent: Thing,
                             agentId: int, state: var AgentState): bool =
  let teamId = getTeamId(agent)
  agent.homeAltar.x >= 0 and
    controller.getBuildingCount(env, teamId, LumberCamp) > 0 and
    controller.getBuildingCount(env, teamId, Wall) < MaxWallsPerTeam and
    env.stockpileCount(teamId, ResourceWood) >= 3

proc shouldTerminateBuilderWallRing(controller: Controller, env: Environment, agent: Thing,
                                    agentId: int, state: var AgentState): bool =
  not canStartBuilderWallRing(controller, env, agent, agentId, state)

proc optBuilderWallRing(controller: Controller, env: Environment, agent: Thing,
                        agentId: int, state: var AgentState): uint16 =
  if not canStartBuilderWallRing(controller, env, agent, agentId, state):
    return 0'u16
  let teamId = getTeamId(agent)
  let altarPos = agent.homeAltar
  var wallTarget = ivec2(-1, -1)
  var doorTarget = ivec2(-1, -1)
  var ringDoorCount = 0
  var bestBlocked = int.high
  var bestDist = int.high
  # Calculate adaptive wall radius based on per-settlement building count
  let baseRadius = calculateWallRingRadius(controller, env, teamId, altarPos)
  let wallRingRadii = [baseRadius, baseRadius - WallRingRadiusSlack, baseRadius + WallRingRadiusSlack]
  for radius in wallRingRadii:
    var blocked = 0
    var doorCount = 0
    var candidateWall = ivec2(-1, -1)
    var candidateDoor = ivec2(-1, -1)
    var candidateWallDist = int.high
    var candidateDoorDist = int.high
    for dx in -radius .. radius:
      for dy in -radius .. radius:
        if max(abs(dx), abs(dy)) != radius:
          continue
        let pos = altarPos + ivec2(dx.int32, dy.int32)
        if not isValidPos(pos):
          inc blocked
          continue
        let posTerrain = env.terrain[pos.x][pos.y]
        if posTerrain == TerrainRoad or isRampTerrain(posTerrain):
          inc blocked
          continue
        let wallThing = env.getThing(pos)
        if not isNil(wallThing) and wallThing.kind == Wall:
          continue
        let doorThing = env.getBackgroundThing(pos)
        if not isNil(doorThing) and doorThing.kind == Door:
          inc doorCount
          continue
        if not env.canPlace(pos):
          inc blocked
          continue
        let dist = int(chebyshevDist(agent.pos, pos))
        let isDoorSlot = (dx == 0 or dy == 0 or abs(dx) == abs(dy))
        if isDoorSlot:
          if dist < candidateDoorDist:
            candidateDoorDist = dist
            candidateDoor = pos
        else:
          if dist < candidateWallDist:
            candidateWallDist = dist
            candidateWall = pos
    let candidateDist = min(candidateWallDist, candidateDoorDist)
    if candidateWall.x < 0 and candidateDoor.x < 0:
      continue
    if blocked < bestBlocked or (blocked == bestBlocked and candidateDist < bestDist):
      bestBlocked = blocked
      bestDist = candidateDist
      wallTarget = candidateWall
      doorTarget = candidateDoor
      ringDoorCount = doorCount
  if wallTarget.x >= 0:
    if env.canAffordBuild(agent, thingItem("Wall")):
      let (did, act) = goToAdjacentAndBuild(
        controller, env, agent, agentId, state, wallTarget, BuildIndexWall
      )
      if did: return act
    else:
      let (didWood, actWood) = controller.ensureWood(env, agent, agentId, state)
      if didWood: return actWood
  if doorTarget.x >= 0:
    if ringDoorCount < WallRingMaxDoors and env.canAffordBuild(agent, thingItem("Door")):
      let (didDoor, actDoor) = goToAdjacentAndBuild(
        controller, env, agent, agentId, state, doorTarget, BuildIndexDoor
      )
      if didDoor: return actDoor
    if env.canAffordBuild(agent, thingItem("Wall")):
      let (didWall, actWall) = goToAdjacentAndBuild(
        controller, env, agent, agentId, state, doorTarget, BuildIndexWall
      )
      if didWall: return actWall
    else:
      let (didWood, actWood) = controller.ensureWood(env, agent, agentId, state)
      if didWood: return actWood
  0'u16

# Coordination-responsive behavior: respond to defense requests by building military structures
proc canStartBuilderDefenseResponse(controller: Controller, env: Environment, agent: Thing,
                                    agentId: int, state: var AgentState): bool =
  ## Check if there's a defense request and we can respond by building
  let teamId = getTeamId(agent)
  if not builderShouldPrioritizeDefense(teamId):
    return false
  # Check if we're missing any defense buildings
  for kind in DefenseRequestBuildingKinds:
    if controller.getBuildingCount(env, teamId, kind) == 0:
      return true
  false

proc shouldTerminateBuilderDefenseResponse(controller: Controller, env: Environment, agent: Thing,
                                           agentId: int, state: var AgentState): bool =
  ## Terminate when no more defense requests or defense buildings built
  let teamId = getTeamId(agent)
  if not builderShouldPrioritizeDefense(teamId):
    return true
  # Check if all defense buildings exist
  for kind in DefenseRequestBuildingKinds:
    if controller.getBuildingCount(env, teamId, kind) == 0:
      return false
  true

proc optBuilderDefenseResponse(controller: Controller, env: Environment, agent: Thing,
                               agentId: int, state: var AgentState): uint16 =
  ## Build military/defensive structures in response to coordination request
  let teamId = getTeamId(agent)
  for kind in DefenseRequestBuildingKinds:
    if controller.getBuildingCount(env, teamId, kind) == 0:
      let (did, act) = controller.tryBuildIfMissing(env, agent, agentId, state, teamId, kind)
      if did:
        # Mark the defense request as fulfilled once we start building
        markDefenseRequestFulfilled(teamId)
        return act
  0'u16

builderGuard(canStartBuilderDock, shouldTerminateBuilderDock):
  let checkPos = if agent.homeAltar.x >= 0: agent.homeAltar else: agent.pos
  controller.getBuildingCount(env, getTeamId(agent), Dock) == 0 and
    hasWaterNearby(env, checkPos, 20)

proc optBuilderDock(controller: Controller, env: Environment, agent: Thing,
                    agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  let (did, act) = controller.tryBuildDockIfMissing(env, agent, agentId, state, teamId)
  if did: return act
  0'u16

proc teamNavalCount(env: Environment, teamId: int): int =
  ## Count alive naval units for a team.
  ## Delegates to canonical countTeamNavalAgents from ai_utils.
  countTeamNavalAgents(env, teamId)

builderGuard(canStartBuilderNavalTrain, shouldTerminateBuilderNavalTrain):
  agent.unitClass == UnitVillager and
    controller.getBuildingCount(env, getTeamId(agent), Dock) > 0 and
    teamNavalCount(env, getTeamId(agent)) < MaxNavalPerTeam and
    env.canSpendStockpile(getTeamId(agent), buildingTrainCosts(Dock))

proc optBuilderNavalTrain(controller: Controller, env: Environment, agent: Thing,
                          agentId: int, state: var AgentState): uint16 =
  ## Send a builder to the Dock to create one naval unit.
  let teamId = getTeamId(agent)
  let dock = env.findNearestFriendlyThingSpiral(state, teamId, Dock)
  if isNil(dock):
    return 0'u16
  # Queue training if no ready entry
  if not dock.productionQueueHasReady() and
     dock.productionQueue.entries.len < ProductionQueueMaxSize:
    discard env.tryBatchQueueTrain(dock, teamId, 1)
  actOrMove(controller, env, agent, agentId, state, dock.pos, 3'u16)

let BuilderNavalTrainOption* = OptionDef(
  name: "BuilderNavalTrain",
  canStart: canStartBuilderNavalTrain,
  shouldTerminate: shouldTerminateBuilderNavalTrain,
  act: optBuilderNavalTrain,
  interruptible: true
)

proc builderShouldBuildSiege(controller: Controller, env: Environment, teamId: int): bool =
  ## Check if builder should build siege workshop due to request
  if not hasSiegeBuildRequest(teamId):
    return false
  # Only if we don't already have one
  controller.getBuildingCount(env, teamId, SiegeWorkshop) == 0

builderGuard(canStartBuilderSiegeResponse, shouldTerminateBuilderSiegeResponse):
  builderShouldBuildSiege(controller, env, getTeamId(agent))

proc optBuilderSiegeResponse(controller: Controller, env: Environment, agent: Thing,
                             agentId: int, state: var AgentState): uint16 =
  ## Build siege workshop in response to coordination request
  let teamId = getTeamId(agent)
  let (did, act) = controller.tryBuildIfMissing(env, agent, agentId, state, teamId, SiegeWorkshop)
  if did:
    markSiegeBuildRequestFulfilled(teamId)
    return act
  0'u16

proc minBasicStockpile(env: Environment, teamId: int): int =
  ## Returns the minimum stockpile count among food, wood, and stone.
  result = env.stockpileCount(teamId, ResourceFood)
  let wood = env.stockpileCount(teamId, ResourceWood)
  let stone = env.stockpileCount(teamId, ResourceStone)
  if wood < result: result = wood
  if stone < result: result = stone

builderGuard(canStartBuilderGatherScarce, shouldTerminateBuilderGatherScarce):
  agent.unitClass == UnitVillager and minBasicStockpile(env, getTeamId(agent)) < 5

proc optBuilderGatherScarce(controller: Controller, env: Environment, agent: Thing,
                            agentId: int, state: var AgentState): uint16 =
  let teamId = getTeamId(agent)
  let food = env.stockpileCount(teamId, ResourceFood)
  let wood = env.stockpileCount(teamId, ResourceWood)
  let stone = env.stockpileCount(teamId, ResourceStone)
  var targetRes = ResourceFood
  var best = food
  if wood < best:
    best = wood
    targetRes = ResourceWood
  if stone < best:
    best = stone
    targetRes = ResourceStone
  if best < 5:
    case targetRes
    of ResourceFood:
      let (didFood, actFood) = controller.ensureWheat(env, agent, agentId, state)
      if didFood: return actFood
    of ResourceWood:
      let (didWood, actWood) = controller.ensureWood(env, agent, agentId, state)
      if didWood: return actWood
    of ResourceStone:
      let (didStone, actStone) = controller.ensureStone(env, agent, agentId, state)
      if didStone: return actStone
    else:
      discard
  0'u16

proc canStartBuilderVisitTradingHub(controller: Controller, env: Environment, agent: Thing,
                                    agentId: int, state: var AgentState): bool =
  if agent.inventory.len != 0:
    return false
  let hub = findNearestNeutralHub(env, agent.pos)
  not isNil(hub) and chebyshevDist(agent.pos, hub.pos) > 6'i32

proc shouldTerminateBuilderVisitTradingHub(controller: Controller, env: Environment, agent: Thing,
                                           agentId: int, state: var AgentState): bool =
  not canStartBuilderVisitTradingHub(controller, env, agent, agentId, state)

proc optBuilderVisitTradingHub(controller: Controller, env: Environment, agent: Thing,
                               agentId: int, state: var AgentState): uint16 =
  let hub = findNearestNeutralHub(env, agent.pos)
  if isNil(hub):
    return 0'u16
  if isAdjacent(agent.pos, hub.pos):
    return 0'u16
  controller.moveTo(env, agent, agentId, state, hub.pos)

# Shared OptionDefs used in both BuilderOptions and BuilderOptionsThreat
let BuilderFleeOption = OptionDef(
  name: "BuilderFlee", canStart: canStartBuilderFlee,
  shouldTerminate: shouldTerminateBuilderFlee, act: optBuilderFlee,
  interruptible: false)
let BuilderPlantOnFertileOption = OptionDef(
  name: "BuilderPlantOnFertile", canStart: canStartBuilderPlantOnFertile,
  shouldTerminate: shouldTerminateBuilderPlantOnFertile, act: optPlantOnFertile,
  interruptible: true)
let BuilderWallRingOption = OptionDef(
  name: "BuilderWallRing", canStart: canStartBuilderWallRing,
  shouldTerminate: shouldTerminateBuilderWallRing, act: optBuilderWallRing,
  interruptible: true)
let BuilderDefenseResponseOption = OptionDef(
  name: "BuilderDefenseResponse", canStart: canStartBuilderDefenseResponse,
  shouldTerminate: shouldTerminateBuilderDefenseResponse, act: optBuilderDefenseResponse,
  interruptible: true)
let BuilderSiegeResponseOption = OptionDef(
  name: "BuilderSiegeResponse", canStart: canStartBuilderSiegeResponse,
  shouldTerminate: shouldTerminateBuilderSiegeResponse, act: optBuilderSiegeResponse,
  interruptible: true)
let BuilderRepairOption = OptionDef(
  name: "BuilderRepair", canStart: canStartBuilderRepair,
  shouldTerminate: shouldTerminateBuilderRepair, act: optBuilderRepair,
  interruptible: true)
let BuilderMillNearResourceOption = OptionDef(
  name: "BuilderMillNearResource", canStart: canStartBuilderMillNearResource,
  shouldTerminate: shouldTerminateBuilderMillNearResource, act: optBuilderMillNearResource,
  interruptible: true)
let BuilderPlantIfMillsOption = OptionDef(
  name: "BuilderPlantIfMills", canStart: canStartBuilderPlantIfMills,
  shouldTerminate: shouldTerminateBuilderPlantIfMills, act: optBuilderPlantIfMills,
  interruptible: true)
let BuilderCampThresholdOption = OptionDef(
  name: "BuilderCampThreshold", canStart: canStartBuilderCampThreshold,
  shouldTerminate: shouldTerminateBuilderCampThreshold, act: optBuilderCampThreshold,
  interruptible: true)
let BuilderDockOption = OptionDef(
  name: "BuilderDock", canStart: canStartBuilderDock,
  shouldTerminate: shouldTerminateBuilderDock, act: optBuilderDock,
  interruptible: true)
let BuilderVisitTradingHubOption = OptionDef(
  name: "BuilderVisitTradingHub", canStart: canStartBuilderVisitTradingHub,
  shouldTerminate: shouldTerminateBuilderVisitTradingHub, act: optBuilderVisitTradingHub,
  interruptible: true)

let BuilderOptions* = [
  TownBellGarrisonOption,  # Highest priority: town bell recall overrides everything
  BuilderFleeOption,
  EmergencyHealOption,
  BuilderPlantOnFertileOption,
  OptionDef(name: "BuilderDropoffCarrying", canStart: canStartBuilderDropoffCarrying,
    shouldTerminate: shouldTerminateBuilderDropoffCarrying, act: optBuilderDropoffCarrying,
    interruptible: true),
  OptionDef(name: "BuilderPopCap", canStart: canStartBuilderPopCap,
    shouldTerminate: shouldTerminateBuilderPopCap, act: optBuilderPopCap,
    interruptible: true),
  OptionDef(name: "BuilderCoreInfrastructure", canStart: canStartBuilderCoreInfrastructure,
    shouldTerminate: shouldTerminateBuilderCoreInfrastructure, act: optBuilderCoreInfrastructure,
    interruptible: true),
  BuilderMillNearResourceOption,
  BuilderPlantIfMillsOption,
  BuilderDefenseResponseOption,      # Defense before drop-off spam (tv-88y)
  BuilderSiegeResponseOption,
  OptionDef(name: "BuilderTechBuildings", canStart: canStartBuilderTechBuildings,
    shouldTerminate: shouldTerminateBuilderTechBuildings, act: optBuilderTechBuildings,
    interruptible: true),
  BuilderRepairOption,
  BuilderCampThresholdOption,        # Drop-offs after tech/military buildings (tv-88y)
  BuilderStrategicDropoffOption,     # Proactive drop-off placement near distant clusters (tv-gn2)
  ResearchUniversityTechOption,
  BuilderDockOption,
  BuilderNavalTrainOption,
  ResearchCastleTechOption,
  ResearchUnitUpgradeOption,
  ResearchBlacksmithUpgradeOption,
  ResearchEconomyTechOption,
  BuilderWallRingOption,
  OptionDef(name: "BuilderGatherScarce", canStart: canStartBuilderGatherScarce,
    shouldTerminate: shouldTerminateBuilderGatherScarce, act: optBuilderGatherScarce,
    interruptible: true),
  MarketTradeOption,
  BuilderVisitTradingHubOption,
  SmeltGoldOption,
  CraftBreadOption,
  StoreValuablesOption,
  FallbackSearchOption
]

# BuilderOptionsThreat: Reordered priorities for when under threat.
# Priority order: Flee -> Defense -> TechBuildings -> Repair -> Infrastructure -> WallRing
# (tv-il11vv: Moved WallRing lower to prioritize military buildings over walls)
let BuilderOptionsThreat* = [
  TownBellGarrisonOption,  # Highest priority: town bell recall overrides everything
  BuilderFleeOption,
  EmergencyHealOption,
  BuilderPlantOnFertileOption,
  OptionDef(name: "BuilderDropoffCarrying", canStart: canStartBuilderDropoffCarrying,
    shouldTerminate: optionsAlwaysTerminate, act: optBuilderDropoffCarrying,
    interruptible: true),
  OptionDef(name: "BuilderPopCap", canStart: canStartBuilderPopCap,
    shouldTerminate: optionsAlwaysTerminate, act: optBuilderPopCap,
    interruptible: true),
  BuilderDefenseResponseOption,  # Military buildings prioritized in threat mode
  BuilderSiegeResponseOption,
  OptionDef(name: "BuilderTechBuildings", canStart: canStartBuilderTechBuildings,
    shouldTerminate: optionsAlwaysTerminate, act: optBuilderTechBuildings,
    interruptible: true),
  BuilderRepairOption,           # Repair existing structures
  ResearchUniversityTechOption,
  BuilderDockOption,
  BuilderNavalTrainOption,
  ResearchCastleTechOption,
  ResearchUnitUpgradeOption,
  ResearchBlacksmithUpgradeOption,
  ResearchEconomyTechOption,
  OptionDef(name: "BuilderCoreInfrastructure", canStart: canStartBuilderCoreInfrastructure,
    shouldTerminate: optionsAlwaysTerminate, act: optBuilderCoreInfrastructure,
    interruptible: true),
  BuilderMillNearResourceOption,
  BuilderPlantIfMillsOption,
  BuilderCampThresholdOption,
  BuilderStrategicDropoffOption,  # Strategic drop-off placement (tv-gn2)
  BuilderWallRingOption,         # Walls after infrastructure (tv-il11vv)
  OptionDef(name: "BuilderGatherScarce", canStart: canStartBuilderGatherScarce,
    shouldTerminate: optionsAlwaysTerminate, act: optBuilderGatherScarce,
    interruptible: true),
  MarketTradeOption,
  BuilderVisitTradingHubOption,
  SmeltGoldOption,
  CraftBreadOption,
  StoreValuablesOption,
  FallbackSearchOption
]
