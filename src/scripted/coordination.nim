# Inter-role coordination system
# Allows agents to communicate needs across role boundaries:
# - Gatherer requests protection from Fighter when under attack
# - Fighter requests defensive structures from Builder when seeing threats
# - Builder responds to defense requests by prioritizing walls/towers

# This file is designed to be used alongside the include-based ai system.
# It doesn't import ai_core to avoid type namespace conflicts.

import vmath
import ../common_types, ../environment

const
  MaxCoordinationRequests* = 16  # Max pending requests per team
  RequestExpirationSteps* = 60  # Requests expire after N steps
  DuplicateWindowSteps* = 30    # Duplicate detection window (steps)
  ProtectionResponseRadius* = 15  # Fighters respond to protection requests within this radius
  DefenseRequestRadius* = 20  # Distance from threat that triggers defense request

type
  CoordinationRequestKind* = enum
    RequestProtection   # Gatherer requests Fighter escort
    RequestDefense      # Fighter requests Builder to build defensive structures
    RequestSiegeBuild   # Fighter requests Builder to build siege workshop

  CoordinationPriority* = enum
    PriorityLow = 0
    PriorityNormal = 1
    PriorityHigh = 2        # Urgent requests (e.g. active combat)

  CoordinationRequest* = object
    kind*: CoordinationRequestKind
    teamId*: int
    requesterId*: int       # Agent ID that made the request
    requesterPos*: IVec2    # Position of requester
    threatPos*: IVec2       # Position of the threat (for defense/protection)
    createdStep*: int       # Step when created
    fulfilled*: bool        # Whether request has been handled
    priority*: CoordinationPriority  # Request priority for fulfillment ordering

  CoordinationState* = object
    requests*: array[MaxCoordinationRequests, CoordinationRequest]
    requestCount*: int

# Team-indexed coordination state (global storage)
var teamCoordination*: array[MapRoomObjectsTeams, CoordinationState]

template validTeamId(teamId: int): bool =
  teamId >= 0 and teamId < MapRoomObjectsTeams

template coordState(teamId: int): var CoordinationState =
  teamCoordination[teamId]

proc hasUnfulfilledRequest(teamId: int, kind: CoordinationRequestKind): bool =
  ## Check if there's an unfulfilled request of the given kind
  if not validTeamId(teamId):
    return false
  let state = addr coordState(teamId)
  for i in 0 ..< state.requestCount:
    if state.requests[i].kind == kind and not state.requests[i].fulfilled:
      return true
  false

proc markRequestFulfilled(teamId: int, kind: CoordinationRequestKind) =
  ## Mark the highest-priority unfulfilled request of the given kind as fulfilled
  if not validTeamId(teamId):
    return
  let state = addr coordState(teamId)
  var bestIdx = -1
  var bestPriority = PriorityLow
  for i in 0 ..< state.requestCount:
    let req = addr state.requests[i]
    if req.kind == kind and not req.fulfilled:
      if bestIdx < 0 or req.priority > bestPriority:
        bestIdx = i
        bestPriority = req.priority
  if bestIdx >= 0:
    state.requests[bestIdx].fulfilled = true

proc clearExpiredRequests*(step: int) =
  ## Remove expired and fulfilled requests
  for teamId in 0 ..< MapRoomObjectsTeams:
    let state = addr coordState(teamId)
    var writeIdx = 0
    for readIdx in 0 ..< state.requestCount:
      let req = state.requests[readIdx]
      if not req.fulfilled and (step - req.createdStep) < RequestExpirationSteps:
        state.requests[writeIdx] = req
        inc writeIdx
    state.requestCount = writeIdx

proc addRequest*(teamId: int, kind: CoordinationRequestKind,
                 requesterId: int, requesterPos, threatPos: IVec2, step: int,
                 priority: CoordinationPriority = PriorityNormal): bool =
  ## Add a coordination request. Returns true if added successfully.
  if not validTeamId(teamId):
    return false
  let state = addr coordState(teamId)
  # Check for duplicate (same requester, same kind, recent)
  for i in 0 ..< state.requestCount:
    let req = state.requests[i]
    if req.requesterId == requesterId and req.kind == kind and
       (step - req.createdStep) < DuplicateWindowSteps:
      return false
  if state.requestCount >= MaxCoordinationRequests:
    # Remove oldest request to make room
    for i in 1 ..< MaxCoordinationRequests:
      state.requests[i-1] = state.requests[i]
    dec state.requestCount
  let idx = state.requestCount
  state.requests[idx] = CoordinationRequest(
    kind: kind,
    teamId: teamId,
    requesterId: requesterId,
    requesterPos: requesterPos,
    threatPos: threatPos,
    createdStep: step,
    fulfilled: false,
    priority: priority
  )
  inc state.requestCount
  true

proc findNearestProtectionRequest*(teamId: int, agentPos: IVec2): ptr CoordinationRequest =
  ## Find the highest-priority, nearest unfulfilled protection request within response radius
  if not validTeamId(teamId):
    return nil
  let state = addr coordState(teamId)
  var bestDist = int.high
  var bestPriority = PriorityLow
  var bestReq: ptr CoordinationRequest = nil
  for i in 0 ..< state.requestCount:
    let req = addr state.requests[i]
    if req.kind != RequestProtection or req.fulfilled:
      continue
    let dx = abs(agentPos.x - req.requesterPos.x)
    let dy = abs(agentPos.y - req.requesterPos.y)
    let dist = int(if dx > dy: dx else: dy)  # chebyshevDist inline
    if dist <= ProtectionResponseRadius:
      if req.priority > bestPriority or
         (req.priority == bestPriority and dist < bestDist):
        bestDist = dist
        bestPriority = req.priority
        bestReq = req
  bestReq

proc hasDefenseRequest*(teamId: int): bool =
  ## Check if there's an unfulfilled defense request
  hasUnfulfilledRequest(teamId, RequestDefense)

proc hasSiegeBuildRequest*(teamId: int): bool =
  ## Check if there's an unfulfilled siege build request
  hasUnfulfilledRequest(teamId, RequestSiegeBuild)

proc markDefenseRequestFulfilled*(teamId: int) =
  ## Mark the highest-priority unfulfilled defense request as fulfilled
  markRequestFulfilled(teamId, RequestDefense)

proc markSiegeBuildRequestFulfilled*(teamId: int) =
  ## Mark the highest-priority unfulfilled siege build request as fulfilled
  markRequestFulfilled(teamId, RequestSiegeBuild)

# --- Coordination request creators (called from role behaviors) ---

proc requestProtectionFromFighter*(env: Environment, agent: Thing, threatPos: IVec2) =
  ## Called by Gatherer when fleeing - requests Fighter escort
  let teamId = getTeamId(agent)
  discard addRequest(teamId, RequestProtection, agent.agentId, agent.pos, threatPos, env.currentStep)

proc requestDefenseFromBuilder*(env: Environment, agent: Thing, threatPos: IVec2) =
  ## Called by Fighter when seeing enemy threat - requests Builder to prioritize defensive structures
  let teamId = getTeamId(agent)
  discard addRequest(teamId, RequestDefense, agent.agentId, agent.pos, threatPos, env.currentStep)

proc requestSiegeFromBuilder*(env: Environment, agent: Thing) =
  ## Called by Fighter when seeing enemy structures - requests Builder to build siege workshop
  let teamId = getTeamId(agent)
  discard addRequest(teamId, RequestSiegeBuild, agent.agentId, agent.pos, agent.pos, env.currentStep)

# --- Coordination behavior helpers ---

proc fighterShouldEscort*(env: Environment, agent: Thing): tuple[should: bool, target: IVec2] =
  ## Check if fighter should respond to a protection request
  ## Returns (true, target position) if should escort
  let teamId = getTeamId(agent)
  let req = findNearestProtectionRequest(teamId, agent.pos)
  if isNil(req):
    return (false, ivec2(-1, -1))
  # Check if the requester is still alive and still needs help
  if req.requesterId >= 0 and req.requesterId < MapAgents:
    let requester = env.agents[req.requesterId]
    if isAgentAlive(env, requester):
      # Move toward the requester's current position
      return (true, requester.pos)
  (false, ivec2(-1, -1))

proc builderShouldPrioritizeDefense*(teamId: int): bool =
  ## Check if builder should prioritize defensive structures
  hasDefenseRequest(teamId)

# Note: builderShouldBuildSiege is implemented in builder.nim with access to Controller

# ============================================================================
# Resource Reservation System
# ============================================================================
# Prevents multiple agents from targeting the same resource node simultaneously.
# Agents 'reserve' a position before moving to gather; other agents skip reserved
# targets. Reservations expire after N steps or when the reserving agent dies.

const
  MaxResourceReservations* = 64  # Max concurrent reservations per team
  ReservationExpirationSteps* = 30  # Reservations expire after N steps

type
  ResourceReservation* = object
    pos*: IVec2           # Reserved resource position
    agentId*: int32       # Agent that holds the reservation
    createdStep*: int32   # Step when reservation was created

  ReservationState* = object
    reservations*: array[MaxResourceReservations, ResourceReservation]
    count*: int

# Team-indexed reservation state (global storage)
var teamReservations*: array[MapRoomObjectsTeams, ReservationState]

template resState(teamId: int): var ReservationState =
  teamReservations[teamId]

proc clearExpiredReservations*(env: Environment) =
  ## Remove reservations that have expired or whose agent is dead.
  let currentStep = env.currentStep
  for teamId in 0 ..< MapRoomObjectsTeams:
    let state = addr resState(teamId)
    var writeIdx = 0
    for readIdx in 0 ..< state.count:
      let res = state.reservations[readIdx]
      let expired = (currentStep - res.createdStep) >= ReservationExpirationSteps
      let agentDead = res.agentId >= 0 and res.agentId < env.agents.len and
                      not isAgentAlive(env, env.agents[res.agentId])
      if not expired and not agentDead:
        if writeIdx != readIdx:
          state.reservations[writeIdx] = res
        inc writeIdx
    state.count = writeIdx

proc isResourceReserved*(teamId: int, pos: IVec2, excludeAgentId: int = -1): bool =
  ## Check if a resource position is reserved by another agent on this team.
  ## excludeAgentId allows the reserving agent to see its own reservation as unreserved.
  if not validTeamId(teamId):
    return false
  let state = addr resState(teamId)
  for i in 0 ..< state.count:
    let res = state.reservations[i]
    if res.pos == pos and res.agentId != excludeAgentId.int32:
      return true
  false

proc reserveResource*(teamId: int, agentId: int, pos: IVec2, step: int): bool =
  ## Reserve a resource position for an agent. Returns true if reserved successfully.
  ## If the agent already has a reservation, it is replaced (one reservation per agent).
  if not validTeamId(teamId):
    return false
  let state = addr resState(teamId)
  # Check if another agent already reserved this position
  for i in 0 ..< state.count:
    let res = state.reservations[i]
    if res.pos == pos and res.agentId != agentId.int32:
      return false  # Already reserved by someone else
  # Remove any existing reservation by this agent (one per agent)
  var writeIdx = 0
  for readIdx in 0 ..< state.count:
    if state.reservations[readIdx].agentId != agentId.int32:
      if writeIdx != readIdx:
        state.reservations[writeIdx] = state.reservations[readIdx]
      inc writeIdx
    # If same agent re-reserving same pos, also skip (will re-add below)
  state.count = writeIdx
  # Add new reservation
  if state.count >= MaxResourceReservations:
    return false  # No space
  let idx = state.count
  state.reservations[idx] = ResourceReservation(
    pos: pos,
    agentId: agentId.int32,
    createdStep: step.int32
  )
  inc state.count
  true

proc releaseReservation*(teamId: int, agentId: int) =
  ## Release any reservation held by this agent.
  if not validTeamId(teamId):
    return
  let state = addr resState(teamId)
  var writeIdx = 0
  for readIdx in 0 ..< state.count:
    if state.reservations[readIdx].agentId != agentId.int32:
      if writeIdx != readIdx:
        state.reservations[writeIdx] = state.reservations[readIdx]
      inc writeIdx
  state.count = writeIdx

proc getReservationPos*(teamId: int, agentId: int): IVec2 =
  ## Get the reserved position for an agent, or (-1,-1) if none.
  if not validTeamId(teamId):
    return ivec2(-1, -1)
  let state = addr resState(teamId)
  for i in 0 ..< state.count:
    if state.reservations[i].agentId == agentId.int32:
      return state.reservations[i].pos
  ivec2(-1, -1)

# ============================================================================
# Resource Patch Tracking (AoE-style gathering clusters)
# ============================================================================
# Tracks gatherer assignment counts around drop-off buildings so that villagers
# prefer undermanned patches and spread out evenly across resource sites.
# A "patch" is defined as the area around a drop-off building (LumberCamp,
# MiningCamp, Quarry, Granary) or TownCenter.

type
  ResourcePatchKind* = enum
    PatchWood, PatchGold, PatchStone, PatchFood

proc countGatherersNearPos*(env: Environment, teamId: int, pos: IVec2, radius: int): int =
  ## Count alive gatherer agents assigned to gather near a position.
  ## Uses reservation positions to determine assignment.
  if not validTeamId(teamId):
    return 0
  let rState = addr resState(teamId)
  for i in 0 ..< rState.count:
    let res = rState.reservations[i]
    let dx = abs(res.pos.x - pos.x)
    let dy = abs(res.pos.y - pos.y)
    if max(dx, dy) <= radius.int32:
      inc result

proc findNearestDropoffForResource*(env: Environment, pos: IVec2, teamId: int,
                                     patchKind: ResourcePatchKind): Thing =
  ## Find the nearest friendly drop-off building appropriate for a resource type.
  ## Returns nil if no drop-off found within search radius.
  case patchKind
  of PatchWood:
    result = findNearestFriendlyThingSpatial(env, pos, teamId, LumberCamp, DropoffProximityRadius)
    if isNil(result):
      result = findNearestFriendlyThingSpatial(env, pos, teamId, TownCenter, DropoffProximityRadius)
  of PatchGold:
    result = findNearestFriendlyThingSpatial(env, pos, teamId, MiningCamp, DropoffProximityRadius)
    if isNil(result):
      result = findNearestFriendlyThingSpatial(env, pos, teamId, TownCenter, DropoffProximityRadius)
  of PatchStone:
    result = findNearestFriendlyThingSpatial(env, pos, teamId, Quarry, DropoffProximityRadius)
    if isNil(result):
      result = findNearestFriendlyThingSpatial(env, pos, teamId, TownCenter, DropoffProximityRadius)
  of PatchFood:
    result = findNearestFriendlyThingSpatial(env, pos, teamId, Granary, DropoffProximityRadius)
    if isNil(result):
      result = findNearestFriendlyThingSpatial(env, pos, teamId, Mill, DropoffProximityRadius)
    if isNil(result):
      result = findNearestFriendlyThingSpatial(env, pos, teamId, TownCenter, DropoffProximityRadius)

proc findUnderstaffedPatchPos*(env: Environment, agentPos: IVec2, teamId: int,
                                patchKind: ResourcePatchKind): IVec2 =
  ## Find the position of the nearest drop-off building with an understaffed patch.
  ## Returns (-1,-1) if no understaffed patch found.
  ## Used for idle villager auto-assignment.
  result = ivec2(-1, -1)
  var bestDist = int.high
  # Check each relevant building type for understaffed patches
  template checkKind(buildingKind: ThingKind) =
    let building = findNearestFriendlyThingSpatial(env, agentPos, teamId, buildingKind, 50)
    if not isNil(building):
      let gatherers = countGatherersNearPos(env, teamId, building.pos, PatchRadius)
      if gatherers < MaxGatherersPerPatch:
        let dist = int(max(abs(agentPos.x - building.pos.x), abs(agentPos.y - building.pos.y)))
        if dist < bestDist:
          bestDist = dist
          result = building.pos

  case patchKind
  of PatchWood:
    checkKind(LumberCamp)
  of PatchGold:
    checkKind(MiningCamp)
  of PatchStone:
    checkKind(Quarry)
  of PatchFood:
    checkKind(Granary)
    checkKind(Mill)
  # Also check TownCenter as fallback
  checkKind(TownCenter)
