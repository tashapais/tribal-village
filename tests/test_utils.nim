import std/[tables]
import environment
import agent_control
import common
import types
import items
import terrain

proc decodeAction*(action: uint16): tuple[verb: int, arg: int] =
  (action.int div ActionArgumentCount, action.int mod ActionArgumentCount)

proc dirIndex*(fromPos, toPos: IVec2): int =
  let dx = toPos.x - fromPos.x
  let dy = toPos.y - fromPos.y
  let sx = if dx > 0: 1'i32 elif dx < 0: -1'i32 else: 0'i32
  let sy = if dy > 0: 1'i32 elif dy < 0: -1'i32 else: 0'i32
  if sx == 0'i32 and sy == -1'i32: return 0
  if sx == 0'i32 and sy == 1'i32: return 1
  if sx == -1'i32 and sy == 0'i32: return 2
  if sx == 1'i32 and sy == 0'i32: return 3
  if sx == -1'i32 and sy == -1'i32: return 4
  if sx == 1'i32 and sy == -1'i32: return 5
  if sx == -1'i32 and sy == 1'i32: return 6
  if sx == 1'i32 and sy == 1'i32: return 7
  0

proc makeEmptyEnv*(): Environment =
  result = Environment(config: defaultEnvironmentConfig())
  result.currentStep = 0
  result.shouldReset = false
  result.observationsInitialized = false
  result.things.setLen(0)
  result.agents.setLen(0)
  result.stats.setLen(0)
  result.thingsByKind = default(array[ThingKind, seq[Thing]])
  # Initialize aura tracking collections
  result.tankUnits.setLen(0)
  result.monkUnits.setLen(0)
  # Initialize agent order array for step shuffle
  for i in 0 ..< MapAgents:
    result.agentOrder[i] = i
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      result.grid[x][y] = nil
      result.backgroundGrid[x][y] = nil
      result.terrain[x][y] = TerrainEmpty
      result.biomes[x][y] = BiomeBaseType
      result.elevation[x][y] = 0
      result.baseTintColors[x][y] = BaseTileColorDefault
      result.computedTintColors[x][y] = TileColor(r: 0, g: 0, b: 0, intensity: 0)
      result.tintMods[x][y] = TintModification(r: 0, g: 0, b: 0)
      result.tintStrength[x][y] = 0
      result.tumorTintMods[x][y] = TintModification(r: 0, g: 0, b: 0)
      result.tumorStrength[x][y] = 0
  result.teamStockpiles = default(array[MapRoomObjectsTeams, TeamStockpile])
  # Initialize civ bonuses to defaults
  for teamId in 0 ..< MapRoomObjectsTeams:
    result.teamCivBonuses[teamId] = defaultCivBonus()
  result.initMarketPrices()  # Initialize AoE2-style market prices
  result.victoryWinner = -1
  result.victoryWinners = NoTeamMask
  for teamId in 0 ..< MapRoomObjectsTeams:
    result.victoryStates[teamId].wonderBuiltStep = -1
    result.victoryStates[teamId].relicHoldStartStep = -1
    result.victoryStates[teamId].kingAgentId = -1
  # Initialize alliance state: each team is allied with itself only
  for teamId in 0 ..< MapRoomObjectsTeams:
    result.teamAlliances[teamId] = TeamMasks[teamId]
  result.actionTintPositions.setLen(0)
  result.activeTiles.positions.setLen(0)
  result.activeTiles.flags = default(array[MapWidth, array[MapHeight, bool]])
  result.tumorActiveTiles.positions.setLen(0)
  result.tumorActiveTiles.flags = default(array[MapWidth, array[MapHeight, bool]])
  result.altarColors = initTable[IVec2, Color]()
  result.teamColors = newSeq[Color](MapRoomObjectsTeams)
  result.agentColors = newSeq[Color](MapAgents)

proc addAgentAt*(env: Environment, agentId: int, pos: IVec2,
                 homeAltar: IVec2 = ivec2(-1, -1), unitClass: AgentUnitClass = UnitVillager,
                 orientation: Orientation = N, stance: AgentStance = StanceNoAttack): Thing =
  while env.agents.len <= agentId:
    let nextId = env.agents.len
    let isTarget = nextId == agentId
    let agent = Thing(
      kind: Agent,
      pos: (if isTarget: pos else: ivec2(-1, -1)),
      agentId: nextId,
      orientation: (if isTarget: orientation else: N),
      inventory: emptyInventory(),
      hp: (if isTarget: AgentMaxHp else: 0),
      maxHp: AgentMaxHp,
      attackDamage: 1,
      unitClass: (if isTarget: unitClass else: UnitVillager),
      stance: (if isTarget: stance else: StanceNoAttack),
      homeAltar: (if isTarget: homeAltar else: ivec2(-1, -1)),
      faith: (if isTarget and unitClass == UnitMonk: MonkMaxFaith else: 0),
      rallyTarget: ivec2(-1, -1)
    )
    env.add(agent)
    env.terminated[nextId] = (if isTarget: 0.0 else: 1.0)
    if isTarget:
      result = agent

proc addBuilding*(env: Environment, kind: ThingKind, pos: IVec2, teamId: int): Thing =
  let thing = Thing(kind: kind, pos: pos, teamId: teamId)
  thing.inventory = emptyInventory()
  thing.rallyPoint = ivec2(-1, -1)  # No rally point by default
  thing.constructed = true  # Test buildings are fully constructed
  let capacity = buildingBarrelCapacity(kind)
  if capacity > 0:
    thing.barrelCapacity = capacity
  env.add(thing)
  thing

proc addBuildings*(env: Environment, teamId: int, start: IVec2, kinds: openArray[ThingKind]) =
  var dx = 0
  for kind in kinds:
    discard addBuilding(env, kind, start + ivec2(dx.int32, 0), teamId)
    inc dx

proc addAltar*(env: Environment, pos: IVec2, teamId: int, hearts: int): Thing =
  let altar = Thing(kind: Altar, pos: pos, teamId: teamId)
  altar.inventory = emptyInventory()
  altar.hearts = hearts
  env.add(altar)
  altar

proc addResource*(env: Environment, kind: ThingKind, pos: IVec2, key: ItemKey,
                  amount: int = ResourceNodeInitial): Thing =
  let node = Thing(kind: kind, pos: pos)
  node.inventory = emptyInventory()
  if key != ItemNone and amount > 0:
    setInv(node, key, amount)
  env.add(node)
  node

proc setStockpile*(env: Environment, teamId: int, res: StockpileResource, count: int) =
  env.teamStockpiles[teamId].counts[res] = count

proc stepNoop*(env: Environment) =
  ## Step the environment with all agents taking NOOP action
  while env.agents.len < MapAgents:
    let nextId = env.agents.len
    let agent = Thing(
      kind: Agent,
      pos: ivec2(-1, -1),
      agentId: nextId,
      orientation: N,
      inventory: emptyInventory(),
      hp: 0,
      maxHp: AgentMaxHp,
      attackDamage: 1,
      unitClass: UnitVillager,
      stance: StanceNoAttack,
      homeAltar: ivec2(-1, -1),
      rallyTarget: ivec2(-1, -1)
    )
    env.add(agent)
    env.terminated[nextId] = 1.0
  var actions: array[MapAgents, uint16]
  for i in 0 ..< MapAgents:
    actions[i] = 0
  env.step(addr actions)
  env.ensureObservations()

proc stepAction*(env: Environment, agentId: int, verb: uint16, argument: int) =
  while env.agents.len < MapAgents:
    let nextId = env.agents.len
    let agent = Thing(
      kind: Agent,
      pos: ivec2(-1, -1),
      agentId: nextId,
      orientation: N,
      inventory: emptyInventory(),
      hp: 0,
      maxHp: AgentMaxHp,
      attackDamage: 1,
      unitClass: UnitVillager,
      stance: StanceNoAttack,
      homeAltar: ivec2(-1, -1),
      rallyTarget: ivec2(-1, -1)
    )
    env.add(agent)
    env.terminated[nextId] = 1.0
  var actions: array[MapAgents, uint16]
  for i in 0 ..< MapAgents:
    actions[i] = 0
  actions[agentId] = encodeAction(verb, argument.uint16)
  env.step(addr actions)
  env.ensureObservations()

proc newTestController*(seed: int): Controller =
  ## Create a controller configured for testing with Brutal difficulty (no decision delays).
  ## This ensures deterministic test behavior without random NOOP actions.
  result = newController(seed)
  for teamId in 0 ..< MapRoomObjectsTeams:
    result.setDifficulty(teamId, DiffBrutal)
