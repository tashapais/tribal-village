import std/[algorithm, strutils, tables, sets], vmath, chroma
import entropy
import envconfig
import terrain, items, common_types, biome
import types, registry
import spatial_index
import formations
import state_dumper
import arena_alloc

# Import split modules
import environment_state
import environment_grid
import environment_agents

when defined(techAudit):
  import tech_audit
when defined(econAudit):
  import econ_audit
when defined(settlerLog):
  import settler_events
when defined(settlerMetrics):
  import settler_metrics
when defined(audio):
  import audio_events

# Re-export split modules for backwards compatibility
export environment_state
export environment_grid
export environment_agents

export terrain, items, common_types
export types, registry
export spatial_index
export formations
export state_dumper
when defined(settlerLog):
  export settler_events
when defined(settlerMetrics):
  export settler_metrics

const
  ## Default tumor behavior constants
  DefaultTumorBranchRange* = 5
  DefaultTumorBranchMinAge* = 2
  DefaultTumorBranchChance* = 0.1
  DefaultTumorAdjacencyDeathChance* = 1.0 / 3.0

  ## Default village spacing constants
  DefaultMinVillageSpacing* = 22
  DefaultSpawnerMinDistance* = 20
  DefaultInitialActiveAgents* = 6

  ## Ramp tile placement constants
  ## Controls how frequently ramps are placed at elevation transitions
  ## and their visual width for clearer elevation feedback.
  RampPlacementSpacing* = 4     # Place ramp every Nth cliff edge (lower = more ramps)
  RampWidthMin* = 1             # Minimum ramp width in tiles
  RampWidthMax* = 3             # Maximum ramp width in tiles

  ## Cliff fall damage - agents take damage when dropping elevation without a ramp
  CliffFallDamage* = 1          # Damage taken per elevation level dropped without ramp

  ## Default combat constants
  DefaultSpearCharges* = 5
  DefaultArmorPoints* = 5
  DefaultBreadHealAmount* = 999

  ## Default market tuning (AoE2-style dynamic pricing)
  ## Prices are in gold per 100 units of resource (scaled for integer math)
  MarketBasePrice* = 100        # Base price: 100 gold per 100 resources
  MarketMinPrice* = 20          # Minimum price floor
  MarketMaxPrice* = 300         # Maximum price ceiling
  MarketBuyPriceIncrease* = 3   # Price increase per buy transaction
  MarketSellPriceDecrease* = 3  # Price decrease per sell transaction
  MarketPriceDecayRate* = 1     # Price drift toward base per decay tick
  MarketPriceDecayInterval* = 50 # Steps between price decay ticks
  DefaultMarketCooldown* = 2
  # Legacy constants (kept for compatibility)
  DefaultMarketSellNumerator* = 1
  DefaultMarketSellDenominator* = 2
  DefaultMarketBuyFoodNumerator* = 1
  DefaultMarketBuyFoodDenominator* = 1

  ## Biome gathering bonus constants
  BiomeGatherBonusChance* = 0.20  # 20% chance for bonus item in matching biomes
  DesertOasisBonusChance* = 0.10  # 10% chance for bonus in desert near water
  DesertOasisRadius* = 3  # Tiles from water to get desert bonus

## Error types and FFI error state management are in environment_state.nim

proc clear[T](s: var openarray[T]) =
  ## Zero out a contiguous buffer (arrays/openarrays) without reallocating.
  zeroMem(cast[pointer](s[0].addr), s.len * sizeof(T))

proc hasWaterNearby*(env: Environment, pos: IVec2, radius: int, includeShallow: bool = true): bool =
  ## Check if there is water terrain within the given radius of a position.
  ## Includes ShallowWater by default since docks can be placed on either.
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      let x = pos.x + dx
      let y = pos.y + dy
      if x >= 0 and x < MapWidth and y >= 0 and y < MapHeight:
        let t = env.terrain[x][y]
        if t == Water or (includeShallow and t == ShallowWater):
          return true
  false

proc getBiomeGatherBonus*(env: Environment, pos: IVec2, itemKey: ItemKey): int =
  ## Calculate bonus items from biome-specific gathering bonuses.
  ## Returns 0 or 1 based on probability roll using deterministic seed.
  ## Forest: +20% wood, Plains: +20% food, Caves: +20% stone, Snow: +20% gold,
  ## Desert: +10% all resources near water (oasis effect)
  if not isValidPos(pos):
    return 0

  let biome = env.biomes[pos.x][pos.y]

  # Check for biome-specific bonus
  var bonusChance = 0.0
  case biome
  of BiomeForestType:
    if itemKey == ItemWood:
      bonusChance = BiomeGatherBonusChance
  of BiomePlainsType:
    if itemKey == ItemWheat:
      bonusChance = BiomeGatherBonusChance
  of BiomeCavesType:
    if itemKey == ItemStone:
      bonusChance = BiomeGatherBonusChance
  of BiomeSnowType:
    if itemKey == ItemGold:
      bonusChance = BiomeGatherBonusChance
  of BiomeDesertType:
    # Desert gives bonus to all resources if near water (oasis effect)
    if itemKey == ItemWood or itemKey == ItemWheat or itemKey == ItemStone or itemKey == ItemGold:
      if env.hasWaterNearby(pos, DesertOasisRadius):
        bonusChance = DesertOasisBonusChance
  else:
    discard

  if bonusChance <= 0.0:
    return 0

  # Use deterministic seed based on position and step for reproducible behavior
  # Cast to int to avoid int32 overflow and ensure positive seed
  let seed = abs(int(pos.x) * 31337 + int(pos.y) * 7919 + env.currentStep * 13) + 1
  var r = initRand(seed)
  # Warm up RNG by discarding first few values to improve distribution
  discard next(r)
  discard next(r)
  if randChance(r, bonusChance):
    return 1
  0

{.push boundChecks: off, overflowChecks: off.}
proc writeTileObs(env: Environment, agentId, obsX, obsY, worldX, worldY: int) {.inline.} =
  ## Write observation data for a single tile. Called from rebuildObservations
  ## which already zeroed all observation memory, so we only set non-zero values.
  ## Bounds checking disabled: caller validates worldX/worldY in [0, MapWidth/Height)
  ## and obsX/obsY in [0, ObservationWidth/Height).

  # Cache base observation pointer - avoid repeated addr computation
  let obs = addr env.observations[agentId]

  # Terrain layer (one-hot encoded)
  let terrain = env.terrain[worldX][worldY]
  obs[][TerrainLayerStart + ord(terrain)][obsX][obsY] = 1

  # Thing layers - cache lookups
  let blockingThing = env.grid[worldX][worldY]
  let backgroundThing = env.backgroundGrid[worldX][worldY]

  if not isNil(blockingThing):
    obs[][ThingLayerStart + ord(blockingThing.kind)][obsX][obsY] = 1

  if not isNil(backgroundThing):
    obs[][ThingLayerStart + ord(backgroundThing.kind)][obsX][obsY] = 1

  # Process based on what's on the tile - single branch structure
  if not isNil(blockingThing):
    if blockingThing.kind == Agent:
      # Agent-specific layers (team, orientation, class, idle, stance)
      obs[][ord(TeamLayer)][obsX][obsY] = uint8(getTeamId(blockingThing) + 1)
      obs[][ord(AgentOrientationLayer)][obsX][obsY] = uint8(ord(blockingThing.orientation) + 1)
      obs[][ord(AgentUnitClassLayer)][obsX][obsY] = uint8(ord(blockingThing.unitClass) + 1)
      obs[][ord(UnitStanceLayer)][obsX][obsY] = uint8(ord(blockingThing.stance) + 1)

      # Idle detection: 1 if agent took NOOP/ORIENT action
      if blockingThing.isIdle:
        obs[][ord(AgentIdleLayer)][obsX][obsY] = 1

      # Monk faith (only if non-zero)
      if blockingThing.unitClass == UnitMonk and blockingThing.faith > 0:
        obs[][ord(MonkFaithLayer)][obsX][obsY] =
          uint8((blockingThing.faith * 255) div MonkMaxFaith)

      # Trebuchet packed state
      if blockingThing.unitClass == UnitTrebuchet and blockingThing.packed:
        obs[][ord(TrebuchetPackedLayer)][obsX][obsY] = 1
    else:
      # Non-agent blocking thing (building/resource/etc)
      # Team ownership: prefer blocking thing, fall back to background
      if blockingThing.kind in TeamOwnedKinds and
         blockingThing.teamId >= 0 and blockingThing.teamId < MapRoomObjectsTeams:
        obs[][ord(TeamLayer)][obsX][obsY] = uint8(blockingThing.teamId + 1)
      elif not isNil(backgroundThing) and backgroundThing.kind in TeamOwnedKinds and
           backgroundThing.teamId >= 0 and backgroundThing.teamId < MapRoomObjectsTeams:
        obs[][ord(TeamLayer)][obsX][obsY] = uint8(backgroundThing.teamId + 1)

      # Building HP (normalized to 0-255)
      if blockingThing.maxHp > 0:
        obs[][ord(BuildingHpLayer)][obsX][obsY] =
          uint8((blockingThing.hp * 255) div blockingThing.maxHp)

      # Garrison count (normalized to 0-255 by capacity)
      let capacity = case blockingThing.kind
        of TownCenter: TownCenterGarrisonCapacity
        of Castle: CastleGarrisonCapacity
        of GuardTower: GuardTowerGarrisonCapacity
        of House: HouseGarrisonCapacity
        else: 0
      if capacity > 0 and blockingThing.garrisonedUnits.len > 0:
        obs[][ord(GarrisonCountLayer)][obsX][obsY] =
          uint8((blockingThing.garrisonedUnits.len * 255) div capacity)

      # Monastery relic count
      if blockingThing.kind == Monastery and blockingThing.garrisonedRelics > 0:
        obs[][ord(RelicCountLayer)][obsX][obsY] =
          uint8(min(blockingThing.garrisonedRelics, 255))

      # Production queue length
      if blockingThing.productionQueue.entries.len > 0:
        obs[][ord(ProductionQueueLenLayer)][obsX][obsY] =
          uint8(min(blockingThing.productionQueue.entries.len, 255))
  else:
    # No blocking thing - check background for team ownership
    if not isNil(backgroundThing) and backgroundThing.kind in TeamOwnedKinds and
       backgroundThing.teamId >= 0 and backgroundThing.teamId < MapRoomObjectsTeams:
      obs[][ord(TeamLayer)][obsX][obsY] = uint8(backgroundThing.teamId + 1)

  # Tint layer
  let tintCode = env.actionTintCode[worldX][worldY]
  if tintCode != 0:
    obs[][ord(TintLayer)][obsX][obsY] = tintCode

  # Biome layer (enum value)
  obs[][ord(BiomeLayer)][obsX][obsY] = uint8(ord(env.biomes[worldX][worldY]))
{.pop.}

proc updateObservations(
  env: Environment,
  layer: ObservationName,
  pos: IVec2,
  value: int
) {.inline.} =
  ## No-op: observations are rebuilt in batch at end of step() for efficiency.
  ## Previously iterated ALL agents per tile update which was O(updates * agents).
  ## Now rebuildObservations is called once at end of step() which is O(agents * tiles).
  discard (env, layer, pos, value)

include "colors"
include "event_log"

const
  DefaultScoreNeutralThreshold = 0.05'f32
  DefaultScoreIncludeWater = false

{.push inline.}
proc updateAgentInventoryObs*(env: Environment, agent: Thing, key: ItemKey) =
  ## No-op: inventory observations are rebuilt in batch at end of step() for efficiency.
  ## Kept for API compatibility - call sites remain to enable future observation changes.
  discard (env, agent, key)

proc updateAgentInventoryObs*(env: Environment, agent: Thing, kind: ItemKind) =
  ## No-op: type-safe overload using ItemKind enum.
  ## See updateAgentInventoryObs(env, agent, ItemKey) for rationale.
  discard (env, agent, kind)

proc stockpileCount*(env: Environment, teamId: int, res: StockpileResource): int =
  env.teamStockpiles[teamId].counts[res]

proc addToStockpile*(env: Environment, teamId: int, res: StockpileResource, amount: int) =
  ## Add resources to team stockpile, applying gather rate modifiers
  let rawModifier = env.teamModifiers[teamId].gatherRateMultiplier
  let modifier = if rawModifier == 0.0'f32: 1.0'f32 else: rawModifier  # Default to 1.0 if uninitialized
  # Apply CivBonus gather rate multiplier
  let civGather = env.teamCivBonuses[teamId].gatherRateMultiplier
  let civModifier = if civGather == 0.0'f32: 1.0'f32 else: civGather
  let adjustedAmount = int(float32(amount) * modifier * civModifier)
  env.teamStockpiles[teamId].counts[res] += adjustedAmount

proc canSpendStockpile*(env: Environment, teamId: int,
                        costs: openArray[tuple[res: StockpileResource, count: int]]): bool =
  for cost in costs:
    if env.teamStockpiles[teamId].counts[cost.res] < cost.count:
      return false
  true

proc spendStockpile*(env: Environment, teamId: int,
                     costs: openArray[tuple[res: StockpileResource, count: int]]): bool =
  if not env.canSpendStockpile(teamId, costs):
    return false
  for cost in costs:
    env.teamStockpiles[teamId].counts[cost.res] -= cost.count
  true

proc canSpendStockpile*(env: Environment, teamId: int,
                        costs: openArray[tuple[key: ItemKey, count: int]]): bool =
  for cost in costs:
    let res = stockpileResourceForItem(cost.key)
    if res == ResourceNone:
      return false
    if env.teamStockpiles[teamId].counts[res] < cost.count:
      return false
  true

proc spendStockpile*(env: Environment, teamId: int,
                     costs: openArray[tuple[key: ItemKey, count: int]]): bool =
  if not env.canSpendStockpile(teamId, costs):
    return false
  for cost in costs:
    let res = stockpileResourceForItem(cost.key)
    env.teamStockpiles[teamId].counts[res] -= cost.count
  true

# ============================================================================
# AoE2-style Market Trading with Dynamic Prices
# ============================================================================

proc initMarketPrices*(env: Environment) =
  ## Initialize market prices to base rates for all teams
  for teamId in 0 ..< MapRoomObjectsTeams:
    for res in StockpileResource:
      if res != ResourceNone and res != ResourceGold:
        env.teamMarketPrices[teamId].prices[res] = MarketBasePrice

proc getMarketPrice*(env: Environment, teamId: int, res: StockpileResource): int {.inline.} =
  ## Get current market price for a resource (gold cost per 100 units)
  if res == ResourceGold or res == ResourceNone:
    return 0
  env.teamMarketPrices[teamId].prices[res]

proc setMarketPrice*(env: Environment, teamId: int, res: StockpileResource, price: int) =
  ## Set market price with clamping to min/max bounds
  if res == ResourceGold or res == ResourceNone:
    return
  env.teamMarketPrices[teamId].prices[res] = clamp(price, MarketMinPrice, MarketMaxPrice)

proc marketBuyResource*(env: Environment, teamId: int, res: StockpileResource,
                        amount: int): tuple[goldCost: int, resourceGained: int] =
  ## Buy resources from market using gold from stockpile.
  ## Returns (gold spent, resources gained). Price increases after buying.
  ## Uses scaled integer math: price is gold per 100 units.
  if res == ResourceGold or res == ResourceNone or amount <= 0:
    return (0, 0)

  let currentPrice = env.getMarketPrice(teamId, res)
  # Cost = (amount * price) / 100, rounding up
  let goldCost = (amount * currentPrice + 99) div 100

  # Check if team has enough gold
  if env.teamStockpiles[teamId].counts[ResourceGold] < goldCost:
    return (0, 0)

  # Execute transaction
  env.teamStockpiles[teamId].counts[ResourceGold] -= goldCost
  env.teamStockpiles[teamId].counts[res] += amount

  # Increase price (supply decreased, demand increased)
  env.setMarketPrice(teamId, res, currentPrice + MarketBuyPriceIncrease)

  when defined(econAudit):
    recordMarketBuy(teamId, res, amount, goldCost, env.currentStep)

  result = (goldCost, amount)

proc marketSellResource*(env: Environment, teamId: int, res: StockpileResource,
                         amount: int): tuple[resourceSold: int, goldGained: int] =
  ## Sell resources to market for gold.
  ## Returns (resources sold, gold gained). Price decreases after selling.
  ## Uses scaled integer math: price is gold per 100 units.
  if res == ResourceGold or res == ResourceNone or amount <= 0:
    return (0, 0)

  let currentPrice = env.getMarketPrice(teamId, res)
  # Gain = (amount * price) / 100, rounding down
  let goldGained = (amount * currentPrice) div 100

  # Check if team has enough resources to sell
  if env.teamStockpiles[teamId].counts[res] < amount:
    return (0, 0)

  # Execute transaction
  env.teamStockpiles[teamId].counts[res] -= amount
  env.teamStockpiles[teamId].counts[ResourceGold] += goldGained

  # Decrease price (supply increased)
  env.setMarketPrice(teamId, res, currentPrice - MarketSellPriceDecrease)

  when defined(econAudit):
    recordMarketSell(teamId, res, amount, goldGained, env.currentStep)

  result = (amount, goldGained)

proc marketSellInventory*(env: Environment, agent: Thing, itemKey: ItemKey):
                          tuple[amountSold: int, goldGained: int] =
  ## Sell all of an item from agent's inventory to their team's market.
  ## Returns (amount sold, gold gained).
  let teamId = getTeamId(agent)
  let res = stockpileResourceForItem(itemKey)
  if res == ResourceGold or res == ResourceNone or res == ResourceWater:
    return (0, 0)

  let amount = getInv(agent, itemKey)
  if amount <= 0:
    return (0, 0)

  let currentPrice = env.getMarketPrice(teamId, res)
  # Gain = (amount * price) / 100, rounding down
  let goldGained = (amount * currentPrice) div 100

  if goldGained > 0:
    # Clear inventory and add gold to stockpile (no gather rate modifier for market trades)
    setInv(agent, itemKey, 0)
    env.teamStockpiles[teamId].counts[ResourceGold] += goldGained
    # Decrease price (supply increased)
    env.setMarketPrice(teamId, res, currentPrice - MarketSellPriceDecrease)
    return (amount, goldGained)

  result = (0, 0)

proc marketBuyFood*(env: Environment, agent: Thing, goldAmount: int):
                    tuple[goldSpent: int, foodGained: int] =
  ## Buy food with gold from agent's inventory.
  ## Returns (gold spent, food gained to stockpile).
  let teamId = getTeamId(agent)
  if goldAmount <= 0:
    return (0, 0)

  let invGold = getInv(agent, ItemGold)
  if invGold < goldAmount:
    return (0, 0)

  let currentPrice = env.getMarketPrice(teamId, ResourceFood)
  # Food gained = (gold * 100) / price
  let foodGained = (goldAmount * 100) div currentPrice

  if foodGained > 0:
    setInv(agent, ItemGold, invGold - goldAmount)
    # No gather rate modifier for market trades
    env.teamStockpiles[teamId].counts[ResourceFood] += foodGained
    # Increase price (demand increased)
    env.setMarketPrice(teamId, ResourceFood, currentPrice + MarketBuyPriceIncrease)
    return (goldAmount, foodGained)

  result = (0, 0)

proc decayMarketPrices*(env: Environment) =
  ## Slowly drift market prices back toward base rate.
  ## Should be called periodically (every MarketPriceDecayInterval steps).
  for teamId in 0 ..< MapRoomObjectsTeams:
    for res in StockpileResource:
      if res == ResourceGold or res == ResourceNone:
        continue
      let currentPrice = env.teamMarketPrices[teamId].prices[res]
      if currentPrice > MarketBasePrice:
        env.teamMarketPrices[teamId].prices[res] = max(MarketBasePrice,
          currentPrice - MarketPriceDecayRate)
      elif currentPrice < MarketBasePrice:
        env.teamMarketPrices[teamId].prices[res] = min(MarketBasePrice,
          currentPrice + MarketPriceDecayRate)

# ============================================================================
# AoE2-style Tribute System (resource transfer between teams)
# ============================================================================

proc tributeResources*(env: Environment, fromTeam, toTeam: int,
                       resource: StockpileResource, amount: int): int =
  ## Transfer resources from one team to another, applying a tax.
  ## Returns the actual amount received after tax (0 if transfer failed).
  ## Coinage tech (researched at University) reduces the tax rate.
  if fromTeam < 0 or fromTeam >= MapRoomObjectsTeams:
    return 0
  if toTeam < 0 or toTeam >= MapRoomObjectsTeams:
    return 0
  if fromTeam == toTeam:
    return 0
  if amount < TributeMinAmount:
    return 0
  if resource == ResourceNone:
    return 0

  # Check if sender has enough resources
  if env.teamStockpiles[fromTeam].counts[resource] < amount:
    return 0

  # Calculate tax rate (Coinage tech reduces it)
  let taxRate = if env.teamUniversityTechs[fromTeam].researched[TechCoinage]:
    TributeTaxRate - CoinageTaxReduction
  else:
    TributeTaxRate

  let taxAmount = int(float(amount) * taxRate)
  let received = amount - taxAmount

  if received <= 0:
    return 0

  # Execute the transfer
  env.teamStockpiles[fromTeam].counts[resource] -= amount
  env.teamStockpiles[toTeam].counts[resource] += received

  # Track cumulative tributes for scoring
  env.teamTributesSent[fromTeam] += amount
  env.teamTributesReceived[toTeam] += received

  received

# ============================================================================
# Alliance System (symmetric team alliances for shared victory)
# ============================================================================

proc formAlliance*(env: Environment, teamA, teamB: int) {.inline.} =
  ## Form a symmetric alliance between two teams.
  ## Both teams will consider each other allied.
  if teamA < 0 or teamA >= MapRoomObjectsTeams: return
  if teamB < 0 or teamB >= MapRoomObjectsTeams: return
  env.teamAlliances[teamA] = env.teamAlliances[teamA] or getTeamMask(teamB)
  env.teamAlliances[teamB] = env.teamAlliances[teamB] or getTeamMask(teamA)

proc breakAlliance*(env: Environment, teamA, teamB: int) {.inline.} =
  ## Break the alliance between two teams (both directions).
  ## Teams cannot break alliance with themselves.
  if teamA < 0 or teamA >= MapRoomObjectsTeams: return
  if teamB < 0 or teamB >= MapRoomObjectsTeams: return
  if teamA == teamB: return  # Cannot un-ally with self
  env.teamAlliances[teamA] = env.teamAlliances[teamA] and (not getTeamMask(teamB))
  env.teamAlliances[teamB] = env.teamAlliances[teamB] and (not getTeamMask(teamA))

proc areAllied*(env: Environment, teamA, teamB: int): bool {.inline.} =
  ## Check if two teams are allied. Teams are always allied with themselves.
  if teamA < 0 or teamA >= MapRoomObjectsTeams: return false
  if teamB < 0 or teamB >= MapRoomObjectsTeams: return false
  isTeamInMask(teamB, env.teamAlliances[teamA])

proc getAllies*(env: Environment, teamId: int): TeamMask {.inline.} =
  ## Return the alliance bitmask for a team (includes self).
  if teamId < 0 or teamId >= MapRoomObjectsTeams: return NoTeamMask
  env.teamAlliances[teamId]

# ============================================================================

proc spendInventory*(env: Environment, agent: Thing,
                     costs: openArray[tuple[key: ItemKey, count: int]]): bool =
  if not canSpendInventory(agent, costs):
    return false
  for cost in costs:
    setInv(agent, cost.key, getInv(agent, cost.key) - cost.count)
    env.updateAgentInventoryObs(agent, cost.key)
  true

proc choosePayment*(env: Environment, agent: Thing,
                    costs: openArray[tuple[key: ItemKey, count: int]]): PaymentSource =
  if costs.len == 0:
    return PayNone
  if canSpendInventory(agent, costs):
    return PayInventory
  let teamId = getTeamId(agent)
  if env.canSpendStockpile(teamId, costs):
    return PayStockpile
  PayNone

proc spendCosts*(env: Environment, agent: Thing, source: PaymentSource,
                 costs: openArray[tuple[key: ItemKey, count: int]]): bool =
  case source
  of PayInventory:
    spendInventory(env, agent, costs)
  of PayStockpile:
    env.spendStockpile(getTeamId(agent), costs)
  of PayNone:
    false

const
  UnitMaxHpByClass: array[AgentUnitClass, int] = [
    VillagerMaxHp,
    ManAtArmsMaxHp,
    ArcherMaxHp,
    ScoutMaxHp,
    KnightMaxHp,
    MonkMaxHp,
    BatteringRamMaxHp,
    MangonelMaxHp,
    TrebuchetMaxHp,
    GoblinMaxHp,
    BoatMaxHp,
    TradeCogMaxHp,
    # Castle unique units
    SamuraiMaxHp,
    LongbowmanMaxHp,
    CataphractMaxHp,
    WoadRaiderMaxHp,
    TeutonicKnightMaxHp,
    HuskarlMaxHp,
    MamelukeMaxHp,
    JanissaryMaxHp,
    KingMaxHp,
    # Unit upgrade tiers
    LongSwordsmanMaxHp,
    ChampionMaxHp,
    LightCavalryMaxHp,
    HussarMaxHp,
    CrossbowmanMaxHp,
    ArbalesterMaxHp,
    # Naval combat units
    GalleyMaxHp,
    FireShipMaxHp,
    FishingShipMaxHp,
    TransportShipMaxHp,
    DemoShipMaxHp,
    CannonGalleonMaxHp,
    # Additional siege unit
    ScorpionMaxHp,
    # Stable cavalry upgrades
    CavalierMaxHp,
    PaladinMaxHp,
    # Camel line
    CamelMaxHp,
    HeavyCamelMaxHp,
    ImperialCamelMaxHp,
    # Archery Range units
    SkirmisherMaxHp,
    EliteSkirmisherMaxHp,
    CavalryArcherMaxHp,
    HeavyCavalryArcherMaxHp,
    HandCannoneerMaxHp
  ]
  UnitAttackDamageByClass: array[AgentUnitClass, int] = [
    VillagerAttackDamage,
    ManAtArmsAttackDamage,
    ArcherAttackDamage,
    ScoutAttackDamage,
    KnightAttackDamage,
    MonkAttackDamage,
    BatteringRamAttackDamage,
    MangonelAttackDamage,
    TrebuchetAttackDamage,
    GoblinAttackDamage,
    BoatAttackDamage,
    TradeCogAttackDamage,
    # Castle unique units
    SamuraiAttackDamage,
    LongbowmanAttackDamage,
    CataphractAttackDamage,
    WoadRaiderAttackDamage,
    TeutonicKnightAttackDamage,
    HuskarlAttackDamage,
    MamelukeAttackDamage,
    JanissaryAttackDamage,
    KingAttackDamage,
    # Unit upgrade tiers
    LongSwordsmanAttackDamage,
    ChampionAttackDamage,
    LightCavalryAttackDamage,
    HussarAttackDamage,
    CrossbowmanAttackDamage,
    ArbalesterAttackDamage,
    # Naval combat units
    GalleyAttackDamage,
    FireShipAttackDamage,
    FishingShipAttackDamage,
    TransportShipAttackDamage,
    DemoShipAttackDamage,
    CannonGalleonAttackDamage,
    # Additional siege unit
    ScorpionAttackDamage,
    # Stable cavalry upgrades
    CavalierAttackDamage,
    PaladinAttackDamage,
    # Camel line
    CamelAttackDamage,
    HeavyCamelAttackDamage,
    ImperialCamelAttackDamage,
    # Archery Range units
    SkirmisherAttackDamage,
    EliteSkirmisherAttackDamage,
    CavalryArcherAttackDamage,
    HeavyCavalryArcherAttackDamage,
    HandCannoneerAttackDamage
  ]

proc defaultStanceForClass*(unitClass: AgentUnitClass): AgentStance =
  ## Returns the default stance for a unit class.
  ## Villagers use NoAttack (won't auto-attack).
  ## Military units use Defensive (attack in range, return to position).
  case unitClass
  of UnitVillager, UnitMonk, UnitFishingShip, UnitTransportShip:
    StanceNoAttack
  of UnitBoat, UnitTradeCog:
    StanceDefensive
  of UnitManAtArms, UnitArcher, UnitScout, UnitKnight, UnitBatteringRam, UnitMangonel, UnitTrebuchet, UnitGoblin,
     UnitSamurai, UnitLongbowman, UnitCataphract, UnitWoadRaider, UnitTeutonicKnight,
     UnitHuskarl, UnitMameluke, UnitJanissary, UnitKing,
     UnitLongSwordsman, UnitChampion, UnitLightCavalry, UnitHussar, UnitCrossbowman, UnitArbalester,
     UnitGalley, UnitFireShip, UnitDemoShip, UnitCannonGalleon, UnitScorpion,
     UnitCavalier, UnitPaladin, UnitCamel, UnitHeavyCamel, UnitImperialCamel,
     UnitSkirmisher, UnitEliteSkirmisher, UnitCavalryArcher, UnitHeavyCavalryArcher, UnitHandCannoneer:
    StanceDefensive

type
  UnitCategory* = enum
    ## Categories for Blacksmith upgrade application
    CategoryNone      ## Units that don't receive upgrades (villagers, siege, monks)
    CategoryInfantry  ## Man-at-arms, Samurai, Woad Raider, Teutonic Knight, Huskarl
    CategoryCavalry   ## Scout, Knight, Cataphract, Mameluke
    CategoryArcher    ## Archer, Longbowman, Janissary

const
  ## Pre-computed lookup table for unit category (eliminates switch/case in hot path)
  UnitCategoryByClass*: array[AgentUnitClass, UnitCategory] = [
    CategoryNone,      # UnitVillager
    CategoryInfantry,  # UnitManAtArms
    CategoryArcher,    # UnitArcher
    CategoryCavalry,   # UnitScout
    CategoryCavalry,   # UnitKnight
    CategoryNone,      # UnitMonk
    CategoryNone,      # UnitBatteringRam
    CategoryNone,      # UnitMangonel
    CategoryNone,      # UnitTrebuchet
    CategoryNone,      # UnitGoblin
    CategoryNone,      # UnitBoat
    CategoryNone,      # UnitTradeCog
    CategoryInfantry,  # UnitSamurai
    CategoryArcher,    # UnitLongbowman
    CategoryCavalry,   # UnitCataphract
    CategoryInfantry,  # UnitWoadRaider
    CategoryInfantry,  # UnitTeutonicKnight
    CategoryInfantry,  # UnitHuskarl
    CategoryCavalry,   # UnitMameluke
    CategoryArcher,    # UnitJanissary
    CategoryNone,      # UnitKing
    CategoryInfantry,  # UnitLongSwordsman
    CategoryInfantry,  # UnitChampion
    CategoryCavalry,   # UnitLightCavalry
    CategoryCavalry,   # UnitHussar
    CategoryArcher,    # UnitCrossbowman
    CategoryArcher,    # UnitArbalester
    CategoryNone,      # UnitGalley
    CategoryNone,      # UnitFireShip
    CategoryNone,      # UnitFishingShip
    CategoryNone,      # UnitTransportShip
    CategoryNone,      # UnitDemoShip
    CategoryNone,      # UnitCannonGalleon
    CategoryNone,      # UnitScorpion
    CategoryCavalry,   # UnitCavalier
    CategoryCavalry,   # UnitPaladin
    CategoryCavalry,   # UnitCamel
    CategoryCavalry,   # UnitHeavyCamel
    CategoryCavalry,   # UnitImperialCamel
    # Archery Range units
    CategoryArcher,    # UnitSkirmisher
    CategoryArcher,    # UnitEliteSkirmisher
    CategoryArcher,    # UnitCavalryArcher (ranged cavalry, benefits from archer upgrades)
    CategoryArcher,    # UnitHeavyCavalryArcher
    CategoryArcher,    # UnitHandCannoneer
  ]

proc getUnitCategory*(unitClass: AgentUnitClass): UnitCategory {.inline.} =
  ## Returns the Blacksmith upgrade category for a unit class.
  ## Uses pre-computed lookup table for O(1) access.
  UnitCategoryByClass[unitClass]

proc getBlacksmithAttackBonus*(env: Environment, teamId: int, unitClass: AgentUnitClass): int {.inline.} =
  ## Returns the attack bonus from Blacksmith upgrades for a unit.
  ## Melee attack (Forging line) applies to infantry + cavalry.
  ## Archer attack (Fletching line) applies to archers.
  ## Bonus varies by tier: level 3 melee gives +2 extra (Blast Furnace).
  let category = UnitCategoryByClass[unitClass]
  case category
  of CategoryInfantry, CategoryCavalry:
    let level = env.teamBlacksmithUpgrades[teamId].levels[UpgradeMeleeAttack]
    BlacksmithMeleeAttackBonus[level]
  of CategoryArcher:
    let level = env.teamBlacksmithUpgrades[teamId].levels[UpgradeArcherAttack]
    BlacksmithArcherAttackBonus[level]
  of CategoryNone:
    0

proc getBlacksmithArmorBonus*(env: Environment, teamId: int, unitClass: AgentUnitClass): int {.inline.} =
  ## Returns the armor bonus from Blacksmith upgrades for a unit.
  ## Bonus varies by tier: level 3 gives +2 extra (Plate/Ring upgrades).
  let category = UnitCategoryByClass[unitClass]
  case category
  of CategoryInfantry:
    let level = env.teamBlacksmithUpgrades[teamId].levels[UpgradeInfantryArmor]
    BlacksmithInfantryArmorBonus[level]
  of CategoryCavalry:
    let level = env.teamBlacksmithUpgrades[teamId].levels[UpgradeCavalryArmor]
    BlacksmithCavalryArmorBonus[level]
  of CategoryArcher:
    let level = env.teamBlacksmithUpgrades[teamId].levels[UpgradeArcherArmor]
    BlacksmithArcherArmorBonus[level]
  of CategoryNone:
    0

proc applyUnitClass*(agent: Thing, unitClass: AgentUnitClass) =
  ## Apply unit class stats without team modifiers (backwards compatibility)
  agent.unitClass = unitClass
  if unitClass != UnitBoat:
    agent.embarkedUnitClass = unitClass
  agent.maxHp = UnitMaxHpByClass[unitClass]
  agent.attackDamage = UnitAttackDamageByClass[unitClass]
  agent.hp = agent.maxHp
  agent.stance = defaultStanceForClass(unitClass)
  # Initialize monk faith
  if unitClass == UnitMonk:
    agent.faith = MonkMaxFaith
  else:
    agent.faith = 0

proc applyUnitClass*(env: Environment, agent: Thing, unitClass: AgentUnitClass) =
  ## Apply unit class stats with team modifier bonuses
  ## Also maintains tankUnits/monkUnits collections for efficient aura iteration
  let oldClass = agent.unitClass
  agent.unitClass = unitClass
  if unitClass != UnitBoat:
    agent.embarkedUnitClass = unitClass
  let teamId = getTeamId(agent)
  let modifiers = env.teamModifiers[teamId]
  # Use base stat override if set (>0), otherwise use default
  let baseHp = if modifiers.unitBaseHpOverride[unitClass] > 0:
    modifiers.unitBaseHpOverride[unitClass]
  else:
    UnitMaxHpByClass[unitClass]
  let baseAttack = if modifiers.unitBaseAttackOverride[unitClass] > 0:
    modifiers.unitBaseAttackOverride[unitClass]
  else:
    UnitAttackDamageByClass[unitClass]
  agent.maxHp = baseHp + modifiers.unitHpBonus[unitClass]
  agent.attackDamage = baseAttack + modifiers.unitAttackBonus[unitClass]
  # Apply CivBonus unit HP and attack multipliers
  if teamId >= 0 and teamId < MapRoomObjectsTeams:
    let civBonus = env.teamCivBonuses[teamId]
    if civBonus.unitHpMultiplier != 0.0'f32 and civBonus.unitHpMultiplier != 1.0'f32:
      agent.maxHp = max(1, int(float32(agent.maxHp) * civBonus.unitHpMultiplier + 0.5))
    if civBonus.unitAttackMultiplier != 0.0'f32 and civBonus.unitAttackMultiplier != 1.0'f32:
      agent.attackDamage = max(0, int(float32(agent.attackDamage) * civBonus.unitAttackMultiplier + 0.5))
  agent.hp = agent.maxHp
  # Initialize monk faith
  if unitClass == UnitMonk:
    agent.faith = MonkMaxFaith
  else:
    agent.faith = 0

  # Update aura unit collections for optimized aura processing
  # Tank units: ManAtArms, Knight, Cavalier, Paladin have shield auras
  let wasTank = oldClass in TankAuraUnits
  let isTank = unitClass in TankAuraUnits
  if wasTank and not isTank:
    # Remove from tankUnits (swap-and-pop for O(1))
    for i in 0 ..< env.tankUnits.len:
      if env.tankUnits[i] == agent:
        env.tankUnits[i] = env.tankUnits[^1]
        env.tankUnits.setLen(env.tankUnits.len - 1)
        break
  elif isTank and not wasTank:
    env.tankUnits.add(agent)

  # Monk units: have heal auras
  let wasMonk = oldClass == UnitMonk
  let isMonk = unitClass == UnitMonk
  if wasMonk and not isMonk:
    # Remove from monkUnits (swap-and-pop for O(1))
    for i in 0 ..< env.monkUnits.len:
      if env.monkUnits[i] == agent:
        env.monkUnits[i] = env.monkUnits[^1]
        env.monkUnits.setLen(env.monkUnits.len - 1)
        break
  elif isMonk and not wasMonk:
    env.monkUnits.add(agent)

  # Villager tracking for town bell garrison optimization
  let wasVillager = oldClass == UnitVillager
  let isVillager = unitClass == UnitVillager
  # teamId already computed above at function start
  if teamId >= 0 and teamId < MapRoomObjectsTeams:
    if wasVillager and not isVillager:
      # Remove from teamVillagers (swap-and-pop for O(1))
      for i in 0 ..< env.teamVillagers[teamId].len:
        if env.teamVillagers[teamId][i] == agent:
          env.teamVillagers[teamId][i] = env.teamVillagers[teamId][^1]
          env.teamVillagers[teamId].setLen(env.teamVillagers[teamId].len - 1)
          break
    elif isVillager and not wasVillager:
      env.teamVillagers[teamId].add(agent)

proc embarkAgent*(env: Environment, agent: Thing) =
  if agent.unitClass in {UnitBoat, UnitTradeCog, UnitGalley, UnitFireShip,
                          UnitFishingShip, UnitTransportShip, UnitDemoShip, UnitCannonGalleon}:
    return
  agent.embarkedUnitClass = agent.unitClass
  applyUnitClass(env, agent, UnitBoat)

proc disembarkAgent*(env: Environment, agent: Thing) =
  if agent.unitClass == UnitTradeCog:
    return  # Trade Cogs never disembark
  if agent.unitClass != UnitBoat:
    return
  var target = agent.embarkedUnitClass
  if target == UnitBoat:
    target = UnitVillager
  applyUnitClass(env, agent, target)
{.pop.}

# Forward declaration - implementation in tint.nim (included below)
proc ensureTintColors*(env: Environment) {.inline.}

proc scoreTerritory*(env: Environment): TerritoryScore =
  ## Compute territory ownership by nearest tint color (teams + clippy).
  ## Ensures tint colors are up-to-date before scoring.
  env.ensureTintColors()
  var score: TerritoryScore
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      if not DefaultScoreIncludeWater and env.terrain[x][y] == Water:
        continue
      let tint = env.computedTintColors[x][y]
      if tint.intensity < DefaultScoreNeutralThreshold:
        inc score.neutralTiles
        continue
      var bestDist = 1.0e9'f32
      var bestTeam = -1
      # Clippy as NPC team
      let drc = tint.r - ClippyTint.r
      let dgc = tint.g - ClippyTint.g
      let dbc = tint.b - ClippyTint.b
      bestDist = drc * drc + dgc * dgc + dbc * dbc
      bestTeam = MapRoomObjectsTeams
      for teamId in 0 ..< min(env.teamColors.len, MapRoomObjectsTeams):
        let teamColor = env.teamColors[teamId]
        let dr = tint.r - teamColor.r
        let dg = tint.g - teamColor.g
        let db = tint.b - teamColor.b
        let dist = dr * dr + dg * dg + db * db
        if dist < bestDist:
          bestDist = dist
          bestTeam = teamId
      if bestTeam == MapRoomObjectsTeams:
        inc score.clippyTiles
      elif bestTeam >= 0 and bestTeam < MapRoomObjectsTeams:
        inc score.teamTiles[bestTeam]
      inc score.scoredTiles
  score


proc setRallyPoint*(building: Thing, pos: IVec2) =
  ## Set a building's rally point. Trained units will auto-move here after spawning.
  building.rallyPoint = pos

proc clearRallyPoint*(building: Thing) =
  ## Clear a building's rally point.
  building.rallyPoint = ivec2(-1, -1)

proc hasRallyPoint*(building: Thing): bool =
  ## Check if a building has an active rally point.
  building.rallyPoint.x >= 0 and building.rallyPoint.y >= 0

proc rebuildObservationsForAgent(env: Environment, agentId: int, agent: Thing) {.inline.} =
  ## Rebuild all observation layers for a single agent.
  ## Optimization: precompute valid observation bounds to avoid per-iteration checks.
  let agentPos = agent.pos
  # Precompute valid observation ranges - eliminates per-iteration boundary checks
  # For obsX: worldX = agentPos.x + (obsX - ObservationRadius) must be in [0, MapWidth)
  # Solving: 0 <= agentPos.x + obsX - ObservationRadius < MapWidth
  #          ObservationRadius - agentPos.x <= obsX < MapWidth - agentPos.x + ObservationRadius
  let minObsX = max(0, ObservationRadius - agentPos.x)
  let maxObsX = min(ObservationWidth, MapWidth - agentPos.x + ObservationRadius)
  let minObsY = max(0, ObservationRadius - agentPos.y)
  let maxObsY = min(ObservationHeight, MapHeight - agentPos.y + ObservationRadius)
  for obsX in minObsX ..< maxObsX:
    let worldX = agentPos.x + (obsX - ObservationRadius)
    for obsY in minObsY ..< maxObsY:
      let worldY = agentPos.y + (obsY - ObservationRadius)
      writeTileObs(env, agentId, obsX, obsY, worldX, worldY)

proc rebuildObservations*(env: Environment) =
  ## Recompute all observation layers from the current environment state.
  ## Optimization: Only zero and rebuild observations for alive agents that moved.
  env.observationsInitialized = false

proc ensureObservations*(env: Environment) {.inline.} =
  ## Ensure observations are up-to-date (lazy rebuild if dirty).
  ## Call this before accessing env.observations directly.
  ## Optimized: uses per-agent dirty bits to skip stationary agents entirely.
  ## Only agents that moved since last rebuild get their observations updated.
  if env.observationsDirty:
    env.rebuildObservations()
    env.observationsDirty = false

  let firstRun = not env.observationsInitialized

  for agentId in 0 ..< env.agents.len:
    let agent = env.agents[agentId]
    if not isAgentAlive(env, agent):
      # Dead agent: zero their observation slot if needed
      if env.observationsInitialized:
        zeroMem(addr env.observations[agentId], sizeof(env.observations[agentId]))
      env.agentObsDirty[agentId] = false  # Clear dirty bit for dead agent
      continue

    # Use per-agent dirty bit instead of position comparison
    # Dirty bit is set when agent moves (in updateSpatialIndex) or at spawn
    if firstRun or env.agentObsDirty[agentId]:
      # Agent moved or first run: zero and full rebuild
      if not firstRun:
        zeroMem(addr env.observations[agentId], sizeof(env.observations[agentId]))
      rebuildObservationsForAgent(env, agentId, agent)
      env.agentObsDirty[agentId] = false  # Clear dirty bit after rebuild
    # else: Agent stationary - terrain/biome unchanged, skip rebuild
    # Note: Things could move in/out of view, but we accept this minor
    # staleness for major perf gain. Next movement will refresh.

  # Rally point layer: mark tiles that are rally targets for friendly buildings
  # Optimization: collect rally points by team first, then iterate agents once
  # This changes O(buildings × agents) to O(buildings + agents × rally_points_per_team)
  var rallyPointsByTeam: array[MapRoomObjectsTeams + 1, seq[IVec2]]
  for i in 0 .. MapRoomObjectsTeams:
    rallyPointsByTeam[i] = @[]

  # First pass: collect all rally points grouped by team
  for bKind in TeamBuildingKinds:
    for thing in env.thingsByKind[bKind]:
      if not thing.hasRallyPoint():
        continue
      let rp = thing.rallyPoint
      if not isValidPos(rp):
        continue
      let buildingTeam = thing.teamId
      if buildingTeam >= 0 and buildingTeam <= MapRoomObjectsTeams:
        rallyPointsByTeam[buildingTeam].add(rp)

  # Second pass: for each agent, mark rally points from their team
  for agentId in 0 ..< env.agents.len:
    let agent = env.agents[agentId]
    if not isAgentAlive(env, agent):
      continue
    let teamId = getTeamId(agent)
    if teamId < 0 or teamId > MapRoomObjectsTeams:
      continue
    var agentObs = addr env.observations[agentId]
    for rp in rallyPointsByTeam[teamId]:
      let obsX = rp.x - agent.pos.x + ObservationRadius
      let obsY = rp.y - agent.pos.y + ObservationRadius
      if obsX >= 0 and obsX < ObservationWidth and obsY >= 0 and obsY < ObservationHeight:
        agentObs[][ord(RallyPointLayer)][obsX][obsY] = 1

  env.observationsInitialized = true

## Grid queries (getThing, getBackgroundThing, isEmpty, hasDoor, etc.) are in environment_grid.nim
## Elevation/movement checks (canTraverseElevation, willCauseCliffFallDamage, isBuildableTerrain, isSpawnable) are in environment_grid.nim

proc canPlace*(env: Environment, pos: IVec2, checkFrozen: bool = true): bool {.inline.} =
  ## Check if a building can be placed at the position.
  ## NOTE: Remains here because it uses isTileFrozen from colors.nim (included file).
  isValidPos(pos) and env.isEmpty(pos) and isNil(env.getBackgroundThing(pos)) and
    (not checkFrozen or not isTileFrozen(pos, env)) and isBuildableTerrain(env.terrain[pos.x][pos.y])

proc canPlaceDock*(env: Environment, pos: IVec2, checkFrozen: bool = true): bool {.inline.} =
  ## Check if a dock can be placed at the position (must be water or shallow water).
  ## NOTE: Remains here because it uses isTileFrozen from colors.nim (included file).
  isValidPos(pos) and env.isEmpty(pos) and isNil(env.getBackgroundThing(pos)) and
    (not checkFrozen or not isTileFrozen(pos, env)) and env.terrain[pos.x][pos.y] in WaterTerrain

## resetTileColor is in environment_grid.nim

# Build craft recipes after registry is available.
CraftRecipes = initCraftRecipesBase()
appendBuildingRecipes(CraftRecipes)

proc stockpileCapacityLeft(agent: Thing): int {.inline.} =
  var total = 0
  for invKey, invCount in agent.inventory.pairs:
    if invCount > 0 and isStockpileResourceKey(invKey):
      total += invCount
  max(0, ResourceCarryCapacity - total)

proc getVillagerCarryCapacity*(env: Environment, teamId: int): int
  ## Forward declaration - defined in economy tech section below

proc stockpileCapacityLeftWithTech(env: Environment, agent: Thing): int {.inline.} =
  ## Stockpile capacity respecting Wheelbarrow/Hand Cart tech bonuses for villagers.
  let capacity = if agent.unitClass == UnitVillager:
    env.getVillagerCarryCapacity(getTeamId(agent))
  else:
    ResourceCarryCapacity
  var total = 0
  for invKey, invCount in agent.inventory.pairs:
    if invCount > 0 and isStockpileResourceKey(invKey):
      total += invCount
  max(0, capacity - total)

proc giveItem(env: Environment, agent: Thing, key: ItemKey, count: int = 1): bool =
  if count <= 0:
    return false
  if isStockpileResourceKey(key):
    if env.stockpileCapacityLeftWithTech(agent) < count:
      return false
  else:
    if getInv(agent, key) + count > MapObjectAgentMaxInventory:
      return false
  setInv(agent, key, getInv(agent, key) + count)
  env.updateAgentInventoryObs(agent, key)
  true

proc useStorageBuilding(env: Environment, agent: Thing, storage: Thing, allowed: openArray[ItemKey]): bool =
  if storage.inventory.len > 0:
    var storedKey = ItemNone
    var storedCount = 0
    for key, count in storage.inventory.pairs:
      if count > 0:
        storedKey = key
        storedCount = count
        break
    if storedKey == ItemNone:
      return false
    if allowed.len > 0:
      var allowedMatch = false
      for allowedKey in allowed:
        if storedKey == allowedKey:
          allowedMatch = true
          break
      if not allowedMatch:
        return false
    let agentCount = getInv(agent, storedKey)
    let storageSpace = max(0, storage.barrelCapacity - storedCount)
    if agentCount > 0 and storageSpace > 0:
      let moved = min(agentCount, storageSpace)
      setInv(agent, storedKey, agentCount - moved)
      setInv(storage, storedKey, storedCount + moved)
      env.updateAgentInventoryObs(agent, storedKey)
      return true
    let capacityLeft =
      if isStockpileResourceKey(storedKey):
        env.stockpileCapacityLeftWithTech(agent)
      else:
        max(0, MapObjectAgentMaxInventory - agentCount)
    if capacityLeft > 0:
      let moved = min(storedCount, capacityLeft)
      if moved > 0:
        setInv(agent, storedKey, agentCount + moved)
        let remaining = storedCount - moved
        setInv(storage, storedKey, remaining)
        env.updateAgentInventoryObs(agent, storedKey)
        return true
    return false

  var choiceKey = ItemNone
  var choiceCount = 0
  if allowed.len == 0:
    for key, count in agent.inventory.pairs:
      if count > choiceCount:
        choiceKey = key
        choiceCount = count
  else:
    for key in allowed:
      let count = getInv(agent, key)
      if count > choiceCount:
        choiceKey = key
        choiceCount = count
  if choiceCount > 0 and choiceKey != ItemNone:
    let moved = min(choiceCount, storage.barrelCapacity)
    setInv(agent, choiceKey, choiceCount - moved)
    setInv(storage, choiceKey, moved)
    env.updateAgentInventoryObs(agent, choiceKey)
    return true
  false

proc useDropoffBuilding(env: Environment, agent: Thing, allowed: set[StockpileResource]): bool =
  let teamId = getTeamId(agent)
  var depositKeys: seq[ItemKey] = @[]
  for key, count in agent.inventory.pairs:
    if count <= 0:
      continue
    if not isStockpileResourceKey(key):
      continue
    let stockpileRes = stockpileResourceForItem(key)
    if stockpileRes in allowed:
      depositKeys.add(key)
  if depositKeys.len == 0:
    return false
  for key in depositKeys:
    let count = getInv(agent, key)
    if count <= 0:
      continue
    let stockpileRes = stockpileResourceForItem(key)
    env.addToStockpile(teamId, stockpileRes, count)
    when defined(eventLog):
      logResourceDeposited(teamId, $stockpileRes, count, env.currentStep)
    when defined(econAudit):
      recordDeposit(teamId, stockpileRes, count, env.currentStep)
    setInv(agent, key, 0)
    env.updateAgentInventoryObs(agent, key)
  true

proc tryTrainUnit(env: Environment, agent: Thing, building: Thing, unitClass: AgentUnitClass,
                  costs: openArray[tuple[res: StockpileResource, count: int]], cooldown: int): bool =
  if agent.unitClass != UnitVillager:
    return false
  let teamId = getTeamId(agent)
  if building.teamId != teamId:
    return false
  if not env.spendStockpile(teamId, costs):
    return false
  applyUnitClass(env, agent, unitClass)
  if agent.inventorySpear > 0:
    agent.inventorySpear = 0
  building.cooldown = cooldown
  true

proc queueTrainUnit*(env: Environment, building: Thing, teamId: int,
                     unitClass: AgentUnitClass,
                     costs: openArray[tuple[res: StockpileResource, count: int]]): bool =
  ## Queue a unit for training at a building (AoE2-style production queue).
  ## Resources are spent when queued. When a villager later interacts with the
  ## building and there's a ready entry, the villager is instantly converted
  ## without additional cost.
  if building.productionQueue.entries.len >= ProductionQueueMaxSize:
    return false
  if building.teamId != teamId:
    return false
  # Apply CivBonus food cost multiplier to training costs
  var adjustedCosts: seq[tuple[res: StockpileResource, count: int]] = @[]
  if teamId >= 0 and teamId < MapRoomObjectsTeams:
    let civBonus = env.teamCivBonuses[teamId]
    let applyFood = civBonus.foodCostMultiplier != 0.0'f32 and civBonus.foodCostMultiplier != 1.0'f32
    let applyWood = civBonus.woodCostMultiplier != 0.0'f32 and civBonus.woodCostMultiplier != 1.0'f32
    if applyFood or applyWood:
      for cost in costs:
        var c = cost
        if applyFood and c.res == ResourceFood:
          c.count = max(1, int(float32(c.count) * civBonus.foodCostMultiplier + 0.5))
        elif applyWood and c.res == ResourceWood:
          c.count = max(1, int(float32(c.count) * civBonus.woodCostMultiplier + 0.5))
        adjustedCosts.add(c)
  if adjustedCosts.len > 0:
    if not env.spendStockpile(teamId, adjustedCosts):
      return false
  else:
    if not env.spendStockpile(teamId, costs):
      return false
  when defined(econAudit):
    recordTrainingCost(teamId, costs, env.currentStep)
  let trainTime = unitTrainTime(unitClass)
  building.productionQueue.entries.add(ProductionQueueEntry(
    unitClass: unitClass,
    totalSteps: trainTime,
    remainingSteps: trainTime
  ))
  true

proc refundTrainCosts(env: Environment, building: Thing) =
  ## Refund training costs for a cancelled queue entry, applying CivBonus multipliers.
  let teamId = building.teamId
  if teamId >= 0 and teamId < MapRoomObjectsTeams:
    let baseCosts = buildingTrainCosts(building.kind)
    let civBonus = env.teamCivBonuses[teamId]
    let applyFood = civBonus.foodCostMultiplier != 0.0'f32 and civBonus.foodCostMultiplier != 1.0'f32
    let applyWood = civBonus.woodCostMultiplier != 0.0'f32 and civBonus.woodCostMultiplier != 1.0'f32
    for cost in baseCosts:
      var refundAmount = cost.count
      if applyFood and cost.res == ResourceFood:
        refundAmount = max(1, int(float32(cost.count) * civBonus.foodCostMultiplier + 0.5))
      elif applyWood and cost.res == ResourceWood:
        refundAmount = max(1, int(float32(cost.count) * civBonus.woodCostMultiplier + 0.5))
      env.teamStockpiles[teamId].counts[cost.res] += refundAmount
    when defined(econAudit):
      recordRefund(teamId, baseCosts, env.currentStep)

proc cancelLastQueued*(env: Environment, building: Thing): bool =
  ## Cancel the last unit in the production queue, refunding resources.
  if building.productionQueue.entries.len == 0:
    return false
  building.productionQueue.entries.setLen(building.productionQueue.entries.len - 1)
  env.refundTrainCosts(building)
  true

proc cancelQueueEntry*(env: Environment, building: Thing, index: int): bool =
  ## Cancel a specific unit in the production queue by index, refunding resources.
  if index < 0 or index >= building.productionQueue.entries.len:
    return false
  building.productionQueue.entries.delete(index)
  env.refundTrainCosts(building)
  true

proc effectiveTrainUnit*(env: Environment, buildingKind: ThingKind, teamId: int): AgentUnitClass =
  ## Returns the effective unit class trained by a building, considering upgrades.
  ## For example, if LongSwordsman upgrade is researched, Barracks trains LongSwordsman instead of ManAtArms.
  ## "Unlock" upgrades (Knight, Skirmisher, CavalryArcher) switch the building's production line.
  let baseUnit = buildingTrainUnit(buildingKind, teamId)
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return baseUnit
  # Check upgrade chain for the base unit
  case baseUnit
  of UnitManAtArms:
    if env.teamUnitUpgrades[teamId].researched[UpgradeChampion]:
      return UnitChampion
    if env.teamUnitUpgrades[teamId].researched[UpgradeLongSwordsman]:
      return UnitLongSwordsman
    return UnitManAtArms
  of UnitScout:
    # Knight line replaces Scout line once researched
    if env.teamUnitUpgrades[teamId].researched[UpgradeKnight]:
      return UnitKnight
    if env.teamUnitUpgrades[teamId].researched[UpgradeHussar]:
      return UnitHussar
    if env.teamUnitUpgrades[teamId].researched[UpgradeLightCavalry]:
      return UnitLightCavalry
    return UnitScout
  of UnitArcher:
    # CavalryArcher line replaces earlier lines once researched
    if env.teamUnitUpgrades[teamId].researched[UpgradeCavalryArcher]:
      if env.teamUnitUpgrades[teamId].researched[UpgradeHeavyCavalryArcher]:
        return UnitHeavyCavalryArcher
      return UnitCavalryArcher
    # Skirmisher line replaces Archer line once researched
    if env.teamUnitUpgrades[teamId].researched[UpgradeSkirmisher]:
      if env.teamUnitUpgrades[teamId].researched[UpgradeEliteSkirmisher]:
        return UnitEliteSkirmisher
      return UnitSkirmisher
    if env.teamUnitUpgrades[teamId].researched[UpgradeArbalester]:
      return UnitArbalester
    if env.teamUnitUpgrades[teamId].researched[UpgradeCrossbowman]:
      return UnitCrossbowman
    return UnitArcher
  else:
    return baseUnit

proc tryBatchQueueTrain*(env: Environment, building: Thing, teamId: int,
                         count: int): int =
  ## Queue multiple units for training (batch/shift-click).
  ## Returns the number of units actually queued.
  if not buildingHasTrain(building.kind):
    return 0
  let unitClass = env.effectiveTrainUnit(building.kind, teamId)
  let costs = buildingTrainCosts(building.kind)
  var queued = 0
  for i in 0 ..< count:
    if not env.queueTrainUnit(building, teamId, unitClass, costs):
      break
    inc queued
  queued

proc productionQueueHasReady*(building: Thing): bool =
  ## Check if the building has a queue entry ready for conversion.
  building.productionQueue.entries.len > 0 and
    building.productionQueue.entries[0].remainingSteps <= 0

proc consumeReadyQueueEntry*(building: Thing): AgentUnitClass =
  ## Consume the front ready entry from the queue. Returns the unit class.
  ## Caller must verify productionQueueHasReady first.
  result = building.productionQueue.entries[0].unitClass
  building.productionQueue.entries.delete(0)

proc processProductionQueue*(building: Thing) =
  ## Tick one step of a building's production queue countdown.
  if building.productionQueue.entries.len > 0 and
     building.productionQueue.entries[0].remainingSteps > 0:
    building.productionQueue.entries[0].remainingSteps -= 1

proc getNextBlacksmithUpgrade*(env: Environment, teamId: int): BlacksmithUpgradeType =
  ## Find the next upgrade to research (lowest level across all types).
  ## Returns the upgrade type with the lowest current level.
  var minLevel = BlacksmithUpgradeMaxLevel + 1
  result = UpgradeMeleeAttack  # Default
  for upgradeType in BlacksmithUpgradeType:
    let level = env.teamBlacksmithUpgrades[teamId].levels[upgradeType]
    if level < minLevel:
      minLevel = level
      result = upgradeType

proc tryResearchBlacksmithUpgrade*(env: Environment, agent: Thing, building: Thing): bool =
  ## Attempt to research the next Blacksmith upgrade for the team.
  ## Costs: Food + Gold, increasing by level.
  ## Returns true if research was successful.
  if agent.unitClass != UnitVillager:
    return false
  let teamId = getTeamId(agent)
  if building.teamId != teamId:
    return false
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false

  # Find the next upgrade to research
  let upgradeType = env.getNextBlacksmithUpgrade(teamId)
  let currentLevel = env.teamBlacksmithUpgrades[teamId].levels[upgradeType]

  # Check if already at max level
  if currentLevel >= BlacksmithUpgradeMaxLevel:
    return false

  # Calculate cost based on current level (level 0->1: base cost, 1->2: 2x, 2->3: 3x)
  let costMultiplier = currentLevel + 1
  let foodCost = BlacksmithUpgradeFoodCost * costMultiplier
  let goldCost = BlacksmithUpgradeGoldCost * costMultiplier

  # Check and spend resources
  let costs = [(ResourceFood, foodCost), (ResourceGold, goldCost)]
  if not env.spendStockpile(teamId, costs):
    return false
  when defined(econAudit):
    recordResearchCost(teamId, costs, env.currentStep)

  # Apply the upgrade
  env.teamBlacksmithUpgrades[teamId].levels[upgradeType] = currentLevel + 1
  building.cooldown = 5  # Short cooldown after research
  when defined(eventLog):
    logTechResearched(teamId, "Blacksmith " & $upgradeType & " Level " & $(currentLevel + 1), env.currentStep)
  when defined(techAudit):
    logBlacksmithUpgrade(teamId, upgradeType, currentLevel + 1, env.currentStep)
  true

proc getNextUniversityTech(env: Environment, teamId: int): UniversityTechType =
  ## Find the next unresearched University tech.
  ## Returns techs in order: Ballistics first (most impactful for ranged combat).
  for techType in UniversityTechType:
    if not env.teamUniversityTechs[teamId].researched[techType]:
      return techType
  # All researched, return first (no-op in caller)
  TechBallistics

proc hasUniversityTech*(env: Environment, teamId: int, tech: UniversityTechType): bool {.inline.} =
  ## Check if a team has researched a specific University tech.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false
  env.teamUniversityTechs[teamId].researched[tech]

proc tryResearchUniversityTech*(env: Environment, agent: Thing, building: Thing): bool =
  ## Attempt to research the next University tech for the team.
  ## Costs: Food + Gold + Wood (varies by tech).
  ## Returns true if research was successful.
  if agent.unitClass != UnitVillager:
    return false
  let teamId = getTeamId(agent)
  if building.teamId != teamId:
    return false
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false

  # Find the next tech to research
  let techType = env.getNextUniversityTech(teamId)

  # Check if already researched
  if env.teamUniversityTechs[teamId].researched[techType]:
    return false

  # Calculate cost - costs increase for later techs
  let techIndex = ord(techType) + 1
  let foodCost = UniversityTechFoodCost * techIndex
  let goldCost = UniversityTechGoldCost * techIndex
  let woodCost = UniversityTechWoodCost * techIndex

  # Check and spend resources
  let costs = [(ResourceFood, foodCost), (ResourceGold, goldCost), (ResourceWood, woodCost)]
  if not env.spendStockpile(teamId, costs):
    return false
  when defined(econAudit):
    recordResearchCost(teamId, costs, env.currentStep)

  # Apply the tech
  env.teamUniversityTechs[teamId].researched[techType] = true
  building.cooldown = 8  # Longer cooldown for tech research
  when defined(eventLog):
    logTechResearched(teamId, "University " & $techType, env.currentStep)
  when defined(techAudit):
    logUniversityTech(teamId, techType, env.currentStep)
  true

proc castleTechsForTeam*(teamId: int): (CastleTechType, CastleTechType) =
  ## Returns the (Castle Age, Imperial Age) tech pair for a team.
  ## Each team has exactly 2 unique techs, interleaved in the enum.
  let base = CastleTechType(teamId * 2)
  let imperial = CastleTechType(teamId * 2 + 1)
  (base, imperial)

proc getNextCastleTech(env: Environment, teamId: int): CastleTechType =
  ## Find the next unresearched Castle tech for this team.
  ## Castle Age tech must be researched before Imperial Age tech.
  let (castleAge, imperialAge) = castleTechsForTeam(teamId)
  if not env.teamCastleTechs[teamId].researched[castleAge]:
    return castleAge
  if not env.teamCastleTechs[teamId].researched[imperialAge]:
    return imperialAge
  # Both researched, return castle age (no-op in caller)
  castleAge

proc hasCastleTech*(env: Environment, teamId: int, tech: CastleTechType): bool {.inline.} =
  ## Check if a team has researched a specific Castle unique tech.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false
  env.teamCastleTechs[teamId].researched[tech]

proc applyCastleTechBonuses*(env: Environment, teamId: int, tech: CastleTechType) =
  ## Apply the bonuses from a Castle unique tech to the team's modifiers.
  ## Called when a tech is researched.
  case tech
  of CastleTechYeomen:
    # +1 archer range (modeled as +1 archer attack), +2 tower attack
    env.teamModifiers[teamId].unitAttackBonus[UnitArcher] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitLongbowman] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitCrossbowman] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitArbalester] += 1
  of CastleTechKataparuto:
    # +3 trebuchet attack
    env.teamModifiers[teamId].unitAttackBonus[UnitTrebuchet] += 3
  of CastleTechLogistica:
    # +1 infantry attack
    env.teamModifiers[teamId].unitAttackBonus[UnitManAtArms] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitSamurai] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitWoadRaider] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitTeutonicKnight] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitHuskarl] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitLongSwordsman] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitChampion] += 1
  of CastleTechCrenellations:
    # +2 castle attack (applied via hasCastleTech check in tower attack)
    discard
  of CastleTechGreekFire:
    # +2 tower attack vs siege (applied via hasCastleTech check in tower attack)
    discard
  of CastleTechFurorCeltica:
    # +2 siege attack
    env.teamModifiers[teamId].unitAttackBonus[UnitBatteringRam] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitMangonel] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitTrebuchet] += 2
  of CastleTechAnarchy:
    # +1 infantry HP
    env.teamModifiers[teamId].unitHpBonus[UnitManAtArms] += 1
    env.teamModifiers[teamId].unitHpBonus[UnitSamurai] += 1
    env.teamModifiers[teamId].unitHpBonus[UnitWoadRaider] += 1
    env.teamModifiers[teamId].unitHpBonus[UnitTeutonicKnight] += 1
    env.teamModifiers[teamId].unitHpBonus[UnitHuskarl] += 1
    env.teamModifiers[teamId].unitHpBonus[UnitLongSwordsman] += 1
    env.teamModifiers[teamId].unitHpBonus[UnitChampion] += 1
  of CastleTechPerfusion:
    # Military units train faster (modeled as +2 all military attack)
    env.teamModifiers[teamId].unitAttackBonus[UnitManAtArms] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitArcher] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitScout] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitKnight] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitLongSwordsman] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitChampion] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitLightCavalry] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitHussar] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitCrossbowman] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitArbalester] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitCavalier] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitPaladin] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitCamel] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitHeavyCamel] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitImperialCamel] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitSkirmisher] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitEliteSkirmisher] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitCavalryArcher] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitHeavyCavalryArcher] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitHandCannoneer] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitScorpion] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitSamurai] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitLongbowman] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitCataphract] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitWoadRaider] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitTeutonicKnight] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitHuskarl] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitMameluke] += 2
    env.teamModifiers[teamId].unitAttackBonus[UnitJanissary] += 2
  of CastleTechIronclad:
    # +3 siege HP
    env.teamModifiers[teamId].unitHpBonus[UnitBatteringRam] += 3
    env.teamModifiers[teamId].unitHpBonus[UnitMangonel] += 3
    env.teamModifiers[teamId].unitHpBonus[UnitTrebuchet] += 3
  of CastleTechCrenellations2:
    # +2 castle attack (applied via hasCastleTech check in tower attack)
    discard
  of CastleTechBerserkergang:
    # +2 infantry HP
    env.teamModifiers[teamId].unitHpBonus[UnitManAtArms] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitSamurai] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitWoadRaider] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitTeutonicKnight] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitHuskarl] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitLongSwordsman] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitChampion] += 2
  of CastleTechChieftains:
    # +1 cavalry attack
    env.teamModifiers[teamId].unitAttackBonus[UnitScout] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitKnight] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitCavalier] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitPaladin] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitCataphract] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitMameluke] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitLightCavalry] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitHussar] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitCamel] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitHeavyCamel] += 1
    env.teamModifiers[teamId].unitAttackBonus[UnitImperialCamel] += 1
  of CastleTechZealotry:
    # +2 cavalry HP
    env.teamModifiers[teamId].unitHpBonus[UnitScout] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitKnight] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitCavalier] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitPaladin] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitCataphract] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitMameluke] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitLightCavalry] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitHussar] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitCamel] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitHeavyCamel] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitImperialCamel] += 2
  of CastleTechMahayana:
    # +1 monk effectiveness (modeled as +1 monk attack)
    env.teamModifiers[teamId].unitAttackBonus[UnitMonk] += 1
  of CastleTechSipahi:
    # +2 archer HP
    env.teamModifiers[teamId].unitHpBonus[UnitArcher] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitLongbowman] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitJanissary] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitCrossbowman] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitArbalester] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitSkirmisher] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitEliteSkirmisher] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitCavalryArcher] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitHeavyCavalryArcher] += 2
    env.teamModifiers[teamId].unitHpBonus[UnitHandCannoneer] += 2
  of CastleTechArtillery:
    # +2 tower and castle attack (applied via hasCastleTech check in tower attack)
    discard

proc tryResearchCastleTech*(env: Environment, agent: Thing, building: Thing): bool =
  ## Attempt to research the next Castle unique tech for the team.
  ## Each team has 2 unique techs (Castle Age first, then Imperial Age).
  ## Only villagers can research. Returns true if research was successful.
  if agent.unitClass != UnitVillager:
    return false
  let teamId = getTeamId(agent)
  if building.teamId != teamId:
    return false
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false

  # Find the next tech to research for this team
  let techType = env.getNextCastleTech(teamId)

  # Check if already researched
  if env.teamCastleTechs[teamId].researched[techType]:
    return false

  # Determine cost based on whether this is Castle Age or Imperial Age tech
  let (castleAge, _) = castleTechsForTeam(teamId)
  let isImperial = techType != castleAge
  let foodCost = if isImperial: CastleTechImperialFoodCost else: CastleTechFoodCost
  let goldCost = if isImperial: CastleTechImperialGoldCost else: CastleTechGoldCost

  # Check and spend resources
  let costs = [(ResourceFood, foodCost), (ResourceGold, goldCost)]
  if not env.spendStockpile(teamId, costs):
    return false
  when defined(econAudit):
    recordResearchCost(teamId, costs, env.currentStep)

  # Apply the tech
  env.teamCastleTechs[teamId].researched[techType] = true
  env.applyCastleTechBonuses(teamId, techType)
  building.cooldown = 10  # Longer cooldown for unique tech research
  when defined(eventLog):
    logTechResearched(teamId, "Castle " & $techType, env.currentStep)
  when defined(techAudit):
    logCastleTech(teamId, techType, isImperial, env.currentStep)
  true

# ---- UI-driven research (no villager required) ----

proc uiResearchBlacksmithUpgrade*(env: Environment, building: Thing,
                                  upgradeType: BlacksmithUpgradeType): bool =
  ## UI-driven research: directly research a specific Blacksmith upgrade.
  ## Used when player clicks research button in command panel.
  if building.kind != Blacksmith:
    return false
  let teamId = building.teamId
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false

  let currentLevel = env.teamBlacksmithUpgrades[teamId].levels[upgradeType]
  if currentLevel >= BlacksmithUpgradeMaxLevel:
    return false

  let costMultiplier = currentLevel + 1
  let foodCost = BlacksmithUpgradeFoodCost * costMultiplier
  let goldCost = BlacksmithUpgradeGoldCost * costMultiplier
  let costs = [(ResourceFood, foodCost), (ResourceGold, goldCost)]

  if not env.spendStockpile(teamId, costs):
    return false
  when defined(econAudit):
    recordResearchCost(teamId, costs, env.currentStep)

  env.teamBlacksmithUpgrades[teamId].levels[upgradeType] = currentLevel + 1
  building.cooldown = 5
  when defined(eventLog):
    logTechResearched(teamId, "Blacksmith " & $upgradeType & " Level " & $(currentLevel + 1), env.currentStep)
  when defined(techAudit):
    logBlacksmithUpgrade(teamId, upgradeType, currentLevel + 1, env.currentStep)
  true

proc uiResearchUniversityTech*(env: Environment, building: Thing,
                               techType: UniversityTechType): bool =
  ## UI-driven research: directly research a specific University tech.
  if building.kind != University:
    return false
  let teamId = building.teamId
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false

  if env.teamUniversityTechs[teamId].researched[techType]:
    return false

  let techIndex = ord(techType) + 1
  let foodCost = UniversityTechFoodCost * techIndex
  let goldCost = UniversityTechGoldCost * techIndex
  let woodCost = UniversityTechWoodCost * techIndex
  let costs = [(ResourceFood, foodCost), (ResourceGold, goldCost), (ResourceWood, woodCost)]

  if not env.spendStockpile(teamId, costs):
    return false
  when defined(econAudit):
    recordResearchCost(teamId, costs, env.currentStep)

  env.teamUniversityTechs[teamId].researched[techType] = true
  building.cooldown = 8
  when defined(eventLog):
    logTechResearched(teamId, "University " & $techType, env.currentStep)
  when defined(techAudit):
    logUniversityTech(teamId, techType, env.currentStep)
  true

proc uiResearchCastleTech*(env: Environment, building: Thing, techIndex: int): bool =
  ## UI-driven research: research Castle unique tech by index (0=Castle Age, 1=Imperial).
  if building.kind != Castle:
    return false
  let teamId = building.teamId
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false

  let (castleAge, imperialAge) = castleTechsForTeam(teamId)
  let techType = if techIndex == 0: castleAge else: imperialAge

  if env.teamCastleTechs[teamId].researched[techType]:
    return false
  # Imperial Age requires Castle Age
  if techIndex == 1 and not env.teamCastleTechs[teamId].researched[castleAge]:
    return false

  let isImperial = techIndex == 1
  let foodCost = if isImperial: CastleTechImperialFoodCost else: CastleTechFoodCost
  let goldCost = if isImperial: CastleTechImperialGoldCost else: CastleTechGoldCost
  let costs = [(ResourceFood, foodCost), (ResourceGold, goldCost)]

  if not env.spendStockpile(teamId, costs):
    return false
  when defined(econAudit):
    recordResearchCost(teamId, costs, env.currentStep)

  env.teamCastleTechs[teamId].researched[techType] = true
  env.applyCastleTechBonuses(teamId, techType)
  building.cooldown = 10
  when defined(eventLog):
    logTechResearched(teamId, "Castle " & $techType, env.currentStep)
  when defined(techAudit):
    logCastleTech(teamId, techType, isImperial, env.currentStep)
  true

proc uiQueueTrainUnit*(env: Environment, building: Thing, unitClass: AgentUnitClass,
                       count: int = 1): int =
  ## UI-driven training: queue units directly from building without villager.
  ## Returns number of units successfully queued.
  if not buildingHasTrain(building.kind):
    return 0
  let teamId = building.teamId
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return 0
  let costs = buildingTrainCosts(building.kind)
  var queued = 0
  for i in 0 ..< count:
    if not env.queueTrainUnit(building, teamId, unitClass, costs):
      break
    inc queued
  queued

# ---- Unit upgrade / promotion chain logic (AoE2-style) ----

proc upgradePrerequisite*(upgrade: UnitUpgradeType): UnitUpgradeType =
  ## Returns the prerequisite upgrade that must be researched first.
  ## Tier-2 upgrades have no prerequisite (returns themselves).
  ## Tier-3 upgrades require the corresponding tier-2.
  case upgrade
  of UpgradeLongSwordsman: UpgradeLongSwordsman  # no prereq
  of UpgradeChampion: UpgradeLongSwordsman
  of UpgradeLightCavalry: UpgradeLightCavalry    # no prereq
  of UpgradeHussar: UpgradeLightCavalry
  of UpgradeKnight: UpgradeKnight                # no prereq (unlocks Knight line)
  of UpgradeCrossbowman: UpgradeCrossbowman      # no prereq
  of UpgradeArbalester: UpgradeCrossbowman
  of UpgradeSkirmisher: UpgradeSkirmisher         # no prereq (unlocks Skirmisher line)
  of UpgradeEliteSkirmisher: UpgradeSkirmisher    # requires Skirmisher unlock
  of UpgradeCavalryArcher: UpgradeCavalryArcher   # no prereq (unlocks CavalryArcher line)
  of UpgradeHeavyCavalryArcher: UpgradeCavalryArcher  # requires CavalryArcher unlock

proc upgradeSourceUnit*(upgrade: UnitUpgradeType): AgentUnitClass =
  ## Returns the unit class that gets upgraded.
  ## For "unlock" upgrades (Knight, Skirmisher, CavalryArcher), source = target
  ## since these unlock new production lines rather than converting existing units.
  case upgrade
  of UpgradeLongSwordsman: UnitManAtArms
  of UpgradeChampion: UnitLongSwordsman
  of UpgradeLightCavalry: UnitScout
  of UpgradeHussar: UnitLightCavalry
  of UpgradeKnight: UnitKnight              # unlock (no conversion)
  of UpgradeCrossbowman: UnitArcher
  of UpgradeArbalester: UnitCrossbowman
  of UpgradeSkirmisher: UnitSkirmisher      # unlock (no conversion)
  of UpgradeEliteSkirmisher: UnitSkirmisher
  of UpgradeCavalryArcher: UnitCavalryArcher  # unlock (no conversion)
  of UpgradeHeavyCavalryArcher: UnitCavalryArcher

proc upgradeTargetUnit*(upgrade: UnitUpgradeType): AgentUnitClass =
  ## Returns the unit class that results from the upgrade.
  ## For "unlock" upgrades, source = target (no existing units to convert).
  case upgrade
  of UpgradeLongSwordsman: UnitLongSwordsman
  of UpgradeChampion: UnitChampion
  of UpgradeLightCavalry: UnitLightCavalry
  of UpgradeHussar: UnitHussar
  of UpgradeKnight: UnitKnight              # unlock
  of UpgradeCrossbowman: UnitCrossbowman
  of UpgradeArbalester: UnitArbalester
  of UpgradeSkirmisher: UnitSkirmisher      # unlock
  of UpgradeEliteSkirmisher: UnitEliteSkirmisher
  of UpgradeCavalryArcher: UnitCavalryArcher  # unlock
  of UpgradeHeavyCavalryArcher: UnitHeavyCavalryArcher

proc upgradeBuilding*(upgrade: UnitUpgradeType): ThingKind =
  ## Returns the building where this upgrade is researched.
  case upgrade
  of UpgradeLongSwordsman, UpgradeChampion: Barracks
  of UpgradeLightCavalry, UpgradeHussar, UpgradeKnight: Stable
  of UpgradeCrossbowman, UpgradeArbalester,
     UpgradeSkirmisher, UpgradeEliteSkirmisher,
     UpgradeCavalryArcher, UpgradeHeavyCavalryArcher: ArcheryRange

proc upgradeCosts*(upgrade: UnitUpgradeType): seq[tuple[res: StockpileResource, count: int]] =
  ## Returns the resource costs for an upgrade.
  case upgrade
  of UpgradeLongSwordsman, UpgradeLightCavalry, UpgradeCrossbowman,
     UpgradeKnight, UpgradeSkirmisher, UpgradeCavalryArcher,
     UpgradeEliteSkirmisher, UpgradeHeavyCavalryArcher:
    @[(res: ResourceFood, count: UnitUpgradeTier2FoodCost),
      (res: ResourceGold, count: UnitUpgradeTier2GoldCost)]
  of UpgradeChampion, UpgradeHussar, UpgradeArbalester:
    @[(res: ResourceFood, count: UnitUpgradeTier3FoodCost),
      (res: ResourceGold, count: UnitUpgradeTier3GoldCost)]

proc hasUnitUpgrade*(env: Environment, teamId: int, upgrade: UnitUpgradeType): bool {.inline.} =
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false
  env.teamUnitUpgrades[teamId].researched[upgrade]

proc getNextUnitUpgrade*(env: Environment, teamId: int, buildingKind: ThingKind): UnitUpgradeType =
  ## Find the next available upgrade for the given building type.
  ## Rotates starting point by teamId to distribute across upgrade lines,
  ## so different teams research different upgrades (Knight vs LightCavalry, etc.)
  let allUpgrades = block:
    var upgrades: seq[UnitUpgradeType]
    for u in UnitUpgradeType:
      if upgradeBuilding(u) == buildingKind:
        upgrades.add(u)
    upgrades
  if allUpgrades.len == 0:
    return UpgradeLongSwordsman  # fallback
  let startIdx = teamId mod allUpgrades.len
  for offset in 0 ..< allUpgrades.len:
    let upgrade = allUpgrades[(startIdx + offset) mod allUpgrades.len]
    if env.teamUnitUpgrades[teamId].researched[upgrade]:
      continue
    # Check prerequisite
    let prereq = upgradePrerequisite(upgrade)
    if prereq != upgrade and not env.teamUnitUpgrades[teamId].researched[prereq]:
      continue
    return upgrade
  # No upgrades available; return first of this building type (caller checks researched)
  allUpgrades[0]

proc upgradeExistingUnits*(env: Environment, teamId: int, fromClass: AgentUnitClass, toClass: AgentUnitClass) =
  ## Upgrade all living units of fromClass on the given team to toClass.
  ## Preserves current HP ratio.
  when defined(techAudit):
    var unitsUpgraded = 0
    let baseHpFrom = UnitMaxHpByClass[fromClass]
    let baseHpTo = UnitMaxHpByClass[toClass]
    let baseAttackFrom = UnitAttackDamageByClass[fromClass]
    let baseAttackTo = UnitAttackDamageByClass[toClass]
  for agent in env.liveAgents:
    if env.terminated[agent.agentId] != 0.0:
      continue
    if getTeamId(agent) != teamId:
      continue
    if agent.unitClass != fromClass:
      continue
    let hpRatio = if agent.maxHp > 0: agent.hp.float / agent.maxHp.float else: 1.0
    # Use applyUnitClass to properly handle CivBonus multipliers, team modifiers,
    # and aura collection tracking (tankUnits, monkUnits, teamVillagers)
    applyUnitClass(env, agent, toClass)
    # Restore HP ratio (applyUnitClass sets hp = maxHp)
    agent.hp = max(1, int(hpRatio * agent.maxHp.float))
    when defined(techAudit):
      inc unitsUpgraded
  when defined(techAudit):
    if unitsUpgraded > 0:
      let attackDelta = baseAttackTo - baseAttackFrom
      let hpDelta = baseHpTo - baseHpFrom
      logUpgradeApplication(teamId, $fromClass & " -> " & $toClass, unitsUpgraded,
                            attackDelta, 0, hpDelta, env.currentStep)

proc tryResearchUnitUpgrade*(env: Environment, agent: Thing, building: Thing): bool =
  ## Attempt to research the next unit upgrade at a military building.
  ## Only villagers can research. Returns true if research was successful.
  if agent.unitClass != UnitVillager:
    return false
  let teamId = getTeamId(agent)
  if building.teamId != teamId:
    return false
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false

  let upgrade = env.getNextUnitUpgrade(teamId, building.kind)

  # Check if already researched
  if env.teamUnitUpgrades[teamId].researched[upgrade]:
    return false

  # Check prerequisite
  let prereq = upgradePrerequisite(upgrade)
  if prereq != upgrade and not env.teamUnitUpgrades[teamId].researched[prereq]:
    return false

  # Check and spend resources
  let costs = upgradeCosts(upgrade)
  if not env.spendStockpile(teamId, costs):
    return false
  when defined(econAudit):
    recordResearchCost(teamId, costs, env.currentStep)

  # Apply the upgrade
  env.teamUnitUpgrades[teamId].researched[upgrade] = true
  env.upgradeExistingUnits(teamId, upgradeSourceUnit(upgrade), upgradeTargetUnit(upgrade))
  building.cooldown = 8
  when defined(eventLog):
    logTechResearched(teamId, "Unit Upgrade " & $upgrade, env.currentStep)
  when defined(techAudit):
    logUnitUpgrade(teamId, upgrade, env.currentStep, costs)
  true

# ---- Economy tech logic (AoE2-style) ----

proc hasEconomyTech*(env: Environment, teamId: int, tech: EconomyTechType): bool {.inline.} =
  ## Check if a team has researched a specific economy tech.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false
  env.teamEconomyTechs[teamId].researched[tech]

proc getWoodGatherBonus*(env: Environment, teamId: int): int =
  ## Calculate total wood gathering bonus percentage from Lumber Camp techs.
  ## Returns bonus as integer percentage (e.g., 50 = +50%).
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return 0
  var bonus = 0
  if env.teamEconomyTechs[teamId].researched[TechDoubleBitAxe]:
    bonus += DoubleBitAxeGatherBonus
  if env.teamEconomyTechs[teamId].researched[TechBowSaw]:
    bonus += BowSawGatherBonus
  if env.teamEconomyTechs[teamId].researched[TechTwoManSaw]:
    bonus += TwoManSawGatherBonus
  bonus

proc getGoldGatherBonus*(env: Environment, teamId: int): int =
  ## Calculate total gold gathering bonus percentage from Mining Camp techs.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return 0
  var bonus = 0
  if env.teamEconomyTechs[teamId].researched[TechGoldMining]:
    bonus += GoldMiningGatherBonus
  if env.teamEconomyTechs[teamId].researched[TechGoldShaftMining]:
    bonus += GoldShaftMiningGatherBonus
  bonus

proc getStoneGatherBonus*(env: Environment, teamId: int): int =
  ## Calculate total stone gathering bonus percentage from Mining Camp techs.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return 0
  var bonus = 0
  if env.teamEconomyTechs[teamId].researched[TechStoneMining]:
    bonus += StoneMiningGatherBonus
  if env.teamEconomyTechs[teamId].researched[TechStoneShaftMining]:
    bonus += StoneShaftMiningGatherBonus
  bonus

proc getVillagerCarryCapacity*(env: Environment, teamId: int): int =
  ## Calculate villager carry capacity including economy tech bonuses.
  ## Base capacity is ResourceCarryCapacity (5).
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return ResourceCarryCapacity
  var capacity = ResourceCarryCapacity
  if env.teamEconomyTechs[teamId].researched[TechWheelbarrow]:
    capacity += WheelbarrowCarryBonus
  if env.teamEconomyTechs[teamId].researched[TechHandCart]:
    capacity += HandCartCarryBonus
  capacity

proc getVillagerSpeedBonus*(env: Environment, teamId: int): int =
  ## Calculate villager speed bonus percentage from economy techs.
  ## Returns bonus as integer percentage (e.g., 20 = +20%).
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return 0
  var bonus = 0
  if env.teamEconomyTechs[teamId].researched[TechWheelbarrow]:
    bonus += WheelbarrowSpeedBonus
  if env.teamEconomyTechs[teamId].researched[TechHandCart]:
    bonus += HandCartSpeedBonus
  bonus

proc getFarmFoodBonus*(env: Environment, teamId: int): int =
  ## Calculate total farm food bonus from Mill techs.
  ## Returns bonus food amount per farm.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return 0
  var bonus = 0
  if env.teamEconomyTechs[teamId].researched[TechHorseCollar]:
    bonus += HorseCollarFarmBonus
  if env.teamEconomyTechs[teamId].researched[TechHeavyPlow]:
    bonus += HeavyPlowFarmBonus
  if env.teamEconomyTechs[teamId].researched[TechCropRotation]:
    bonus += CropRotationFarmBonus
  bonus

proc canAutoReseed*(env: Environment, teamId: int): bool {.inline.} =
  ## Check if team has researched Horse Collar (enables auto-reseed).
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false
  env.teamEconomyTechs[teamId].researched[TechHorseCollar]

proc economyTechBuilding*(tech: EconomyTechType): ThingKind =
  ## Returns the building where this economy tech is researched.
  case tech
  of TechWheelbarrow, TechHandCart: TownCenter
  of TechDoubleBitAxe, TechBowSaw, TechTwoManSaw: LumberCamp
  of TechGoldMining, TechGoldShaftMining, TechStoneMining, TechStoneShaftMining: MiningCamp
  of TechHorseCollar, TechHeavyPlow, TechCropRotation: Mill

proc economyTechCost*(tech: EconomyTechType): seq[tuple[res: StockpileResource, count: int]] =
  ## Returns the resource costs for an economy tech.
  case tech
  of TechWheelbarrow:
    @[(res: ResourceFood, count: WheelbarrowFoodCost),
      (res: ResourceWood, count: WheelbarrowWoodCost)]
  of TechHandCart:
    @[(res: ResourceFood, count: HandCartFoodCost),
      (res: ResourceWood, count: HandCartWoodCost)]
  of TechDoubleBitAxe:
    @[(res: ResourceFood, count: DoubleBitAxeFoodCost),
      (res: ResourceWood, count: DoubleBitAxeWoodCost)]
  of TechBowSaw:
    @[(res: ResourceFood, count: BowSawFoodCost),
      (res: ResourceWood, count: BowSawWoodCost)]
  of TechTwoManSaw:
    @[(res: ResourceFood, count: TwoManSawFoodCost),
      (res: ResourceWood, count: TwoManSawWoodCost)]
  of TechGoldMining:
    @[(res: ResourceFood, count: GoldMiningFoodCost),
      (res: ResourceWood, count: GoldMiningWoodCost)]
  of TechGoldShaftMining:
    @[(res: ResourceFood, count: GoldShaftMiningFoodCost),
      (res: ResourceWood, count: GoldShaftMiningWoodCost)]
  of TechStoneMining:
    @[(res: ResourceFood, count: StoneMiningFoodCost),
      (res: ResourceWood, count: StoneMiningWoodCost)]
  of TechStoneShaftMining:
    @[(res: ResourceFood, count: StoneShaftMiningFoodCost),
      (res: ResourceWood, count: StoneShaftMiningWoodCost)]
  of TechHorseCollar:
    @[(res: ResourceFood, count: HorseCollarFoodCost),
      (res: ResourceWood, count: HorseCollarWoodCost)]
  of TechHeavyPlow:
    @[(res: ResourceFood, count: HeavyPlowFoodCost),
      (res: ResourceWood, count: HeavyPlowWoodCost)]
  of TechCropRotation:
    @[(res: ResourceFood, count: CropRotationFoodCost),
      (res: ResourceWood, count: CropRotationWoodCost)]

proc economyTechPrerequisite*(tech: EconomyTechType): EconomyTechType =
  ## Returns the prerequisite tech that must be researched first.
  ## Returns itself if no prerequisite.
  case tech
  of TechWheelbarrow: TechWheelbarrow  # no prereq
  of TechHandCart: TechWheelbarrow
  of TechDoubleBitAxe: TechDoubleBitAxe  # no prereq
  of TechBowSaw: TechDoubleBitAxe
  of TechTwoManSaw: TechBowSaw
  of TechGoldMining: TechGoldMining  # no prereq
  of TechGoldShaftMining: TechGoldMining
  of TechStoneMining: TechStoneMining  # no prereq
  of TechStoneShaftMining: TechStoneMining
  of TechHorseCollar: TechHorseCollar  # no prereq
  of TechHeavyPlow: TechHorseCollar
  of TechCropRotation: TechHeavyPlow

proc getNextEconomyTech*(env: Environment, teamId: int, buildingKind: ThingKind): EconomyTechType =
  ## Find the next available economy tech for the given building type.
  ## Returns the first unresearched tech whose prerequisites are met.
  for tech in EconomyTechType:
    if economyTechBuilding(tech) != buildingKind:
      continue
    if env.teamEconomyTechs[teamId].researched[tech]:
      continue
    # Check prerequisite
    let prereq = economyTechPrerequisite(tech)
    if prereq != tech and not env.teamEconomyTechs[teamId].researched[prereq]:
      continue
    return tech
  # No techs available; return first of this building type (caller checks researched)
  for tech in EconomyTechType:
    if economyTechBuilding(tech) == buildingKind:
      return tech
  TechWheelbarrow  # fallback

proc tryResearchEconomyTech*(env: Environment, agent: Thing, building: Thing): bool =
  ## Attempt to research the next economy tech at a building.
  ## Only villagers can research. Returns true if research was successful.
  if agent.unitClass != UnitVillager:
    return false
  let teamId = getTeamId(agent)
  if building.teamId != teamId:
    return false
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false

  let tech = env.getNextEconomyTech(teamId, building.kind)

  # Verify this tech belongs to this building type (handles fallback case)
  if economyTechBuilding(tech) != building.kind:
    return false

  # Check if already researched
  if env.teamEconomyTechs[teamId].researched[tech]:
    return false

  # Check prerequisite
  let prereq = economyTechPrerequisite(tech)
  if prereq != tech and not env.teamEconomyTechs[teamId].researched[prereq]:
    return false

  # Check and spend resources
  let costs = economyTechCost(tech)
  if not env.spendStockpile(teamId, costs):
    return false
  when defined(econAudit):
    recordResearchCost(teamId, costs, env.currentStep)

  # Apply the tech
  env.teamEconomyTechs[teamId].researched[tech] = true
  building.cooldown = 6
  when defined(eventLog):
    logTechResearched(teamId, "Economy " & $tech, env.currentStep)
  true

proc addFarmToMillQueue*(env: Environment, mill: Thing, farmPos: IVec2) =
  ## Add a farm position to a mill's auto-reseed queue.
  ## Only adds if the farm is within the mill's fertile radius.
  if mill.kind != Mill:
    return
  let dist = max(abs(farmPos.x - mill.pos.x), abs(farmPos.y - mill.pos.y))
  if dist > buildingFertileRadius(Mill):
    return
  # Avoid duplicates
  for pos in mill.farmQueue:
    if pos == farmPos:
      return
  mill.farmQueue.add(farmPos)

proc findNearestMill*(env: Environment, pos: IVec2, teamId: int): Thing =
  ## Find the nearest mill belonging to the given team within range.
  ## Returns nil if no mill found.
  var bestMill: Thing = nil
  var bestDist = high(int32)
  for mill in env.thingsByKind[Mill]:
    if mill.teamId != teamId:
      continue
    let dist = max(abs(pos.x - mill.pos.x), abs(pos.y - mill.pos.y))
    if dist <= buildingFertileRadius(Mill) and dist < bestDist:
      bestDist = dist
      bestMill = mill
  bestMill

proc tryAutoReseedFarm*(env: Environment, mill: Thing): bool
  ## Forward declaration - implemented after placement include

proc queueFarmReseed*(env: Environment, mill: Thing, teamId: int): bool =
  ## Queue a farm reseed at the Mill (pre-pay wood cost).
  ## Returns true if successful.
  if mill.kind != Mill:
    return false
  if mill.teamId != teamId:
    return false
  # Check and spend cost
  let costs = @[(res: ResourceWood, count: FarmReseedWoodCost)]
  if FarmReseedWoodCost > 0 and not env.spendStockpile(teamId, costs):
    return false
  when defined(econAudit):
    if FarmReseedWoodCost > 0:
      recordFarmReseed(teamId, FarmReseedWoodCost, env.currentStep)
  mill.queuedFarmReseeds += 1
  true

proc tryCraftAtStation(env: Environment, agent: Thing, station: CraftStation, stationThing: Thing): bool =
  for recipe in CraftRecipes:
    if recipe.station != station:
      continue
    var hasThingOutput = false
    for output in recipe.outputs:
      if isThingKey(output.key):
        hasThingOutput = true
        break
    if hasThingOutput:
      continue
    let useStockpile = block:
      if recipe.station == StationSiegeWorkshop:
        false
      else:
        var uses = false
        for output in recipe.outputs:
          if isThingKey(output.key):
            uses = true
            break
        uses
    let teamId = getTeamId(agent)
    var canApply = true
    for input in recipe.inputs:
      if useStockpile and isStockpileResourceKey(input.key):
        let stockpileRes = stockpileResourceForItem(input.key)
        if env.stockpileCount(teamId, stockpileRes) < input.count:
          canApply = false
          break
      elif getInv(agent, input.key) < input.count:
        canApply = false
        break
    if canApply:
      for output in recipe.outputs:
        if getInv(agent, output.key) + output.count > MapObjectAgentMaxInventory:
          canApply = false
          break
    if not canApply:
      continue
    if useStockpile:
      var costs: seq[tuple[res: StockpileResource, count: int]] = @[]
      for input in recipe.inputs:
        if isStockpileResourceKey(input.key):
          costs.add((res: stockpileResourceForItem(input.key), count: input.count))
      discard env.spendStockpile(teamId, costs)
    for input in recipe.inputs:
      if useStockpile and isStockpileResourceKey(input.key):
        continue
      setInv(agent, input.key, getInv(agent, input.key) - input.count)
      env.updateAgentInventoryObs(agent, input.key)
    for output in recipe.outputs:
      setInv(agent, output.key, getInv(agent, output.key) + output.count)
      env.updateAgentInventoryObs(agent, output.key)
    if not isNil(stationThing):
      stationThing.cooldown = 0
    return true
  false

include "placement"

proc tryAutoReseedFarm*(env: Environment, mill: Thing): bool =
  ## Try to auto-reseed a farm from the mill's queue.
  ## Returns true if a farm was reseeded.
  if mill.kind != Mill:
    return false
  if mill.farmQueue.len == 0:
    return false
  let teamId = mill.teamId
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return false
  if not env.canAutoReseed(teamId):
    return false

  let farmPos = mill.farmQueue[0]

  # Validate position and terrain BEFORE spending resources or dequeuing
  if not isValidPos(farmPos):
    mill.farmQueue.delete(0)  # Invalid pos can be discarded
    return false
  if env.grid[farmPos.x][farmPos.y] != nil:
    return false  # Something blocking; keep in queue, may clear later
  let terrain = env.terrain[farmPos.x][farmPos.y]
  if terrain != Fertile:
    return false  # Terrain not yet fertile; keep in queue, Mill will refresh

  # Check cost (only after validation passes)
  let costs = @[(res: ResourceWood, count: FarmReseedWoodCost)]
  if FarmReseedWoodCost > 0 and not env.spendStockpile(teamId, costs):
    return false
  when defined(econAudit):
    if FarmReseedWoodCost > 0:
      recordFarmReseed(teamId, FarmReseedWoodCost, env.currentStep)

  # Remove from queue only on successful reseed
  mill.farmQueue.delete(0)

  # Create the farm (wheat crop)
  let crop = Thing(kind: Wheat, pos: farmPos)
  crop.inventory = emptyInventory()
  let farmFood = ResourceNodeInitial + env.getFarmFoodBonus(teamId)
  setInv(crop, ItemWheat, farmFood)
  env.add(crop)
  true

proc grantItem(env: Environment, agent: Thing, key: ItemKey, amount: int = 1): bool =
  if amount <= 0:
    return true
  for _ in 0 ..< amount:
    if not env.giveItem(agent, key):
      return false
  true

# Forward declaration for sparkle effect (defined later in file)
proc spawnGatherSparkle*(env: Environment, pos: IVec2)

proc harvestTree(env: Environment, agent: Thing, tree: Thing): bool =
  if not env.grantItem(agent, ItemWood):
    return false
  env.rewards[agent.agentId] += env.config.woodReward
  # Apply biome gathering bonus
  let bonus = env.getBiomeGatherBonus(tree.pos, ItemWood)
  if bonus > 0:
    discard env.grantItem(agent, ItemWood, bonus)
  # Apply lumber camp tech gathering bonus (AoE2-style)
  let teamId = getTeamId(agent)
  let techBonusPct = env.getWoodGatherBonus(teamId)
  if techBonusPct > 0 and (env.currentStep mod (100 div max(1, techBonusPct))) == 0:
    discard env.grantItem(agent, ItemWood)
  when defined(eventLog):
    logResourceGathered(teamId, "Wood", 1 + bonus, env.currentStep)
  let stumpPos = tree.pos  # Capture before pool release
  removeThing(env, tree)
  let stump = acquireThing(env, Stump)
  stump.pos = stumpPos
  stump.inventory = emptyInventory()
  let remaining = ResourceNodeInitial - 1
  if remaining > 0:
    setInv(stump, ItemWood, remaining)
  env.add(stump)
  # Spawn sparkle effect at harvest location
  env.spawnGatherSparkle(stumpPos)
  true

proc spawnDamageNumber*(env: Environment, pos: IVec2, amount: int,
                        kind: DamageNumberKind = DmgNumDamage) =
  ## Spawn a floating damage number at the given position.
  ## Numbers float upward and fade out over DamageNumberLifetime frames.
  if amount <= 0 or not isValidPos(pos):
    return
  env.damageNumbers.add(DamageNumber(
    pos: pos, amount: amount, kind: kind,
    countdown: DamageNumberLifetime, lifetime: DamageNumberLifetime))

proc spawnRagdoll*(env: Environment, pos: IVec2, direction: Vec2,
                   unitClass: AgentUnitClass, teamId: int) =
  ## Spawn a ragdoll body at the death position.
  ## The body tumbles away from the damage source direction.
  if not isValidPos(pos):
    return
  # Normalize direction and apply initial speed
  let dirLen = sqrt(direction.x * direction.x + direction.y * direction.y)
  let normalizedDir = if dirLen > 0.001:
    vec2(direction.x / dirLen, direction.y / dirLen)
  else:
    vec2(1.0, 0.0)  # Default direction if no attacker
  # Add randomness to angular velocity (clockwise or counter-clockwise)
  let angularDir = if (pos.x + pos.y) mod 2 == 0: 1.0'f32 else: -1.0'f32
  env.ragdolls.add(RagdollBody(
    pos: vec2(pos.x.float32, pos.y.float32),
    velocity: vec2(normalizedDir.x * RagdollInitialSpeed, normalizedDir.y * RagdollInitialSpeed),
    angle: 0.0'f32,
    angularVel: RagdollAngularSpeed * angularDir,
    unitClass: unitClass,
    teamId: teamId,
    countdown: RagdollLifetime,
    lifetime: RagdollLifetime))

proc spawnDebris*(env: Environment, pos: IVec2, buildingKind: ThingKind) =
  ## Spawn debris particles at the given position when a building is destroyed.
  ## Particles spread outward and fade over DebrisLifetime frames.
  if not isValidPos(pos):
    return
  # Determine debris kind based on building type
  let debrisKind = case buildingKind
    of Wall, Outpost, GuardTower, Castle, Monastery, University: DebrisStone
    of TownCenter, House, Barracks, ArcheryRange, Stable, Market: DebrisBrick
    else: DebrisWood  # Default for wooden structures
  # Spawn multiple debris particles with random-ish directions
  for i in 0 ..< DebrisParticlesPerBuilding:
    # Create outward velocity with some variation using simple deterministic spread
    let angle = (i.float32 / DebrisParticlesPerBuilding.float32) * 6.28318  # 2*PI
    let speed = 0.08 + (i mod 3).float32 * 0.03  # Vary speed slightly
    let velocity = vec2(cos(angle) * speed, sin(angle) * speed - 0.02)  # Slight downward drift
    env.debris.add(Debris(
      pos: vec2(pos.x.float32, pos.y.float32),
      velocity: velocity,
      kind: debrisKind,
      countdown: DebrisLifetime,
      lifetime: DebrisLifetime))

proc spawnSpawnEffect*(env: Environment, pos: IVec2) =
  ## Spawn a visual effect at the given position when a unit is created.
  ## Shows a pulsing/expanding effect that fades out.
  if not isValidPos(pos):
    return
  env.spawnEffects.add(SpawnEffect(
    pos: pos,
    countdown: SpawnEffectLifetime, lifetime: SpawnEffectLifetime))

proc spawnGatherSparkle*(env: Environment, pos: IVec2) =
  ## Spawn sparkle particles at the given position when a worker collects resources.
  ## Particles burst outward and fade over GatherSparkleLifetime frames.
  if not isValidPos(pos):
    return
  # Spawn multiple particles in a burst pattern
  for i in 0 ..< GatherSparkleParticleCount:
    let angle = (i.float32 / GatherSparkleParticleCount.float32) * 6.28318  # 2*PI
    let speed = 0.06 + (i mod 3).float32 * 0.02  # Vary speed slightly
    let velocity = vec2(cos(angle) * speed, sin(angle) * speed)
    env.gatherSparkles.add(GatherSparkle(
      pos: vec2(pos.x.float32, pos.y.float32),
      velocity: velocity,
      countdown: GatherSparkleLifetime,
      lifetime: GatherSparkleLifetime))

proc spawnConstructionDust*(env: Environment, pos: IVec2) =
  ## Spawn dust particles at the given position during building construction.
  ## Particles rise upward and fade over ConstructionDustLifetime frames.
  if not isValidPos(pos):
    return
  # Spawn multiple dust particles rising from the construction site
  for i in 0 ..< ConstructionDustParticleCount:
    # Horizontal spread: particles start at random-ish positions around the building
    let xOffset = (i.float32 - 1.0) * 0.25  # Spread horizontally (-0.25, 0, 0.25)
    # Upward velocity with slight variation
    let ySpeed = -0.04 - (i mod 2).float32 * 0.02  # Negative Y = upward
    let xDrift = (i mod 3).float32 * 0.01 - 0.01  # Slight horizontal drift
    env.constructionDust.add(ConstructionDust(
      pos: vec2(pos.x.float32 + xOffset, pos.y.float32 + 0.3),  # Start at base of building
      velocity: vec2(xDrift, ySpeed),
      countdown: ConstructionDustLifetime,
      lifetime: ConstructionDustLifetime))

proc spawnUnitTrail*(env: Environment, pos: IVec2, teamId: int) =
  ## Spawn a dust/footprint trail particle at the given position when a unit moves.
  ## Creates a fading trail effect behind moving units.
  if not isValidPos(pos):
    return
  # Add slight random drift for more natural dust dispersal
  let xDrift = ((env.currentStep mod 3).float32 - 1.0) * 0.005
  let yDrift = ((env.currentStep mod 5).float32 - 2.0) * 0.003
  env.unitTrails.add(UnitTrail(
    pos: vec2(pos.x.float32, pos.y.float32),
    velocity: vec2(xDrift, yDrift),
    countdown: UnitTrailLifetime,
    lifetime: UnitTrailLifetime,
    teamId: teamId.int8))

proc spawnDustParticles*(env: Environment, pos: IVec2, terrain: TerrainType) =
  ## Spawn dust particles when a unit walks on dusty terrain.
  ## Particle color varies based on terrain type.
  if not isValidPos(pos):
    return
  # Map terrain to color index: 0=sand/dune (tan), 1=snow (white), 2=mud (brown), 3=grass/fertile (green-brown), 4=road (gray)
  let colorIdx = case terrain
    of Sand, Dune: 0'u8
    of Snow: 1'u8
    of Mud: 2'u8
    of Grass, Fertile: 3'u8
    of Road: 4'u8
    else: 0'u8
  # Spawn multiple small dust particles
  for i in 0 ..< DustParticleCount:
    # Horizontal spread around footstep
    let xOffset = (i.float32 - 1.0) * 0.15
    # Upward drift with slight variation
    let ySpeed = -0.03 - (i mod 2).float32 * 0.01  # Negative Y = upward
    let xDrift = ((env.currentStep + i) mod 3).float32 * 0.008 - 0.008
    env.dustParticles.add(DustParticle(
      pos: vec2(pos.x.float32 + xOffset, pos.y.float32 + 0.2),
      velocity: vec2(xDrift, ySpeed),
      countdown: DustParticleLifetime,
      lifetime: DustParticleLifetime,
      terrainColor: colorIdx))

proc spawnWaterRipple*(env: Environment, pos: IVec2) =
  ## Spawn a ripple effect when a unit walks through water.
  ## Creates an expanding ring that fades out.
  if not isValidPos(pos):
    return
  env.waterRipples.add(WaterRipple(
    pos: vec2(pos.x.float32, pos.y.float32),
    countdown: WaterRippleLifetime,
    lifetime: WaterRippleLifetime))

proc spawnAttackImpact*(env: Environment, pos: IVec2) =
  ## Spawn burst particles at the given position when an attack hits a target.
  ## Particles radiate outward and fade quickly for a sharp impact effect.
  if not isValidPos(pos):
    return
  # Spawn multiple particles in a radial burst pattern
  for i in 0 ..< AttackImpactParticleCount:
    let angle = (i.float32 / AttackImpactParticleCount.float32) * 6.28318  # 2*PI
    let speed = 0.12 + (i mod 3).float32 * 0.04  # Vary speed for organic burst
    let velocity = vec2(cos(angle) * speed, sin(angle) * speed)
    env.attackImpacts.add(AttackImpact(
      pos: vec2(pos.x.float32, pos.y.float32),
      velocity: velocity,
      countdown: AttackImpactLifetime,
      lifetime: AttackImpactLifetime))

proc spawnConversionEffect*(env: Environment, pos: IVec2, teamColor: Color) =
  ## Spawn a pulsing glow effect at the given position when a monk converts a unit.
  ## The effect uses the new team's color for visual feedback.
  if not isValidPos(pos):
    return
  env.conversionEffects.add(ConversionEffect(
    pos: vec2(pos.x.float32, pos.y.float32),
    countdown: ConversionEffectLifetime,
    lifetime: ConversionEffectLifetime,
    teamColor: teamColor))

include "combat_audit"
include "tumor_audit"
include "action_audit"
include "action_freq_counter"
include "combat"

# ============== CLIPPY AI ==============




{.push inline.}
proc isValidEmptyPosition(env: Environment, pos: IVec2): bool =
  ## Check if a position is within map bounds, empty, and not blocked terrain
  pos.x >= MapBorder and pos.x < MapWidth - MapBorder and
    pos.y >= MapBorder and pos.y < MapHeight - MapBorder and
    env.isEmpty(pos) and isNil(env.getBackgroundThing(pos)) and
    not isBlockedTerrain(env.terrain[pos.x][pos.y])

proc generateRandomMapPosition(r: var Rand): IVec2 =
  ## Generate a random position within map boundaries
  ivec2(
    int32(randIntExclusive(r, MapBorder, MapWidth - MapBorder)),
    int32(randIntExclusive(r, MapBorder, MapHeight - MapBorder))
  )
{.pop.}

proc findEmptyPositionsAround*(env: Environment, center: IVec2, radius: int): seq[IVec2] =
  ## Find empty positions around a center point within a given radius
  result = @[]
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      if dx == 0 and dy == 0:
        continue
      let pos = ivec2(center.x + dx, center.y + dy)
      if env.isValidEmptyPosition(pos):
        result.add(pos)

proc findFirstEmptyPositionAround*(env: Environment, center: IVec2, radius: int): IVec2 =
  ## Find first empty position around center (no allocation)
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      if dx == 0 and dy == 0:
        continue
      let pos = ivec2(center.x + dx, center.y + dy)
      if env.isValidEmptyPosition(pos):
        return pos
  ivec2(-1, -1)

# Tumor constants from shared tuning defaults.
const
  TumorBranchRange = DefaultTumorBranchRange
  TumorBranchMinAge = DefaultTumorBranchMinAge
  TumorBranchChance = DefaultTumorBranchChance
  TumorAdjacencyDeathChance = DefaultTumorAdjacencyDeathChance
  TumorProcessStagger* = 4  ## Process 1/N tumors per step for branching (perf optimization)

let TumorBranchOffsets = block:
  var offsets: seq[IVec2] = @[]
  for dx in -TumorBranchRange .. TumorBranchRange:
    for dy in -TumorBranchRange .. TumorBranchRange:
      if dx == 0 and dy == 0:
        continue
      if max(abs(dx), abs(dy)) > TumorBranchRange:
        continue
      offsets.add(ivec2(dx, dy))
  offsets

proc randomEmptyPos(r: var Rand, env: Environment): IVec2 =
  # Try with moderate attempts first
  for i in 0 ..< 100:
    let pos = r.generateRandomMapPosition()
    if env.isValidEmptyPosition(pos):
      return pos
  # Try harder with more attempts
  for i in 0 ..< 1000:
    let pos = r.generateRandomMapPosition()
    if env.isValidEmptyPosition(pos):
      return pos
  raiseMapFullError()

include "tint"

proc buildCostsForKey*(key: ItemKey): seq[tuple[key: ItemKey, count: int]] =
  var kind: ThingKind
  if parseThingKey(key, kind) and isBuildingKind(kind):
    var costs: seq[tuple[key: ItemKey, count: int]] = @[]
    for input in BuildingRegistry[kind].buildCost:
      costs.add((key: input.key, count: input.count))
    return costs
  for recipe in CraftRecipes:
    for output in recipe.outputs:
      if output.key != key:
        continue
      var costs: seq[tuple[key: ItemKey, count: int]] = @[]
      for input in recipe.inputs:
        costs.add((key: input.key, count: input.count))
      return costs
  @[]

let BuildChoices*: array[ActionArgumentCount, ItemKey] = block:
  var choices: array[ActionArgumentCount, ItemKey]
  for i in 0 ..< choices.len:
    choices[i] = ItemNone
  for kind in ThingKind:
    if not isBuildingKind(kind):
      continue
    if not buildingBuildable(kind):
      continue
    let buildIndex = BuildingRegistry[kind].buildIndex
    if buildIndex >= 0 and buildIndex < choices.len:
      choices[buildIndex] = thingItem($kind)
  choices[BuildIndexWall] = thingItem("Wall")
  choices[BuildIndexRoad] = thingItem("Road")
  choices[BuildIndexDoor] = thingItem("Door")
  choices

proc render*(env: Environment): string =
  for y in 0 ..< MapHeight:
    for x in 0 ..< MapWidth:
      # First check terrain
      var cell = $TerrainCatalog[env.terrain[x][y]].ascii
      # Then override with objects if present (blocking first, background second)
      let blockingThing = env.grid[x][y]
      let thing = if not isNil(blockingThing): blockingThing else: env.backgroundGrid[x][y]
      if not isNil(thing):
        let kind = thing.kind
        let ascii = if isBuildingKind(kind): BuildingRegistry[kind].ascii else: ThingCatalog[kind].ascii
        cell = $ascii
      result.add(cell)
    result.add("\n")

include "connectivity"
include "spawn"
include "console_viz"
include "gather_heatmap"
include "step"
