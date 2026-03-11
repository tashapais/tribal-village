## renderer_building_ui.nim - Building construction, overlays, and placement
##
## Contains: building construction scaffolding, building UI overlays (stockpiles,
## population, garrison, production queue), building ghost preview, placement validation.

import
  boxy, pixie, vmath, std/math,
  common, constants, environment

import renderer_core, label_cache
from renderer_effects import drawBuildingSmoke

# ─── Building Construction Rendering ─────────────────────────────────────────

proc renderBuildingConstruction*(pos: IVec2, constructionRatio: float32) =
  ## Render construction scaffolding for a building under construction.
  ##
  ## Draws scaffolding posts at the four corners of the building and horizontal
  ## bars connecting them, plus a progress bar showing construction completion.
  ##
  ## Parameters:
  ##   pos: World position of the building
  ##   constructionRatio: Progress from 0.0 (just started) to 1.0 (complete)
  let scaffoldTint = ScaffoldTint
  let scaffoldScale = ScaffoldingPostScale
  let offsets = [vec2(-ScaffoldPostOffset, -ScaffoldPostOffset), vec2(ScaffoldPostOffset, -ScaffoldPostOffset),
                 vec2(-ScaffoldPostOffset, ScaffoldPostOffset), vec2(ScaffoldPostOffset, ScaffoldPostOffset)]
  for offset in offsets:
    bxy.drawImage("floor", pos.vec2 + offset, angle = 0,
                  scale = scaffoldScale, tint = scaffoldTint)
  # Draw horizontal scaffold bars connecting posts
  let barTint = ScaffoldBarTint
  for yOff in [-ScaffoldPostOffset, ScaffoldPostOffset]:
    bxy.drawImage("floor", pos.vec2 + vec2(0, yOff), angle = 0,
                  scale = scaffoldScale, tint = barTint)
  # Draw construction progress bar below the building
  drawSegmentBar(pos.vec2, vec2(0, ConstructionBarOffsetY), constructionRatio,
                 ConstructionBarFill, BarBgColor)

proc renderBuildingUI*(thing: Thing, pos: IVec2,
                       teamPopCounts, teamHouseCounts: array[MapRoomObjectsTeams, int]) =
  ## Render UI overlays for a building (stockpiles, population, garrison).
  ##
  ## Handles:
  ## - Production queue progress bars for buildings training units
  ## - Resource stockpile icons showing team resource counts
  ## - Population display on TownCenters (current/max pop)
  ## - Garrison indicators showing garrisoned unit counts
  ##
  ## Parameters:
  ##   thing: The building Thing
  ##   pos: World position of the building
  ##   teamPopCounts: Array of population counts per team
  ##   teamHouseCounts: Array of house counts per team

  # Production queue progress bar (AoE2-style)
  if thing.productionQueue.entries.len > 0:
    let entry = thing.productionQueue.entries[0]
    if entry.totalSteps > 0 and entry.remainingSteps > 0:
      let ratio = clamp(1.0'f32 - entry.remainingSteps.float32 / entry.totalSteps.float32, 0.0, 1.0)
      drawSegmentBar(pos.vec2, vec2(0, ProductionBarOffsetY), ratio,
                     ProductionBarFill, BarBgColor)
      # Draw smoke/chimney effect for active production buildings
      drawBuildingSmoke(pos.vec2, thing.id)
  let res = buildingStockpileRes(thing.kind)
  if res != ResourceNone:
    let teamId = thing.teamId
    if teamId < 0 or teamId >= MapRoomObjectsTeams:
      return
    let icon = case res
      of ResourceFood: itemSpriteKey(ItemWheat)
      of ResourceWood: itemSpriteKey(ItemWood)
      of ResourceStone: itemSpriteKey(ItemStone)
      of ResourceGold: itemSpriteKey(ItemGold)
      of ResourceWater: itemSpriteKey(ItemWater)
      of ResourceNone: ""
    let count = env.teamStockpiles[teamId].counts[res]
    let iconPos = pos.vec2 + vec2(BuildingIconOffsetX, BuildingIconOffsetY)
    if icon.len > 0 and icon in bxy:
      bxy.drawImage(icon, iconPos, angle = 0, scale = OverlayIconScale * resourceUiIconScale(res),
                    tint = withAlpha(TintWhite, (if count > 0: 1.0 else: ResourceIconDimAlpha)))
    if count > 0:
      let labelKey = ensureHeartCountLabel(count)
      if labelKey.len > 0 and labelKey in bxy:
        bxy.drawImage(labelKey, iconPos + vec2(BuildingLabelOffsetX, BuildingLabelOffsetY), angle = 0,
                      scale = OverlayLabelScale, tint = TintWhite)
  if thing.kind == TownCenter:
    let teamId = thing.teamId
    if teamId >= 0 and teamId < MapRoomObjectsTeams:
      let iconPos = pos.vec2 + vec2(BuildingIconOffsetX, BuildingIconOffsetY)
      if "oriented/gatherer.s" in bxy:
        bxy.drawImage("oriented/gatherer.s", iconPos, angle = 0,
                      scale = OverlayIconScale, tint = TintWhite)
      let popText = "x " & $teamPopCounts[teamId] & "/" &
                    $min(MapAgentsPerTeam, teamHouseCounts[teamId] * HousePopCap)
      let popLabel = ensureLabel("overlay", popText, overlayLabelStyle).imageKey
      if popLabel.len > 0 and popLabel in bxy:
        bxy.drawImage(popLabel, iconPos + vec2(BuildingLabelOffsetX, BuildingLabelOffsetY), angle = 0,
                      scale = OverlayLabelScale, tint = TintWhite)
  # Garrison indicator for buildings that can garrison units
  if thing.kind in {TownCenter, Castle, GuardTower, House}:
    let garrisonCount = thing.garrisonedUnits.len
    if garrisonCount > 0:
      # Position on right side of building to avoid overlap with stockpile icons
      let garrisonIconPos = pos.vec2 + vec2(BuildingGarrisonOffsetX, BuildingIconOffsetY)
      if "oriented/fighter.s" in bxy:
        bxy.drawImage("oriented/fighter.s", garrisonIconPos, angle = 0,
                      scale = OverlayIconScale, tint = TintWhite)
      let garrisonText = "x" & $garrisonCount
      let garrisonLabel = ensureLabel("overlay", garrisonText, overlayLabelStyle).imageKey
      if garrisonLabel.len > 0 and garrisonLabel in bxy:
        bxy.drawImage(garrisonLabel, garrisonIconPos + vec2(BuildingGarrisonLabelOffsetX, BuildingLabelOffsetY), angle = 0,
                      scale = OverlayLabelScale, tint = TintWhite)

# ─── Building Ghost Preview ──────────────────────────────────────────────────

proc canPlaceBuildingAt*(pos: IVec2, kind: ThingKind): bool =
  ## Check if a building can be placed at the given position.
  if not isValidPos(pos):
    return false
  # Check terrain
  let terrain = env.terrain[pos.x][pos.y]
  if isWaterTerrain(terrain):
    return false
  # Check for existing objects
  let blocking = env.grid[pos.x][pos.y]
  if not isNil(blocking):
    return false
  let background = env.backgroundGrid[pos.x][pos.y]
  if not isNil(background) and background.kind in CliffKinds:
    return false
  true

proc drawBuildingGhost*(worldPos: Vec2) =
  ## Draw a transparent building preview at the given world position.
  ## Shows green if placement is valid, red if invalid.
  if not buildingPlacementMode:
    return

  let gridPos = (worldPos + vec2(GridSnapOffset, GridSnapOffset)).ivec2
  let spriteKey = buildingSpriteKey(buildingPlacementKind)
  if spriteKey.len == 0 or spriteKey notin bxy:
    return

  let valid = canPlaceBuildingAt(gridPos, buildingPlacementKind)
  buildingPlacementValid = valid

  # Ghost tint: green for valid, red for invalid
  let tint = if valid:
    GhostValidColor
  else:
    GhostInvalidColor

  bxy.drawImage(spriteKey, gridPos.vec2, angle = 0, scale = SpriteScale, tint = tint)
