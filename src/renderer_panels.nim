## renderer_panels.nim - Resource bar, unit info panel, and minimap
##
## Contains: resource bar display, unit info panel, minimap rendering.

import
  boxy, bumpy, pixie, vmath, std/math,
  common, environment

import renderer_core, label_cache

# ─── Shared Constants ─────────────────────────────────────────────────────────

const
  MinimapSizeConst = MinimapSize  # alias to common constant
  MinimapPadding = MinimapPanelPadding
  MinimapUpdateInterval = MinimapUpdateFrameInterval
  MinimapBorderWidth = MinimapPanelBorderWidth

# ─── Unit Info Panel ──────────────────────────────────────────────────────────

proc getUnitInfoLabel(text: string, fontSize: float32 = UnitInfoFontSize.float32): (string, IVec2) =
  let style = labelStyle(InfoLabelFontPath, fontSize, UnitInfoLabelPadding, UnitInfoLabelLineSpacing)
  let cached = ensureLabel("unit_info", text, style)
  return (cached.imageKey, cached.size)

proc drawUnitInfoPanel*(panelRect: IRect) =
  ## Draw unit info panel showing details about selected unit/building.
  ## Positioned in bottom-right area of the screen.
  if selection.len == 0:
    return

  let selected = selection[0]
  if selected.isNil:
    return

  let panelW = UnitInfoPanelW
  let panelH = UnitInfoPanelH
  let panelX = panelRect.x.float32 + panelRect.w.float32 - panelW - MinimapPadding
  let panelY = panelRect.y.float32 + panelRect.h.float32 - panelH - MinimapPadding - FooterHeight.float32

  # Draw panel border and background
  bxy.drawRect(rect = Rect(x: panelX - 1.0, y: panelY - 1.0, w: panelW + 2.0, h: panelH + 2.0),
               color = UiBorder)
  bxy.drawRect(rect = Rect(x: panelX, y: panelY, w: panelW, h: panelH),
               color = UiBgPanel)

  var yOffset = UnitInfoPanelPadding
  let xPadding = UnitInfoPanelPadding

  # Draw name/type
  let name = if selected.kind == Agent:
    UnitClassLabels[selected.unitClass]
  elif isBuildingKind(selected.kind):
    BuildingRegistry[selected.kind].displayName
  else:
    $selected.kind
  let (nameKey, nameSize) = getUnitInfoLabel(name, UnitInfoNameFontSize)
  drawUiImageScaled(nameKey, vec2(panelX + xPadding, panelY + yOffset),
                    vec2(nameSize.x.float32, nameSize.y.float32))
  yOffset += nameSize.y.float32 + UnitInfoLineSpacingLarge

  # Draw HP with visual health bar
  if selected.maxHp > 0:
    let hpText = "HP: " & $selected.hp & "/" & $selected.maxHp
    let (hpKey, hpSize) = getUnitInfoLabel(hpText)
    drawUiImageScaled(hpKey, vec2(panelX + xPadding, panelY + yOffset),
                      vec2(hpSize.x.float32, hpSize.y.float32))
    yOffset += hpSize.y.float32 + UnitInfoLineSpacingSmall

    # Visual health bar beneath HP text
    let hpBarW = panelW - xPadding * 2
    let hpBarH = 4.0'f32
    let hpRatio = selected.hp.float32 / selected.maxHp.float32
    # Background
    bxy.drawRect(rect = Rect(x: panelX + xPadding, y: panelY + yOffset, w: hpBarW, h: hpBarH),
                 color = UiHealthBg)
    # Filled portion with color gradient
    let hpBarColor = getHealthBarColor(hpRatio)
    bxy.drawRect(rect = Rect(x: panelX + xPadding, y: panelY + yOffset,
                             w: hpBarW * hpRatio, h: hpBarH),
                 color = hpBarColor)
    yOffset += hpBarH + UnitInfoLineSpacingLarge

  # Draw attack and range for units
  if selected.kind == Agent:
    let atkText = "Atk: " & $selected.attackDamage & "  Rng: " & $getUnitAttackRange(selected)
    let (atkKey, atkSize) = getUnitInfoLabel(atkText)
    drawUiImageScaled(atkKey, vec2(panelX + xPadding, panelY + yOffset),
                      vec2(atkSize.x.float32, atkSize.y.float32))
    yOffset += atkSize.y.float32 + UnitInfoLineSpacingSmall

  # Draw team
  let teamText = "Team: " & $selected.teamId
  let (teamKey, teamSize) = getUnitInfoLabel(teamText)
  drawUiImageScaled(teamKey, vec2(panelX + xPadding, panelY + yOffset),
                    vec2(teamSize.x.float32, teamSize.y.float32))
  yOffset += teamSize.y.float32 + UnitInfoLineSpacingSmall

  # Draw position
  let posText = "Pos: " & $selected.pos.x & ", " & $selected.pos.y
  let (posKey, posSize) = getUnitInfoLabel(posText)
  drawUiImageScaled(posKey, vec2(panelX + xPadding, panelY + yOffset),
                    vec2(posSize.x.float32, posSize.y.float32))

# ─── Resource Bar ────────────────────────────────────────────────────────────

  # Resource bar constants are defined in renderer_core.nim

proc ensureResourceBarLabel(text: string): (string, IVec2) =
  let cached = ensureLabel("res_bar", text, resourceBarLabelStyle)
  return (cached.imageKey, cached.size)

proc drawResourceBar*(panelRect: IRect, teamId: int) =
  ## Draw resource bar at top of viewport showing team resources.
  if teamId < 0 or teamId >= MapRoomObjectsTeams:
    return

  let barY = panelRect.y.float32
  let barH = ResourceBarHeight.float32
  let barX = panelRect.x.float32

  # Draw background
  bxy.drawRect(rect = Rect(x: barX, y: barY, w: panelRect.w.float32, h: barH),
               color = UiBgBar)

  let stockpile = env.teamStockpiles[teamId]
  var xOffset = ResourceBarXStart

  # Draw each resource type (all 5 resources)
  for res in [ResourceFood, ResourceWood, ResourceStone, ResourceGold, ResourceWater]:
    let icon = case res
      of ResourceFood: itemSpriteKey(ItemWheat)
      of ResourceWood: itemSpriteKey(ItemWood)
      of ResourceStone: itemSpriteKey(ItemStone)
      of ResourceGold: itemSpriteKey(ItemGold)
      of ResourceWater: itemSpriteKey(ItemWater)
      of ResourceNone: ""

    if icon.len > 0 and icon in bxy:
      # Draw icon
      let iconBaseSize = bxy.getImageSize(icon)
      let fitScale = min(ResourceBarIconMaxSize / iconBaseSize.x.float32,
                         ResourceBarIconMaxSize / iconBaseSize.y.float32) * resourceUiIconScale(res)
      let iconSize = vec2(iconBaseSize.x.float32, iconBaseSize.y.float32) * fitScale
      let iconX = barX + xOffset + (ResourceBarIconSlotW - iconSize.x) * 0.5
      let iconY = barY + (barH - iconSize.y) * 0.5
      drawUiImageScaled(icon, vec2(iconX, iconY), iconSize)
      xOffset += ResourceBarIconSlotW + ResourceBarIconGap

      # Draw count
      let count = stockpile.counts[res]
      let (labelKey, labelSize) = ensureResourceBarLabel($count)
      let labelY = barY + (barH - labelSize.y.float32) / 2.0
      drawUiImageScaled(labelKey, vec2(barX + xOffset, labelY),
                        vec2(labelSize.x.float32, labelSize.y.float32))
      xOffset += labelSize.x.float32 + ResourceBarItemSpacing

  # Draw separator before population counter
  xOffset += ResourceBarItemSpacing
  bxy.drawRect(rect = Rect(x: barX + xOffset, y: barY + 4.0, w: 1.0, h: barH - 8.0),
               color = UiBorder)
  xOffset += ResourceBarItemSpacing

  # Draw population counter (current/max)
  let popCount = env.stepTeamPopCounts[teamId]
  let popCap = env.stepTeamPopCaps[teamId]
  let popText = $popCount & "/" & $popCap
  let popTextColor = if popCap > 0 and popCount >= popCap: UiDanger else: UiFgText
  let popStyle = labelStyleColored(FooterFontPath, FooterFontSize, ResourceBarLabelPadding, popTextColor)
  let popCached = ensureLabel("res_bar_pop", popText, popStyle)
  let popLabelY = barY + (barH - popCached.size.y.float32) / 2.0
  drawUiImageScaled(popCached.imageKey, vec2(barX + xOffset, popLabelY),
                    vec2(popCached.size.x.float32, popCached.size.y.float32))
  xOffset += popCached.size.x.float32 + ResourceBarItemSpacing

# ─── Minimap ─────────────────────────────────────────────────────────────────

var
  minimapTerrainImage: Image     # cached base terrain (invalidated on mapGeneration change)
  minimapTerrainGeneration = -1
  minimapCompositeImage: Image   # terrain + units + fog composite
  minimapLastUnitFrame = -1
  minimapImageKey = "minimap_composite"
  # Pre-computed minimap scale factors
  minimapScaleX: float32 = MinimapSizeConst.float32 / MapWidth.float32
  minimapScaleY: float32 = MinimapSizeConst.float32 / MapHeight.float32
  minimapInvScaleX: float32 = MapWidth.float32 / MinimapSizeConst.float32
  minimapInvScaleY: float32 = MapHeight.float32 / MinimapSizeConst.float32
  # Cached team colors for minimap (avoid Color -> ColorRGBX conversion each frame)
  minimapTeamColors: array[MapRoomObjectsTeams, ColorRGBX]
  minimapTeamBrightColors: array[MapRoomObjectsTeams, ColorRGBX]  # For buildings
  minimapTeamColorsInitialized = false

proc toMinimapColor(terrain: TerrainType, biome: BiomeType): ColorRGBX =
  ## Map a terrain+biome to a minimap pixel color.
  case terrain
  of Water:
    MinimapPanelWater
  of ShallowWater:
    MinimapPanelShallowWater
  of Bridge:
    MinimapPanelBridge
  of Road:
    MinimapPanelRoad
  of Snow:
    MinimapPanelSnow
  of Dune, Sand:
    MinimapPanelSandy
  of Mud:
    MinimapPanelMud
  of Mountain:
    MinimapPanelMountain
  else:
    # Use biome tint for base terrain (Empty, Grass, Fertile, ramps, etc.)
    let tc = case biome
      of BiomeForestType: BiomeColorForest
      of BiomeDesertType: BiomeColorDesert
      of BiomeCavesType: BiomeColorCaves
      of BiomeCityType: BiomeColorCity
      of BiomePlainsType: BiomeColorPlains
      of BiomeSwampType: BiomeColorSwamp
      of BiomeDungeonType: BiomeColorDungeon
      of BiomeSnowType: BiomeColorSnow
      else: BaseTileColorDefault
    let i = min(tc.intensity, MinimapIntensityCap)
    rgbx(
      uint8(clamp(tc.r * i * 255, 0, 255)),
      uint8(clamp(tc.g * i * 255, 0, 255)),
      uint8(clamp(tc.b * i * 255, 0, 255)),
      255
    )

proc rebuildMinimapTerrain() =
  ## Rebuild the cached terrain layer. Called when mapGeneration changes.
  if minimapTerrainImage.isNil or
     minimapTerrainImage.width != MinimapSizeConst or
     minimapTerrainImage.height != MinimapSizeConst:
    minimapTerrainImage = newImage(MinimapSizeConst, MinimapSizeConst)

  # Scale factors: map coords -> minimap pixel
  let scaleX = MinimapSizeConst.float32 / MapWidth.float32
  let scaleY = MinimapSizeConst.float32 / MapHeight.float32

  for py in 0 ..< MinimapSizeConst:
    for px in 0 ..< MinimapSizeConst:
      let mx = clamp(int(px.float32 / scaleX), 0, MapWidth - 1)
      let my = clamp(int(py.float32 / scaleY), 0, MapHeight - 1)
      let terrain = env.terrain[mx][my]
      let biome = env.biomes[mx][my]
      # Check for trees at this tile
      let bg = env.backgroundGrid[mx][my]
      let c = if bg.isKind(Tree):
        MinimapPanelTree
      else:
        toMinimapColor(terrain, biome)
      minimapTerrainImage.unsafe[px, py] = c

  minimapTerrainGeneration = env.mapGeneration

proc initMinimapTeamColors() =
  ## Pre-compute team colors for minimap to avoid per-frame conversions.
  for i in 0 ..< MapRoomObjectsTeams:
    let tc = if i < env.teamColors.len: env.teamColors[i] else: NeutralGray
    minimapTeamColors[i] = colorToRgbx(tc)
    minimapTeamBrightColors[i] = colorToRgbx(color(
      min(tc.r * MinimapBrightMul + MinimapBrightAdd, 1.0),
      min(tc.g * MinimapBrightMul + MinimapBrightAdd, 1.0),
      min(tc.b * MinimapBrightMul + MinimapBrightAdd, 1.0),
      1.0
    ))
  minimapTeamColorsInitialized = true

# Building kinds that commonly have instances (skip iteration for unlikely kinds)
const MinimapBuildingKinds = [
  TownCenter, House, Mill, LumberCamp, MiningCamp, Market, Blacksmith,
  Barracks, ArcheryRange, Stable, SiegeWorkshop, Castle, Monastery,
  GuardTower, Door
]

proc rebuildMinimapComposite(fogTeamId: int) =
  ## Composite terrain + units + buildings + fog into final minimap image.
  if minimapTerrainGeneration != env.mapGeneration:
    rebuildMinimapTerrain()

  # Ensure team colors are initialized
  if not minimapTeamColorsInitialized:
    initMinimapTeamColors()

  if minimapCompositeImage.isNil or
     minimapCompositeImage.width != MinimapSizeConst or
     minimapCompositeImage.height != MinimapSizeConst:
    minimapCompositeImage = newImage(MinimapSizeConst, MinimapSizeConst)

  # Start from cached terrain
  copyMem(addr minimapCompositeImage.data[0],
          addr minimapTerrainImage.data[0],
          MinimapSizeConst * MinimapSizeConst * MinimapBytesPerPixel)

  # Use pre-computed scale factors
  let scaleX = minimapScaleX
  let scaleY = minimapScaleY

  # Draw buildings (team-colored, 2x2 pixel blocks)
  # Only iterate over building kinds that are likely to have instances
  for kind in MinimapBuildingKinds:
    for thing in env.thingsByKind[kind]:
      if not isValidPos(thing.pos):
        continue
      let teamId = thing.teamId
      # Use pre-computed team colors
      let bright = if teamId >= 0 and teamId < MapRoomObjectsTeams:
        minimapTeamBrightColors[teamId]
      else:
        rgbx(MinimapNeutralGrayBright.uint8, MinimapNeutralGrayBright.uint8, MinimapNeutralGrayBright.uint8, 255)
      let px = int(thing.pos.x.float32 * scaleX)
      let py = int(thing.pos.y.float32 * scaleY)
      # Unrolled 2x2 block drawing
      let fx0 = clamp(px, 0, MinimapSizeConst - 1)
      let fx1 = clamp(px + 1, 0, MinimapSizeConst - 1)
      let fy0 = clamp(py, 0, MinimapSizeConst - 1)
      let fy1 = clamp(py + 1, 0, MinimapSizeConst - 1)
      minimapCompositeImage.unsafe[fx0, fy0] = bright
      minimapCompositeImage.unsafe[fx1, fy0] = bright
      minimapCompositeImage.unsafe[fx0, fy1] = bright
      minimapCompositeImage.unsafe[fx1, fy1] = bright

  # Draw units (team-colored dots) - use pre-computed colors
  for agent in env.agents:
    if not isAgentAlive(env, agent):
      continue
    let teamId = getTeamId(agent)
    let dot = if teamId >= 0 and teamId < MapRoomObjectsTeams:
      minimapTeamColors[teamId]
    else:
      MinimapPanelUnknownGray
    let px = clamp(int(agent.pos.x.float32 * scaleX), 0, MinimapSizeConst - 1)
    let py = clamp(int(agent.pos.y.float32 * scaleY), 0, MinimapSizeConst - 1)
    minimapCompositeImage.unsafe[px, py] = dot

  # Apply fog of war with edge smoothing
  if fogTeamId >= 0 and fogTeamId < MapRoomObjectsTeams:
    let invScaleX = minimapInvScaleX
    let invScaleY = minimapInvScaleY
    const
      MinimapFogEdgeSmoothFactor = MinimapFogEdgeFactor  # How much to lighten edge tiles
      Neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for py in 0 ..< MinimapSizeConst:
      let my = clamp(int(py.float32 * invScaleY), 0, MapHeight - 1)
      for px in 0 ..< MinimapSizeConst:
        let mx = clamp(int(px.float32 * invScaleX), 0, MapWidth - 1)
        if not fogVisibility[mx][my]:
          # Check if this is an edge tile (adjacent to visible)
          var isEdge = false
          for (dx, dy) in Neighbors:
            let nx = mx + dx
            let ny = my + dy
            if nx >= 0 and nx < MapWidth and ny >= 0 and ny < MapHeight:
              if fogVisibility[nx][ny]:
                isEdge = true
                break
          # Darken fogged areas
          let c = minimapCompositeImage.unsafe[px, py]
          let factor = if isEdge: MinimapFogEdgeSmoothFactor else: MinimapFogDarkFactor
          minimapCompositeImage.unsafe[px, py] = rgbx(
            uint8(c.r.float32 * factor),
            uint8(c.g.float32 * factor),
            uint8(c.b.float32 * factor),
            c.a
          )

  minimapLastUnitFrame = frame

proc drawMinimap*(panelRect: IRect, panel: Panel) =
  ## Draw the minimap in the bottom-left corner of the panel.
  let minimapX = panelRect.x.float32 + MinimapPadding
  let minimapY = panelRect.y.float32 + panelRect.h.float32 - MinimapSizeConst.float32 - MinimapPadding - FooterHeight.float32

  # Rebuild composite if needed (every MinimapUpdateInterval frames or on mapgen change)
  let fogTeamId = if settings.showFogOfWar: playerTeam else: -1
  if frame - minimapLastUnitFrame >= MinimapUpdateInterval or
     minimapTerrainGeneration != env.mapGeneration:
    rebuildMinimapComposite(fogTeamId)
    bxy.addImage(minimapImageKey, minimapCompositeImage)

  # Draw border
  let borderColor = UiMinimapBorder
  bxy.drawRect(
    rect = Rect(x: minimapX - MinimapBorderWidth, y: minimapY - MinimapBorderWidth,
                w: MinimapSizeConst.float32 + MinimapBorderWidth * 2,
                h: MinimapSizeConst.float32 + MinimapBorderWidth * 2),
    color = borderColor
  )

  # Draw minimap image
  drawUiImageScaled(minimapImageKey, vec2(minimapX, minimapY),
                    vec2(MinimapSizeConst.float32, MinimapSizeConst.float32))

  # Draw viewport rectangle
  if currentViewport.valid:
    let scaleX = minimapScaleX
    let scaleY = minimapScaleY
    let vpX = minimapX + currentViewport.minX.float32 * scaleX
    let vpY = minimapY + currentViewport.minY.float32 * scaleY
    let vpW = (currentViewport.maxX - currentViewport.minX + 1).float32 * scaleX
    let vpH = (currentViewport.maxY - currentViewport.minY + 1).float32 * scaleY
    let vpColor = UiViewportOutline
    # Draw viewport outline as 4 thin rectangles
    let lineW = MinimapViewportLineW
    bxy.drawRect(rect = Rect(x: vpX, y: vpY, w: vpW, h: lineW), color = vpColor)  # top
    bxy.drawRect(rect = Rect(x: vpX, y: vpY + vpH - lineW, w: vpW, h: lineW), color = vpColor)  # bottom
    bxy.drawRect(rect = Rect(x: vpX, y: vpY, w: lineW, h: vpH), color = vpColor)  # left
    bxy.drawRect(rect = Rect(x: vpX + vpW - lineW, y: vpY, w: lineW, h: vpH), color = vpColor)  # right
