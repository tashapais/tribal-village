## console_viz.nim - ANSI console map renderer for headless servers
##
## Renders the game state as colored ANSI text to stdout.
## Gated behind env var TV_CONSOLE_VIZ=1, prints every TV_VIZ_INTERVAL steps.
## Pure ANSI escape codes — no X11, no curses.

# Included by environment.nim — types, strutils are in scope.
import std/os

let consoleVizEnabled* = parseEnvBool("TV_CONSOLE_VIZ", false)
let consoleVizInterval* = max(1, parseEnvInt("TV_VIZ_INTERVAL", 10))

# ANSI escape helpers
const
  Esc = "\e["
  Reset = Esc & "0m"
  Bold = Esc & "1m"

proc fg(r, g, b: int): string = Esc & "38;2;" & $r & ";" & $g & ";" & $b & "m"
proc bg(r, g, b: int): string = Esc & "48;2;" & $r & ";" & $g & ";" & $b & "m"

# Team ANSI colors (approximate WarmTeamPalette as 0-255 RGB)
const TeamFg: array[8, array[3, int]] = [
  [232, 107, 107],  # 0: red
  [240, 166, 107],  # 1: orange
  [240, 209, 107],  # 2: yellow
  [153, 214, 128],  # 3: olive-lime
  [199, 97, 224],   # 4: magenta
  [107, 184, 240],  # 5: sky
  [222, 222, 222],  # 6: gray
  [237, 143, 209],  # 7: pink
]

# Terrain background colors (muted)
proc terrainBg(t: TerrainType): string =
  case t
  of Water:                    bg(20, 40, 120)
  of ShallowWater:             bg(40, 80, 150)
  of Bridge:                   bg(120, 90, 50)
  of Fertile:                  bg(50, 100, 30)
  of Road:                     bg(130, 120, 100)
  of Grass:                    bg(40, 80, 30)
  of Dune, Sand:               bg(160, 140, 80)
  of Snow:                     bg(200, 200, 210)
  of Mud:                      bg(80, 60, 40)
  of Mountain:                 bg(60, 55, 50)
  of Empty:                    bg(10, 10, 10)
  of RampUpN, RampUpS, RampUpW, RampUpE,
     RampDownN, RampDownS, RampDownW, RampDownE:
                               bg(90, 85, 75)

proc thingChar(kind: ThingKind): char =
  case kind
  of Agent:             'A'
  of Wall:              '#'
  of Door:              '+'
  of Tree:              'T'
  of Wheat:             'w'
  of Fish:              'f'
  of Stone:             'o'
  of Gold:              '$'
  of Bush, Cactus:      ','
  of Stalagmite:        '^'
  of Relic:             '*'
  of Magma:             '~'
  of Altar:             '&'
  of Spawner:           'S'
  of Tumor:             '%'
  of Cow:               'c'
  of Bear:              'B'
  of Wolf:              'W'
  of Corpse, Skeleton:  'x'
  of TownCenter:        'H'
  of House:             'h'
  of Barracks:          'b'
  of ArcheryRange:      'a'
  of Stable:            's'
  of SiegeWorkshop, MangonelWorkshop, TrebuchetWorkshop: 'E'
  of Castle:            'C'
  of Monastery:         'm'
  of Temple:            'P'
  of University:        'u'
  of Wonder:            '!'
  of ControlPoint:      '@'
  of Market:            'M'
  of Blacksmith:        'K'
  of GuardTower, Outpost: 't'
  of Dock:              'D'
  of Mill:              'i'
  of Granary:           'g'
  of LumberCamp:        'l'
  of Quarry:            'q'
  of MiningCamp:        'n'
  of Lantern:           '.'
  of Barrel:            'O'
  of Stump, Stubble:    '`'
  of ClayOven:          'v'
  of WeavingLoom:       '='
  of GoblinHive:        'Z'
  of GoblinHut:         'z'
  of GoblinTotem:       'j'
  of CliffEdgeN, CliffEdgeE, CliffEdgeS, CliffEdgeW,
     CliffCornerInNE, CliffCornerInSE, CliffCornerInSW, CliffCornerInNW,
     CliffCornerOutNE, CliffCornerOutSE, CliffCornerOutSW, CliffCornerOutNW:
                        '|'
  of WaterfallN, WaterfallE, WaterfallS, WaterfallW:
                        '~'

proc unitClassChar(uc: AgentUnitClass): char =
  case uc
  of UnitVillager:      'v'
  of UnitManAtArms, UnitLongSwordsman, UnitChampion: 'm'
  of UnitArcher, UnitCrossbowman, UnitArbalester:    'a'
  of UnitScout, UnitLightCavalry, UnitHussar:        's'
  of UnitKnight:        'k'
  of UnitMonk:          'p'
  of UnitBatteringRam:  'r'
  of UnitMangonel:      'g'
  of UnitTrebuchet:     't'
  of UnitGoblin:        'G'
  of UnitBoat:          'b'
  of UnitTradeCog:      'c'
  of UnitSamurai:       'S'
  of UnitLongbowman:    'L'
  of UnitCataphract:    'C'
  of UnitWoadRaider:    'W'
  of UnitTeutonicKnight:'T'
  of UnitHuskarl:       'H'
  of UnitMameluke:      'M'
  of UnitJanissary:     'J'
  of UnitKing:          'K'
  of UnitGalley:        'y'  # Galley warship
  of UnitFireShip:      'f'  # Fire Ship
  of UnitFishingShip:   'F'  # Fishing Ship
  of UnitTransportShip: 'P'  # Transport Ship (P for passenger)
  of UnitDemoShip:      'D'  # Demolition Ship
  of UnitCannonGalleon: 'N'  # Cannon Galleon (N for naval artillery)
  of UnitScorpion:      'x'  # Scorpion ballista
  of UnitCavalier:      'k'  # Cavalier (uses knight char)
  of UnitPaladin:       'k'  # Paladin (uses knight char)
  of UnitCamel:         'l'  # Camel Rider (l for camel)
  of UnitHeavyCamel:    'l'  # Heavy Camel (uses camel char)
  of UnitImperialCamel: 'l'  # Imperial Camel (uses camel char)
  of UnitSkirmisher:    'i'  # Skirmisher (anti-archer)
  of UnitEliteSkirmisher: 'i' # Elite Skirmisher
  of UnitCavalryArcher: 'q'  # Cavalry Archer (mounted ranged)
  of UnitHeavyCavalryArcher: 'q' # Heavy Cavalry Archer
  of UnitHandCannoneer: 'h'  # Hand Cannoneer (gunpowder)

proc printGameMap*(env: Environment) =
  ## Render the map as ANSI colored text. One char per tile.
  ## Prioritizes: agent > grid thing > background thing > terrain.
  var buf = newStringOfCap(MapWidth * MapHeight * 20)
  buf.add(Esc & "2J" & Esc & "H")  # clear screen, cursor home

  buf.add(Bold & "=== tribal-village step " & $env.currentStep & " ===" & Reset & "\n")

  for y in 0 ..< MapHeight:
    for x in 0 ..< MapWidth:
      let t = env.terrain[x][y]
      let gridThing = env.grid[x][y]
      let bgThing = env.backgroundGrid[x][y]

      let tbg = terrainBg(t)

      if not gridThing.isNil:
        if gridThing.kind == Agent:
          let teamId = getTeamId(gridThing)
          let tc = TeamFg[teamId mod 8]
          buf.add(tbg & fg(tc[0], tc[1], tc[2]) & Bold)
          buf.add(unitClassChar(gridThing.unitClass))
          buf.add(Reset)
        else:
          # Building or wall — use team color if available
          if gridThing.teamId >= 0 and gridThing.teamId < 8:
            let tc = TeamFg[gridThing.teamId]
            buf.add(tbg & fg(tc[0], tc[1], tc[2]) & Bold)
          else:
            buf.add(tbg & fg(180, 180, 180))
          buf.add(thingChar(gridThing.kind))
          buf.add(Reset)
      elif not bgThing.isNil:
        # Background objects: resources, trees, etc.
        case bgThing.kind
        of Tree, Bush, Cactus, Stalagmite, Stubble:
          buf.add(tbg & fg(30, 140, 30))
        of Gold:
          buf.add(tbg & fg(255, 215, 0))
        of Stone:
          buf.add(tbg & fg(160, 160, 160))
        of Wheat:
          buf.add(tbg & fg(220, 200, 80))
        of Fish:
          buf.add(tbg & fg(100, 180, 220))
        of Relic:
          buf.add(tbg & fg(255, 100, 255))
        of Magma:
          buf.add(tbg & fg(255, 80, 20))
        else:
          buf.add(tbg & fg(140, 140, 140))
        buf.add(thingChar(bgThing.kind))
        buf.add(Reset)
      else:
        # Empty terrain
        case t
        of Water, ShallowWater:
          buf.add(tbg & " " & Reset)
        else:
          buf.add(tbg & " " & Reset)
    buf.add("\n")

  stdout.write(buf)
  stdout.flushFile()

proc printGameHUD*(env: Environment) =
  ## Print per-team resource summary below the map.
  var buf = newStringOfCap(2048)
  buf.add("\n" & Bold & "--- HUD ---" & Reset & "\n")

  for teamId in 0 ..< MapRoomObjectsTeams:
    let tc = TeamFg[teamId]
    let stockpile = env.teamStockpiles[teamId]

    # Count population, military, buildings for this team
    var pop = 0
    var mil = 0
    for agent in env.liveAgents:
      if not isAgentAlive(env, agent): continue
      if getTeamId(agent) != teamId: continue
      inc pop
      if agent.unitClass != UnitVillager:
        inc mil

    var bldgCount = 0
    for kind in ThingKind:
      if not isBuildingKind(kind): continue
      for thing in env.thingsByKind[kind]:
        if not thing.hasValue: continue
        if thing.teamId == teamId:
          inc bldgCount

    buf.add(fg(tc[0], tc[1], tc[2]) & Bold)
    buf.add("T" & $teamId)
    buf.add(Reset & " ")
    buf.add("F:" & $stockpile.counts[ResourceFood])
    buf.add(" W:" & $stockpile.counts[ResourceWood])
    buf.add(" G:" & $stockpile.counts[ResourceGold])
    buf.add(" S:" & $stockpile.counts[ResourceStone])
    buf.add(" | Pop:" & $pop)
    buf.add(" Mil:" & $mil)
    buf.add(" Bld:" & $bldgCount)
    buf.add("\n")

  stdout.write(buf)
  stdout.flushFile()

proc maybeRenderConsole*(env: Environment) =
  ## Called from step(). Renders if TV_CONSOLE_VIZ=1 and interval matches.
  if not consoleVizEnabled:
    return
  if env.currentStep mod consoleVizInterval != 0:
    return
  env.printGameMap()
  env.printGameHUD()
