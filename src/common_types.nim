## common_types.nim - Game-logic types shared across modules
##
## This module contains types, constants, and procs used by both game logic
## and rendering code, but which do NOT depend on rendering libraries
## (boxy, windy, pixie, opengl, etc.). Modules that only need game logic
## (e.g., agent_control, AI code) can import this instead of common.nim
## to avoid pulling in the entire graphics stack.

when defined(emscripten):
  import windy/platforms/emscripten/emdefs
else:
  import std/[times]

import vmath

proc nowSeconds*(): float64 =
  when defined(emscripten):
    emscripten_get_now() / 1000.0
  else:
    epochTime()

type
  IRect* = object
    x*: int
    y*: int
    w*: int
    h*: int

  Orientation* = enum
    N = 0  # North (Up)
    S = 1  # South (Down)
    W = 2  # West (Left)
    E = 3  # East (Right)
    NW = 4 # Northwest (Up-Left)
    NE = 5 # Northeast (Up-Right)
    SW = 6 # Southwest (Down-Left)
    SE = 7 # Southeast (Down-Right)

  CommandButtonKind* = enum
    CmdNone
    # Unit commands (common)
    CmdMove
    CmdAttack
    CmdStop
    CmdPatrol
    CmdStance
    CmdHoldPosition
    # Formation commands
    CmdFormationLine
    CmdFormationBox
    CmdFormationStaggered
    CmdFormationRangedSpread
    # Villager-specific
    CmdBuild
    CmdGather
    CmdBuildBack  # Return from build submenu
    # Building placement commands (build submenu)
    CmdBuildHouse
    CmdBuildMill
    CmdBuildLumberCamp
    CmdBuildMiningCamp
    CmdBuildBarracks
    CmdBuildArcheryRange
    CmdBuildStable
    CmdBuildWall
    CmdBuildBlacksmith
    CmdBuildMarket
    # Building commands
    CmdSetRally
    CmdUngarrison
    # Production (for military buildings)
    CmdTrainVillager
    CmdTrainManAtArms
    CmdTrainArcher
    CmdTrainScout
    CmdTrainKnight
    CmdTrainMonk
    CmdTrainBatteringRam
    CmdTrainMangonel
    CmdTrainTrebuchet
    CmdTrainBoat
    CmdTrainTradeCog
    CmdTrainGalley
    CmdTrainFireShip
    CmdTrainFishingShip
    CmdTrainTransportShip
    CmdTrainDemoShip
    CmdTrainCannonGalleon
    # Research commands (Blacksmith upgrades - 5 lines)
    CmdResearchMeleeAttack      # Forging → Iron Casting → Blast Furnace
    CmdResearchArcherAttack     # Fletching → Bodkin Arrow → Bracer
    CmdResearchInfantryArmor    # Scale Mail → Chain Mail → Plate Mail
    CmdResearchCavalryArmor     # Scale Barding → Chain Barding → Plate Barding
    CmdResearchArcherArmor      # Padded Archer → Leather Archer → Ring Archer
    # Research commands (University techs)
    CmdResearchBallistics
    CmdResearchMurderHoles
    CmdResearchMasonry
    CmdResearchArchitecture
    CmdResearchTreadmillCrane
    CmdResearchArrowslits
    CmdResearchHeatedShot
    CmdResearchSiegeEngineers
    CmdResearchChemistry
    CmdResearchCoinage
    # Research commands (Castle unique techs)
    CmdResearchCastleTech1      # Team's first unique tech (Castle Age)
    CmdResearchCastleTech2      # Team's second unique tech (Imperial Age)
    # Mill commands
    CmdQueueFarm                # Queue farm reseed (pre-pay wood for auto-reseed)

{.push inline.}
proc ivec2*(x, y: int): IVec2 =
  result.x = x.int32
  result.y = y.int32

# Distance calculations - use these instead of inline expressions
template chebyshevDist*(a, b: IVec2): int32 =
  ## Chebyshev distance (max of abs differences) - used for tower/building ranges
  max(abs(a.x - b.x), abs(a.y - b.y))

template manhattanDist*(a, b: IVec2): int32 =
  ## Manhattan distance (sum of abs differences) - used for pathfinding costs
  abs(a.x - b.x) + abs(a.y - b.y)
{.pop.}

const OrientationDeltas*: array[8, IVec2] = [
  ivec2(0, -1),   # N (North)
  ivec2(0, 1),    # S (South)
  ivec2(-1, 0),   # W (West)
  ivec2(1, 0),    # E (East)
  ivec2(-1, -1),  # NW (Northwest)
  ivec2(1, -1),   # NE (Northeast)
  ivec2(-1, 1),   # SW (Southwest)
  ivec2(1, 1)     # SE (Southeast)
]

const
  ActionVerbCount* = 11  # Added set rally point action (verb 10)
  ActionArgumentCount* = 28

  # Action verb indices (used by replay_writer, replay_analyzer, ai_audit)
  ActionNoop* = 0
  ActionMove* = 1
  ActionAttack* = 2
  ActionUse* = 3
  ActionSwap* = 4
  ActionPut* = 5
  ActionPlantLantern* = 6
  ActionPlantResource* = 7
  ActionBuild* = 8
  ActionOrient* = 9
  ActionSetRallyPoint* = 10

  ActionNames*: array[ActionVerbCount, string] = [
    "noop", "move", "attack", "use", "swap", "put",
    "plant_lantern", "plant_resource", "build", "orient", "set_rally_point"
  ]

proc encodeAction*(verb: uint16, argument: uint16): uint16 =
  uint16(verb.int * ActionArgumentCount + argument.int)

{.push inline.}
proc orientationToVec*(orientation: Orientation): IVec2 =
  OrientationDeltas[orientation.int]
{.pop.}

# Screen shake state for combat feedback
const
  ScreenShakeDecayRate* = 0.85'f32  ## Multiplicative decay per frame
  ScreenShakeMaxIntensity* = 8.0'f32  ## Maximum shake intensity in pixels
  ScreenShakeDeathIntensity* = 4.0'f32  ## Shake intensity when a unit dies

var
  screenShakeIntensity*: float32 = 0.0  ## Current shake intensity
  screenShakeOffset*: Vec2 = vec2(0, 0)  ## Current frame's random offset
  screenShakeRng: uint32 = 12345  ## Simple RNG state for shake

proc screenShakeLcg(): float32 =
  ## Simple LCG for screen shake randomness (deterministic per frame).
  screenShakeRng = screenShakeRng * 1103515245'u32 + 12345'u32
  let normalized = (screenShakeRng.float32 / uint32.high.float32) * 2.0 - 1.0
  normalized

proc triggerScreenShake*(intensity: float32 = ScreenShakeDeathIntensity) =
  ## Trigger a screen shake effect. Intensity is additive up to max.
  screenShakeIntensity = min(ScreenShakeMaxIntensity,
                              screenShakeIntensity + intensity)

proc updateScreenShake*() =
  ## Update screen shake each frame: decay intensity and generate new offset.
  if screenShakeIntensity > 0.1:
    screenShakeOffset = vec2(
      screenShakeLcg() * screenShakeIntensity,
      screenShakeLcg() * screenShakeIntensity
    )
    screenShakeIntensity *= ScreenShakeDecayRate
  else:
    screenShakeIntensity = 0.0
    screenShakeOffset = vec2(0, 0)

var
  play*: bool = true
  playSpeed*: float32 = 0.1  # slower default playback
  lastSimTime*: float64 = nowSeconds()

const
  SlowPlaySpeed* = 0.2
  FastPlaySpeed* = 0.02
  FasterPlaySpeed* = 0.005
  SuperPlaySpeed* = 0.001

const
  CardinalOffsets* = [
    ivec2(0, -1),
    ivec2(1, 0),
    ivec2(0, 1),
    ivec2(-1, 0)
  ]
  AdjacentOffsets8* = [
    ivec2(0, -1),
    ivec2(1, 0),
    ivec2(0, 1),
    ivec2(-1, 0),
    ivec2(1, -1),
    ivec2(1, 1),
    ivec2(-1, 1),
    ivec2(-1, -1)
  ]
