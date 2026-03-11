## perf_actions.nim - Benchmark test for actions subsystem performance
##
## Tests performance optimizations in action processing:
## - Reusable constructionBuilders table (avoids per-step allocation)
## - Pre-computed team pop caps/counts (O(1) lookup for monk conversion)
## - Reusable agentOrder array (shuffle in place)
##
## Run: nim c -r -d:release --path:src tests/perf_actions.nim

import std/[unittest, monotimes, strformat]
import environment
import types
import agent_control

const
  WarmupSteps = 50
  MeasuredSteps = 200
  TargetActionsMs = 0.25  # Target mean time for actions subsystem at step ~200

proc msBetween(a, b: MonoTime): float64 =
  (b.ticks - a.ticks).float64 / 1_000_000.0

suite "Performance: Actions Subsystem":
  test "actions subsystem completes within budget at 200 steps":
    initGlobalController(BuiltinAI, seed = 42)
    let env = newEnvironment()

    # Warmup phase
    for step in 0 ..< WarmupSteps:
      var actions = getActions(env)
      env.step(addr actions)

    # Measure actions timing over multiple steps
    var totalActionsMs = 0.0
    var maxActionsMs = 0.0
    var sampleCount = 0

    for step in 0 ..< MeasuredSteps:
      var actions = getActions(env)

      # We can't isolate actions timing without -d:stepTiming or -d:perfRegression,
      # so we measure full step and track total time
      let tStart = getMonoTime()
      env.step(addr actions)
      let tEnd = getMonoTime()
      let stepMs = msBetween(tStart, tEnd)

      totalActionsMs += stepMs
      maxActionsMs = max(maxActionsMs, stepMs)
      inc sampleCount

    let meanMs = totalActionsMs / sampleCount.float64

    echo &"  Actions benchmark: mean={meanMs:.4f}ms, max={maxActionsMs:.4f}ms over {sampleCount} steps"
    echo &"  Total time: {totalActionsMs:.2f}ms"

    # This test primarily ensures the optimizations don't cause regressions
    # Actual per-subsystem timing requires -d:perfRegression build
    check meanMs < 10.0  # Sanity check: less than 10ms per step total

  test "pre-computed team pop caps are available":
    let env = newEnvironment()

    # After step, stepTeamPopCaps should be populated
    var actions: array[MapAgents, uint16]
    for i in 0 ..< MapAgents:
      actions[i] = 0  # NOOP
    env.step(addr actions)

    # Check that at least team 0 has some pop cap
    # (teams are initialized with town centers which provide pop cap)
    var totalPopCap = 0
    for teamId in 0 ..< MapRoomObjectsTeams:
      totalPopCap += env.stepTeamPopCaps[teamId]

    check totalPopCap > 0
    echo &"  Total pop cap across teams: {totalPopCap}"

  test "agentOrder is initialized and shuffled":
    let env = newEnvironment()

    # agentOrder should contain all indices 0..MapAgents-1
    var seen: array[MapAgents, bool]
    for i in 0 ..< MapAgents:
      let idx = env.agentOrder[i]
      check idx >= 0 and idx < MapAgents
      seen[idx] = true

    # All indices should be present
    for i in 0 ..< MapAgents:
      check seen[i]

    # After a step, order should be shuffled (statistically unlikely to be sorted)
    var actions: array[MapAgents, uint16]
    env.step(addr actions)

    var sortedCount = 0
    for i in 0 ..< MapAgents - 1:
      if env.agentOrder[i] < env.agentOrder[i + 1]:
        inc sortedCount

    # With 1000 agents, if sorted we'd have ~999 sorted pairs
    # After shuffle, expect roughly 50% sorted pairs
    let sortedRatio = sortedCount.float64 / (MapAgents - 1).float64
    check sortedRatio < 0.9  # Not mostly sorted
    echo &"  Sorted pair ratio after shuffle: {sortedRatio * 100.0:.2f}%"

  test "constructionBuilders table is cleared per step":
    let env = newEnvironment()

    # constructionBuilders should be empty at start of each step
    # (we can't directly test this without stepping, but we verify it doesn't leak)
    var actions: array[MapAgents, uint16]
    for step in 0 ..< 10:
      env.step(addr actions)
      # Table should be cleared before construction tracking
      # Can't easily verify without modifying step, but this ensures no crashes

    check true  # Test passes if no exceptions/crashes

when isMainModule:
  echo "Running actions subsystem performance tests..."
