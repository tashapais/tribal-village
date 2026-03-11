## AI decision audit logging system.
## Compile with -d:aiAudit to enable. At runtime:
##   TV_AI_LOG=1  -> summary every 50 steps
##   TV_AI_LOG=2  -> verbose per-agent per-step logging
## When compiled without -d:aiAudit, this file is a no-op.

import ai_types
export ai_types

when defined(aiAudit):
  import std/[os, strformat, strutils]

  const
    AuditActionNames*: array[ActionVerbCount, string] = [
      "noop", "move", "attack", "use", "swap", "put",
      "plant_lantern", "plant_resource", "build", "orient", "set_rally_point"
    ]
    AuditRoleNames*: array[4, string] = ["Gatherer", "Builder", "Fighter", "Scripted"]
    AuditSummaryInterval* = 50

  type
    AuditDecisionBranch* = enum
      BranchInactive       ## Agent dead/inactive
      BranchDecisionDelay  ## Difficulty-based NOOP delay
      BranchStopped        ## Agent stopped via stop command
      BranchGoblinRelic    ## Goblin relic collection behavior
      BranchGoblinAvoid    ## Goblin threat avoidance
      BranchGoblinSearch   ## Goblin relic search/wander
      BranchEscape         ## Stuck escape mode
      BranchAttackOpportunity ## Adjacent attack opportunity
      BranchPatrolChase    ## Patrol - chasing enemy
      BranchPatrolMove     ## Patrol - moving between waypoints
      BranchRallyPoint     ## Moving to rally point
      BranchAttackMoveEngage ## Attack-move engaging enemy
      BranchAttackMoveAdvance ## Attack-move advancing to target
      BranchSettlerMigrate ## Settler migrating to new town site
      BranchHearts         ## Hearts prioritization (gatherer)
      BranchPopCapWood     ## Pop cap house - gathering wood
      BranchPopCapBuild    ## Pop cap house - building
      BranchRoleCatalog    ## Role-based catalog decision

    AuditRecord* = object
      agentId*: int
      teamId*: int
      role*: AgentRole
      verb*: int
      arg*: int
      branch*: AuditDecisionBranch

    AuditSummaryState* = object
      logLevel*: int  # 0=off, 1=summary, 2=verbose
      stepDecisions*: seq[AuditRecord]
      # Accumulator counters for summary mode
      verbCounts*: array[ActionVerbCount, int]
      roleCounts*: array[MapRoomObjectsTeams, array[4, int]]  # [team][role]
      branchCounts*: array[AuditDecisionBranch, int]
      stepsAccumulated*: int
      totalDecisions*: int

  var
    auditSummary*: AuditSummaryState
    auditCurrentBranch*: AuditDecisionBranch  # Set during decideAction

  proc initAuditLog*() =
    let level = getEnv("TV_AI_LOG", "0")
    auditSummary.logLevel = try: parseInt(level) except: 0
    auditSummary.stepDecisions = @[]
    auditSummary.stepsAccumulated = 0
    auditSummary.totalDecisions = 0

  proc setAuditBranch*(branch: AuditDecisionBranch) {.inline.} =
    auditCurrentBranch = branch

  proc recordAuditDecision*(agentId: int, teamId: int, role: AgentRole,
                            action: uint16) =
    if auditSummary.logLevel <= 0:
      return
    let verb = action.int div ActionArgumentCount
    let arg = action.int mod ActionArgumentCount
    let branch = auditCurrentBranch

    # Accumulate counters
    if verb >= 0 and verb < ActionVerbCount:
      inc auditSummary.verbCounts[verb]
    if teamId >= 0 and teamId < MapRoomObjectsTeams:
      inc auditSummary.roleCounts[teamId][ord(role)]
    inc auditSummary.branchCounts[branch]
    inc auditSummary.totalDecisions

    # Verbose mode: record individual decisions for printing
    if auditSummary.logLevel >= 2:
      auditSummary.stepDecisions.add(AuditRecord(
        agentId: agentId,
        teamId: teamId,
        role: role,
        verb: verb,
        arg: arg,
        branch: branch
      ))

  proc printVerboseDecisions*(step: int) =
    if auditSummary.logLevel < 2 or auditSummary.stepDecisions.len == 0:
      return
    echo &"[AI_AUDIT step={step}] {auditSummary.stepDecisions.len} decisions:"
    for d in auditSummary.stepDecisions:
      let verbName = if d.verb >= 0 and d.verb < ActionVerbCount:
                       AuditActionNames[d.verb]
                     else: $d.verb
      let roleName = if ord(d.role) < AuditRoleNames.len:
                       AuditRoleNames[ord(d.role)]
                     else: $d.role
      echo &"  agent={d.agentId} team={d.teamId} role={roleName} action={verbName}:{d.arg} branch={d.branch}"
    auditSummary.stepDecisions.setLen(0)

  proc printAuditSummary*(step: int) =
    if auditSummary.logLevel <= 0:
      return
    inc auditSummary.stepsAccumulated

    # Verbose: print every step
    if auditSummary.logLevel >= 2:
      printVerboseDecisions(step)

    # Summary: print every N steps
    if auditSummary.stepsAccumulated mod AuditSummaryInterval != 0:
      return

    let total = auditSummary.totalDecisions
    if total == 0:
      return

    echo &"\n[AI_AUDIT SUMMARY steps={step - AuditSummaryInterval + 1}..{step}] total_decisions={total}"

    # Action distribution
    echo "  Action distribution:"
    for i in 0 ..< ActionVerbCount:
      let count = auditSummary.verbCounts[i]
      if count > 0:
        let pct = (count.float * 100.0) / total.float
        echo &"    {AuditActionNames[i]}: {count} ({pct:.1f}%)"

    # Role distribution per team
    echo "  Role distribution per team:"
    for teamId in 0 ..< MapRoomObjectsTeams:
      var teamTotal = 0
      for r in 0 ..< 4:
        teamTotal += auditSummary.roleCounts[teamId][r]
      if teamTotal > 0:
        var parts: seq[string] = @[]
        for r in 0 ..< 4:
          let c = auditSummary.roleCounts[teamId][r]
          if c > 0:
            let pct = (c.float * 100.0) / teamTotal.float
            parts.add(&"{AuditRoleNames[r]}={c}({pct:.0f}%)")
        echo &"    team {teamId}: {parts.join(\", \")}"

    # Decision branch distribution
    echo "  Decision branches:"
    for branch in AuditDecisionBranch:
      let count = auditSummary.branchCounts[branch]
      if count > 0:
        let pct = (count.float * 100.0) / total.float
        echo &"    {branch}: {count} ({pct:.1f}%)"

    # Reset accumulators
    for i in 0 ..< ActionVerbCount:
      auditSummary.verbCounts[i] = 0
    for teamId in 0 ..< MapRoomObjectsTeams:
      for r in 0 ..< 4:
        auditSummary.roleCounts[teamId][r] = 0
    for branch in AuditDecisionBranch:
      auditSummary.branchCounts[branch] = 0
    auditSummary.totalDecisions = 0

  # No-op stubs when not compiled with aiAudit
else:
  template setAuditBranch*(branch: untyped) = discard
  template initAuditLog*() = discard
  template recordAuditDecision*(agentId, teamId: int, role: untyped, action: uint16) = discard
  template printAuditSummary*(step: int) = discard
