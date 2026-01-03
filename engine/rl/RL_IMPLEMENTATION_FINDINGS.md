# RL Implementation Findings - Training Performance Issues

## Executive Summary

This document outlines findings from an investigation of the Reinforcement Learning (RL) implementation in `/engine/rl`, focusing specifically on **issues that impact the quality of RL training results**. The analysis identifies problems with reward attribution, observation space design, action space composition, and exploration that would prevent the agent from learning optimal policies or converging efficiently.

## Architecture Overview

The RL implementation consists of:
1. **C++ Core** (`rl_interface.cpp`, `rl_spec_config.cpp`, `rl_potential_functions.cpp`)
2. **Python Gymnasium Bridge** (`python_rl_env/rl_bridge.py`)
3. **Integration Point** (`player.cpp::select_action()`)

The system operates by:
- Intercepting action selection in the player's main control loop
- Building an observation of game state (resources, buffs, dots, cooldowns)
- Exposing a discrete action space (all castable abilities + pseudo-actions)
- Communicating with external policies via JSON over stdin/stdout
- Attributing reward as damage delta since the previous decision

---

## Critical Issues Affecting RL Training Quality

### 1. **Python Blacklist Removes Core Rotation Abilities**

**Location:** `rl_bridge.py:45-59`

```python
GLOBAL_ACTION_BLACKLIST = {
    "invoke_external_buff",
    "snapshot_stats",
    "cancel_buff",
    "use_item_arazs_ritual_forge",
    "do_treacherous_transmitter_task",
    "run_action_list",
    "entangling_roots",
    "shred",      # <- CORE Feral ability removed!
    "rake",       # <- CORE Feral ability removed!
    "skull_bash",
    "cat_form",
    "rip",        # <- CORE Feral finisher removed!
    "cataclysm",
}
```

**Problem:** The Python environment removes essential rotation abilities from the action space, making it impossible for the agent to learn optimal play.

**Impact on RL Training:**
- **Feral Druid cannot learn proper rotation**: Core builders (Shred, Rake) and finisher (Rip) are unavailable
- Agent is forced to use sub-optimal alternatives or wait actions
- Training will converge to incorrect policies that don't reflect actual gameplay
- DPS results will be artificially low and meaningless

**Why This Matters:** The agent literally cannot execute the intended rotation. It's like trying to teach someone to play piano with half the keys removed.

**Recommendation:** Remove this blacklist entirely or make it configurable per-spec. The C++ filtering (`is_exposed_action`) should be the only authority on what actions are available.

---

### 2. **Reward Sparsity - Only Total Damage, No Intermediate Signals**

**Location:** `player.cpp:14372-14380`

```cpp
// Compute reward: damage delta since last decision
const double current_dmg = priority_iteration_dmg;
const double reward      = current_dmg - rl_last_priority_iteration_dmg;
rl_last_priority_iteration_dmg = current_dmg;
```

**Problem:** Reward is purely damage-based with no shaping for important intermediate actions.

**Impact on RL Training:**
- **Credit assignment problem**: Damage from a Rip tick 10 seconds later is credited to whatever action was taken at that moment, not to the Rip cast that set it up
- **Sparse rewards for setup actions**: Applying dots, building combo points, activating buffs all receive zero or negative reward (GCD cost) until damage happens
- **Exploration is discouraged**: The agent learns to spam high-damage instant casts rather than complex rotations with setup phases
- **Long-term planning is not rewarded**: Optimal play (e.g., pooling energy for Tiger's Fury windows) looks worse than spamming available abilities

**Why This Matters:** PPO and similar algorithms struggle with sparse rewards and long time horizons. The agent needs feedback on whether its actions are setting up future damage, not just whether damage happened right now.

**Recommendation:** Implement reward shaping with:
1. Small positive rewards for maintaining important dots/buffs
2. Rewards for building/spending combo points efficiently  
3. Penalties for letting important dots fall off
4. Potential-based shaping (infrastructure exists but is disabled by default)

---

### 3. **Observation Space Missing Critical Information**

**Location:** `rl_interface.cpp:655-743`

**Problem:** Several critical pieces of information needed for optimal decision-making are missing from observations.

**Missing Information:**

1. **No visibility into upcoming major cooldowns**
   - Agent can't see that Convoke is coming off cooldown in 5 seconds
   - Can't plan to pool resources for burst windows
   - Results in wasted cooldown usage or resource capping

2. **No energy/resource regeneration rate**
   - Observation includes current energy (as percentage) but not regen rate
   - Can't distinguish between base 10 energy/sec vs. buffed 20 energy/sec
   - Impacts pooling decisions and wait action choices

3. **No visibility of DoT/buff pandemic windows**
   - Agent sees remaining time but not refresh thresholds
   - Can't learn to refresh dots in the last 30% (pandemic mechanic)
   - Will either refresh too early (wasting ticks) or too late (losing uptime)

4. **No GCD state for other queued actions**
   - Can't see if auto-attacks or off-GCD abilities are active
   - Missing context for whether to wait or cast immediately

**Impact on RL Training:**
- Agent learns reactive play instead of proactive planning
- Can't optimize around cooldown timings
- Resource management becomes trial-and-error instead of calculated
- Fails to learn game mechanics (pandemic, resource pooling)

**Recommendation:** Add to observation space:
```cpp
// Upcoming cooldowns (next 3 major abilities coming off CD in <15s)
std::array<double, 3> major_cd_upcoming_s;  

// Resource regeneration (current regen rate / base rate)
double resource_regen_rate_norm;

// Pandemic/refresh windows for active dots
std::vector<bool> dot_in_pandemic_window;

// Active GCD state
bool has_queued_action;
```

---

### 4. **Action Space Includes Uncastable "Trap" Actions**

**Location:** `rl_interface.cpp:151-253`, `rl_interface.cpp:600-650`

**Problem:** DBC-based action discovery adds abilities that require forms/stances the player doesn't have, creating "trap" actions that waste the agent's time exploring.

**Example - Feral Druid sees:**
- `moonkin_form` (requires Balance talent)
- `bear_form` abilities (suboptimal for DPS)
- `moonfire` (weak outside Moonkin Form)
- Various abilities from class tree that aren't specialized

**Impact on RL Training:**
- **Expanded action space slows learning**: More actions = more exploration needed = slower convergence
- **Trap actions waste episodes**: Agent tries Moonkin Form, realizes it's terrible, has to unlearn it
- **Masking doesn't help enough**: Actions are technically "legal" but strategically terrible, so the agent must learn through trial-and-error that they shouldn't be used
- **Curriculum learning is disrupted**: Agent should focus on core rotation first, but gets distracted by tangential abilities

**Why This Matters:** Exploration complexity grows exponentially with action space size. Every useless action added to the space makes learning significantly slower. It's like teaching chess but including "illegal moves" that are technically allowed but lose immediately.

**Recommendation:** 
1. Filter actions by spec relevance (not just legality)
2. Consider form requirements: don't expose Moonkin abilities to Feral unless they have Fluid Form talent
3. Start with a minimal core action set and expand progressively (curriculum learning)

---

### 5. **Wait Actions Create State-Space Ambiguity**

**Location:** `rl_interface.cpp:30-36`, `player.cpp:14263-14289`

```cpp
constexpr double WAIT_DURATIONS[] = { 0.1, 0.2, 0.5, 1.0, 1.5 };
```

**Problem:** Multiple wait durations and the "pass" action create overlapping/ambiguous states that confuse the learning process.

**Impact on RL Training:**
- **Equivalent actions**: `wait_0.1` twice ≈ `wait_0.2` once, but the agent must learn this equivalence
- **Pass semantics unclear**: Returns `nullptr` which triggers APL-based waiting, not a true no-op
- **Resource waiting is implicit**: Agent can't specify "wait until 50 energy" directly, must learn to chain wait actions
- **Value function ambiguity**: Different sequences of waits leading to the same state have different Q-values

**Example Problem:**
```
State: 30 energy, no GCD
Action A: wait_0.5 → 35 energy → cast Shred
Action B: wait_0.2 → wait_0.3 → 35 energy → cast Shred
```
These should have identical value, but the agent sees them as different state-action sequences.

**Why This Matters:** The agent wastes learning capacity discovering equivalences between wait patterns instead of learning rotation mechanics. PPO's policy network struggles to generalize across these equivalent trajectories.

**Recommendation:**
1. **Simplify to one parameterized wait action** with the duration as a continuous parameter, OR
2. **Remove wait actions entirely** and let the agent learn that doing nothing (if that's an option) pools resources naturally, OR  
3. **Add explicit "wait until resource X" actions** (wait_until_energy_50, etc.) for common thresholds

---

### 6. **No Partial Observability of Enemy Mechanics**

**Location:** `rl_interface.cpp:656-680` (only observes TTD)

**Problem:** Agent only sees target time-to-die (TTD), not enemy mechanics, phases, or damage patterns.

**Impact on RL Training:**
- Can't learn to save cooldowns for execute phases
- Can't learn to pool resources before phase transitions
- Can't learn movement-intensive mechanics (if implemented)
- Missing context for when to defensive vs. offensive

**Example:** 
- Boss at 35% health → burn phase begins
- Agent doesn't know this is happening
- Uses cooldowns at 36% health
- Cooldowns aren't available when they matter most
- Agent never learns to optimize around phase transitions

**Why This Matters:** Real gameplay has distinct phases with different optimal strategies. The agent learns an "average" policy that's suboptimal in all phases rather than learning phase-specific strategies.

**Recommendation:** Add to observation:
```cpp
double target_health_pct;        // Current health percentage
bool target_in_execute_phase;   // <35% or similar threshold
double target_phase_id;          // Phase indicator if available
```

---

### 7. **Potential Functions Are Placeholders (Reward Shaping Disabled)**

**Location:** `rl_potential_functions.cpp:10-30`

```cpp
double feral_potential( const observation_t& obs, const player_t* /*player*/ )
{
  // Placeholder: no shaping yet (returns 0)
  (void)obs;  // Suppress unused parameter warning
  return 0.0;
}
```

**Problem:** The infrastructure for reward shaping exists but all potential functions return 0, providing no shaping benefit.

**Impact on RL Training:**
- All the issues from #2 (reward sparsity) remain unaddressed
- Agent struggles with credit assignment
- Long-term planning is not rewarded
- Setup actions (buffs, dots) appear to have negative value

**Why This Matters:** Potential-based shaping is **provably policy-invariant** (doesn't change the optimal policy) while providing denser feedback. Without it, the agent is flying blind during the setup phase of rotations.

**Example of What's Missing:**
```cpp
double feral_potential( const observation_t& obs, const player_t* player )
{
  double potential = 0.0;
  
  // Reward maintaining Rip
  if (obs.dot_remains_norm[RIP_INDEX] > 0.3)  // In pandemic window
    potential += 500.0;
  
  // Reward having Tiger's Fury active
  if (obs.buff_remains_norm[TIGERS_FURY_INDEX] > 0)
    potential += 300.0;
  
  // Reward high combo points when ready to spend
  if (player->resources.current[COMBO_POINT] >= 5)
    potential += 200.0 * obs.resource_pct[COMBO_POINT];
    
  return potential;
}
```

**Recommendation:** Implement spec-specific potential functions that reward:
- Maintaining important dots/buffs
- Having resources available when needed  
- Being in advantageous states (high combo points, energy pooled, etc.)

---

### 8. **Deterministic Simulation with Fixed Seeds Prevents Stochastic Learning**

**Location:** `rl_bridge.py:104`, `sim.cpp` (seed handling)

```python
def __init__(self, ..., seed: Optional[int] = None, ...):
    self.seed_value = seed
    # If seed is set, every episode is identical
```

**Problem:** If a fixed seed is used (for reproducibility), the agent sees the exact same simulation every episode, limiting exploration.

**Impact on RL Training:**
- **Overfitting to specific RNG outcomes**: Agent learns to exploit specific proc sequences that won't generalize
- **Poor exploration**: Stochastic policies can't discover diverse trajectories if the environment is deterministic
- **Brittle policies**: Agent's learned policy only works for that specific seed's proc pattern

**Why This Matters:** RL algorithms rely on experiencing diverse scenarios to learn robust policies. If every trinket proc happens at exactly the same time in every episode, the agent learns to time abilities around those specific procs rather than learning adaptive strategies.

**Recommendation:** 
1. Use different seeds per episode (or no seed) during training
2. Only fix seed for evaluation/testing
3. Consider adding seed as part of environment state to ensure diversity

---

### 9. **No Curriculum Learning or Staged Training**

**Problem:** The agent is immediately exposed to the full complexity of the action space and rotation mechanics with no gradual ramp-up.

**Impact on RL Training:**
- **Overwhelming action space**: 50+ actions to choose from at every step
- **No structured learning**: Agent must discover basic concepts (builders before spenders) through random exploration
- **Slow initial learning**: Many episodes are wasted on nonsensical action sequences
- **Local optima**: Agent may converge to simple but suboptimal strategies (spam one ability) because discovering the full rotation requires very specific exploration paths

**Why This Matters:** Curriculum learning (starting simple, gradually increasing complexity) is proven to dramatically improve sample efficiency and final performance in complex RL tasks. Current implementation is like teaching calculus before arithmetic.

**Example Curriculum:**
```
Stage 1: Core builders only (Shred, Rake) + finisher (Rip)
         → Learn basic combo point system

Stage 2: Add Tiger's Fury cooldown
         → Learn cooldown management

Stage 3: Add full action space
         → Learn complete rotation

Stage 4: Multi-target scenarios
         → Learn target switching
```

**Recommendation:** Implement staged training with progressively unlocked actions, or use action masking that initially restricts the agent to core abilities only.

---

### 10. **Off-GCD and Cast-While-Casting Actions Are Ignored**

**Location:** `player.cpp:14326-14329`

```cpp
// Skip OFF_GCD and CAST_WHILE_CASTING contexts
if ( ... && et == execute_type::FOREGROUND && ...)
```

**Problem:** The RL agent only makes decisions at foreground GCD points, ignoring opportunities for off-GCD abilities and cast-while-casting windows.

**Impact on RL Training:**
- **Misses free damage**: Off-GCD abilities (trinkets, racials, some cooldowns) are never used by the RL agent
- **Suboptimal compared to APL**: Hand-written APLs use off-GCD abilities, making RL results artificially worse
- **Can't learn advanced techniques**: Ability weaving and GCD optimization are core to high-level play
- **Incomplete observation of game state**: Agent doesn't see the full picture of when abilities are available

**Example - Feral Druid:**
- `berserk_cat` is usable off-GCD
- Optimal play: cast Shred → immediately use Berserk (no GCD) → cast Shred again
- RL agent: casts Shred → next decision only at the following GCD → Berserk is never considered

**Why This Matters:** This is a fundamental architectural limitation that caps the agent's maximum possible performance below APL level. Even with perfect learning, the agent cannot match hand-crafted rotations.

**Recommendation:** 
1. **Option A**: Extend RL decision points to off-GCD and CWC contexts (increases decision frequency)
2. **Option B**: Expose off-GCD actions as part of the main action space with special timing rules
3. **Option C**: Implement a hybrid mode where APL handles off-GCD and RL handles main rotation

---

## Moderate Issues Affecting Training Efficiency

### 11. **Hard-Coded Normalization Constants May Clip Important Information**

**Location:** `rl_interface.cpp:705`

```cpp
constexpr double BUFF_NORM_MAX = 30.0;  // Normalize durations by 30 seconds
```

**Problem:** Buffs/dots longer than 30 seconds are clamped to 1.0, losing information about their actual duration.

**Impact on RL Training:**
- Long-duration effects (Rip with 24s base, extended by Pandemic to 30s+) lose resolution
- Agent can't distinguish between "Rip has 30s left" vs. "Rip has 60s left"
- Refresh decisions become suboptimal for long dots
- The observation space artificially limits what the agent can learn

**Recommendation:** Either increase the normalization constant or use a non-linear scaling (log scale) that preserves information across a wider range.

---

### 12. **Single-Target Only - No AoE or Cleave Training**

**Location:** `rl_interface.cpp:676-680` (only observes primary target)

```cpp
// Target TTD
if ( sim->target )
  obs.target_ttd_s = sim->target->time_to_percent( 0.0 ).total_seconds();
```

**Problem:** Agent only receives information about the primary target, ignoring multi-target scenarios.

**Impact on RL Training:**
- Cannot learn AoE rotation priorities
- Cannot learn target switching strategies
- Cannot learn when to use single-target vs. AoE abilities
- Training is limited to Patchwerk-style single-target only

**Recommendation:** Add multi-target observations (count of enemies, total health pool, highest priority target) to enable cleave/AoE learning.

---

### 13. **Action Discovery May Miss Dynamically Available Abilities**

**Location:** `rl_interface.cpp:152-253`

**Problem:** DBC-based discovery happens once at initialization but some abilities become available dynamically (trinket effects, temporary buffs granting new abilities).

**Impact on RL Training:**
- Agent can't use abilities granted by gear/buffs mid-fight
- Action space is static when it should be dynamic
- Missing opportunities for damage from temporary abilities
- Can't learn to optimize around gear procs that grant new actions

**Recommendation:** Implement runtime action space updates or periodic re-discovery to capture dynamically available abilities.

---

### 14. **No Explicit Terminal State Handling**

**Problem:** Episodes end when target dies (TTD reaches 0), but there's no special handling or observation of terminal states.

**Impact on RL Training:**
- Agent doesn't learn "endgame" behavior (execute phase optimization)
- Value function doesn't properly handle terminal states in Bellman equation
- No special reward/penalty for fight length
- Agent might waste resources at the end of fights not knowing it's about to terminate

**Recommendation:** Add terminal state indicator to observations and consider adding fight-length penalty/bonus to reward function.

---

## Minor Issues Affecting Training Quality

### 15. **Limited Spec Coverage Means Lessons Don't Transfer**

**Location:** `rl_spec_config.cpp` (only 3 specs configured)

**Problem:** Only Feral Druid, Guardian Druid, and Destruction Warlock have observation configurations.

**Impact on RL Training:**
- Cannot train agents for other specs without code changes
- No cross-spec transfer learning possible
- Limited validation of the RL approach across different playstyles

**Recommendation:** Add observation configs for more specs, prioritizing those with diverse mechanics (healer, ranged, melee, tank).

---

### 16. **Episode Length May Be Too Long for Effective Learning**

**Problem:** Default fight length is 300 seconds (5 minutes), which is a very long episode for RL.

**Impact on RL Training:**
- Each episode takes a long time to simulate
- Fewer episodes per training hour = slower learning
- Long episodes increase variance in returns
- Discounting over 300s may cause issues with credit assignment

**Recommendation:** Consider shorter episodes (30-60s fights) during early training, gradually increasing as the agent improves. Or use episode truncation with proper value bootstrapping.

---

### 17. **No Observation Normalization Verification**

**Problem:** Observation features are normalized by different constants (GCD by 1.5s, buffs by 30s, resources by max), but there's no validation that these produce a well-scaled observation space.

**Impact on RL Training:**
- Features with different scales can bias neural network training
- Some features might dominate gradient updates
- Network may ignore important but small-scale features
- Slows convergence and hurts final performance

**Recommendation:** Log observation statistics (mean, variance) during training and verify all features are in similar ranges (e.g., [0, 1] or [-1, 1]). Add observation normalization wrapper if needed.

---

### 18. **Dummy Policy is Pure Random (No Heuristic Baseline)**

**Location:** `rl_interface.cpp:399-414`

```cpp
std::size_t dummy_policy( const step_input_t& input, void* /*user_data*/ )
{
  // Default: pick a pseudo-random legal action using round-robin
  std::vector<std::size_t> legal;
  for ( std::size_t i = 0; i < input.action_space.action_mask.size(); ++i )
  {
    if ( input.action_space.action_mask[ i ] )
      legal.push_back( i );
  }
  if ( legal.empty() )
    return input.action_space.actions.size();
  
  static thread_local uint64_t tl_counter = 0;
  return legal[ ( tl_counter++ ) % legal.size() ];
}
```

**Problem:** The default policy is pure random (round-robin through legal actions), providing no useful baseline.

**Impact on RL Training:**
- No comparison to a reasonable baseline (naive but functional rotation)
- Can't measure if the agent is actually learning vs. random play
- Starting from random initialization means many early episodes are pure noise
- No warm-start or imitation learning from expert demonstrations

**Recommendation:** Implement a simple heuristic policy as baseline:
- Always maintain important dots/buffs
- Spend combo points at 5
- Use cooldowns on cooldown
- Fill with builder actions
This provides a sanity check and potential warm-start for RL training.

---

## Design Limitations

### 19. **No Support for Implicit APL Fallback or Hybrid Policies**

**Problem:** RL either controls everything or nothing - no hybrid mode where RL handles strategy and APL handles routine execution.

**Impact on RL Training:**
- Agent must learn low-level execution details instead of high-level strategy
- Missing the benefits of combining expert rules (APL) with learned optimization (RL)
- Increases action space complexity unnecessarily

**Recommendation:** Implement hierarchical control where RL selects high-level actions (cooldown usage, resource pooling decisions) and APL handles routine rotation filling.

---

### 20. **No Mechanism for Goal Conditioning or Multi-Objective Optimization**

**Problem:** Agent optimizes purely for damage, with no way to trade off other objectives (survival, mana efficiency, etc.)

**Impact on RL Training:**
- Cannot train tank agents (need to balance damage vs. mitigation)
- Cannot train healer agents (need to balance healing vs. damage)
- Cannot learn adaptive strategies (maximize DPS while ensuring survival)

**Recommendation:** Support multi-objective reward functions and goal-conditioned policies where the agent can be told what to optimize for.

---

## Positive Aspects

Despite the issues identified, the implementation has strong foundations:

1. **Action Masking**: Properly prevents illegal actions, crucial for learning valid policies
2. **Modular Architecture**: Clean separation between C++ engine and Python training
3. **Gymnasium Compatibility**: Easy integration with modern RL libraries
4. **Extensible Spec Configs**: Easy to add new specs once the pattern is established
5. **Tracing Support**: JSONL logs enable offline analysis and debugging
6. **Reward Shaping Infrastructure**: Potential functions exist, just need implementation

---

## Recommendations Priority

### Critical (Blocks Effective Training)
1. **Remove Python action blacklist** (Issue #1) - Agent cannot learn without core abilities
2. **Implement reward shaping** (Issue #2, #7) - Sparse rewards prevent learning complex rotations
3. **Add missing observations** (Issue #3) - Agent needs cooldown and mechanics information
4. **Include off-GCD actions** (Issue #10) - Otherwise agent cannot match APL performance

### High Priority (Significantly Improves Learning)
5. **Filter action space by spec relevance** (Issue #4) - Reduces exploration burden
6. **Simplify wait actions** (Issue #5) - Removes ambiguous state-action pairs
7. **Add enemy phase information** (Issue #6) - Enables strategic planning
8. **Implement curriculum learning** (Issue #9) - Dramatically speeds up initial learning

### Medium Priority (Efficiency Improvements)
9. **Use diverse seeds** (Issue #8) - Prevents overfitting to specific RNG
10. **Normalize observation features** (Issue #17) - Improves neural network training
11. **Add multi-target observations** (Issue #12) - Enables AoE scenarios
12. **Implement heuristic baseline** (Issue #18) - Provides sanity checks

### Low Priority (Polish)
13. Adjust normalization constants (Issue #11)
14. Shorter training episodes (Issue #16)
15. Dynamic action discovery (Issue #13)
16. More spec coverage (Issue #15)

---

## Conclusion

The RL implementation has solid architectural foundations but suffers from **critical observation and action space design issues** that prevent effective learning. The most severe problems are:

1. **Incomplete observations**: Agent is missing crucial information about cooldowns, mechanics, and timing
2. **Reward sparsity**: No shaping for intermediate actions means setup phases appear worthless
3. **Action space pollution**: Irrelevant actions slow exploration; essential actions are blacklisted
4. **Architectural limitation**: Off-GCD actions ignored means agent cannot reach APL performance

With the critical issues addressed (especially #1, #2, #3, #10), the implementation could successfully train agents that match or exceed hand-crafted APL performance. The current state will produce agents that converge to suboptimal policies due to missing information and incorrect reward signals.

---

*Document generated: 2026-01-03*
*Focus: Issues affecting RL training performance and convergence quality*
*Reviewer: GitHub Copilot Agent*
