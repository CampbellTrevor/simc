# RL Implementation Findings and Analysis

## Executive Summary

This document outlines the findings from an investigation of the Reinforcement Learning (RL) implementation in `/engine/rl`. The implementation provides a Gymnasium-compatible environment for training RL agents to control SimulationCraft character actions. While the implementation is functional and well-structured, several issues were identified that may impact correctness, performance, and maintainability.

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

## Critical Issues

### 1. **Global State Mutation in Policy Functions**

**Location:** `rl_interface.cpp:357-358`

```cpp
rl::policy_fn_t g_policy = &rl::dummy_policy;
void* g_policy_user_data = nullptr;
```

**Problem:** Global mutable state for the policy function pointer creates thread-safety issues and coupling problems.

**Impact:**
- Prevents safe multi-threaded operation with different policies per thread
- Violates the principle that policy should be per-player or per-sim, not global
- Creates hidden coupling between different parts of the codebase

**Recommendation:** Move policy state to either `sim_t` or `player_t` to enable per-instance control.

---

### 2. **Unsafe Const-Cast in Action Creation**

**Location:** `rl_interface.cpp:758`

```cpp
create_baseline_actions( const_cast<player_t&>( player ) );
```

**Problem:** Modifying a const-qualified player within `build_action_list()` violates const-correctness.

**Impact:**
- Breaks const contract, leading to undefined behavior potential
- Makes the codebase harder to reason about
- Could cause issues if called from genuinely const contexts

**Recommendation:** Either:
1. Change `build_action_list()` to take non-const `player_t&`, OR
2. Separate action list building into two phases: initialization (non-const) and reading (const)

---

### 3. **Default-Enabled RL Mode is Dangerous**

**Location:** `sim.cpp:1548-1551`

```cpp
rl_enable( true ),
rl_trace( true ),
rl_trace_file( "rl_trace.jsonl" ),
rl_stdio( true ),
```

**Problem:** RL mode is enabled by default, which fundamentally changes simulation behavior.

**Impact:**
- Normal SimulationCraft usage requires explicit `rl_enable=0` to disable RL
- Creates trace files by default, polluting working directories
- The dummy policy (random legal actions) will produce incorrect DPS numbers
- Violates principle of least surprise for existing users

**Recommendation:** Default all RL options to `false`. Make RL opt-in, not opt-out.

---

### 4. **Hard-Coded Action Blacklist in Python Bridge**

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
    "shred",    # <- Why are core Feral abilities blacklisted?
    "rake",     # <- Why are core Feral abilities blacklisted?
    "skull_bash",
    "cat_form",
    "rip",      # <- Why are core Feral abilities blacklisted?
    "cataclysm",
}
```

**Problem:** Hard-coded blacklist in Python code prevents the agent from learning core class mechanics.

**Impact:**
- Feral Druid's core abilities (Shred, Rake, Rip) are blacklisted, preventing proper gameplay
- Blacklist should either be configuration-driven or removed entirely
- Forces the agent to rely on sub-optimal actions
- Creates a mismatch between what C++ exposes and what Python sees

**Recommendation:** Remove this global blacklist or make it configurable per-spec. The C++ filtering (`is_exposed_action`) should be the authoritative source.

---

### 5. **Memory Management Issues with Baseline Actions**

**Location:** `rl_interface.cpp:298-309`

```cpp
action_t* a = player.create_action( action_name, "" );
if ( !a )
  continue;  // Silently skip unrecognized actions

// Mark this action with the RL baseline APL so it passes is_exposed_action
a->action_list = rl_apl;
```

**Problem:** Actions created by RL baseline discovery are not properly tracked for cleanup.

**Impact:**
- Potential memory leaks if actions are not properly deleted
- Actions are added to `player.action_list`, but ownership is unclear
- No clear documentation of lifecycle management

**Recommendation:** Add explicit documentation of ownership and ensure proper cleanup in player destructor.

---

### 6. **Mutable Caching in Const Methods**

**Location:** `rl_spec_config.hpp:37-42`

```cpp
// Cached pointers (populated on first use, per-player)
// Mutable because caching happens during const observation building
mutable std::vector<buff_t*> cached_buffs;
mutable std::vector<dot_t*> cached_dots;
mutable bool buffs_cached = false;
mutable bool dots_cached = false;
mutable const player_t* cached_for_player = nullptr;
mutable const player_t* cached_for_target = nullptr;
```

**Problem:** Using `mutable` to bypass const-correctness is a code smell.

**Impact:**
- Hides the fact that observation building has side effects
- Makes thread-safety analysis harder
- Violates principle that const methods should not modify observable state

**Recommendation:** Use external caching (e.g., in `player_t`) or accept that observation building is non-const.

---

### 7. **Insufficient Thread Safety in Stdio Bridge**

**Location:** `rl_interface.cpp:361-362`, `561-587`

```cpp
std::mutex g_stdio_mutex;  // Only protects stdio operations

std::size_t stdio_policy( const step_input_t& input, void* /*user_data*/ )
{
  std::lock_guard<std::mutex> lock( g_stdio_mutex );
  // Write state as JSON line to stdout
  write_step_json_to_stream( std::cout, input );
  // Read action index from stdin
  ...
}
```

**Problem:** The stdio mutex only protects the I/O operations, not the policy state itself.

**Impact:**
- Forces `threads=1` (see `sim.cpp:4399-4402`) to work around thread safety issues
- Prevents parallel training from within a single simc process
- Global policy state can still be modified during stdio operations

**Recommendation:** Document that stdio bridge requires single-threaded mode and consider per-sim policy state.

---

## Moderate Issues

### 8. **Incomplete Error Handling in Baseline Action Initialization**

**Location:** `rl_interface.cpp:327-350`

```cpp
try
{
  a->init();
}
catch ( const std::exception& e )
{
  a->background = true;
  std::cout << "Warning: RL baseline action '" << a->name_str << "' initialization failed: " << e.what() << "\n";
  continue;
}
```

**Problem:** Failed initialization silently converts actions to background, making them invisible.

**Impact:**
- Difficult to debug when actions mysteriously disappear
- Error messages go to stdout instead of proper logging
- No way to know if critical abilities failed to initialize

**Recommendation:** Use proper logging mechanism and consider failing more explicitly for critical abilities.

---

### 9. **Normalization Constants are Hard-Coded**

**Location:** `rl_interface.cpp:705`, `rl_bridge.py:41`

```cpp
constexpr double BUFF_NORM_MAX = 30.0;  // Normalize durations by 30 seconds
```

```python
BASE_GCD_SECONDS = 1.5
```

**Problem:** Hard-coded normalization constants may not be appropriate for all specs/scenarios.

**Impact:**
- Buffs/dots longer than 30 seconds will be clamped to 1.0
- GCD normalization assumes 1.5s base GCD, incorrect for some classes
- Makes the observation space less informative for long-duration effects

**Recommendation:** Make normalization constants configurable or derive them from game data.

---

### 10. **Pass Pseudo-Action Design is Questionable**

**Location:** `rl_interface.cpp:42-46`, `player.cpp:14291-14315`

```cpp
// A "pass" action that does nothing and has zero duration.
// This allows the RL agent to skip a decision point
constexpr std::size_t NUM_PASS_PSEUDO_ACTIONS = 1;
const char* PASS_LABEL = "pass";
```

**Problem:** The pass action returns `nullptr` which triggers "no action selected" behavior. This is different from a true no-op.

**Impact:**
- Undefined behavior: `nullptr` return from `select_action()` means "try the next APL"
- The comment says "zero duration" but actually triggers resource-based waiting
- Creates confusion about what "pass" actually does

**Recommendation:** Either:
1. Make pass truly a zero-duration no-op, OR
2. Remove pass and let the agent learn to use short wait actions instead

---

### 11. **Reward Attribution is Single-Player Only**

**Location:** `player.cpp:14372-14380`

```cpp
// Compute reward: damage delta since last decision
const double current_dmg = priority_iteration_dmg;
const double reward      = current_dmg - rl_last_priority_iteration_dmg;
rl_last_priority_iteration_dmg = current_dmg;
```

**Problem:** Reward is based solely on damage output, ignoring healing, tanking, and support roles.

**Impact:**
- Cannot train tank or healer agents
- Ignores defensive cooldown usage
- No consideration for survival vs damage trade-offs

**Recommendation:** Add configurable reward functions for different roles. Consider:
- Tank: survival time, threat generation, damage taken
- Healer: healing output, mana efficiency, raid member survival
- DPS: damage output (current), target priority switching

---

### 12. **Spec-Specific Configurations are Hard-Coded**

**Location:** `rl_spec_config.cpp:29-133`

```cpp
g_spec_configs[ DRUID_FERAL ] = {
    { "cat_form", "bear_form", "moonkin_form", "tigers_fury", ... },
    { "rip", "rake", "thrash_cat", ... },
    RESOURCE_COMBO_POINT
};
```

**Problem:** Spec configurations are hard-coded in C++, requiring recompilation for changes.

**Impact:**
- Cannot experiment with different buff/dot tracking without rebuilding
- Adding new specs requires C++ changes
- No way to customize observation space for experimentation

**Recommendation:** Consider loading spec configs from JSON/YAML files or making them runtime-configurable.

---

### 13. **Action Discovery May Miss Dynamic Actions**

**Location:** `rl_interface.cpp:152-253`

```cpp
std::set<std::string> discover_class_actions( const player_t& player )
{
  // Discovers actions from DBC data and talent trees
  // But may miss dynamically-created actions
}
```

**Problem:** DBC-based discovery may not capture all available actions, especially those created dynamically based on gear, buffs, or temporary effects.

**Impact:**
- Some abilities may not be available to the RL agent
- Trinket effects and temporary abilities might be missed
- Convoke-style effects that grant temporary spells won't be discovered

**Recommendation:** Add runtime action discovery that updates when new actions become available.

---

## Minor Issues and Suggestions

### 14. **Inconsistent Comment Style**

**Problem:** Mix of C++ (`//`) and documentation (`///`) comment styles.

**Recommendation:** Use Doxygen-style (`///`) consistently for API documentation.

---

### 15. **Magic Numbers in Reward Shaping**

**Location:** `player.cpp:14390`

```cpp
input.reward = rl::compute_shaped_reward( reward, rl_prev_observation, input.observation, 0.99, this );
```

**Problem:** Gamma value (0.99) is hard-coded.

**Recommendation:** Make gamma configurable via sim options.

---

### 16. **Limited Spec Coverage**

**Location:** `rl_spec_config.cpp`

**Problem:** Only Feral Druid, Guardian Druid, and Destruction Warlock have configurations.

**Impact:** Limited applicability to other specs.

**Recommendation:** Add configurations for more popular specs or provide a template/generator.

---

### 17. **No Validation of Action Space Stability**

**Problem:** The Python bridge assumes action space size is stable across episodes, but there's no validation that spec, talents, or gear haven't changed.

**Impact:** Model trained on one configuration will fail silently on another.

**Recommendation:** Add action space fingerprinting/validation.

---

### 18. **Stderr Handling Could Miss Early Errors**

**Location:** `rl_bridge.py:250-260`

```python
def _stderr_reader(self) -> None:
    """Background thread that reads stderr and accumulates output."""
    try:
        if self._process and self._process.stderr:
            for line in self._process.stderr:
                with self._stderr_lock:
                    self._stderr_buffer.append(line)
```

**Problem:** Background thread might not capture early initialization errors before the main thread reads the first message.

**Recommendation:** Add a small delay or synchronization to ensure early errors are captured.

---

## Design Concerns

### 19. **RL and APL are Orthogonal, Not Integrated**

**Observation:** The RL system completely bypasses APL evaluation when enabled, rather than augmenting it.

**Concern:** This prevents hybrid approaches where:
- APL handles routine rotations
- RL handles cooldown optimization
- APL provides fallback when RL is uncertain

**Suggestion:** Consider a hybrid mode where APL and RL can cooperate.

---

### 20. **No Support for Multi-Target or Cleave Scenarios**

**Problem:** Observation only tracks a single target's dots and TTD.

**Impact:** Cannot learn optimal target switching or AoE priority.

**Recommendation:** Extend observation to include multiple targets or aggregated AoE metrics.

---

### 21. **Limited Observability for Timing Windows**

**Problem:** No explicit observation of:
- Upcoming major cooldowns
- Enemy mechanic timers
- Optimal burst window indicators

**Impact:** Agent must learn these implicitly from damage patterns.

**Recommendation:** Add features for upcoming cooldowns and important timing windows.

---

## Positive Aspects

Despite the issues identified, the implementation has several strengths:

1. **Clean Separation of Concerns:** The RL system is well-isolated in its own namespace
2. **Gymnasium Compatibility:** Standard RL interface makes it easy to use existing algorithms
3. **Action Masking:** Proper handling of illegal actions via masking
4. **Extensible Design:** Easy to add new specs via spec_obs_config
5. **Comprehensive Documentation:** README.md is thorough and helpful
6. **Reward Shaping Support:** Infrastructure for potential-based shaping is in place
7. **Trace Logging:** JSONL tracing aids debugging and analysis

---

## Recommendations Priority

### High Priority (Correctness/Safety)
1. Fix global policy state (Issue #1)
2. Change RL default to disabled (Issue #3)
3. Remove unsafe const-cast (Issue #2)
4. Fix hard-coded Python blacklist (Issue #4)

### Medium Priority (Functionality)
5. Improve memory management documentation (Issue #5)
6. Extend reward function for non-DPS roles (Issue #11)
7. Fix pass pseudo-action semantics (Issue #10)
8. Add action space validation (Issue #17)

### Low Priority (Polish)
9. Make normalization constants configurable (Issue #9)
10. Improve error handling (Issue #8)
11. Use consistent comment style (Issue #14)
12. Make gamma configurable (Issue #15)

---

## Conclusion

The RL implementation is a solid foundation for training agents to play SimulationCraft characters. The core architecture is sound and the integration with the player selection loop is well-designed. However, several critical issues need to be addressed before this can be considered production-ready:

- **Global state management** needs to be refactored for thread safety and proper encapsulation
- **Default behavior** should not enable experimental features
- **Action filtering** should be consistent between C++ and Python
- **const-correctness** violations should be resolved

With these issues addressed, the RL implementation would be a powerful tool for exploring optimal rotations and cooldown usage patterns.

---

## Appendix: Testing Recommendations

To improve confidence in the implementation:

1. **Unit Tests:** Test action filtering, observation building, reward computation
2. **Integration Tests:** Test full episodes with known-good policies
3. **Validation:** Compare RL-trained agents against hand-crafted APLs
4. **Stress Testing:** Multi-threaded scenarios, long episodes, edge cases
5. **Performance Profiling:** Measure overhead of RL decision-making

---

*Document generated: 2026-01-03*
*Reviewer: GitHub Copilot Agent*
*Codebase: SimulationCraft RL Implementation (commit: HEAD)*
