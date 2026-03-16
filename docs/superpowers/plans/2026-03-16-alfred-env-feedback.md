# ALFRED Env Feedback Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make invalid `env_feedback` in ALFRED explain what failed, which target was involved, what the target state was, and what the next likely recovery action should be.

**Architecture:** Keep `EBAlfEnv.get_env_feedback()` as the outer renderer and enrich the failure payloads produced inside `ThorConnector`. Add a small set of formatting helpers in `thor_connector.py`, then route named-object actions and point-based actions through them so existing calling code keeps working.

**Tech Stack:** Python, stdlib `unittest`, lightweight import stubs for missing simulation dependencies

---

## Chunk 1: Tests And Failure Formatter

### Task 1: Add failing tests for detailed ALFRED invalid feedback

**Files:**
- Create: `tests/envs/eb_alfred/test_feedback_messages.py`
- Modify: `embodiedbench/envs/eb_alfred/thor_connector.py`
- Modify: `embodiedbench/envs/eb_alfred/EBAlfEnv.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:
- `open_by_point()` failure includes reason code, target object, point, search radius, object state, raw AI2-THOR error, and a recovery hint.
- `open()` failure by object name includes target object, object id, object state, raw AI2-THOR error, and a recovery hint.
- `EBAlfEnv.get_env_feedback()` preserves the richer failure text inside the existing `Last action is invalid.` wrapper.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.envs.eb_alfred.test_feedback_messages -v`
Expected: FAIL because the richer ALFRED failure formatting does not exist yet.

- [ ] **Step 3: Write minimal implementation**

In `embodiedbench/envs/eb_alfred/thor_connector.py`:
- add helpers to summarize target objects and object state
- add a shared formatter for interaction failures
- update ALFRED invalid-action paths to use the shared formatter

In `embodiedbench/envs/eb_alfred/EBAlfEnv.py`:
- keep the existing wrapper text and ensure richer messages pass through unchanged

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.envs.eb_alfred.test_feedback_messages -v`
Expected: PASS
