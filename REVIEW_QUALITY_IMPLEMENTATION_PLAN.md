# Plan: Add Review Time & Quality Controls

This document is the implementation plan for the tips that increase review time and editorial quality (deeper reasoning, mandatory blackboard, optional reflection). **All checklist items below have been implemented.**

---

## Summary of changes

| Phase | What was done | Files |
|-------|----------------|-------|
| **1** | Ollama parameters updated for longer, more rigorous output | Both scripts + Modelfile |
| **2** | Deep Thinking Protocol added and injected into every prompt | Both scripts |
| **3** | Blackboard made mandatory via `read_full_blackboard()` | `journal_editor_coach.py` |
| **4** | Optional reflection pass (second LLM call per stage) | `journal_editor_coach.py` |

---

## Phase 1: Ollama parameters ✅

**Goal:** Allow ~1200–2500+ tokens of reasoning per stage and slightly lower creativity for editorial rigor.

**Changes:**

- **`journal_editor_coach.py`** and **`journal_editor_coach_20-30page.py`**  
  - `num_ctx`: 4096 → **8192**  
  - `num_predict`: 950 → **2800**  
  - `temperature`: 0.65 → **0.55**  
  - Added **`top_p`: 0.92**, **`repeat_penalty`: 1.12**

- **`JournalEditorCoach.Modelfile`**  
  - Same parameter updates so the custom model matches.  
  - **Action required:** Re-create the model:
    ```bash
    ollama create JournalEditorCoach -f JournalEditorCoach.Modelfile
    ```

---

## Phase 2: Deep Thinking Protocol ✅

**Goal:** Force a structured 6-step reasoning process before any JSON output.

**Changes:**

- **`journal_editor_coach.py`**
  - New constant **`DEEP_THINKING_PROTOCOL`** (after `REVIEWER_ARTICLE_FIRST_INSTRUCTION`) with the 6 steps: Paper-Level Understanding → Section Claims → Rigorous Evaluation → Cross-Check & Blackboard → Self-Critique → Synthesis.
  - **`_build_user_prompt()`** now builds the user message as:  
    **Full blackboard** → **DEEP_THINKING_PROTOCOL** → **REVIEWER_ARTICLE_FIRST_INSTRUCTION** → global context → stage task → manuscript content.

- **`journal_editor_coach_20-30page.py`**
  - Same **`DEEP_THINKING_PROTOCOL`** constant (with “Cross-Check & Context” for the 20–30 page flow).
  - **`SYSTEM_PROMPT`** is now **`DEEP_THINKING_PROTOCOL +`** existing system prompt text, so every section review gets the protocol at the top.

---

## Phase 3: Blackboard mandatory & logging ✅

**Goal:** Every agent explicitly reads the full blackboard before thinking, and we log that it happened.

**Changes in `journal_editor_coach.py`:**

- **`Blackboard`**
  - New method **`read_full_blackboard()`** that returns a single string with:
    - Previous stage summaries (`get_context_log(2500)`)
    - Orchestrator notes
    - Last 3 editor board posts (if any; key `editor_board_posts` used via `.get(..., [])`)

- **`ReviewAgent.act()`**
  - Right after `"Starting Stage {id}..."`:
    - Call **`read_full_blackboard()`** and assign to a variable (used when building the user prompt).
    - Log: **"Read full blackboard (Xk chars) — forcing deep context"**.

- **`_build_user_prompt()`**
  - User prompt now starts with **`read_full_blackboard()`** output, so the model always sees the full blackboard first.

---

## Phase 4: Optional reflection pass ✅

**Goal:** One extra LLM call per stage to refine the draft analysis (adds ~30–90 s per stage, improves depth).

**Changes in `journal_editor_coach.py`:**

- New constant **`ENABLE_REFLECTION_PASS = True`** (set to **`False`** to disable).
- After a stage produces valid JSON:
  - If **`ENABLE_REFLECTION_PASS`** and result has **`stage_name`**:
    - Build **`reflect_prompt`**: “Here is my draft analysis: \<JSON\>. Improve it. Make the reasoning deeper and more editorial. Output improved JSON only.”
    - Call **`ollama.chat`** with the **same** `system_prompt` and this user message.
    - Parse JSON from the response; if valid, set **`result = improved`** and keep **`stage_name`**.
    - Log **"Reflection pass applied"** or, on exception, **"Reflection pass skipped: \<error\>"**.

---

## Testing checklist (your steps)

- [ ] Re-create custom model:  
  `ollama create JournalEditorCoach -f JournalEditorCoach.Modelfile`
- [ ] Run one paper with **`--no-cleanup-env`** (e.g. 20–30 page script or main 7-stage script).
- [ ] In the console, confirm:
  - “Read full blackboard (Xk chars) — forcing deep context” each stage (main script).
  - Each stage taking roughly **6–18 minutes** instead of 1–4 (depending on hardware and paper length).
- [ ] Inspect **Evidence_Trail.md** and **Review_Step*.md**: expect 2–3× more reasoning, clearer cross-section links, and a more “ASAC editor” tone.

---

## Summary checklist (all implemented in code)

- [x] Update **OLLAMA_OPTIONS** in both scripts + Modelfile (num_predict 2800, temp 0.55, ctx 8192, top_p, repeat_penalty).
- [x] Add **DEEP_THINKING_PROTOCOL** constant.
- [x] Inject protocol + full blackboard read into every prompt (main script: user prompt; 20–30 page: system prompt).
- [x] Add **`read_full_blackboard()`** and use it at the start of each stage + in user prompt.
- [x] Add optional **reflection pass** (controlled by **`ENABLE_REFLECTION_PASS`**).
- [ ] **You:** Re-create **JournalEditorCoach** model.
- [ ] **You:** Test on one 20–30 page paper and confirm longer, richer reviews.

These changes are additive: existing structure, JSON output, and resume logic are unchanged; they only encourage the model to spend more time and follow a strict editorial thinking protocol.
