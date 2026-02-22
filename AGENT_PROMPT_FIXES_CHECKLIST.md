# Agent Prompt Fixes Checklist

This checklist documents issues identified in the agent prompt system and provides actionable fixes.

---

## 1. Stage 8 ("Reflector & Depth Critic") Underspecified

**Status:** [x] Done

### Problem
Stage 8 exists in `build_stage_specs()` (lines 1368-1377 in `journal_editor_coach.py`) but has:
- No defined JSON output schema
- Vague criteria ("deep enough")
- Never called by OrchestratorAgent (only stages 1-7 are processed)

```python
# Current underspecified prompt:
"system_prompt": """You are the Referee who reads the entire blackboard and challenges weak or shallow findings.
For every major finding from previous agents, ask: "Is this deep enough for an ASAC editor? What evidence is missing?"
Then write improved, deeper versions and post them back to the blackboard.
Output JSON with "improved_findings" list."""
```

### Fix
Option A: Implement Stage 8 properly with a complete JSON schema:

```python
"system_prompt": """IMPORTANT: You have already seen this Paper Structure Map from the desk screen:
{STRUCTURE_MAP_PLACEHOLDER}

You are a senior ASAC referee performing a final depth review of all stage findings.

**CRITERIA**:
1. **Evidence Quality**: Are quotes sufficient, accurate, and fairly represented?
2. **Reasoning Depth**: Does each finding explain WHY it matters editorially?
3. **Blind Spots**: What did previous stages miss or under-analyze?
4. **Actionability**: Are recommendations specific enough to act on?

**PROCESS**:
Review each stage's major findings. For any that are shallow, missing evidence, or underspecified:
- Quote the original finding
- Explain what's missing or weak
- Provide an improved, deeper version

**OUTPUT JSON**:
{
  "depth_score": 1-10,
  "stage_reviews": [
    {
      "stage_id": 1,
      "stage_name": "Stage name",
      "original_finding": "Quote from the stage output",
      "weakness": "Why this finding is shallow or incomplete",
      "improved_finding": "Deeper, more rigorous version with better evidence"
    }
  ],
  "overall_blind_spots": ["Topics completely missed by all stages"],
  "final_editorial_verdict": "One paragraph summarizing whether this review is rigorous enough for ASAC"
}
```

Option B: Remove Stage 8 entirely if not needed.

### How to Apply
1. Open `journal_editor_coach.py`
2. Locate `build_stage_specs()` function (around line 1092)
3. Find Stage 8 definition (lines 1368-1377)
4. Replace with the improved prompt above (Option A) OR delete the stage (Option B)
5. If keeping Stage 8, update `OrchestratorAgent.act()` to include stage 8 in the loop (line 1040)

---

## 2. Inconsistent JSON Schemas Across Stages

**Status:** [ ] Not Started | [ ] In Progress | [ ] Done

### Problem
Each stage uses different decision field names:
- Stage 1: `decision` (PASS/FAIL/WARNING)
- Stage 2: `decision` (STRONG FIT/WEAK FIT/OFF TOPIC)
- Stage 3: `theoretical_contribution` (High/Medium/Low)
- Stage 4: `method_rigor` (High/Medium/Low)
- Stage 5: `readability_score` (High/Medium/Low)
- Stage 6: `final_recommendation` (ACCEPT/MINOR REVISE/MAJOR REVISE/REJECT)
- Stage 7: `coherence_score` (1-10)

This makes:
- Progress display inconsistent
- Report generation complex (`generate_professional_report` lines 1535-1620)
- Downstream processing fragile

### Fix
Add a standardized `stage_decision` field to ALL stages while keeping domain-specific fields:

```python
# Add to every stage's OUTPUT JSON:
"stage_decision": "PASS" | "MINOR_REVISION" | "MAJOR_REVISION" | "FAIL",
```

Mapping:
- Stage 1: PASS/FAIL/WARNING → PASS/MINOR_REVISION/MAJOR_REVISION/FAIL
- Stage 2: STRONG FIT/WEAK FIT/OFF TOPIC → PASS/MINOR_REVISION/MAJOR_REVISION/FAIL
- Stage 3-5: High/Medium/Low → PASS/MINOR_REVISION/MAJOR_REVISION/FAIL
- Stage 6: Keep `final_recommendation` as source of truth
- Stage 7: coherence_score 8-10=PASS, 5-7=MINOR_REVISION, 3-4=MAJOR_REVISION, 1-2=FAIL

### How to Apply
1. Open `journal_editor_coach.py`
2. For each stage spec in `build_stage_specs()`, add `"stage_decision"` to the OUTPUT JSON
3. Update `generate_professional_report()` to use standardized field
4. Test with a sample manuscript

---

## 3. Hardcoded ASAC Theme String

**Status:** [ ] Not Started | [ ] In Progress | [ ] Done

### Problem
The conference theme is hardcoded at line 46:
```python
ASAC_THEME = "Blue Sky Thinking: Harnessing the Future"
```

This requires code changes every year when the theme changes.

### Fix
Make the theme configurable via:
1. Command-line argument
2. Config file
3. Environment variable

### How to Apply
1. Open `journal_editor_coach.py`
2. Add argparse argument in `main()`:
   ```python
   parser.add_argument("--theme", default=os.environ.get("ASAC_THEME", "Blue Sky Thinking: Harnessing the Future"),
                       help="ASAC conference theme for Stage 2 evaluation")
   ```
3. Pass theme to `build_stage_specs()` or inject into Stage 2 prompt
4. Update Stage 2 system prompt to use `{THEME_PLACEHOLDER}` instead of hardcoded text
5. Optionally create `config.json`:
   ```json
   {
     "asac_theme": "Blue Sky Thinking: Harnessing the Future",
     "conference_year": 2026
   }
   ```

---

## 4. Editorial Mentor Questions All Focus on "whole"

**Status:** [ ] Not Started | [ ] In Progress | [ ] Done

### Problem
All questions in `editorial_mentor_config.json` have `"focus": "whole"`, which means the mentor always sees the entire manuscript trimmed to budget. This wastes context and reduces specificity.

The system supports section-specific focus: `abstract`, `introduction`, `methods`, `results`, `discussion`.

### Fix
Update question focuses to target relevant sections:

```json
{
  "id": "strongest_weakest",
  "label": "Strongest and weakest section",
  "question": "Which section is strongest and why? Which section needs the most work?",
  "focus": "whole"
},
{
  "id": "method_clarity",
  "label": "Method section clarity",
  "question": "Is the methodology described clearly enough to replicate? What's missing?",
  "focus": "methods"
},
{
  "id": "discussion_impact",
  "label": "Discussion impact",
  "question": "Do the implications follow logically from the findings? What practical takeaways are missing?",
  "focus": "discussion"
}
```

### How to Apply
1. Open `editorial_mentor_config.json`
2. Review each question and determine which section it targets
3. Update `"focus"` field to appropriate value
4. Add new section-specific questions where helpful

---

## 5. Socratic Topic Resolution Too Simple

**Status:** [ ] Not Started | [ ] In Progress | [ ] Done

### Problem
In `socratic_mentor.py`, topics are marked resolved after a single exchange (lines 186-195):
```python
def is_topic_resolved(self) -> bool:
    if self.current_topic_idx >= len(self.topics):
        return True
    return self.topics[self.current_topic_idx].get("discussion_done", False)

def mark_topic_resolved(self):
    if self.current_topic_idx < len(self.topics):
        self.topics[self.current_topic_idx]["discussion_done"] = True
    self.current_topic_idx += 1
```

This means even if the user gives a shallow 2-word response, the topic is "done" and moves on.

### Fix
Implement smarter resolution detection:
1. Track response quality/length
2. Count follow-up questions asked
3. Let LLM judge if topic is exhausted

### How to Apply
1. Open `socratic_mentor.py`
2. Modify `is_topic_resolved()` to check:
   ```python
   def is_topic_resolved(self) -> bool:
       if self.current_topic_idx >= len(self.topics):
           return True
       topic = self.topics[self.current_topic_idx]

       # Require minimum engagement
       exchange_count = len([h for h in self.dialogue_history
                            if topic["stage_name"] in str(h)])

       if exchange_count < 1:
           return False  # At least one exchange required

       # Let LLM judge if deeper discussion needed
       if exchange_count >= 2:
           return topic.get("discussion_done", False)

       return topic.get("discussion_done", False)
   ```

3. Add user option to continue exploring topic:
   ```python
   # After mentor reply, ask:
   continue_topic = Prompt.ask("[dim]Explore this topic further? [y/N][/dim]", default="n")
   if continue_topic.lower() != "y":
       dialogue.mark_topic_resolved()
   ```

---

## 6. Temperature and Top_p Conflict

**Status:** [ ] Not Started | [ ] In Progress | [ ] Done

### Problem
In `JournalEditorCoach.Modelfile` (lines 5-7):
```
PARAMETER temperature 0.55
PARAMETER top_p 0.92
```

These sampling methods conflict. When using temperature sampling, top_p has reduced effect. Using both adds complexity without clear benefit.

### Fix
Choose one primary sampling method:
- **Option A (Temperature-focused):** Remove top_p or set to 1.0
- **Option B (Nucleus-focused):** Set temperature to 1.0 and rely on top_p

For editorial work requiring consistency, Option A is recommended.

### How to Apply
1. Open `JournalEditorCoach.Modelfile`
2. Option A: Remove `PARAMETER top_p 0.92` or set to 1.0
3. Recreate model: `ollama create JournalEditorCoach -f JournalEditorCoach.Modelfile`
4. Test with sample manuscript to compare output quality

---

## 7. Fact Verification Prompt Defined But Never Used

**Status:** [ ] Not Started | [ ] In Progress | [ ] Done

### Problem
`FACT_VERIFICATION_PROMPT` is defined (lines 111-146) but never called in the review pipeline. This is a valuable feature that could catch hallucinated quotes or misrepresentations.

### Fix
Integrate fact verification as optional Stage 9 or as a post-processing step.

### How to Apply
1. Open `journal_editor_coach.py`
2. Add CLI flag: `--verify-facts`
3. In `main()`, after all stages complete, optionally run verification:
   ```python
   if args.verify_facts:
       rprint("[bold cyan]Running fact verification pass...[/bold cyan]")
       verification_result = run_fact_verification(full_text, results, MODEL_NAME)
       blackboard.set("fact_verification", verification_result)
   ```
4. Implement `run_fact_verification()` using the existing prompt
5. Include verification results in final report

---

## 8. Missing Quote Length Enforcement

**Status:** [ ] Not Started | [ ] In Progress | [ ] Done

### Problem
The Modelfile specifies max 35 words for quotes, but this isn't consistently enforced in stage prompts. Some stages may output longer quotes.

### Fix
Add explicit character/word limit to all stage prompts' quote fields.

### How to Apply
1. Open `journal_editor_coach.py`
2. In each stage's OUTPUT JSON schema, add note:
   ```json
   "quote": "Exact text snippet (MAX 35 WORDS)",
   ```
3. Optionally add post-processing to truncate quotes:
   ```python
   def enforce_quote_limit(quote: str, max_words: int = 35) -> str:
       words = quote.split()
       if len(words) > max_words:
           return " ".join(words[:max_words]) + "..."
       return quote
   ```

---

## 9. No Retry on Malformed Socratic Questions

**Status:** [ ] Not Started | [ ] In Progress | [ ] Done

### Problem
In `socratic_mentor.py`, if LLM fails to generate a valid question, it falls back to a generic template without retry:
```python
question = (response.get("message") or {}).get("content", "").strip()
return question or q  # Falls back to template if empty
```

### Fix
Add retry logic similar to `ReviewAgent`:

### How to Apply
1. Open `socratic_mentor.py`
2. Modify `next_question()` method:
   ```python
   def next_question(self, max_retries: int = 2) -> str:
       for attempt in range(max_retries):
           # ... existing LLM call ...
           if question and len(question) > 10:  # Valid question
               return question
           time.sleep(1)
       return q  # Fallback after retries
   ```

---

## Summary Table

| # | Issue | Priority | Effort | Files Affected |
|---|-------|----------|--------|----------------|
| 1 | Stage 8 underspecified | High | Medium | `journal_editor_coach.py` |
| 2 | Inconsistent JSON schemas | High | High | `journal_editor_coach.py` |
| 3 | Hardcoded theme | Medium | Low | `journal_editor_coach.py` |
| 4 | Mentor focus "whole" | Medium | Low | `editorial_mentor_config.json` |
| 5 | Socratic topic resolution | Medium | Medium | `socratic_mentor.py` |
| 6 | Temperature/top_p conflict | Low | Low | `JournalEditorCoach.Modelfile` |
| 7 | Unused fact verification | Medium | Medium | `journal_editor_coach.py` |
| 8 | Quote length enforcement | Low | Low | `journal_editor_coach.py` |
| 9 | Socratic retry missing | Low | Low | `socratic_mentor.py` |

---

## Recommended Implementation Order

1. **Fix #2** (JSON schemas) - Foundation for everything else
2. **Fix #3** (Configurable theme) - Quick win, annual maintenance
3. **Fix #1** (Stage 8) - Either implement or remove
4. **Fix #7** (Fact verification) - Adds quality assurance
5. **Fix #4** (Mentor focus) - Improves mentor effectiveness
6. **Fix #5** (Socratic resolution) - Better learning experience
7. **Fixes #6, #8, #9** - Polish items
