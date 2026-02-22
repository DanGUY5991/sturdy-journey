# Plan: Qualitative vs Quantitative Paper Review Path

## Overview

Add a research-design-aware review flow:
1. **Classify** the paper as Qualitative or Quantitative (path choice)
2. **Methodology review** — check alignment with the chosen paradigm
3. **Findings review** — accuracy checks by path: mathematical for quantitative, semantic for qualitative

Once one style is identified, use only that path (no dual evaluation).

---

## Current State

| Script | Architecture | Relevant Sections/Stages |
|--------|--------------|--------------------------|
| **journal_editor_coach_20-30page.py** | 8 flat sections | Setup → Title & Abstract … Methods (idx 3), Results (idx 4) … |
| **journal_editor_coach.py** | 8 stages | Stage 1 (Compliance), Stage 4 (Methodology & Analysis) |

---

## Implementation Steps (do one at a time)

### Step 1: Extend setup pass to classify paper type

**File:** `journal_editor_coach_20-30page.py`

- Add `paper_type` to the setup JSON: `"QUALITATIVE" | "QUANTITATIVE" | "MIXED"`
- Update `SETUP_PAPER_PROMPT` to ask for classification and evidence
- Change `run_setup_pass()` to return `(global_mem: str, paper_type: str)` instead of just `str`
- Store `paper_type` in a variable used by later steps
- Update `main()` to unpack and pass `paper_type` through the flow

**Prompt addition:**
```json
"paper_type": "QUALITATIVE" | "QUANTITATIVE" | "MIXED",
"paper_type_evidence": "1–2 sentences: why this classification (e.g. survey/experiment vs interviews/thematic analysis)"
```

---

### Step 2: Add paper_type to global context for section prompts

**File:** `journal_editor_coach_20-30page.py`

- Append `paper_type` (and optionally evidence) to `global_mem` so every section sees it
- Example: `global_mem += f" Paper type: {paper_type}. {evidence}"`
- No change to section logic yet; just expose the classification

---

### Step 3: Methodology section — paradigm alignment

**File:** `journal_editor_coach_20-30page.py`

- For section "Methods (rigor + replicability)" (SECTIONS_CHECKLIST index 3):
  - Add `paper_type` to `process_section()` (new optional param)
  - If `paper_type` is QUALITATIVE: inject instructions for trustworthiness, coding, saturation, reflexivity
  - If `paper_type` is QUANTITATIVE: inject instructions for validity, reliability, sampling, statistics
  - If MIXED: include both but flag which parts use which paradigm
- Update `SYSTEM_PROMPT` or add a `METHODOLOGY_PARADIGM_PROMPT` block that varies by `paper_type`

**Logic:**
- QUALITATIVE: "Evaluate methodology for qualitative rigor: trustworthiness, thematic/coding process, saturation, reflexivity, alignment with qualitative research question"
- QUANTITATIVE: "Evaluate methodology for quantitative rigor: validity, reliability, sampling, statistical design, alignment with hypotheses/quantitative research question"

---

### Step 4: Results section — path-specific accuracy review

**File:** `journal_editor_coach_20-30page.py`

- For section "Results & Evidence Alignment" (index 4):
  - Pass `paper_type` into the prompt
  - **QUANTITATIVE path:** "Review findings for mathematical/statistical accuracy: do reported stats match methods, are calculations/effect sizes/confidence intervals consistent, are claims supported by the numbers?"
  - **QUALITATIVE path:** "Review findings for semantic accuracy: do themes/codes connect logically to data excerpts, are quotes used fairly, is the analytic process traceable?"
  - MIXED: evaluate quantitative results with math lens and qualitative results with semantic lens
- Add `paper_type` to the `process_section()` call for Methods and Results

---

### Step 5: Refactor process_section to accept paper_type

**File:** `journal_editor_coach_20-30page.py`

- Add `paper_type: Optional[str] = None` to `process_section()`
- When `paper_type` is set and section is Methods or Results, prepend path-specific instructions to the user prompt
- Ensure other sections (Title, Theory, Discussion, etc.) still receive `paper_type` in global_mem for coherence, but use the standard evaluation

---

### Step 6: Journal editor coach (7-stage) — add paper_type to Stage 1

**File:** `journal_editor_coach.py`

- Extend Stage 1 (Compliance & Formatting) output JSON with `paper_type: "QUALITATIVE" | "QUANTITATIVE" | "MIXED"`
- Store in blackboard: `blackboard.set("paper_type", result.get("paper_type", ""))`

---

### Step 7: Journal editor coach — Stage 4 Methodology uses paper_type

**File:** `journal_editor_coach.py`

- In Stage 4 (Methodology & Analysis) system prompt, inject `paper_type` from blackboard
- Add conditional instructions: if QUALITATIVE, emphasize qualitative criteria; if QUANTITATIVE, emphasize quantitative criteria
- Update `_build_user_prompt` or stage spec to include paper_type context

---

### Step 8: Journal editor coach — Findings accuracy by path

**File:** `journal_editor_coach.py`

- Stage 4 already covers Methods + Results. Either:
  - Split into Methodology-only and Results-only sub-evaluations when paper_type is set, or
  - Enhance the single Stage 4 prompt with path-specific Results criteria (math vs semantic) based on paper_type

---

### Step 9: Report generation — show paper type and path

**Files:** `journal_editor_coach_20-30page.py`, `journal_editor_coach.py`

- Add a "Paper type" / "Review path" line to the dashboard or summary: e.g. "Qualitative" or "Quantitative"
- Include in `*_editor_review.md` and `*_ASAC_Report.json` when available

---

### Step 10: Fact verification — path-aware checks

**Files:** Both scripts

- When fact-checking Results findings, use paper_type:
  - QUANTITATIVE: flag mismatches between stated numbers and methods (e.g. wrong test, missing correction)
  - QUALITATIVE: flag misuse of quotes or thematic leaps without evidence

---

## Summary Table

| Step | Description | Script(s) |
|------|-------------|-----------|
| 1 | Setup pass returns paper_type | 20-30page |
| 2 | Add paper_type to global_mem | 20-30page |
| 3 | Methodology section — paradigm alignment | 20-30page |
| 4 | Results section — path-specific accuracy | 20-30page |
| 5 | Refactor process_section for paper_type | 20-30page |
| 6 | Stage 1 outputs paper_type | journal_editor_coach |
| 7 | Stage 4 Methodology uses paper_type | journal_editor_coach |
| 8 | Stage 4 Results accuracy by path | journal_editor_coach |
| 9 | Report shows paper type | Both |
| 10 | Fact verification path-aware | Both |

---

## Path Logic (reminder)

- **One path only:** Once paper_type = QUALITATIVE or QUANTITATIVE, use that paradigm. Do not evaluate Results with both math and semantic criteria—use the one that matches.
- **MIXED:** Apply quantitative criteria to quantitative parts and qualitative criteria to qualitative parts; the prompt should distinguish which sections/findings are which.
