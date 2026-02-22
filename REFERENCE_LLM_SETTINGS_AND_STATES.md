# Reference: Local LLM Settings & System States for Optimization

This document provides a comprehensive reference for all local LLM settings and system states used across the ASAC Academic Reviewer application. Use this guide when optimizing performance, debugging, or modifying the system.

---

## Table of Contents

1. [Local LLM Configuration](#local-llm-configuration)
2. [System States & Memory Management](#system-states--memory-management)
3. [Optimization Considerations](#optimization-considerations)
4. [Hardware Constraints](#hardware-constraints)

---

## Local LLM Configuration

### Model Names

| Script | Model Name | Base Model | Notes |
|--------|-----------|------------|-------|
| `journal_editor_coach.py` | `qwen2.5:14b-instruct-q5_k_m` | N/A | Direct model usage |
| `journal_editor_coach_20-30page.py` | `JournalEditorCoach` | `qwen2.5:14b-instruct-q5_k_m` | Custom model with Modelfile |
| `socratic_mentor.py` | `qwen2.5:14b-instruct-q5_k_m` | N/A | Inherits from `journal_editor_coach` |
| `editorial_mentor_review.py` | `qwen2.5:14b-instruct-q5_k_m` | N/A | Inherits from `journal_editor_coach` |

**Custom Model Creation:**
```bash
ollama create JournalEditorCoach -f JournalEditorCoach.Modelfile
```

### Ollama Options (OLLAMA_OPTIONS)

#### journal_editor_coach.py (7-stage review)
```python
OLLAMA_OPTIONS = {
    "num_ctx": 8192,           # Context window size (tokens)
    "num_predict": 2000,        # Max tokens to generate
    "temperature": 0.6,         # Creativity/randomness (0.0-1.0)
    "num_gpu": 999,             # Use all available GPU layers
    "top_p": 0.95,              # Nucleus sampling threshold
    "repeat_penalty": 1.1       # Penalty for repetition
}
```

#### journal_editor_coach_20-30page.py (20-30 page support)
```python
OLLAMA_OPTIONS = {
    "num_ctx": 8192,            # Doubled context window
    "num_predict": 2800,        # Increased for longer reasoning
    "temperature": 0.55,        # Lower = more focused editor thinking
    "num_gpu": 999,
    "top_p": 0.92,              # Slightly lower for rigor
    "repeat_penalty": 1.12      # Higher penalty for repetition
}
```

#### JournalEditorCoach.Modelfile Parameters
```modelfile
PARAMETER num_ctx 8192
PARAMETER temperature 0.55
PARAMETER num_predict 2800
PARAMETER top_p 0.92
PARAMETER repeat_penalty 1.12
```

### Parameter Impact on Performance

| Parameter | Current Value | Impact if Increased | Impact if Decreased | Optimization Notes |
|-----------|--------------|---------------------|---------------------|-------------------|
| `num_ctx` | 8192 | More context, higher memory | Less context, lower memory | **Critical**: B580 12GB limit |
| `num_predict` | 2000-2800 | Longer outputs, slower | Shorter outputs, faster | Balance quality vs speed |
| `temperature` | 0.55-0.6 | More creative, less consistent | More deterministic | Lower for editorial rigor |
| `top_p` | 0.92-0.95 | More diverse tokens | More focused tokens | Lower for precision |
| `repeat_penalty` | 1.1-1.12 | Less repetition | More repetition | Higher reduces redundancy |

### Ollama Server Configuration

**Default Paths:**
- `OLLAMA_DIR`: `F:\ollama-ipex-llm-2.2.0-win` (Windows)
- `OLLAMA_SERVE_PATH`: `{OLLAMA_DIR}\ollama-serve.bat`
- Server URL: `http://localhost:11434`

**Environment Variables:**
```bash
set OLLAMA_DIR=F:\ollama-ipex-llm-2.2.0-win
set OLLAMA_SERVE_PATH=F:\ollama-ipex-llm-2.2.0-win\ollama-serve.bat
```

**Server Startup:**
- Automatic cleanup kills existing `ollama.exe` processes
- Waits up to 30 seconds (15 attempts × 2s) for server readiness
- Uses IPEX-LLM build optimized for Intel GPUs

---

## System States & Memory Management

### Blackboard System (journal_editor_coach.py)

The `Blackboard` class maintains shared state across all review agents.

#### Key State Keys

| Key | Type | Purpose | Access Pattern |
|-----|------|---------|----------------|
| `sections_map` | `Dict[str, str]` | Manuscript sections by name | Read by all agents |
| `structure_map` | `str` | Paper structure from Stage 1 | Written by Stage 1, read by all |
| `stage_{N}_result` | `Dict` | Result from agent N | Written by agent N, read by later stages |
| `stage_{N}_chunks` | `List[str]` | Chunked content for long sections | Written when section exceeds token limit |
| `context_log` | `List[str]` | Running log of stage summaries | Appended by each agent |
| `orchestrator_notes` | `List[str]` | Orchestrator decisions/flags | Appended by orchestrator |
| `chunk_summaries` | `Dict[str, List[str]]` | Summary per section per chunk | For 20-30 page papers |
| `cross_chunk_contradictions` | `List[Dict]` | Detected inconsistencies | Written by Stage 7 |
| `running_themes` | `Dict[str, str]` | Key themes/terms tracked | Cross-section tracking |
| `current_stage` | `int` | Current stage ID | Set by active agent |
| `editor_board_posts` | `List[Dict]` | Standardized agent posts | All agents post here |

#### Context Log Management

- **Max chars for context log**: 1500-2000 chars (truncated from end)
- **Format**: Each entry is a stage summary (~400 chars)
- **Usage**: Agents read full blackboard before analysis

#### Memory Budgets

| Component | Budget | Notes |
|-----------|--------|-------|
| Context log | 1500-2000 chars | Truncated from end |
| Stage input | 2800-4000 tokens | Varies by stage |
| Section chunks | 2500 tokens | With 200 token overlap |
| Stage 7 input | 4000 tokens | Aggregates all prior stages |
| Stage 8 input | 6000 tokens | Full blackboard review |

### Progress Files

#### editor_progress.json (7-stage review)
```json
[
  {
    "stage_name": "Compliance & Formatting",
    "decision": "PASS",
    "paper_structure_map": "...",
    "violations": [...],
    ...
  },
  ...
]
```

**Resume Logic:**
- File checked at startup
- If all 7 stages complete → start fresh
- Otherwise → resume from last completed stage
- Progress saved after each stage

#### progress_20-30page.json (20-30 page review)
```json
{
  "results": [
    {
      "section_name": "Title & Abstract",
      "strengths": [...],
      "weaknesses": [...],
      "section_memory_summary": "...",
      "global_memory_summary": "..."
    },
    ...
  ],
  "global_mem": "Running summary of whole paper",
  "sec_mem": "Previous section summary"
}
```

**Resume Logic:**
- Loads `results`, `global_mem`, `sec_mem`
- Resumes from `len(results)` index
- Autosaves after each section

### Memory States (20-30 page script)

#### Global Memory (`global_mem`)
- **Purpose**: Running summary of entire manuscript
- **Max size**: 400 tokens (updated by each section)
- **Content**: Overall argument, key themes, cross-section connections
- **Update**: Each section appends/updates global summary

#### Section Memory (`sec_mem`)
- **Purpose**: Summary of previous section
- **Max size**: 200-250 tokens
- **Content**: Key findings, quotes, editorial reasoning
- **Update**: Set by each section's `section_memory_summary`

#### Memory Flow
```
Section 1 → global_mem (initial) + sec_mem (N/A)
Section 2 → global_mem (updated) + sec_mem (Section 1 summary)
Section 3 → global_mem (updated) + sec_mem (Section 2 summary)
...
```

### Chunking Strategy

#### When Chunking Occurs
- Section exceeds `max_input_tokens` (typically 2800)
- Chunks created with 200 token overlap
- First chunk sent to LLM with truncation note

#### Chunk Parameters
```python
max_tokens = 2500          # Per chunk
overlap_tokens = 200       # Overlap between chunks
max_input_tokens = 2800    # Total budget (includes system prompt overhead)
```

#### Chunk Storage
- Stored in `blackboard.set(f"stage_{N}_chunks", chunks)`
- Summaries stored in `chunk_summaries[section_key]`
- Cross-chunk contradictions tracked separately

---

## Optimization Considerations

### Performance Bottlenecks

1. **LLM Inference Time**
   - **Current**: ~30-90 seconds per stage (with reflection pass)
   - **Bottleneck**: Model size (14B parameters) + context window (8192)
   - **Optimization**: Reduce `num_predict` or disable reflection pass

2. **Context Window Management**
   - **Current**: 8192 tokens (safe for B580 12GB)
   - **Risk**: Exceeding causes OOM or fallback to CPU
   - **Optimization**: Reduce `num_ctx` or improve chunking

3. **Reflection Pass**
   - **Current**: Enabled (`ENABLE_REFLECTION_PASS = True`)
   - **Impact**: Adds 30-90 seconds per stage
   - **Optimization**: Set to `False` for faster reviews

4. **Chunking Overhead**
   - **Current**: 200 token overlap between chunks
   - **Impact**: Redundant processing for long sections
   - **Optimization**: Reduce overlap or improve summarization

### Memory Optimization

#### GPU Memory (B580 12GB)
- **Model**: ~8-10GB (qwen2.5:14b-instruct-q5_k_m)
- **Context**: ~1-2GB (8192 tokens × 2 bytes/token)
- **Buffer**: ~1GB for operations
- **Total**: ~10-13GB (may spill to system RAM)

**Optimization Strategies:**
1. Reduce `num_ctx` to 4096 (saves ~1GB)
2. Use smaller quantization (q4_k_m instead of q5_k_m)
3. Reduce `num_predict` to limit generation memory
4. Disable reflection pass (saves memory per stage)

#### System RAM
- **Current**: May use system RAM if GPU memory exhausted
- **Impact**: Slower inference (CPU fallback)
- **Monitoring**: Check GPU utilization vs system RAM usage

### Speed Optimization

| Optimization | Speed Gain | Quality Impact | Implementation |
|-------------|------------|---------------|----------------|
| Disable reflection pass | +30-90s/stage | Minimal | Set `ENABLE_REFLECTION_PASS = False` |
| Reduce `num_predict` | +10-30s/stage | Lower output quality | Set to 1500 |
| Reduce `num_ctx` | +5-15s/stage | Less context | Set to 4096 |
| Parallel stage processing | +50-70% overall | None | Requires refactoring |
| Cache structure map | +2-5s/stage | None | Already cached |

### Quality Optimization

| Optimization | Quality Gain | Speed Impact | Implementation |
|-------------|--------------|--------------|----------------|
| Enable reflection pass | +10-20% depth | -30-90s/stage | Already enabled |
| Increase `num_predict` | Longer reasoning | -10-30s/stage | Set to 3000+ |
| Increase `num_ctx` | More context | -5-15s/stage | Risk OOM |
| Lower `temperature` | More consistent | None | Already optimized (0.55) |
| Higher `repeat_penalty` | Less repetition | None | Already optimized (1.12) |

### Token Budget Management

#### Token Counting
```python
def count_tokens(text: str) -> int:
    return int(len(text.split()) * 1.35)  # Academic English estimate
```

#### Budget Allocation (per stage)

| Stage | Input Budget | System Prompt | User Prompt | Output Budget |
|-------|-------------|---------------|-------------|---------------|
| Stage 1 | 2800 tokens | ~500 tokens | ~300 tokens | 2000 tokens |
| Stage 2 | 2800 tokens | ~500 tokens | ~300 tokens | 2000 tokens |
| Stage 3 | 2800 tokens | ~500 tokens | ~300 tokens | 2000 tokens |
| Stage 4 | 3200 tokens | ~500 tokens | ~300 tokens | 2000 tokens |
| Stage 5 | 2800 tokens | ~500 tokens | ~300 tokens | 2000 tokens |
| Stage 6 | 2800 tokens | ~500 tokens | ~300 tokens | 2000 tokens |
| Stage 7 | 4000 tokens | ~500 tokens | ~300 tokens | 2000 tokens |
| Stage 8 | 6000 tokens | ~500 tokens | ~300 tokens | 2000 tokens |

**Total Context Usage:**
- Per stage: ~3600-6500 tokens (input + prompts)
- Full review: ~30,000-40,000 tokens total
- With reflection: ~60,000-80,000 tokens total

---

## Hardware Constraints

### B580 12GB GPU Specifications

- **VRAM**: 12GB
- **Model**: Intel Arc B580 (or similar)
- **Optimization**: IPEX-LLM build for Intel GPUs

### Memory Constraints

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Model weights | ~8-10GB | qwen2.5:14b q5_k_m quantization |
| Context buffer | ~1-2GB | 8192 tokens × 2 bytes/token |
| Generation buffer | ~0.5-1GB | Depends on `num_predict` |
| System overhead | ~0.5GB | CUDA/IPEX overhead |
| **Total** | **~10-13GB** | May spill to system RAM |

### Stability Constraints

**Current Settings (Stable):**
- `num_ctx`: 8192 (maximum safe)
- `num_predict`: 2000-2800 (prevents OOM)
- `num_gpu`: 999 (use all GPU layers)
- Model: q5_k_m quantization (balance quality/speed)

**If Experiencing OOM:**
1. Reduce `num_ctx` to 4096
2. Reduce `num_predict` to 1500
3. Use q4_k_m quantization
4. Disable reflection pass

### Performance Expectations

| Operation | Expected Time | Notes |
|----------|---------------|-------|
| Single stage (no reflection) | 30-60 seconds | Depends on section length |
| Single stage (with reflection) | 60-150 seconds | Adds second LLM call |
| Full 7-stage review | 5-15 minutes | Varies by paper length |
| 20-30 page review | 10-25 minutes | 8 sections × 1-3 minutes each |
| Socratic dialogue (per turn) | 10-30 seconds | Shorter prompts |
| Editorial mentor (per question) | 20-60 seconds | Depends on excerpt size |

---

## Configuration Files

### JournalEditorCoach.Modelfile
```
FROM qwen2.5:14b-instruct-q5_k_m

PARAMETER num_ctx 8192
PARAMETER temperature 0.55
PARAMETER num_predict 2800
PARAMETER top_p 0.92
PARAMETER repeat_penalty 1.12

SYSTEM """
[System prompt for editorial coaching]
"""
```

### Environment Variables
```bash
# Ollama directory (Windows)
set OLLAMA_DIR=F:\ollama-ipex-llm-2.2.0-win

# Ollama serve script path
set OLLAMA_SERVE_PATH=F:\ollama-ipex-llm-2.2.0-win\ollama-serve.bat
```

---

## Debugging & Monitoring

### Key Metrics to Monitor

1. **GPU Memory Usage**
   - Target: <12GB VRAM
   - Alert: >11GB (risk of OOM)
   - Tool: `nvidia-smi` or Intel GPU monitoring

2. **Inference Time**
   - Baseline: 30-60s per stage
   - Alert: >120s (may indicate CPU fallback)
   - Log: Check stage completion times

3. **Token Usage**
   - Monitor: Actual tokens vs budget
   - Alert: Frequent truncation warnings
   - Tool: `count_tokens()` function

4. **Context Log Size**
   - Target: <2000 chars
   - Alert: Frequent truncation
   - Impact: Agents may miss earlier context

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| OOM Error | GPU memory exhausted | Reduce `num_ctx` or `num_predict` |
| Slow inference | >120s per stage | Check GPU utilization, reduce context |
| JSON parse errors | Invalid output format | Increase `num_predict` or retry |
| Missing context | Agents miss prior findings | Increase context log size |
| Chunking issues | Incomplete analysis | Adjust overlap or chunk size |

---

## Recommendations for Optimization

### For Speed
1. Disable reflection pass (`ENABLE_REFLECTION_PASS = False`)
2. Reduce `num_predict` to 1500-1800
3. Reduce `num_ctx` to 4096 (if acceptable)
4. Use smaller quantization (q4_k_m)

### For Quality
1. Keep reflection pass enabled
2. Increase `num_predict` to 3000+
3. Maintain `num_ctx` at 8192
4. Use q5_k_m or higher quantization

### For Memory Efficiency
1. Reduce `num_ctx` to 4096
2. Reduce `num_predict` to 1500
3. Use q4_k_m quantization
4. Improve chunking strategy (less overlap)

### For Stability
1. Keep current settings (proven stable)
2. Monitor GPU memory usage
3. Add retry logic for OOM errors
4. Implement graceful degradation (CPU fallback)

---

## Version History

- **2026-02-19**: Initial reference document created
- Documents settings for:
  - `journal_editor_coach.py` (7-stage review)
  - `journal_editor_coach_20-30page.py` (20-30 page support)
  - `socratic_mentor.py` (Socratic dialogue)
  - `editorial_mentor_review.py` (Editorial mentor)

---

## References

- **Main Scripts**: See `README_AI_Reviewer.md` for usage
- **Model Files**: `JournalEditorCoach.Modelfile`
- **Config Files**: `editorial_mentor_config.json`, `socratic_questions.json`
- **Hardware**: B580 12GB GPU with IPEX-LLM optimized Ollama build
