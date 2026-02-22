# Local Academic AI Reviewer - User Guide

This tool automates the editorial review process for academic manuscripts (specifically ASAC conference submissions) using a local LLM via Ollama.

## Ready to run — quick checklist

Before your first run for **good editorial feedback**:

1. **Ollama**  
   - Ollama (or IPEX-LLM build) installed.  
   - Default path: `F:\ollama-ipex-llm-2.2.0-win\ollama-serve.bat` (set `OLLAMA_DIR` if different).

2. **Model**  
   - Pull the model: `ollama pull qwen2.5:14b-instruct-q5_k_m`  
   - Confirm with: `ollama list`

3. **Python**  
   - From the project folder: `pip install -r requirements.txt`  
   - You need: `ollama`, `PyPDF2`, `python-docx`, `rich`.

4. **Manuscript**  
   - One PDF, DOCX, or TXT file (e.g. 20–30 pages).  
   - Place it anywhere; you will pass the full path.

5. **Run**  
   - **Option A:** Double-click `run_full_review.bat`, then drag the manuscript onto the window (or paste the path when prompted).  
   - **Option B:**  
     `cd "F:\ai asac review folder"`  
     `python journal_editor_coach.py "path\to\your\manuscript.pdf"`  
   - The script will start Ollama if needed, run 7 stages, then write:  
     `*_ASAC_Report.json`, `*_ASAC_Final_Decision.md`, `*_ASAC_Evidence_Trail.md`.  
   - For **editor-style feedback with reasoning and quotes**, open `*_ASAC_Evidence_Trail.md`.

6. **Optional second step (Socratic learning)**  
   - After the review, run `run_socratic_learning.bat` with the same manuscript (report is auto-detected).  
   - Or: `python socratic_mentor.py "path\to\manuscript.pdf"`

7. **Optional challenge mode (argue with the mentor)**  
   - `python editorial_mentor_review.py "path\to\manuscript.pdf" --challenge`  
   - You answer a question; the mentor uses the article (and quotes) to agree or push back.

If a stage fails (e.g. JSON parse error), the script retries up to 3 times and then continues; partial results are saved in `editor_progress.json` and you can re-run to resume.

## Scripts

| Script | Purpose |
|--------|--------|
| **journal_editor_coach.py** | **7-stage** ASAC review (Compliance → Implications → **Coherence**). Agent-based; supports **20–30 page** papers via chunking. Produces `*_ASAC_Report.json`, `*_ASAC_Final_Decision.md`, and `*_ASAC_Evidence_Trail.md` (editor perspective with direct quotes and reasoning). |
| **journal_editor_coach_20-30page.py** | 8-section “editor coach” with running memory. Produces `*_editor_review.md` and `*_structured_quotes.json`. |
| **editorial_mentor_review.py** | Second-pass “editorial mentor”: answers **customizable questions** or **challenge mode** (argue with the mentor). Produces `*_Mentor_Feedback.md` / `*_Mentor_Challenge.md`. |
| **socratic_mentor.py** | **Socratic learning**: multi-turn dialogue on the review findings. Mentor asks probing questions; you respond; mentor agrees or challenges. Produces `*_Socratic_Learning.md` and a learning summary. |

**Batch files:** **run_full_review.bat** (7-stage review + evidence trail), **run_reviewer.bat** (same), **run_mentor.bat** (editorial mentor), **run_socratic_learning.bat** (Socratic dialogue after review).

## Prerequisites

1. **Ollama**: Installed and running (`ollama serve`), or use the IPEX-LLM Ollama build (see below).
2. **Model**: The 7-stage script and Socratic mentor use `qwen2.5:14b-instruct-q5_k_m`. The 20–30 page editor coach script uses a custom model name `JournalEditorCoach` (create from `JournalEditorCoach.Modelfile`).
   ```bash
   ollama pull qwen2.5:14b-instruct-q5_k_m
   ollama create JournalEditorCoach -f JournalEditorCoach.Modelfile   # optional, for 20-30 page script
   ```
3. **Python**: 3.8+
   ```bash
   pip install -r requirements.txt
   ```
   For the 20–30 page script you also need: `pymupdf`, `docx2txt`.

## Configuration

- **Ollama: IPEX-LLM only (no system Ollama)**  
  This project is set up to use **only** the IPEX-LLM build at `F:\ollama-ipex-llm-2.2.0-win`. It does **not** use any Ollama installed on C: (e.g. from the Windows installer). If you uninstalled the system Ollama, that’s correct. The scripts start or wait for `ollama-serve.bat` in that folder and talk to `http://localhost:11434`. If you had `OLLAMA_HOST` set for the old install, unset it so the client uses localhost. Creating the custom model must use the **same** Ollama executable (e.g. run `create_JournalEditorCoach_model.bat` or `F:\ollama-ipex-llm-2.2.0-win\ollama.exe create ...`).

- **Ollama directory (Windows)**: By default Ollama is run from `F:\ollama-ipex-llm-2.2.0-win` (the scripts use `ollama-serve.bat` inside that folder). To use a different directory:
  ```bat
  set OLLAMA_DIR=F:\ollama-ipex-llm-2.2.0-win
  ```
  To point directly at the serve script instead:
  ```bat
  set OLLAMA_SERVE_PATH=F:\ollama-ipex-llm-2.2.0-win\ollama-serve.bat
  ```
- **Cleanup**: By default, the scripts kill existing Ollama and other Python processes, then start Ollama. If you want to leave other processes running (e.g. shared machine), use:
  ```bash
  python journal_editor_coach.py "path/to/manuscript.pdf" --no-cleanup-env
  ```

## Recommended workflow (20–30 page papers)

1. **Full review** (7 stages, chunked for long papers):  
   `run_full_review.bat` or `python journal_editor_coach.py "manuscript.pdf"`  
   → Produces ASAC_Report.json, ASAC_Final_Decision.md, **ASAC_Evidence_Trail.md** (findings + direct quotes + editorial reasoning).

2. **Socratic learning** (argue with the agents, learn editorial reasoning):  
   `run_socratic_learning.bat` or `python socratic_mentor.py "manuscript.pdf"`  
   → Uses the report to ask probing questions; you respond; mentor agrees or challenges. Produces a learning summary.

3. **Optional — Editorial mentor** (custom Q&A or challenge your view):  
   `run_mentor.bat` or `python editorial_mentor_review.py "manuscript.pdf" [--challenge]`

## Usage

**7-stage ASAC review (20–30 page support, Stage 7 = Coherence):**
```bash
python journal_editor_coach.py "path/to/manuscript.pdf"
# Or: run_full_review.bat or run_reviewer.bat (drag-drop file or enter path)
```

**Socratic learning (after the review):**
```bash
python socratic_mentor.py "path/to/manuscript.pdf"
# Report is auto-detected if same base name as manuscript. Or: --first-pass-report "BaseName_ASAC_Report.json"
# Or: run_socratic_learning.bat
```

**20–30 page editor coach (alternative flow):**
```bash
python journal_editor_coach_20-30page.py "path/to/manuscript.pdf"
```

**Editorial mentor (second pass, customizable questions):**
```bash
python editorial_mentor_review.py "path/to/manuscript.pdf"
# Challenge mode: script asks you a question → you answer → mentor uses the article (and quotes) to respond; they may argue you're wrong or agree and back you up, depending on the text:
python editorial_mentor_review.py "path/to/manuscript.pdf" --challenge
# With first-pass blackboard (recommended after the 6-stage review). If the report has the same base name as the manuscript (e.g. manuscript_ASAC_Report.json), it is auto-detected:
python editorial_mentor_review.py "path/to/manuscript.pdf"
# Or pass the report explicitly:
python editorial_mentor_review.py "path/to/manuscript.pdf" --first-pass-report "BaseName_ASAC_Report.json"
# Custom question set:
python editorial_mentor_review.py "path/to/manuscript.pdf" --config my_questions.json
```

Arguments:
- `filepath`: Path to .pdf, .docx, or .txt.
- `--no-cleanup-env`: Do not kill Ollama/Python or restart Ollama; only wait for an existing server.
- **Mentor only**: `--first-pass-report` path to `*_ASAC_Report.json`; `--config` path to JSON with mentor questions (default: `editorial_mentor_config.json`).

## Outputs

**journal_editor_coach.py**
- `[Filename]_ASAC_Report.json`: Full structured results (stages 1–7).
- `[Filename]_ASAC_Final_Decision.md`: Final report.
- `[Filename]_ASAC_Evidence_Trail.md`: **Evidence trail** — each finding with direct quote, editorial reasoning, and connection to other sections.
- `Review_StepN_*.md`: Intermediate stage artifacts.
- `editor_progress.json`: Resume state. Long sections are chunked automatically for 20–30 page papers; Stage 7 runs a Coherence & Cross-Section Consistency check.

**journal_editor_coach_20-30page.py**
- `[Filename]_editor_review.md`: Section-by-section review.
- `[Filename]_structured_quotes.json`: Structured quotes and findings.
- `progress_20-30page.json`: Resume state.

**editorial_mentor_review.py**
- `[Filename]_Mentor_Feedback.md` / `[Filename]_Mentor_Challenge.md`: Mentor answers or challenge dialogue.
- `[Filename]_Mentor_Feedback.json`: Structured data. Uses first-pass blackboard when report is present.

**socratic_mentor.py**
- `[Filename]_Socratic_Learning.md`: Dialogue transcript and **learning summary** (what you learned as an editor).
- `[Filename]_Socratic_Learning.json`: Dialogue history and topics covered. Requires the 7-stage ASAC_Report.json (auto-detected by base name).

## Troubleshooting

- **Slow performance**: The 14B model on a 12GB GPU may use some system RAM. Expect several minutes per stage.
- **“Model not found”**: Run `ollama list` and pull or create the model as in Prerequisites.
- **Ollama path not found**: Set `OLLAMA_SERVE_PATH` or place Ollama in the default path (see Configuration).
- **Paths with spaces**: Use quotes: `python journal_editor_coach.py "F:\my path\file.pdf"`. The batch file passes the path quoted.

## Customization

- **7-stage criteria**: Edit `journal_editor_coach.py` and modify `build_stage_specs()` (system prompts and section keys). Stage 7 (Coherence) reads from prior stage results; long sections use `chunk_section()` and blackboard keys `chunk_summaries`, `cross_chunk_contradictions`, `running_themes`.
- **Socratic questions**: Edit **socratic_questions.json** to add or change probing questions per stage (`stage_1` … `stage_7`). Each stage can have `topics` and `questions`; the mentor uses these to prompt you.
- **20–30 page persona**: Edit `JournalEditorCoach.Modelfile`, then run `ollama create JournalEditorCoach -f JournalEditorCoach.Modelfile`.
- **Editorial mentor questions**: Edit **editorial_mentor_config.json** to add, remove, or change questions. Each item can have:
  - `id`: short key (e.g. `"one_change"`).
  - `label`: title in the report (e.g. `"One change that would strengthen the paper most"`).
  - `question`: the exact question the mentor answers (you can tailor this to your journal or course).
  - `focus`: optional — `"whole"`, `"introduction"`, `"methods"`, `"results"`, `"discussion"`, or `"abstract"` to emphasize that section.
  You can also set `mentor_role` in the config to change the mentor’s tone.
- **Challenge mode (--challenge)**: The script prompts you with **challenge_question** (in config); you type your view; the mentor uses the article and **direct quotes** to respond. They may argue your view is wrong or incomplete, or agree and support you with evidence—they can be convinced when you’re correct. Customize **challenge_question** and **challenge_instruction** in the config to change the prompt and the mentor’s behavior.

## Future Roadmap & References

- **Future Upgrades Reference**: [ChatGPT Conversation Link](https://chatgpt.com/share/6996677d-5668-8004-aef8-9e8bf7b37c7f) - Ideas for improving the reviewer logic and capabilities.

### Free Web Search & Fact-Checking MCP Servers
*Potential integrations for verifying claims without API keys:*
- **[pskill9/web-search](https://github.com/pskill9/web-search)**: Free Google search scraper (no API key).
- **[Aas-ee/open-webSearch](https://github.com/Aas-ee/open-webSearch)**: Multi-engine support (Bing, DuckDuckGo, etc.).
- **[mrkrsl/web-search-mcp](https://github.com/mrkrsl/web-search-mcp)**: Locally hosted search for privacy.
- **[Semantic Scholar API](https://www.factiverse.ai/blog/enhancing-fact-checking-with-semantic-scholar-api)**: For academic fact-checking.

### Academic Research & Literature Review MCP Servers
*High-fidelity tools for graduate-level research verification:*
- **[OpenAlex MCP Server](https://github.com/oksure/openalex-research-mcp)**: Comprehensive literature review, citation analysis, and systematic verification.
- **[Academia MCP](https://mcpservers.org/servers/IlyaGusev/academia_mcp)**: Multi-source research (ArXiv, Semantic Scholar, Hugging Face) + PDF processing.
- **[Scholarly MCP](https://mcpservers.org/servers/ywwAHU/mcp-scholarly)**: Lightweight arXiv search for quick claim verification.
- **[Anti-Bullshit MCP Server](https://himcp.ai/server/anti-bullshit-mcp-server)**: Epistemological analysis of scientific claims and ethical dimensions.
- **[Google Fact Check Tools API](https://glama.ai/mcp/servers/@ag2-mcp-servers/fact-check-tools-api)**: Query verified publishers for existing investigated claims.
