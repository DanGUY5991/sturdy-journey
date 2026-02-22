import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
try:
    import ollama
    import fitz  # PyMuPDF
    import docx2txt
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich import print as rprint
except ImportError as e:
    print(f"Error: Missing dependency. Please install required packages: {e}")
    print("pip install ollama pymupdf docx2txt rich")
    sys.exit(1)

# ==================================================================================
# CONFIGURATION & CONSTANTS
# ==================================================================================

MODEL_NAME = "JournalEditorCoach" # Custom model
BASE_MODEL = "qwen2.5:14b-instruct-q5_k_m"
CONSOLE = Console()

# Tuned for ~45–90 min full review: long reasoning per section (B580 12GB)
OLLAMA_OPTIONS = {
    "num_ctx": 8192,
    "num_predict": 2800,   # Allow 1200–2500 tokens reasoning + JSON per section
    "temperature": 0.55,
    "num_gpu": 999,
    "top_p": 0.92,
    "repeat_penalty": 1.12
}

# Ollama directory (IPEX-LLM build). Set OLLAMA_DIR env var to override.
OLLAMA_DIR = os.environ.get("OLLAMA_DIR", r"F:\ollama-ipex-llm-2.2.0-win")
# Full path to serve script; override with OLLAMA_SERVE_PATH if needed
OLLAMA_SERVE_PATH = os.environ.get(
    "OLLAMA_SERVE_PATH",
    os.path.join(OLLAMA_DIR, "ollama-serve.bat")
)

# All generated files (progress, reports) go here. Delete this folder for a fresh start.
OUTPUT_DIR = os.environ.get("REVIEW_OUTPUT_DIR", "review_output")

# The 8 Sections as requested (standard display names for review)
SECTIONS_CHECKLIST = [
    "Title & Abstract",
    "Research Question & Contribution",
    "Theory & Literature Positioning",
    "Methods (rigor + replicability)",
    "Results & Evidence Alignment",
    "Discussion & Conclusions",
    "Writing Quality, Flow & Clarity",
    "Ethics, Anonymization & Integrity Flags"
]

# Maps checklist index -> bucket key for original heading lookup
CHECKLIST_TO_BUCKET = {0: "Title & Abstract", 1: "Introduction", 2: "Introduction",
                       3: "Methods", 4: "Results", 5: "Discussion",
                       6: "Introduction", 7: "Conclusion"}  # 6=Writing uses Intro+Discussion; 7=Ethics uses Methods+Conclusion

# ASAC/CJAS editor mindset (same as main journal_editor_coach.py)
REVIEWER_ARTICLE_FIRST_INSTRUCTION = """
**ASAC/CJAS EDITOR MINDSET** (read this every single time before analyzing):
You are a 15-year veteran Division Chair and CJAS Associate Editor.
Your job is NOT to summarize — it is to decide whether this manuscript should be accepted, revised, or rejected for the ASAC conference and potential CJAS track.

ALWAYS start your thinking with:
1. What is the SINGLE core claim/contribution the author is making?
2. Does the manuscript deliver a coherent story from gap → theory → method → evidence → implications?
3. Would I desk-reject this in the first 5 minutes? Why or why not?

Use these exact ASAC/CJAS criteria for every evaluation:
- Originality & Gap (must fill a clear, important hole)
- Theoretical Rigor (clear framework, not just citations)
- Methodological Soundness (replicable, appropriate, limitations honest)
- Contribution to Theory/Practice (especially Canada-relevant or interdisciplinary)
- Clarity & APA compliance
- Overall Impact (would colleagues cite this?)

Every strength/weakness MUST be anchored in a direct quote + page/section reference (even if you have to infer from structure).
Be rigorous but developmental — especially for student papers.
"""

# Chain-of-thought prefix so the model takes real time
TAKE_ALL_THE_TIME_PREFIX = (
    "TAKE ALL THE TIME YOU NEED — this can be 500–2000+ tokens of reasoning. "
    "Use explicit chain-of-thought:\n"
    "Step 1: Re-read the entire provided text and restate the paper's core argument in your own words.\n"
    "Step 2: List every key claim in this section.\n"
    "Step 3: Evaluate each claim against the ASAC/CJAS criteria above with direct evidence.\n"
    "Step 4: Consider counter-arguments or what is missing.\n"
    "Step 5: Self-critique: 'What might I be missing as an editor?'\n"
    "Step 6: Only then produce the JSON.\n\n"
    "Quality > speed. Depth is required.\n\n"
)

SYSTEM_PROMPT = TAKE_ALL_THE_TIME_PREFIX + REVIEWER_ARTICLE_FIRST_INSTRUCTION + """

You are a Senior Academic Journal Editor with 15+ years of experience.
Your goal is to coach the user to become a better editor by reviewing their manuscript section-by-section.

**CORE RULES:**
1.  **Direct Quotes**: Every single evaluative claim (strength or weakness) MUST be anchored in a short, exact direct quote (max 35 words) from the text.
2.  **Deep Reasoning**: Explain *why* a quote is strong or weak using editorial logic (clarity, impact, flow, gap-filling, rigor).
3.  **Editor Lesson**: For every point, teach a practical lesson the user can apply to future editing work.
4.  **Tone**: Professional, constructive, educational, and precise.
5.  **Global Context**: Use the "Global Summary" to understand how this section fits into the whole paper.

**Flexible Section Recognition (REQUIRED):**
- Academic papers frequently use non-standard headings, numbering, or slight variations. NEVER rely on exact string matching.
- Automatically map the following (and any close semantic equivalents):
  - Methods family: "Methods", "Methodology", "3 Methodology", "3. Methods", "Methods and Materials", "Materials and Methods", "Experimental Section", "Experimental Methods", "Methodological Approach", "Procedure"
  - Results family: "Results", "4 Results", "Findings", "4. Findings", "Empirical Results", "Data Analysis", "Results & Findings"
  - Discussion family: "Discussion", "Conclusions", "5 Discussion", "Discussion and Conclusions", "6 Discussion and Implications", "Implications", "Interpretation", "Final Remarks"
- Always refer to the section using the **exact original heading + numbering** as it appears in the manuscript (e.g. "3 Methodology", "4.2 Findings", "Section 5: Discussion and Implications").
- In your analysis you may add a short mapping note: "3 Methodology (mapped to standard Methods – rigor + replicability)"
- Never flag a section as "missing", "too short", or output "N/A" for quote if semantically equivalent content exists under any of the variants above.
- Only mark a core section as truly missing if there is genuinely no content that logically belongs there.
- Never use "N/A" for the quote field when real content exists—extract an actual quote or omit the weakness.

**OUTPUT FORMAT**:
Output VALID JSON ONLY. No markdown outside JSON.
{
  "section_name": "Paper's exact heading (e.g. 3 Methodology) or standard name",
  "strengths": [
    {
      "quote": "extract exact short quote—never N/A when content exists",
      "location": "location ref",
      "editor_reasoning": "analysis",
      "editor_lesson": "generalizable rule"
    }
  ],
  "weaknesses": [
    {
      "quote": "extract exact short quote—never N/A when content exists",
      "location": "location ref",
      "editor_reasoning": "analysis",
      "editor_lesson": "generalizable rule"
    }
  ],
  "verification_tests": ["specific test 1"],
  "section_memory_summary": "200-250 token summary of this section",
  "global_memory_summary": "Updated max 400 token running summary of the whole paper so far"
}
"""

# Setup: paper overview pass to seed context for section analysis
SETUP_PAPER_PROMPT = """You are a senior academic editor doing a quick first pass to understand a manuscript before detailed section review.

Read the provided manuscript excerpt and output VALID JSON ONLY. No markdown outside JSON.
{
  "main_argument": "One sentence: the paper's single core claim or contribution.",
  "paper_structure": "2-4 sentences: how the paper is organized (sections, flow from problem to conclusion).",
  "key_terms": ["term1", "term2", "term3"],
  "extraction_quality": "GOOD or CONCERN - note if text seems garbled, truncated, or unreadable.",
  "editor_note": "One sentence: what a section reviewer should keep in mind (e.g. 'Focus on whether the methods match the research question')."
}

Be concise. This overview seeds the section-by-section analysis."""

# Fact verification: review findings against full manuscript and flag items for human review
FACT_VERIFICATION_PROMPT = """You are a senior editor performing a FACT VERIFICATION pass on an existing section-by-section review.

You receive:
1. The FULL manuscript text (complete article).
2. The review findings from each of 8 sections (strengths/weaknesses with quotes and reasoning).

Your task:
1. For each finding that cites a quote: check whether the quote appears in the full manuscript and whether it actually supports the claim made. Note if the quote is taken out of context or if the full article contradicts or reframes it.
2. Flag possible misrepresentations: claims where the evidence is weak, the quote doesn't match the finding, or the full manuscript suggests a different interpretation.
3. List "items to review": findings that are interesting or important enough that a human editor should double-check them (even if you cannot confirm a problem).

Output VALID JSON only. No markdown outside JSON.
{
  "verification_summary": "1-3 sentence overall assessment of whether the section findings are well-grounded in the full manuscript.",
  "verified_findings": [
    {"section": "Section Name", "finding": "brief finding text", "status": "VERIFIED"}
  ],
  "flagged_for_review": [
    {
      "section": "Section Name",
      "original_finding": "The exact finding or claim from the review",
      "quote_used": "The quote the section used as evidence",
      "issue": "Why this may be misrepresented or misunderstood (e.g. quote doesn't support claim, context changes meaning)",
      "full_context": "Relevant passage from the full manuscript that matters",
      "priority": "HIGH or MEDIUM or LOW",
      "suggested_action": "What the human should do (e.g. Re-examine this finding)"
    }
  ],
  "possible_misrepresentations": [
    {"section": "Section Name", "claim": "The claim", "why_suspicious": "Brief reason"}
  ],
  "items_to_review": [
    {"section": "Section Name", "topic": "What to check", "why_interesting": "Why a human should look"}
  ]
}
"""

# ==================================================================================
# OLLAMA PREFLIGHT — ensure local AI is running before starting
# ==================================================================================

def is_ollama_running(timeout_sec: float = 3) -> bool:
    """Quick check if Ollama is already responding at localhost:11434. Does not start anything."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434", timeout=int(timeout_sec)) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_ollama(timeout_sec: int = 60, check_interval: int = 2) -> bool:
    """Wait for Ollama server at localhost:11434. Returns True when ready."""
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout_sec
    rprint("[dim]Waiting for Ollama (local AI)...[/dim]")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://localhost:11434", timeout=5) as resp:
                if resp.status == 200:
                    rprint("[bold green]Ollama is reachable.[/bold green]")
                    return True
        except Exception:
            pass
        time.sleep(check_interval)
    rprint("[bold red]Ollama did not become ready in time.[/bold red]")
    rprint("[yellow]Start Ollama manually (e.g. run your IPEX-LLM ollama-serve.bat or 'ollama serve'), then run this script again with --no-cleanup-env[/yellow]")
    return False


def unload_model(model_name: str) -> bool:
    """Unload the model from GPU memory so the first chat gets a fresh load (helps avoid VRAM fragmentation)."""
    import urllib.request
    try:
        data = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        rprint("[dim]Unloaded model from memory (fresh load on first request).[/dim]")
        time.sleep(2)  # Let VRAM settle
        return True
    except Exception as e:
        rprint(f"[dim]Could not unload model (non-fatal): {e}[/dim]")
        return False


def check_model_available() -> bool:
    """Verify the review model exists. Returns True if OK."""
    try:
        models = ollama.list()
        model_list = models.get("models") if isinstance(models, dict) else getattr(models, "models", [])
        if not model_list:
            rprint("[yellow]No models listed (Ollama may still be loading).[/yellow]")
            rprint("[yellow]Create the model: run create_JournalEditorCoach_model.bat in this folder (with Ollama already running).[/yellow]")
            return False
        # Handle dict {"name"/"model": "..."} or object with .name / .model (Ollama can use "model" not "name")
        names = []
        for m in model_list:
            if isinstance(m, dict):
                names.append(m.get("model") or m.get("name", ""))
            else:
                names.append(getattr(m, "model", None) or getattr(m, "name", "") or str(m))
        names = [n for n in names if n]
        # Match exact name or "name:tag" (e.g. JournalEditorCoach:latest)
        if MODEL_NAME in names:
            return True
        if any(str(n).split(":")[0] == MODEL_NAME for n in names):
            return True
        rprint(f"[bold red]Model '{MODEL_NAME}' not found.[/bold red]")
        rprint(f"[dim]Models currently available: {', '.join(names) or '(none)'}[/dim]")
        rprint(f"[yellow]Create it: run create_JournalEditorCoach_model.bat in this folder (start Ollama first from F:\\ollama-ipex-llm-2.2.0-win).[/yellow]")
        return False
    except Exception as e:
        rprint(f"[yellow]Could not list models: {e}[/yellow]")
        rprint("[yellow]Create the model: run create_JournalEditorCoach_model.bat (with Ollama running from F:\\ollama-ipex-llm-2.2.0-win).[/yellow]")
        return False


# ==================================================================================
# TEXT EXTRACTION & SECTIONING
# ==================================================================================

def extract_text(filepath: str) -> str:
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    try:
        if ext == '.pdf':
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
        elif ext == '.docx':
            return docx2txt.process(filepath)
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported format: {ext}")
    except Exception as e:
        rprint(f"[bold red]Error reading file: {e}[/bold red]")
        sys.exit(1)

def chunk_text_by_sections(text: str) -> tuple:
    """
    Robust splitting for long papers.
    Uses flexible semantic/keyword matching so variant headings (e.g. "3 Methodology",
    "4. Findings", "6 Discussion and Implications") map correctly.
    Returns (sections_dict, original_headings_dict).
    """
    lines = text.split('\n')
    
    buckets = {
        "Title & Abstract": "",
        "Introduction": "",
        "Methods": "",
        "Results": "",
        "Discussion": "",
        "Conclusion": "",
        "Other": ""
    }
    # Capture paper's actual heading text for each bucket (preserves numbering)
    original_bucket_headings: Dict[str, str] = {}

    current_bucket = "Introduction"
    
    # Flexible header patterns — semantic equivalents recognized
    # Methods: Methodology, Methods & Materials, Experimental Methods, Procedure, etc.
    # Results: Findings, Results & Findings, Empirical Results, Data Analysis, etc.
    # Discussion: Discussion and Conclusions, Implications, Interpretation, Final Remarks, etc.
    # Order matters: Conclusion before Discussion so "6 Conclusions" maps to Conclusion
    header_patterns = [
        (re.compile(r'^\s*(?:1\.\s*)?(?:Abstract|Executive Summary)\s*$', re.I), "Title & Abstract"),
        (re.compile(r'^\s*(?:1\.\s*)?Introduction\s*$', re.I), "Introduction"),
        (re.compile(r'^\s*(?:2\.\s*)?(?:Literature|Theory|Background)\s*$', re.I), "Introduction"),
        (re.compile(r'^\s*(?:\d+\.?\s*)?(?:Methods?|Methodology|Methods?\s*&\s*Materials?|'
                    r'Experimental\s+Methods?|Methodological\s+Approach|Procedure|Data|Research\s+Design)\s*(?:\s+[-–—:].*)?$', re.I), "Methods"),
        (re.compile(r'^\s*(?:\d+\.?\s*)?(?:Results?|Findings?|Results?\s*&\s*Findings?|'
                    r'Empirical\s+Results?|Data\s+Analysis|Analysis)\s*(?:\s+[-–—:].*)?$', re.I), "Results"),
        (re.compile(r'^\s*(?:\d+\.?\s*)?(?:Conclusion|Conclusions)\s*$', re.I), "Conclusion"),
        (re.compile(r'^\s*(?:\d+\.?\s*)?(?:Discussion(?:\s+(?:and|&)\s+(?:Conclusions?|Implications?))?|'
                    r'Implications?|Interpretation|Final\s+Remarks)\s*(?:\s+[-–—:].*)?$', re.I), "Discussion"),
        (re.compile(r'^\s*(?:References?|Bibliography)\s*$', re.I), "Other"),
    ]

    if lines and "abstract" in (lines[0] or "").lower():
        current_bucket = "Title & Abstract"

    for line in lines:
        clean_line = line.strip()
        if len(clean_line) < 80:
            for pattern, bucket in header_patterns:
                if pattern.match(clean_line):
                    current_bucket = bucket
                    original_bucket_headings[bucket] = clean_line
                    break
        buckets[current_bucket] += line + "\n"

    # Build original_headings per checklist key
    original_headings: Dict[str, str] = {}
    for i, key in enumerate(SECTIONS_CHECKLIST):
        b = CHECKLIST_TO_BUCKET.get(i, key)
        original_headings[key] = original_bucket_headings.get(b, key)

    mapping = {
        SECTIONS_CHECKLIST[0]: buckets["Title & Abstract"] if len(buckets["Title & Abstract"]) > 200 else buckets["Introduction"][:3000],
        SECTIONS_CHECKLIST[1]: buckets["Introduction"],
        SECTIONS_CHECKLIST[2]: buckets["Introduction"],
        SECTIONS_CHECKLIST[3]: buckets["Methods"],
        SECTIONS_CHECKLIST[4]: buckets["Results"],
        SECTIONS_CHECKLIST[5]: buckets["Discussion"],
        SECTIONS_CHECKLIST[6]: buckets["Introduction"][:5000] + "\n...\n" + buckets["Discussion"][:5000],
        SECTIONS_CHECKLIST[7]: buckets["Methods"] + "\n...\n" + buckets["Conclusion"]
    }

    return mapping, original_headings


def extract_json_from_content(content: str) -> Optional[Dict]:
    """
    Extract a single JSON object from LLM output. Tries: (1) ```json ... ``` block,
    (2) first balanced { ... } in the text. Returns None if parsing fails.
    """
    text = content.strip()
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if code_block:
        block_text = code_block.group(1).strip()
        start = block_text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(block_text)):
                if block_text[i] == "{":
                    depth += 1
                elif block_text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(block_text[start : i + 1])
                        except json.JSONDecodeError:
                            break
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ==================================================================================
# SETUP PASS — prepare paper for analysis
# ==================================================================================

def validate_extraction(full_text: str, sections: Dict[str, str]) -> Tuple[bool, List[str]]:
    """Pre-flight checks on extraction. Returns (ok, list of warnings)."""
    warnings = []
    if len(full_text.strip()) < 500:
        warnings.append("Manuscript is very short (<500 chars). Extraction may be incomplete.")
    if full_text.count("\x00") > 10:
        warnings.append("Null bytes detected—possible encoding/PDF extraction issues.")
    empty_sections = [k for k, v in sections.items() if len((v or "").strip()) < 50]
    if len(empty_sections) > 4:
        warnings.append(f"Many sections empty or minimal: {empty_sections[:5]}...")
    return len(warnings) == 0, warnings


def run_setup_pass(full_text: str, sections: Dict[str, str], original_headings: Dict[str, str]) -> str:
    """
    One LLM pass to build paper overview. Returns initial global_mem for section analysis.
    """
    # Trim for context; include title/abstract + first parts of key sections
    max_chars = 8000
    setup_text = full_text.strip()
    if len(setup_text) > max_chars:
        setup_text = setup_text[:max_chars] + "\n\n[... manuscript trimmed for setup ...]"

    structure_map = "\n".join(
        f"- {h}: mapped to {k}" for k, h in original_headings.items()
        if h and h != k
    ) or "Standard headings used."

    user_prompt = f"""**MANUSCRIPT EXCERPT** (first ~{max_chars} chars):

{setup_text}

---

**SECTION MAPPING** (paper headings → review criteria):
{structure_map}

---

Produce the JSON overview (main_argument, paper_structure, key_terms, extraction_quality, editor_note)."""

    rprint("[dim]Setup: Analyzing paper structure...[/dim]")
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SETUP_PAPER_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options=OLLAMA_OPTIONS
        )
        content = response["message"]["content"]
        data = extract_json_from_content(content)
        if data:
            main = data.get("main_argument", "")
            struct = data.get("paper_structure", "")
            quality = data.get("extraction_quality", "")
            note = data.get("editor_note", "")
            global_mem = (
                f"Initial State (from setup pass): "
                f"Main argument: {main}. Structure: {struct}. "
                f"Extraction: {quality}. Editor note: {note}"
            )
            rprint("[green]Setup complete. Paper overview seeded.[/green]")
            return global_mem
    except Exception as e:
        rprint(f"[yellow]Setup pass failed (non-fatal): {e}[/yellow]")
    return "Initial State: New manuscript review."


# ==================================================================================
# AI PROCESSING
# ==================================================================================

def process_section(section_name: str, text_content: str, previous_section_mem: str, global_mem: str,
                   original_heading: Optional[str] = None) -> Dict:
    """
    Process a single section with retry logic and JSON parsing.
    original_heading: the paper's actual heading (e.g. "3 Methodology") for display and mapping.
    """
    # Max size clamp
    if len(text_content) > 15000:
        text_content = text_content[:15000] + "\n[...Truncated for Stability...]"

    # Only flag as "missing" when content is truly absent; do not flag if semantically equivalent content exists
    if len(text_content.strip()) < 50:
        mapping_note = (
            " Note: Headings like '3 Methodology', 'Methods & Materials', or '4. Findings' are acceptable variants—"
            " only flag as missing if no such section exists at all."
        )
        return {
            "section_name": section_name,
            "original_heading": original_heading or section_name,
            "checklist_section": section_name,
            "strengths": [],
            "weaknesses": [{
                "quote": "N/A",
                "location": "N/A",
                "editor_reasoning": "Section content appears missing or too short.",
                "editor_lesson": "Ensure all standard manuscript sections are present." + mapping_note
            }],
            "verification_tests": ["Check manuscript completeness."],
            "section_memory_summary": "Content missing.",
            "global_memory_summary": global_mem
        }

    display_ctx = ""
    if original_heading and original_heading != section_name:
        display_ctx = f"\n**Paper uses heading**: \"{original_heading}\" (conceptually: {section_name})\n"

    user_prompt = f"""
**BLACKBOARD (previous sections)**:
{global_mem}

**ASAC EDITOR INSTRUCTIONS** (read every time):
{REVIEWER_ARTICLE_FIRST_INSTRUCTION}

**CURRENT SECTION**: {section_name}{display_ctx}

**PREVIOUS SECTION SUMMARY**:
{previous_section_mem}

**TEXT CONTENT TO ANALYZE**:
{text_content}

Analyze for the section the paper calls \"{original_heading or section_name}\" (review criteria: {section_name}).
Use the paper's actual heading in your section_name output when it differs. Output strict JSON.
"""
    
    rprint(f"[dim]Sending {len(user_prompt)} chars to model...[/dim]")
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt}
            ],
            options=OLLAMA_OPTIONS
        )
        
        content = response['message']['content']
        
        # Extract JSON (robust for markdown-wrapped or trailing text)
        data = extract_json_from_content(content)
        if data is not None:
            if "global_memory_summary" not in data:
                data["global_memory_summary"] = global_mem
            data["original_heading"] = data.get("original_heading") or original_heading or data.get("section_name", section_name)
            data["checklist_section"] = section_name  # canonical review criteria name for mapping display
            return data
        
        return {
            "section_name": section_name,
            "original_heading": original_heading or section_name,
            "checklist_section": section_name,
            "error": "Failed to parse JSON", 
            "section_memory_summary": "Error", 
            "global_memory_summary": global_mem,
            "raw_output": content
        }

    except Exception as e:
        rprint(f"[bold red]Ollama Error: {e}[/bold red]")
        rprint("[yellow]If local AI did not load: start Ollama (e.g. run ollama-serve.bat or 'ollama serve'), then run this script again with --no-cleanup-env[/yellow]")
        return {
            "section_name": section_name,
            "original_heading": original_heading or section_name,
            "checklist_section": section_name,
            "error": str(e), 
            "section_memory_summary": "Error", 
            "global_memory_summary": global_mem
        }


def run_fact_verification(results: List[Dict], full_text: str) -> Optional[Dict]:
    """
    Run fact verification pass: review each section's findings against the full manuscript.
    Returns structured JSON with verified_findings, flagged_for_review, possible_misrepresentations, items_to_review.
    """
    # Trim to fit context (num_ctx 8192 ~ 30k chars for content)
    max_manuscript_chars = 18000
    max_results_chars = 10000
    manuscript = full_text.strip()
    if len(manuscript) > max_manuscript_chars:
        manuscript = manuscript[:max_manuscript_chars] + "\n\n[... manuscript trimmed for context ...]"
    results_str = json.dumps(results, indent=0, ensure_ascii=False)
    if len(results_str) > max_results_chars:
        results_str = results_str[:max_results_chars] + "\n... [truncated]"

    user_prompt = f"""**FULL MANUSCRIPT** (use this to verify that section findings are grounded in the whole article):

{manuscript}

---

**SECTION-BY-SECTION REVIEW FINDINGS** (check each finding's quote and claim against the full manuscript above):

{results_str}

---

Perform the fact verification pass. For each finding that uses a quote, verify the quote exists in the manuscript and supports the claim. Flag misrepresentations and list items for human review. Output valid JSON only."""

    rprint("[dim]Running fact verification pass (one LLM call)...[/dim]")
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': FACT_VERIFICATION_PROMPT},
                {'role': 'user', 'content': user_prompt}
            ],
            options=OLLAMA_OPTIONS
        )
        content = response['message']['content']
        data = extract_json_from_content(content)
        if data is not None:
            return data
        rprint("[yellow]Fact verification: could not parse JSON; skipping.[/yellow]")
        return None
    except Exception as e:
        rprint(f"[yellow]Fact verification error: {e}[/yellow]")
        return None


# ==================================================================================
# REPORT GENERATION
# ==================================================================================

def _format_section_display(res: Dict) -> str:
    """Use paper's actual heading when available; add mapping note if different from checklist name."""
    orig = res.get('original_heading') or res.get('section_name', 'Unknown')
    checklist = res.get('checklist_section', res.get('section_name', orig))
    if orig and checklist and orig != checklist:
        return f"{orig} (mapped to {checklist})"
    return orig or checklist or 'Unknown'


def generate_markdown_report(results: List[Dict], output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Journal Editor Coach Review\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Model**: {MODEL_NAME}\n\n")
        
        # Dashboard
        f.write("## 1. Editorial Dashboard\n")
        f.write("| Section | Status | Issues |\n")
        f.write("| :--- | :--- | :--- |\n")
        for res in results:
            issues = len(res.get('weaknesses', []))
            status = "🟢 Strong" if issues == 0 else "🟡 Needs Work" if issues < 3 else "🔴 Critical"
            f.write(f"| {_format_section_display(res)} | {status} | {issues} |\n")
        f.write("\n")
        
        # Details — preserve paper's numbering/heading, add mapping note when different
        f.write("## 2. Detailed Section Analysis\n")
        for res in results:
            sec = _format_section_display(res)
            f.write(f"### {sec}\n")
            
            # Strengths
            if res.get('strengths'):
                f.write("#### What's Working\n")
                for s in res['strengths']:
                    f.write(f"- **{s.get('editor_reasoning', '')}**\n")
                    f.write(f"  - *Quote*: \"{s.get('quote', '')}\"\n")
                    f.write(f"  - 🎓 *Lesson*: {s.get('editor_lesson', '')}\n\n")
            
            # Weaknesses
            if res.get('weaknesses'):
                f.write("#### What Needs Attention\n")
                for w in res['weaknesses']:
                    f.write(f"- 🚩 **{w.get('editor_reasoning', '')}**\n")
                    f.write(f"  - *Quote*: \"{w.get('quote', '')}\"\n")
                    f.write(f"  - 🎓 *Lesson*: {w.get('editor_lesson', '')}\n\n")
            
            f.write("---\n")
            
        # Tests
        f.write("## 3. Editor Verification Tests\n")
        for res in results:
            tests = res.get('verification_tests', [])
            if tests:
                f.write(f"**{_format_section_display(res)}**\n")
                for t in tests:
                    f.write(f"- [ ] {t}\n")
                f.write("\n")


def generate_fact_verification_report(verification: Dict, output_path: str) -> None:
    """Write fact verification results to a markdown file for human review."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Fact Verification — Items for Your Review\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(verification.get("verification_summary", "No summary.") + "\n\n")
        verified = verification.get("verified_findings") or []
        flagged = verification.get("flagged_for_review") or []
        misrep = verification.get("possible_misrepresentations") or []
        items = verification.get("items_to_review") or []
        f.write(f"**Summary**: {len(verified)} findings verified, {len(flagged)} flagged for review, "
                f"{len(misrep)} possible misrepresentations, {len(items)} items worth reviewing.\n\n")
        f.write("---\n\n")
        f.write("## Flagged for Review (priority order)\n\n")
        for item in sorted(flagged, key=lambda x: (0 if x.get("priority") == "HIGH" else 1 if x.get("priority") == "MEDIUM" else 2, x.get("section", ""))):
            f.write(f"### {item.get('section', 'Unknown')} — {item.get('priority', '')}\n\n")
            f.write(f"- **Original finding**: {item.get('original_finding', '')}\n")
            f.write(f"- **Quote used**: \"{item.get('quote_used', '')}\"\n")
            f.write(f"- **Issue**: {item.get('issue', '')}\n")
            if item.get('full_context'):
                f.write(f"- **Full context**: {item.get('full_context', '')[:500]}...\n" if len(str(item.get('full_context', ''))) > 500 else f"- **Full context**: {item.get('full_context', '')}\n")
            f.write(f"- **Suggested action**: {item.get('suggested_action', '')}\n\n")
        f.write("---\n\n")
        f.write("## Possible Misrepresentations\n\n")
        for item in misrep:
            f.write(f"- **{item.get('section', '')}**: {item.get('claim', '')}\n")
            f.write(f"  - *Why suspicious*: {item.get('why_suspicious', '')}\n\n")
        f.write("---\n\n")
        f.write("## Items Worth Your Review\n\n")
        for item in items:
            f.write(f"- **{item.get('section', '')}** — {item.get('topic', '')}\n")
            f.write(f"  - {item.get('why_interesting', '')}\n\n")


# ==================================================================================
# MAIN
# ==================================================================================

import subprocess

def cleanup_environment(do_cleanup: bool = True) -> None:
    """
    Optionally kills conflicting processes and starts Ollama, then waits for it.
    If do_cleanup is False, only waits for existing Ollama (no kill/restart).
    Preserves the current process when killing Python.
    """
    current_pid = os.getpid()
    if do_cleanup:
        rprint(f"[bold yellow]Performing environment cleanup (PID: {current_pid})...[/bold yellow]")

        # If Ollama is already running, do not kill it or start another — use existing server
        if is_ollama_running():
            rprint("[bold green]Ollama already running at localhost:11434. Using existing (will not start another).[/bold green]")
            if not wait_for_ollama(timeout_sec=10):
                sys.exit(1)
        else:
            try:
                subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                rprint("[dim]- Killed running ollama processes.[/dim]")
            except Exception as e:
                rprint(f"[dim red]- Failed to kill ollama: {e}[/dim red]")

            try:
                subprocess.run(f'taskkill /F /IM python.exe /FI "PID ne {current_pid}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                rprint("[dim]- Killed other python processes.[/dim]")
            except Exception as e:
                rprint(f"[dim red]- Failed to kill other python processes: {e}[/dim red]")

            rprint(f"[bold green]Using Ollama from: {OLLAMA_DIR} (IPEX-LLM)[/bold green]")
            try:
                ollama_path = OLLAMA_SERVE_PATH
                if os.path.exists(ollama_path):
                    # Run the .bat with its directory as cwd (IPEX-LLM only; no system/PATH ollama)
                    subprocess.Popen(
                        f'start "IPEX-LLM Ollama" /min /D "{OLLAMA_DIR}" "{ollama_path}"',
                        shell=True,
                        cwd=OLLAMA_DIR
                    )
                    rprint(f"[dim]- Launched: {ollama_path}[/dim]")
                else:
                    rprint(f"[bold red]IPEX-LLM Ollama not found: {ollama_path}[/bold red]")
                    rprint("[yellow]This project uses only F:\\ollama-ipex-llm-2.2.0-win (set OLLAMA_DIR / OLLAMA_SERVE_PATH if yours is elsewhere). Do not use system Ollama from PATH.[/yellow]")
                    sys.exit(1)

                rprint("[dim]- Waiting for Ollama to be ready (up to 60 sec)...[/dim]")
                if not wait_for_ollama(timeout_sec=60):
                    sys.exit(1)
            except Exception as e:
                rprint(f"[dim red]- Failed to start Ollama: {e}[/dim red]")
    else:
        rprint(f"[dim]Skipping cleanup (--no-cleanup-env). Waiting for Ollama at localhost:11434.[/dim]")
        rprint(f"[dim]If using IPEX-LLM, start it from: {OLLAMA_DIR}[/dim]")
        if not wait_for_ollama(timeout_sec=60):
            sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Journal Editor Coach (20-30 Page Support)")
    parser.add_argument("filepath", help="Path to manuscript")
    parser.add_argument("--no-cleanup-env", action="store_true",
                        help="Do not kill Ollama/Python or restart Ollama; only wait for existing server.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore saved progress and run a full new review (all 8 sections with LLM).")
    args = parser.parse_args()

    progress_file = os.path.join(OUTPUT_DIR, "progress_20-30page.json")
    if getattr(args, "fresh", False) and os.path.exists(progress_file):
        try:
            os.remove(progress_file)
            rprint("[yellow]Removed saved progress (--fresh). Starting full new review.[/yellow]\n")
        except OSError as e:
            rprint(f"[red]Could not remove {progress_file}: {e}[/red]")

    cleanup_environment(do_cleanup=not args.no_cleanup_env)

    # Require a real file path (avoid 1-sec finish when .bat passes empty path)
    if not (args.filepath and args.filepath.strip()):
        rprint("[bold red]No manuscript file provided.[/bold red]")
        rprint("Drag and drop a PDF/DOCX onto the .bat file, or run: python journal_editor_coach_20-30page.py \"path\\to\\manuscript.pdf\"")
        return
    if not os.path.exists(args.filepath):
        rprint(f"[bold red]File not found: {args.filepath}[/bold red]")
        return

    # Preflight: ensure model exists so first request doesn't fail silently
    rprint("[dim]Checking that review model is available...[/dim]")
    if not check_model_available():
        sys.exit(1)
    rprint("[bold green]Model ready. This review will take approximately 45–90 minutes.[/bold green]\n")

    rprint("[dim]Unloading model from memory for a fresh start...[/dim]")
    unload_model(MODEL_NAME)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rprint(f"[dim]Output folder: {os.path.abspath(OUTPUT_DIR)} (delete for fresh start)[/dim]\n")

    rprint(Panel("[bold blue]Journal Editor Coach[/bold blue]\n[dim]B580 Optimized • 20-30 Page Support[/dim]", expand=False))

    # Ingest
    text_full = extract_text(args.filepath)
    rprint(f"[green]Loaded {len(text_full)} chars.[/green]")
    
    # Split (flexible section matching + original heading capture)
    sections, original_headings = chunk_text_by_sections(text_full)
    rprint(f"[green]Parsed {len(sections)} sections/checkpoints.[/green]")

    # Pre-flight validation
    ok, warnings = validate_extraction(text_full, sections)
    for w in warnings:
        rprint(f"[yellow]Validation: {w}[/yellow]")

    # State Vars
    results = []
    global_mem = "Initial State: New manuscript review."
    sec_mem = "N/A"
    
    # Resume Logic + cleanup when changing articles
    progress_file = os.path.join(OUTPUT_DIR, "progress_20-30page.json")
    current_filepath = os.path.abspath(args.filepath)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                saved_state = json.load(f)
            saved_path = saved_state.get('filepath') or saved_state.get('manuscript_path')
            if saved_path and os.path.abspath(saved_path) != current_filepath:
                os.remove(progress_file)
                rprint("[yellow]Different article detected. Cleared previous progress for a fresh review.[/yellow]")
            else:
                results = saved_state.get('results', [])
                global_mem = saved_state.get('global_mem', global_mem)
                sec_mem = saved_state.get('sec_mem', sec_mem)
                rprint(f"[yellow]Resuming from section {len(results)+1}...[/yellow]")
        except Exception:
            rprint("[red]Corrupt save file. Starting fresh.[/red]")
            try:
                os.remove(progress_file)
            except OSError:
                pass

    start_idx = len(results)

    # Setup pass: paper overview to seed context (only when starting fresh)
    if start_idx == 0:
        global_mem = run_setup_pass(text_full, sections, original_headings)

    # If a previous run already completed all sections, we skip AI and just regenerate reports
    if start_idx >= len(SECTIONS_CHECKLIST):
        rprint("[bold yellow]All 8 sections were already completed in a previous run. No AI calls this time.[/bold yellow]")
        rprint(f"[dim]Regenerating reports from saved progress. To run a full new review, delete the {OUTPUT_DIR} folder and run again.[/dim]\n")
    else:
        rprint(f"[green]Starting AI review from section {start_idx + 1} of {len(SECTIONS_CHECKLIST)} (this will take a long time).[/green]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=CONSOLE
    ) as progress:
        first_sec = SECTIONS_CHECKLIST[start_idx] if start_idx < len(SECTIONS_CHECKLIST) else "Reviewing"
        first_orig = original_headings.get(first_sec, first_sec)
        initial_desc = f"Analyzing: {first_orig}" if first_orig != first_sec else f"Analyzing: {first_sec}"
        task = progress.add_task(f"[cyan]{initial_desc}[/cyan]", total=len(SECTIONS_CHECKLIST))
        progress.update(task, completed=start_idx)

        for i in range(start_idx, len(SECTIONS_CHECKLIST)):
            section_key = SECTIONS_CHECKLIST[i]
            section_text = sections.get(section_key, "")
            orig_h = original_headings.get(section_key, section_key)
            
            progress.update(task, description=f"Analyzing: {orig_h}" if orig_h != section_key else f"Analyzing: {section_key}")
            
            # AI Inference (pass original heading for display and mapping notes)
            result_data = process_section(section_key, section_text, sec_mem, global_mem, original_heading=orig_h)
            
            # Update Memory
            global_mem = result_data.get('global_memory_summary', global_mem)
            sec_mem = result_data.get('section_memory_summary', sec_mem)
            
            results.append(result_data)
            
            # Autosave (include filepath so we can detect article change on next run)
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "filepath": os.path.abspath(args.filepath),
                    "results": results,
                    "global_mem": global_mem,
                    "sec_mem": sec_mem
                }, f, indent=2, ensure_ascii=False)
                
            progress.advance(task)
    
    # Output (all in OUTPUT_DIR)
    base = os.path.splitext(os.path.basename(args.filepath))[0]
    generate_markdown_report(results, os.path.join(OUTPUT_DIR, f"{base}_editor_review.md"))
    with open(os.path.join(OUTPUT_DIR, f"{base}_structured_quotes.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Fact verification pass (only when all 8 sections were run this time)
    if len(results) >= len(SECTIONS_CHECKLIST):
        verification = run_fact_verification(results, text_full)
        if verification is not None:
            with open(os.path.join(OUTPUT_DIR, f"{base}_fact_verification.json"), 'w', encoding='utf-8') as f:
                json.dump(verification, f, indent=2, ensure_ascii=False)
            generate_fact_verification_report(verification, os.path.join(OUTPUT_DIR, f"{base}_fact_verification.md"))
            n_verified = len(verification.get("verified_findings") or [])
            n_flagged = len(verification.get("flagged_for_review") or [])
            rprint(f"\n[bold green]Fact verification complete:[/bold green] {n_verified} verified, {n_flagged} flagged for your review. See {OUTPUT_DIR}/{base}_fact_verification.md")
        else:
            rprint("\n[dim]Fact verification skipped or failed.[/dim]")
        
    rprint(f"\n[bold green]Done![/bold green] Output saved to {OUTPUT_DIR}/{base}_editor_review.md")

if __name__ == "__main__":
    main()
