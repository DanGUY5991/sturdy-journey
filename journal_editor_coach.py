import argparse
import json
import os
import re
import sys
import time
import glob
import subprocess
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional

# Third-party imports
try:
    import ollama
    import PyPDF2
    from docx import Document
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
except ImportError as e:
    print(f"Error: Missing dependency. Please install required packages: {e}")
    print("pip install ollama PyPDF2 python-docx rich")
    sys.exit(1)

# ==================================================================================
# CONFIGURATION & CONSTANTS
# ==================================================================================

MODEL_NAME = "JournalEditorCoach"  # Custom model — use for ~1 hr deep review (same as 20-30 page)
CONSOLE = Console()

# Tuned for ~45–90 min full review: long reasoning per stage (B580 12GB)
OLLAMA_OPTIONS = {
    "num_ctx": 8192,
    "num_predict": 2800,   # Allow 1200–2500 tokens reasoning + JSON per stage
    "temperature": 0.55,   # Focused, rigorous editor thinking
    "num_gpu": 999,
    "top_p": 0.92,
    "repeat_penalty": 1.12
}

ASAC_THEME = "Blue Sky Thinking: Harnessing the Future"
MAX_AGENT_RETRIES = 3
ENABLE_REFLECTION_PASS = True   # Second LLM call to deepen analysis (adds ~30–90 s per stage)

# Shared instruction so every agent reads for article-level meaning first (ASAC/CJAS editor mindset)
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

# Enhanced thinking protocol for all agents - maximizes deliberative thinking
THINKING_PROTOCOL = """
**THINKING PROTOCOL** (MANDATORY - follow this sequence):

1. **COMPREHENSION**: First, summarize what you understand about this section in 2-3 sentences.
   - What is the main purpose of this section?
   - How does it connect to the paper's overall argument?

2. **QUOTE EXTRACTION**: List 3-5 key claims with exact quotes from the text.
   - Extract direct quotes (max 35 words each) that support or challenge the section's claims
   - Note the location/context of each quote

3. **EVALUATION**: For each claim, evaluate against the stage criteria.
   - Does it meet the criteria? Why or why not?
   - What evidence supports your evaluation?
   - What is missing or unclear?

4. **AMBIGUITY CHECK**: Note any ambiguities or uncertainties.
   - What questions remain unanswered?
   - What assumptions are being made?
   - What alternative interpretations exist?

5. **CROSS-REFERENCE**: Check consistency with:
   - The paper structure map (from Stage 1)
   - Prior stage findings (from blackboard)
   - Other sections of the manuscript

6. **SELF-CRITIQUE**: Before finalizing, ask:
   - "What might I be missing as an editor?"
   - "Am I being too lenient or too harsh?"
   - "Would another editor agree with my assessment?"

7. **STRUCTURED OUTPUT**: Only after completing steps 1-6, produce the required JSON output.

**IMPORTANT**: This process should take time. Use 500-2000+ tokens for reasoning. Quality and depth are more important than speed.
"""

# Fact verification: review stage findings against full manuscript and flag items for human review
FACT_VERIFICATION_PROMPT = """You are a senior editor performing a FACT VERIFICATION pass on an existing stage-by-stage review.

You receive:
1. The FULL manuscript text (complete article).
2. The review findings from each stage (strengths, major_issues, violations, with quotes and reasoning).

Your task:
1. For each finding that cites a quote: check whether the quote appears in the full manuscript and whether it actually supports the claim made. Note if the quote is taken out of context or if the full article contradicts or reframes it.
2. Flag possible misrepresentations: claims where the evidence is weak, the quote doesn't match the finding, or the full manuscript suggests a different interpretation.
3. List "items to review": findings that are interesting or important enough that a human editor should double-check them (even if you cannot confirm a problem).

Output VALID JSON only. No markdown outside JSON.
{
  "verification_summary": "1-3 sentence overall assessment of whether the stage findings are well-grounded in the full manuscript.",
  "verified_findings": [
    {"section": "Stage Name", "finding": "brief finding text", "status": "VERIFIED"}
  ],
  "flagged_for_review": [
    {
      "section": "Stage Name",
      "original_finding": "The exact finding or claim from the review",
      "quote_used": "The quote the stage used as evidence",
      "issue": "Why this may be misrepresented or misunderstood",
      "full_context": "Relevant passage from the full manuscript that matters",
      "priority": "HIGH or MEDIUM or LOW",
      "suggested_action": "What the human should do"
    }
  ],
  "possible_misrepresentations": [
    {"section": "Stage Name", "claim": "The claim", "why_suspicious": "Brief reason"}
  ],
  "items_to_review": [
    {"section": "Stage Name", "topic": "What to check", "why_interesting": "Why a human should look"}
  ]
}
"""

# Ollama directory (IPEX-LLM build). Set OLLAMA_DIR env var to override.
OLLAMA_DIR = os.environ.get("OLLAMA_DIR", r"F:\ollama-ipex-llm-2.2.0-win")
# Full path to serve script; override with OLLAMA_SERVE_PATH if needed
OLLAMA_SERVE_PATH = os.environ.get(
    "OLLAMA_SERVE_PATH",
    os.path.join(OLLAMA_DIR, "ollama-serve.bat")
)

# All generated files (progress, reports, artifacts) go here. Delete this folder for a fresh start.
OUTPUT_DIR = os.environ.get("REVIEW_OUTPUT_DIR", "review_output")

# ==================================================================================
# UTILITIES
# ==================================================================================

def extract_json_from_content(content: str) -> Optional[Dict]:
    """
    Extract a single JSON object from LLM output. Tries: (1) ```json ... ``` block,
    (2) first balanced { ... } in the text. Returns None if parsing fails.
    """
    text = content.strip()
    # Try markdown code block first (capture raw content, then extract balanced JSON)
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
            pass
    # Find first { and then matching } by brace balance
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

def count_tokens(text: str) -> int:
    """Rough but very reliable estimate for academic English."""
    return int(len(text.split()) * 1.35)

def trim_to_budget(text: str, max_tokens: int = 2800) -> str:
    """Trim text to a token budget, appending a note if truncated."""
    if count_tokens(text) <= max_tokens:
        return text.strip()
    words = text.split()
    max_words = int(max_tokens / 1.35)
    return " ".join(words[:max_words]).strip() + \
           "\n\n[Note: Section trimmed for context limit.]"


def chunk_section(text: str, max_tokens: int = 2500, overlap_tokens: int = 200) -> List[str]:
    """
    Split long sections into overlapping chunks for LLM context limits.
    Overlap helps preserve continuity at chunk boundaries.
    Returns a list of chunk strings (may be a single chunk if under limit).
    """
    text = text.strip()
    if not text:
        return [""]
    if count_tokens(text) <= max_tokens:
        return [text]
    words = text.split()
    max_words = int(max_tokens / 1.35)
    overlap_words = max(0, int(overlap_tokens / 1.35))
    step = max_words - overlap_words
    if step <= 0:
        step = max_words
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


# ==================================================================================
# TEXT EXTRACTION & SECTIONING
# ==================================================================================

def extract_text(filepath: str) -> str:
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext == '.pdf':
        text = ""
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    elif ext == '.docx':
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported format: {ext}")

def heuristic_section_split(text: str) -> Dict[str, str]:
    """
    Splits text into logical sections by heading heuristics,
    then maps to the 8 review-checklist buckets.
    Uses flexible semantic/keyword matching so variant headings (e.g. "3 Methodology",
    "4. Findings", "6 Discussion and Implications") map correctly.
    """
    lines = text.split('\n')
    buckets = {
        "Abstract": "", "Introduction": "", "Methods": "",
        "Results": "", "Discussion": "", "Conclusion": "", "Other": ""
    }
    current_bucket = "Introduction"
    # Flexible header patterns — semantic equivalents recognized
    # Methods: Methodology, Methods & Materials, Experimental Methods, Procedure, etc.
    # Results: Findings, Results & Findings, Empirical Results, Data Analysis, etc.
    # Discussion: Discussion and Conclusions, Implications, Interpretation, Final Remarks, etc.
    header_patterns = [
        (re.compile(r'^\s*(?:1\.)?\s*Abstract\s*$', re.I), "Abstract"),
        (re.compile(r'^\s*(?:1\.)?\s*Introduction\s*$', re.I), "Introduction"),
        (re.compile(r'^\s*(?:2\.)?\s*(?:Literature|Theory|Background)\s*$', re.I), "Introduction"),
        (re.compile(r'^\s*(?:\d+\.?\s*)?(?:Methods?|Methodology|Methods?\s*&\s*Materials?|'
                    r'Experimental\s+Methods?|Methodological\s+Approach|Procedure|Data|Research\s+Design)\s*(?:\s+[-–—:].*)?$', re.I), "Methods"),
        (re.compile(r'^\s*(?:\d+\.?\s*)?(?:Results?|Findings?|Results?\s*&\s*Findings?|'
                    r'Empirical\s+Results?|Data\s+Analysis|Analysis)\s*(?:\s+[-–—:].*)?$', re.I), "Results"),
        (re.compile(r'^\s*(?:\d+\.?\s*)?(?:Conclusion|Conclusions)\s*$', re.I), "Conclusion"),
        (re.compile(r'^\s*(?:\d+\.?\s*)?(?:Discussion(?:\s+(?:and|&)\s+(?:Conclusions?|Implications?))?|'
                    r'Implications?|Interpretation|Final\s+Remarks)\s*(?:\s+[-–—:].*)?$', re.I), "Discussion"),
        (re.compile(r'^\s*(?:References?|Bibliography)\s*$', re.I), "Other"),
    ]
    for line in lines:
        clean_line = line.strip()
        if len(clean_line) < 80:
            for pattern, bucket in header_patterns:
                if pattern.match(clean_line):
                    current_bucket = bucket
                    break
        buckets[current_bucket] += line + "\n"

    return {
        "Title & Abstract": buckets["Abstract"] if len(buckets["Abstract"]) > 10
                            else buckets["Introduction"][:2000],
        "Introduction": buckets["Introduction"],
        "Theory & Literature Positioning": buckets["Introduction"],
        "Methods (rigor + replicability)": buckets["Methods"],
        "Results & Evidence Alignment": buckets["Results"],
        "Discussion & Conclusions": buckets["Discussion"] + "\n" + buckets["Conclusion"],
        "Other": buckets["Other"],
        "Methods": buckets["Methods"],
        "Results": buckets["Results"],
        "Discussion": buckets["Discussion"],
    }

# ==================================================================================
# BLACKBOARD — shared agent memory
# ==================================================================================

class Blackboard:
    """
    Shared state store that all agents read from and write to.
    Keys used:
      - "sections_map"             : Dict[str,str] — manuscript sections
      - "structure_map"            : str           — paper structure (from Stage 1)
      - "stage_{N}_result"         : Dict          — result from agent N
      - "stage_{N}_chunks"         : List[str]     — chunked content for long sections (Stage N)
      - "context_log"             : List[str]     — running log of stage summaries
      - "orchestrator_notes"       : List[str]     — orchestrator decisions/flags
      - "chunk_summaries"          : Dict[str, List[str]] — summary per section per chunk (for 20-30 page)
      - "cross_chunk_contradictions": List[Dict]   — detected inconsistencies between chunks
      - "running_themes"          : Dict[str, str] — key themes/terms tracked across sections
      - "reasoning_notes"          : Dict[int, List[str]] — intermediate thoughts per stage
      - "quote_evidence"           : Dict[int, List[Dict]] — extracted quotes with analysis per stage
      - "cross_stage_observations" : List[Dict]     — connections between stages
      - "open_questions"           : List[Dict]     — items needing deeper investigation
    """

    def __init__(self):
        self._store: Dict[str, Any] = {
            "context_log": [],
            "orchestrator_notes": [],
            "chunk_summaries": {},
            "cross_chunk_contradictions": [],
            "running_themes": {},
            "reasoning_notes": {},
            "quote_evidence": {},
            "cross_stage_observations": [],
            "open_questions": [],
        }

    def get(self, key: str, default=None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any):
        self._store[key] = value

    def append_log(self, entry: str):
        self._store["context_log"].append(entry)

    def append_note(self, note: str):
        self._store["orchestrator_notes"].append(note)

    def get_context_log(self, max_chars: int = 1500) -> str:
        full = "\n".join(self._store["context_log"])
        return full[-max_chars:] if len(full) > max_chars else full

    def read_full_blackboard(self) -> str:
        """Agents MUST call this first — forces them to see everything previous agents wrote."""
        log = self.get_context_log(2000)
        notes = "\n".join(self._store.get("orchestrator_notes", []))
        return f"""=== BLACKBOARD (read this BEFORE any analysis) ===
{log}

Orchestrator Notes:
{notes}"""

    def post_to_blackboard(self, agent_name: str, content: Dict):
        """Standardized posting format so every agent speaks the same language."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "stage": self.get("current_stage", "unknown"),
            "content": content
        }
        self._store.setdefault("editor_board_posts", []).append(entry)

    def get_stage_result(self, stage_id: int) -> Optional[Dict]:
        return self._store.get(f"stage_{stage_id}_result")

    def set_stage_result(self, stage_id: int, result: Dict):
        self._store[f"stage_{stage_id}_result"] = result

    def to_progress_list(self) -> List[Dict]:
        """Collect all stage results in order for progress file."""
        results = []
        i = 1
        while True:
            r = self.get_stage_result(i)
            if r is None:
                break
            results.append(r)
            i += 1
        return results

    def load_from_progress_list(self, results: List[Dict]):
        """Restore Blackboard from a saved progress list."""
        for i, res in enumerate(results, 1):
            self.set_stage_result(i, res)
            stage_name = res.get("stage_name", f"Stage {i}")
            summary = f"Stage {stage_name}: {json.dumps(res, indent=None)[:400]}..."
            self.append_log(summary)

    def append_chunk_summary(self, section_key: str, summary: str):
        """Append a chunk summary for a section (used for long papers)."""
        key = "chunk_summaries"
        if key not in self._store or not isinstance(self._store[key], dict):
            self._store[key] = {}
        if section_key not in self._store[key]:
            self._store[key][section_key] = []
        self._store[key][section_key].append(summary)

    def append_cross_chunk_contradiction(self, contradiction: Dict):
        """Record a contradiction detected between chunks/sections."""
        self._store.setdefault("cross_chunk_contradictions", []).append(contradiction)

    def set_running_theme(self, theme_key: str, description: str):
        """Track a key theme or term across sections."""
        self._store.setdefault("running_themes", {})[theme_key] = description

    def add_reasoning_note(self, stage_id: int, note: str):
        """Add an intermediate reasoning note for a stage."""
        if "reasoning_notes" not in self._store:
            self._store["reasoning_notes"] = {}
        if stage_id not in self._store["reasoning_notes"]:
            self._store["reasoning_notes"][stage_id] = []
        self._store["reasoning_notes"][stage_id].append({
            "timestamp": datetime.now().isoformat(),
            "note": note
        })

    def get_reasoning_notes(self, stage_id: int) -> List[str]:
        """Get all reasoning notes for a stage."""
        notes = self._store.get("reasoning_notes", {}).get(stage_id, [])
        return [n["note"] if isinstance(n, dict) else n for n in notes]

    def add_quote_evidence(self, stage_id: int, quote: str, analysis: str, location: str = ""):
        """Add extracted quote with analysis for a stage."""
        if "quote_evidence" not in self._store:
            self._store["quote_evidence"] = {}
        if stage_id not in self._store["quote_evidence"]:
            self._store["quote_evidence"][stage_id] = []
        self._store["quote_evidence"][stage_id].append({
            "quote": quote,
            "analysis": analysis,
            "location": location,
            "timestamp": datetime.now().isoformat()
        })

    def get_quote_evidence(self, stage_id: int) -> List[Dict]:
        """Get all quote evidence for a stage."""
        return self._store.get("quote_evidence", {}).get(stage_id, [])

    def add_cross_stage_observation(self, from_stage: int, to_stage: int, observation: str, evidence: str = ""):
        """Record an observation connecting two stages."""
        self._store.setdefault("cross_stage_observations", []).append({
            "from_stage": from_stage,
            "to_stage": to_stage,
            "observation": observation,
            "evidence": evidence,
            "timestamp": datetime.now().isoformat()
        })

    def get_cross_stage_observations(self, stage_id: int = None) -> List[Dict]:
        """Get cross-stage observations, optionally filtered by stage."""
        observations = self._store.get("cross_stage_observations", [])
        if stage_id is None:
            return observations
        return [obs for obs in observations if obs.get("from_stage") == stage_id or obs.get("to_stage") == stage_id]

    def add_open_question(self, stage_id: int, question: str, context: str = "", priority: str = "medium"):
        """Add an open question that needs deeper investigation."""
        self._store.setdefault("open_questions", []).append({
            "stage_id": stage_id,
            "question": question,
            "context": context,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "resolved": False
        })

    def get_open_questions(self, stage_id: int = None, unresolved_only: bool = True) -> List[Dict]:
        """Get open questions, optionally filtered by stage or resolution status."""
        questions = self._store.get("open_questions", [])
        if stage_id is not None:
            questions = [q for q in questions if q.get("stage_id") == stage_id]
        if unresolved_only:
            questions = [q for q in questions if not q.get("resolved", False)]
        return questions

    def resolve_open_question(self, question_index: int):
        """Mark an open question as resolved."""
        questions = self._store.get("open_questions", [])
        if 0 <= question_index < len(questions):
            questions[question_index]["resolved"] = True

# ==================================================================================
# TOOL REGISTRY
# ==================================================================================

@dataclass
class Tool:
    name: str
    description: str
    fn: Callable

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name].fn(**kwargs)

    def descriptions(self) -> str:
        return "\n".join(f"  - {t.name}: {t.description}"
                         for t in self._tools.values())

# ==================================================================================
# BASE AGENT
# ==================================================================================

class BaseAgent(ABC):
    """
    Abstract base for all agents. Agents have:
      - A name and role description
      - Access to the shared Blackboard
      - A ToolRegistry for calling helper functions
      - An act() method that runs the agent's logic
    """

    def __init__(self, name: str, blackboard: Blackboard, tools: ToolRegistry):
        self.name = name
        self.blackboard = blackboard
        self.tools = tools

    @abstractmethod
    def act(self) -> Dict:
        """Execute the agent's task. Returns a result dict."""
        ...

    def _log(self, msg: str):
        rprint(f"[bold cyan][Agent: {self.name}][/bold cyan] {msg}")

# ==================================================================================
# REVIEW AGENT — one per ASAC stage
# ==================================================================================

class ReviewAgent(BaseAgent):
    """
    Autonomous agent for a single ASAC review stage.

    act() loop:
      1. Pull relevant manuscript sections from Blackboard via tools
      2. Fetch structure_map from Blackboard (written by Stage 1 agent)
      3. Build context from previous stage results
      4. Call Ollama LLM with the stage's system prompt + user prompt
      5. Parse JSON from response
      6. Write result to Blackboard and save intermediate artifact
      7. Retry up to MAX_AGENT_RETRIES on LLM or JSON errors
    """

    def __init__(self, stage_spec: Dict, blackboard: Blackboard, tools: ToolRegistry):
        super().__init__(
            name=stage_spec["name"],
            blackboard=blackboard,
            tools=tools
        )
        self.stage_id: int = stage_spec["id"]
        self.system_prompt: str = stage_spec["system_prompt"]
        self.section_keys: List[str] = stage_spec["section_keys"]
        self.max_input_tokens: int = stage_spec.get("max_input_tokens", 2800)

    def _build_input_text(self) -> str:
        """Assemble manuscript text from Blackboard sections. For Stage 7 (ALL),
        build input from stage 1-6 results. For 20-30 page papers, long content
        is chunked; we pass the first chunk with context and store chunks."""
        if self.section_keys == ["ALL"]:
            return self._build_stage7_input()
        content = ""
        for key in self.section_keys:
            chunk = self.tools.call("get_section", key=key)
            content += chunk + "\n\n"
        content = content.strip()
        if not content:
            return ""
        tokens = count_tokens(content)
        if tokens <= self.max_input_tokens:
            return content
        chunks = chunk_section(content, max_tokens=self.max_input_tokens - 150, overlap_tokens=200)
        self.blackboard.set(f"stage_{self.stage_id}_chunks", chunks)
        first = chunks[0]
        if len(chunks) > 1:
            first += "\n\n[Note: Section trimmed for context limit. This is the first part of the section; later parts omitted. Focus on the excerpt above.]"
        return first

    def _build_stage7_input(self) -> str:
        """Build input for Stage 7/8 from prior stage results (no manuscript text). Stage 7 sees 1-6, Stage 8 sees 1-7."""
        parts = []
        for i in range(1, self.stage_id):
            res = self.blackboard.get_stage_result(i)
            if res is None:
                parts.append(f"Stage {i}: [Not yet run]")
                continue
            name = res.get("stage_name", f"Stage {i}")
            parts.append(f"--- {name} ---")
            parts.append(json.dumps(res, indent=0, ensure_ascii=False)[:2500])
            parts.append("")
        return trim_to_budget("\n".join(parts), max_tokens=self.max_input_tokens)

    def _build_user_prompt(self, input_text: str) -> str:
        structure_map = self.blackboard.get("structure_map", "Not yet created (Stage 1).")
        bb_full = self.blackboard.read_full_blackboard()

        return f"""
**BLACKBOARD READ** (mandatory):
{bb_full}

{REVIEWER_ARTICLE_FIRST_INSTRUCTION}

**GLOBAL CONTEXT (always reference this)**:
- Paper Structure Map: {structure_map}

**STAGE**: {self.name}

**YOUR TASK**: First, in one sentence, state your understanding of the paper's main argument and how the content below fits into it. Then evaluate according to the System Prompt criteria. Check that this section is semantically consistent with the structure map and prior findings.

**MANUSCRIPT CONTENT FOR REVIEW**:
{input_text}

Analyze strictly according to the System Prompt criteria. Output VALID JSON only.
"""

    def _inject_structure_map(self) -> str:
        """Inject structure_map and thinking protocol into system prompt."""
        structure_map = self.blackboard.get("structure_map", "Not available yet.")
        prompt = self.system_prompt
        
        # Inject thinking protocol at the beginning
        if THINKING_PROTOCOL not in prompt:
            prompt = THINKING_PROTOCOL + "\n\n" + prompt
        
        # Inject structure map placeholder if present
        if "{STRUCTURE_MAP_PLACEHOLDER}" in prompt:
            prompt = prompt.replace("{STRUCTURE_MAP_PLACEHOLDER}", structure_map)
        
        return prompt

    def _run_llm_pass(self, user_prompt: str, system_prompt: str, pass_name: str) -> Optional[Dict]:
        """Run a single LLM pass and return parsed JSON result."""
        try:
            self._log(f"[dim]Running {pass_name}...[/dim]")
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options=OLLAMA_OPTIONS
            )
            content = response['message']['content']
            data = extract_json_from_content(content)
            if data is not None:
                data["stage_name"] = self.name
                return data
            return None
        except Exception as e:
            self._log(f"[yellow]{pass_name} error: {e}[/yellow]")
            self._log("[dim]If local AI did not load: start Ollama (e.g. ollama-serve.bat or 'ollama serve'), then run with --no-cleanup-env[/dim]")
            return None

    def act(self) -> Dict:
        """Multi-pass analysis: initial assessment → deep-dive → critique."""
        self._log(f"Starting Stage {self.stage_id} (Multi-Pass Analysis)...")
        self.blackboard.set("current_stage", self.stage_id)
        bb_context = self.blackboard.read_full_blackboard()
        self._log(f"Read full blackboard ({len(bb_context)} chars)")

        input_text = self._build_input_text()
        if len(input_text.strip()) < 50:
            result = {"stage_name": self.name, "error": "Insufficient text content for this stage."}
            self.blackboard.set_stage_result(self.stage_id, result)
            return result

        system_prompt = self._inject_structure_map()
        result = None

        # PASS 1: Initial Assessment
        self._log("[bold cyan]Pass 1: Initial Assessment[/bold cyan]")
        self.blackboard.add_reasoning_note(self.stage_id, "Starting initial assessment pass")
        
        user_prompt_pass1 = self._build_user_prompt(input_text)
        for attempt in range(1, MAX_AGENT_RETRIES + 1):
            result = self._run_llm_pass(user_prompt_pass1, system_prompt, f"Pass 1 (attempt {attempt})")
            if result is not None:
                self.blackboard.add_reasoning_note(self.stage_id, f"Pass 1 completed: Initial assessment generated")
                break
            if attempt < MAX_AGENT_RETRIES:
                time.sleep(2)

        if result is None:
            result = {"stage_name": self.name, "error": "Failed to generate initial assessment after retries"}
            self.blackboard.set_stage_result(self.stage_id, result)
            return result

        # Extract quotes and add to blackboard
        for key in ["strengths", "major_issues", "violations"]:
            items = result.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("quote"):
                        self.blackboard.add_quote_evidence(
                            self.stage_id,
                            item.get("quote", ""),
                            item.get("reasoning", item.get("point", "")),
                            item.get("location", "")
                        )

        # PASS 2: Deep-Dive into Major Findings
        if ENABLE_REFLECTION_PASS:
            self._log("[bold cyan]Pass 2: Deep-Dive Analysis[/bold cyan]")
            self.blackboard.add_reasoning_note(self.stage_id, "Starting deep-dive pass on major findings")
            
            # Build deep-dive prompt focusing on major issues and strengths
            major_findings = []
            for key in ["major_issues", "violations", "strengths"]:
                items = result.get(key, [])
                if isinstance(items, list):
                    major_findings.extend(items[:3])  # Top 3 from each category
            
            deep_dive_prompt = f"""**DEEP-DIVE ANALYSIS** (Pass 2)

You have completed an initial assessment. Now perform a deeper analysis focusing on:

**Initial Assessment Summary:**
{json.dumps(result, indent=2)[:2000]}

**Manuscript Content:**
{input_text[:3000]}

**Your Task:**
1. Review your initial assessment critically
2. For each major finding, dig deeper:
   - Extract additional supporting quotes
   - Consider alternative interpretations
   - Identify what might be missing
   - Evaluate the strength of your evidence
3. Check for cross-section connections (use blackboard context)
4. Note any ambiguities or open questions
5. Refine your evaluation based on deeper analysis

Output improved JSON with deeper reasoning and more comprehensive evidence."""
            
            deep_result = self._run_llm_pass(deep_dive_prompt, system_prompt, "Pass 2")
            if deep_result is not None:
                # Merge deep-dive findings into result
                for key in ["strengths", "major_issues", "violations", "actionable_recommendations"]:
                    if key in deep_result:
                        existing = result.get(key, [])
                        if isinstance(existing, list) and isinstance(deep_result[key], list):
                            # Merge, avoiding duplicates
                            existing_ids = {json.dumps(item, sort_keys=True) for item in existing if isinstance(item, dict)}
                            for item in deep_result[key]:
                                if isinstance(item, dict):
                                    item_id = json.dumps(item, sort_keys=True)
                                    if item_id not in existing_ids:
                                        existing.append(item)
                                        existing_ids.add(item_id)
                            result[key] = existing
                        else:
                            result[key] = deep_result[key]
                
                # Add any new reasoning notes
                if deep_result.get("reasoning_notes"):
                    for note in deep_result["reasoning_notes"]:
                        self.blackboard.add_reasoning_note(self.stage_id, note)
                
                self.blackboard.add_reasoning_note(self.stage_id, "Pass 2 completed: Deep-dive analysis integrated")
                self._log("[dim]Deep-dive pass completed and integrated.[/dim]")

        # PASS 3: Critique and Refinement
        if ENABLE_REFLECTION_PASS:
            self._log("[bold cyan]Pass 3: Self-Critique & Refinement[/bold cyan]")
            self.blackboard.add_reasoning_note(self.stage_id, "Starting critique and refinement pass")
            
            critique_prompt = f"""**SELF-CRITIQUE & REFINEMENT** (Pass 3)

You have completed initial assessment and deep-dive analysis. Now critique your own work:

**Current Analysis:**
{json.dumps(result, indent=2)[:2500]}

**Your Task:**
1. Self-critique: What might you be missing or overlooking?
2. Check for consistency: Do your findings align with the structure map and prior stages?
3. Evaluate evidence strength: Are your quotes and reasoning sufficient?
4. Consider alternative perspectives: Would another editor disagree? Why?
5. Identify open questions: What needs further investigation?
6. Refine recommendations: Make them more specific and actionable

Output refined JSON with improved reasoning and any corrections."""
            
            critique_result = self._run_llm_pass(critique_prompt, system_prompt, "Pass 3")
            if critique_result is not None:
                # Update result with critique refinements
                for key in critique_result:
                    if key not in ["stage_name"]:
                        result[key] = critique_result[key]
                
                self.blackboard.add_reasoning_note(self.stage_id, "Pass 3 completed: Critique and refinement applied")
                self._log("[dim]Critique pass completed.[/dim]")

        # Save intermediate artifact into output folder
        out_dir = self.blackboard.get("output_dir", ".")
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', self.name)
        artifact_filename = f"Review_Step{self.stage_id}_{safe_name}.md"
        artifact_path = os.path.join(out_dir, artifact_filename)
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(f"# {self.name}\n\n")
            f.write(f"## Multi-Pass Analysis Results\n\n")
            f.write(f"```json\n{json.dumps(result, indent=2)}\n```\n\n")
            reasoning_notes = self.blackboard.get_reasoning_notes(self.stage_id)
            if reasoning_notes:
                f.write(f"## Reasoning Notes\n\n")
                for note in reasoning_notes:
                    f.write(f"- {note}\n")
        self._log(f"Saved artifact: {artifact_path}")

        # Write result to Blackboard
        self.blackboard.set_stage_result(self.stage_id, result)
        self.blackboard.post_to_blackboard(agent_name=self.name, content=result)

        # If Stage 1, extract and store the structure_map
        if self.stage_id == 1:
            structure_map = result.get("paper_structure_map", "")
            if structure_map:
                self.blackboard.set("structure_map", str(structure_map))
                self._log("Structure map stored on Blackboard.")

        # Append summary to context log
        summary = f"Stage {self.name}: {json.dumps(result, indent=None)[:400]}..."
        self.blackboard.append_log(summary)

        return result

# ==================================================================================
# CRITIQUE AGENT — Inter-agent peer review
# ==================================================================================

class CritiqueAgent(BaseAgent):
    """
    Peer critique agent that reviews findings from other stages to identify blind spots,
    inconsistencies, and areas needing deeper investigation.
    Runs after all main review stages are complete.
    """

    def __init__(self, target_stage_id: int, blackboard: Blackboard, tools: ToolRegistry):
        super().__init__(
            name=f"CritiqueAgent_Stage{target_stage_id}",
            blackboard=blackboard,
            tools=tools
        )
        self.target_stage_id = target_stage_id

    def act(self) -> Dict:
        """Review a specific stage's findings from a different perspective."""
        self._log(f"Critiquing Stage {self.target_stage_id} findings...")
        
        target_result = self.blackboard.get_stage_result(self.target_stage_id)
        if not target_result:
            return {"error": f"No result found for Stage {self.target_stage_id}"}

        # Get all other stage results for context
        other_stages = []
        for i in range(1, 8):
            if i != self.target_stage_id:
                other = self.blackboard.get_stage_result(i)
                if other:
                    other_stages.append((i, other))

        structure_map = self.blackboard.get("structure_map", "")
        bb_context = self.blackboard.read_full_blackboard()

        critique_system_prompt = f"""You are a senior editorial peer reviewer providing a critique of another reviewer's work.

{THINKING_PROTOCOL}

**Your Role**: Review the findings from Stage {self.target_stage_id} with fresh eyes. Look for:
1. Blind spots or overlooked issues
2. Inconsistencies with other stages
3. Weak evidence or reasoning
4. Missing connections or context
5. Alternative interpretations

Be constructive but rigorous. Your goal is to improve the quality of the review."""

        critique_prompt = f"""**PEER CRITIQUE TASK**

**Target Stage {self.target_stage_id} Findings:**
{json.dumps(target_result, indent=2)[:3000]}

**Other Stage Results (for consistency checking):**
{chr(10).join(f"Stage {i}: {json.dumps(r, indent=0)[:500]}..." for i, r in other_stages[:3])}

**Paper Structure Map:**
{structure_map[:1000]}

**Blackboard Context:**
{bb_context[:1500]}

**Your Critique Should:**
1. Identify any blind spots or missing considerations
2. Check consistency with other stages (flag contradictions)
3. Evaluate the strength of evidence and reasoning
4. Suggest alternative interpretations if applicable
5. Note any open questions that need investigation
6. Provide constructive feedback for improvement

Output JSON with:
{{
  "critique_summary": "Overall assessment of the review quality",
  "blind_spots": ["List of overlooked issues or considerations"],
  "inconsistencies": [{{"issue": "Description", "with_stage": N, "evidence": "..."}}],
  "evidence_strength": "Assessment of quote quality and reasoning",
  "alternative_interpretations": ["Alternative ways to view the findings"],
  "open_questions": ["Questions that need deeper investigation"],
  "recommendations": ["Specific improvements to the review"]
}}"""

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': critique_system_prompt},
                    {'role': 'user', 'content': critique_prompt}
                ],
                options=OLLAMA_OPTIONS
            )
            content = response['message']['content']
            critique_data = extract_json_from_content(content)
            
            if critique_data:
                critique_data["target_stage_id"] = self.target_stage_id
                critique_data["critique_agent"] = self.name
                
                # Record cross-stage observations
                for inconsistency in critique_data.get("inconsistencies", []):
                    self.blackboard.add_cross_stage_observation(
                        self.target_stage_id,
                        inconsistency.get("with_stage", 0),
                        inconsistency.get("issue", ""),
                        inconsistency.get("evidence", "")
                    )
                
                # Add open questions
                for question in critique_data.get("open_questions", []):
                    self.blackboard.add_open_question(
                        self.target_stage_id,
                        question,
                        context="Identified by critique agent",
                        priority="high"
                    )
                
                self._log(f"Critique completed for Stage {self.target_stage_id}")
                return critique_data
            else:
                return {"error": "Failed to parse critique response", "raw_content": content}
        except Exception as e:
            self._log(f"[red]Critique error: {e}[/red]")
            return {"error": str(e)}

# ==================================================================================
# ORCHESTRATOR AGENT
# ==================================================================================

class OrchestratorAgent(BaseAgent):
    """
    Routes and runs ReviewAgents in dependency order.

    Decision logic after each agent:
      - If Stage 1 returns FAIL → log a note but continue (to generate full report).
      - If any agent returns an error → log note, continue (partial report is better than none).
      - Saves progress to disk after each agent so runs can be resumed.
    """

    def __init__(self, agents: List[ReviewAgent], blackboard: Blackboard,
                 tools: ToolRegistry, progress_file: str = "editor_progress.json",
                 manuscript_filepath: Optional[str] = None):
        super().__init__("Orchestrator", blackboard, tools)
        self.agents = agents
        self.progress_file = progress_file
        self.manuscript_filepath = manuscript_filepath

    def _save_progress(self):
        results = self.blackboard.to_progress_list()
        if self.manuscript_filepath:
            payload = {"filepath": os.path.abspath(self.manuscript_filepath), "results": results}
        else:
            payload = results
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _already_done(self, stage_id: int) -> bool:
        return self.blackboard.get_stage_result(stage_id) is not None

    def act(self) -> List[Dict]:
        rprint(Panel(f"[bold green]OrchestratorAgent starting. "
                     f"Running {len(self.agents)} review agents.[/bold green]",
                     title="ASAC Review System"))

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=CONSOLE) as progress:
            task = progress.add_task("[cyan]ASAC Review in progress...[/cyan]",
                                     total=len(self.agents))

            completed = sum(1 for a in self.agents if self._already_done(a.stage_id))
            progress.update(task, completed=completed)

            for agent in self.agents:
                if self._already_done(agent.stage_id):
                    self._log(f"Skipping Stage {agent.stage_id} (already complete).")
                    progress.advance(task)
                    continue

                progress.update(task, description=f"Stage {agent.stage_id}: {agent.name}")
                stage_start = time.time()
                result = agent.act()
                stage_elapsed = time.time() - stage_start
                self._log(f"Stage {agent.stage_id} completed in {stage_elapsed:.1f}s ({stage_elapsed / 60:.1f} min)")
                self._save_progress()

                # Post-agent orchestrator decision
                if result.get("error"):
                    note = (f"Stage {agent.stage_id} ({agent.name}) returned error: "
                            f"{result['error']}. Continuing with partial data.")
                    self.blackboard.append_note(note)
                    rprint(f"[bold yellow]Orchestrator note:[/bold yellow] {note}")

                if agent.stage_id == 1:
                    decision = result.get("decision", "")
                    if decision == "FAIL":
                        note = ("Stage 1 compliance FAIL. Continuing full review "
                                "to provide complete developmental feedback.")
                        self.blackboard.append_note(note)
                        rprint(f"[bold yellow]Orchestrator note:[/bold yellow] {note}")

                progress.advance(task)

        # Inter-agent critique phase: review each stage's findings
        if ENABLE_REFLECTION_PASS:
            rprint("\n[bold cyan]Starting Inter-Agent Critique Phase...[/bold cyan]")
            critique_results = []
            for agent in self.agents:
                if agent.stage_id <= 7:  # Critique main review stages
                    critique_agent = CritiqueAgent(agent.stage_id, self.blackboard, self.tools)
                    critique_result = critique_agent.act()
                    critique_results.append(critique_result)
                    self._log(f"Critique completed for Stage {agent.stage_id}")
            
            # Store critique results on blackboard
            self.blackboard.set("critique_results", critique_results)
            rprint("[bold green]Critique phase complete.[/bold green]")

        rprint("[bold green]All agents complete.[/bold green]")
        return self.blackboard.to_progress_list()

# ==================================================================================
# STAGE SPECIFICATIONS (prompts unchanged from original)
# ==================================================================================

def build_stage_specs() -> List[Dict]:
    """
    Returns the stage specs (Stages 1-8). Each spec defines id, name, section_keys,
    max_input_tokens, and the system_prompt.
    Section-key resolution and content assembly are handled by ReviewAgent.
    """
    specs = [
        {
            "id": 1,
            "name": "Compliance & Formatting (Desk Screen)",
            "section_keys": ["Title & Abstract", "Introduction", "Other"],
            "max_input_tokens": 2800,
            "system_prompt": """You are a veteran ASAC Divisional Editor doing a 60-second desk screen.
            **STRICT RULES FOR ASAC 2026**:
            1. **APA 6th Edition EXACTLY** (NOT 7th).
            2. **Max 30 pages** (inclusive of everything EXCEPT references, tables, diagrams, captions).
            3. **Anonymized**: No author names, university names, or self-revealing citations ("in our previous work").
            4. **Format**: Times New Roman 11pt, Double Spaced.
            5. **Structure**: Check for standard high-level headings (Introduction, Literature Review, Method, Results, Discussion).
            
            **Flexible Section Recognition**:
            Papers often use non-standard headings. Map these as equivalent: Methods ↔ Methodology, Methods & Materials, Procedure, etc.; Results ↔ Findings, Data Analysis, Empirical Results; Discussion ↔ Conclusions, Implications, Interpretation, Discussion and Conclusions. Only list a section in "missing_sections" if there is genuinely no content that belongs there. Do NOT flag "Methods" as missing when the author used "3 Methodology" or "4. Findings" instead of "Results".
            
            **PROCESS**:
            1. Create a "Paper Structure Map" that lists every major heading AND, for each, the **main claim or function** of that section (e.g. "Introduction: positions the gap as X and states the research question as Y"). This semantic skeleton helps later reviewers judge whether the whole paper tells one coherent story.
            2. In one sentence, state what you understand to be the paper's **overall argument or contribution** (so later stages can check consistency).
            3. Check for missing or mislabeled sections (apply flexible recognition above).
            4. Apply strict compliance rules.

            **CRITICAL: EVIDENCE REQUIRED**
            For every violation or warning, you MUST extract the exact text snippet or page number that proves it.

            **OUTPUT JSON**:
            {
              "decision": "PASS" | "FAIL" | "WARNING",
              "paper_structure_map": "For each heading: heading name + 1 sentence on main claim/function of that section. Then 1 sentence: paper's overall argument/contribution.",
              "structure_audit": {
                  "missing_sections": ["list missing"],
                  "labeling_suggestions": ["e.g. Change 'My Analysis' to 'Results'"]
              },
              "violations": [
                {"issue": "Description of violation", "quote": "Exact text snippet or page number proof", "fix": "How to correct it"}
              ],
              "developmental_feedback": "Kind advice for students if they failed."
            }"""
        },
        {
            "id": 2,
            "name": "Relevance & Interest (Theme Fit)",
            "section_keys": ["Title & Abstract", "Introduction", "Discussion & Conclusions"],
            "max_input_tokens": 2800,
            "system_prompt": """IMPORTANT: You have already seen this Paper Structure Map from the desk screen:
{STRUCTURE_MAP_PLACEHOLDER}

You are an ASAC Editor evaluating fit for the 2026 theme: "Blue Sky Thinking: Harnessing the Future".
            **CRITERIA**:
            1. **Theme Fit**: Does it relate to future-focused business issues, emerging tech, sustainability, or new ways of thinking?
            2. **Canadian/Global Relevance**: Is there a "So What?" for the ASAC community (academics + practitioners).
            3. **Novelty**: Avoid "me-too" descriptive work.
            
            **SEMANTIC COHERENCE**: Check that the paper's stated theme/contribution (from the Structure Map and abstract) matches what the Introduction and Discussion actually argue. Flag if the title/abstract promise something the body does not deliver.
            
            **PROCESS**:
            Think step-by-step. Form a one-sentence view of the paper's overall argument, then assess theme fit. Use the Paper Structure Map and previous findings (e.g. if the Title (Stage 1) is weak, does it hurt the Theme Fit here?).
            
            **CRITICAL: EVIDENCE REQUIRED**
            For EVERY strength, issue, or recommendation you make, you MUST include:
            1. The exact direct quote from the manuscript (in "quotes").
            2. A one-sentence 'Reasoning' explaining why this quote supports your comment.

            **OUTPUT JSON**:
            {
              "decision": "STRONG FIT" | "WEAK FIT" | "OFF TOPIC",
              "strengths": [
                {"point": "Why it fits/is relevant", "quote": "Exact text snippet", "reasoning": "Why this matters", "fix": "N/A or Enhancement"}
              ],
              "major_issues": [
                {"point": "Why it misses the mark", "quote": "Exact text snippet", "reasoning": "Why this is a problem", "fix": "Specific correction"}
              ],
              "connection_analysis": "1-2 sentences explicitly linking this section to others.",
              "actionable_recommendations": ["1. ...", "2. ..."],
              "theme_alignment_score": 1-10
            }"""
        },
        {
            "id": 3,
            "name": "Conceptual & Theoretical Foundation",
            "section_keys": ["Introduction", "Theory & Literature Positioning"],
            "max_input_tokens": 2800,
            "system_prompt": """IMPORTANT: You have already seen this Paper Structure Map from the desk screen:
{STRUCTURE_MAP_PLACEHOLDER}

You are an ASAC Editor evaluating the theoretical contribution.
            **CRITERIA**:
            1. **Gap**: Is there a genuine gap identified in the lit review?
            2. **Theory**: Is a clear theoretical framework used (e.g., RBV, Institutional Theory)?
            3. **Logic**: Are hypotheses/propositions logically derived?
            
            **SEMANTIC COHERENCE**: The theory and gap should align with the paper's overall argument (from the Structure Map). If the intro claims one contribution but the theory section supports a different one, flag it. Check that the same key terms and constructs are used consistently.
            
            **PROCESS**:
            Think step-by-step. State how this section supports (or fails to support) the paper's main argument. Contrast their claims with standard literature requirements. Link to other sections.
            **THEORY RECOMMENDATION**: If the current theory is weak, suggest a SPECIFIC relevant theory (e.g. Social Identity Theory).
            
            **CRITICAL: EVIDENCE REQUIRED**
            For EVERY strength, issue, or recommendation you make, you MUST include:
            1. The exact direct quote from the manuscript (in "quotes").
            2. A one-sentence 'Reasoning' explaining why this quote supports your comment.

            **OUTPUT JSON**:
            {
              "theoretical_contribution": "High" | "Medium" | "Low",
              "strengths": [
                 {"point": "Strong theoretical aspect", "quote": "Exact text snippet", "reasoning": "Why this is good"}
              ],
              "major_issues": [
                 {"point": "Missing theory or logic", "quote": "Text snippet where it should be or is weak", "reasoning": "Why this is a problem", "fix": "Specific correction"}
              ],
              "connection_analysis": "1-2 sentences linking this to other sections.",
              "minor_issues": ["list smaller points"],
              "actionable_recommendations": ["1. ...", "2. ..."],
              "developmental_suggestions": "Specific advice to strengthen the theory (Student Focus)."
            }"""
        },
        {
            "id": 4,
            "name": "Methodology & Analysis",
            "section_keys": ["Methods (rigor + replicability)", "Results & Evidence Alignment"],
            "max_input_tokens": 3200,
            "system_prompt": """IMPORTANT: You have already seen this Paper Structure Map from the desk screen:
{STRUCTURE_MAP_PLACEHOLDER}

You are an ASAC Methodology Expert.
            **CRITERIA**:
            1. **Appropriateness**: Is the method (Quant/Qual/Mixed) right for the question?
            2. **Rigor**: Valid measures? Trustworthy qualitative coding?
            3. **Ethics/Sampling**: No unjustified convenience samples. Robustness checks?
            
            **SEMANTIC COHERENCE**: The method should match the research question and theory (from the Structure Map). If the paper claims to test X but the design actually measures Y, or if Results are discussed before the method is clear, flag the inconsistency.
            
            **PROCESS**:
            Think step-by-step. State how the method serves (or fails to serve) the paper's overall argument. Act like a reviewer checking for fatal flaws.
            
            **CRITICAL: EVIDENCE REQUIRED**
            For EVERY strength, issue, or recommendation you make, you MUST include:
            1. The exact direct quote from the manuscript (in "quotes").
            2. A one-sentence 'Reasoning' explaining why this quote supports your comment.

            **OUTPUT JSON**:
            {
              "method_rigor": "High" | "Medium" | "Low",
              "strengths": [
                 {"point": "Solid methodological choice", "quote": "Text describing the method", "reasoning": "Why this is rigorous"}
              ],
              "major_issues": [
                {"point": "Major validity threat", "quote": "Text describing the flawed method", "reasoning": "Why this invalidates results", "fix": "How to fix it"}
              ],
              "connection_analysis": "1-2 sentences linking Method to Theory or Implications.",
              "minor_issues": ["list smaller points"],
              "actionable_recommendations": ["1. ...", "2. ..."]
            }"""
        },
        {
            "id": 5,
            "name": "Readability & Writing Quality",
            "section_keys": ["Introduction", "Discussion & Conclusions", "Results & Evidence Alignment"],
            "max_input_tokens": 2800,
            "system_prompt": """IMPORTANT: You have already seen this Paper Structure Map from the desk screen:
{STRUCTURE_MAP_PLACEHOLDER}

You are an ASAC Copy Editor.
            **CRITERIA**:
            1. **Clarity**: Logical flow (Intro -> Lit -> Method -> etc). Does the reader always know "why this section now"?
            2. **Tone**: Scholarly but accessible. No jargon overload.
            3. **Bilingual Friendly**: Is the English clear enough for a bilingual audience?
            
            **SEMANTIC COHERENCE**: Does the writing make the paper's **one main argument** easy to follow? Flag places where the narrative drifts or where key terms change meaning. Check that the Structure Map's "overall argument" is actually readable in the prose.
            
            **PROCESS**:
            Think step-by-step. Assess flow and sentence structure. Link to Stage 1 (Compliance) and to the paper's stated argument.
            
            **CRITICAL: EVIDENCE REQUIRED**
            For EVERY strength, issue, or recommendation you make, you MUST include:
            1. The exact direct quote from the manuscript (in "quotes").
            2. A one-sentence 'Reasoning' explaining why this quote supports your comment.

            **OUTPUT JSON**:
            {
              "readability_score": "High" | "Medium" | "Low",
              "major_issues": [
                {"point": "Grammar/Style issue", "quote": "Example of the bad writing", "reasoning": "Why it's unclear", "fix": "Suggested rewrite"}
              ],
              "strengths": [
                 {"point": "Good writing style", "quote": "Example of clear writing", "reasoning": "Why it works"}
              ],
              "connection_analysis": "1-2 sentences linking Writing quality to clarity of the Theory/Method.",
              "actionable_recommendations": ["1. ..."]
            }"""
        },
        {
            "id": 6,
            "name": "Implications & Overall Impact",
            "section_keys": ["Discussion & Conclusions"],
            "max_input_tokens": 2800,
            "system_prompt": """IMPORTANT: You have already seen this Paper Structure Map from the desk screen:
{STRUCTURE_MAP_PLACEHOLDER}

You are the ASAC Division Chair making a final recommendation.
            **CRITERIA**:
            1. **Implications**: Are there clear Practical/Policy AND Theoretical implications?
            2. **Overall Value**: Does this advance the field?
            3. **Student Potential**: If this is a student paper, is it promising?
            
            **NARRATIVE ARC**: Synthesize the paper as one story. Does it have a single clear argument from problem → theory → method → evidence → implications? If not, say so explicitly: "The paper does not yet tell one coherent story because ..." Use the Paper Structure Map and Prior Stage Summaries to check consistency across the whole manuscript.
            
            **PROCESS**:
            Think step-by-step. First state the paper's intended contribution and whether the manuscript delivers it as a coherent whole. Then synthesize all previous findings into your recommendation.
            
            **CRITICAL: EVIDENCE REQUIRED**
            For EVERY strength, issue, or recommendation you make, you MUST include:
            1. The exact direct quote from the manuscript (in "quotes").
            2. A one-sentence 'Reasoning' explaining why this quote supports your comment.

            **OUTPUT JSON**:
            {
              "final_recommendation": "ACCEPT" | "MINOR REVISE" | "MAJOR REVISE" | "REJECT",
              "impact_score": 1-5,
              "impact_summary": "2-3 sentences on the paper's value and whether it tells one coherent story.",
              "strengths": [
                {"point": "Key contribution", "quote": "Author's claim of contribution", "reasoning": "Why it matters"}
              ],
              "major_issues": [
                 {"point": "Missed opportunity", "quote": "Text showing the missed implication", "reasoning": "Why it matters", "fix": "Suggestion"}
              ],
              "connection_analysis": "Summary connecting the dots across all sections.",
              "actionable_recommendations": ["1. ...", "2. ..."],
              "student_developmental_feedback": "Warm, constructive mentorship feedback.",
              "confidential_comments_to_editor": "Honest assessment for the chair."
            }"""
        },
        {
            "id": 7,
            "name": "Coherence & Cross-Section Consistency",
            "section_keys": ["ALL"],
            "max_input_tokens": 4000,
            "system_prompt": """IMPORTANT: You have already seen this Paper Structure Map from the desk screen:
{STRUCTURE_MAP_PLACEHOLDER}

You are a senior editor checking the whole manuscript for coherence and consistency. You receive summaries of Stages 1-6 (compliance, theme fit, theory, method, readability, implications). Your job is to spot contradictions and terminology drift across sections.

**CRITERIA**:
1. **Contradictions**: Do any stage findings contradict each other or the structure map? (e.g. Stage 1 says "no abstract" but Stage 2 discusses the abstract.)
2. **Title/Abstract vs. Body**: Do the claims in the abstract and title match what the body actually argues and finds?
3. **Terminology**: Are key terms and constructs used consistently, or do they shift meaning between sections?
4. **One coherent story**: Does the paper tell a single clear argument from problem → theory → method → evidence → implications?

**PROCESS**:
Review the Stage 1-6 summaries below. For every contradiction or inconsistency, cite the exact finding or quote from the stage summaries. Rate overall coherence.

**CRITICAL: EVIDENCE REQUIRED**
For every contradiction or issue, you MUST reference which stage(s) and what was said (quote or paraphrase from the summaries).

**OUTPUT JSON**:
{
  "coherence_score": 1-10,
  "structure_coherence": "1-2 sentences: Does the paper tell one coherent story?",
  "contradictions": [
    {"quote_a": "Exact or paraphrased finding from one stage", "quote_b": "Conflicting finding from another", "issue": "Why this is a contradiction"}
  ],
  "terminology_consistency": [
    {"term": "key term", "issue": "How it shifts or is inconsistent", "locations": "Which stages/sections"}
  ],
  "abstract_body_alignment": "Do title/abstract match the body? Quote or paraphrase evidence.",
  "actionable_recommendations": ["1. ...", "2. ..."]
}"""
        },
        {
            "id": 8,
            "name": "Reflector & Depth Critic",
            "section_keys": ["ALL"],
            "max_input_tokens": 6000,
            "system_prompt": """You are the Referee who reads the entire blackboard and challenges weak or shallow findings.
For every major finding from previous agents, ask: "Is this deep enough for an ASAC editor? What evidence is missing?"
Then write improved, deeper versions and post them back to the blackboard.
Output JSON with "improved_findings" list."""
        }
    ]
    # Note: THINKING_PROTOCOL is injected automatically in _inject_structure_map()
    return specs

# ==================================================================================
# OLLAMA PREFLIGHT — ensure local AI is running before starting
# ==================================================================================

def unload_model(model_name: str) -> bool:
    """Unload the model from GPU memory so the first chat gets a fresh load (helps avoid VRAM fragmentation)."""
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


def is_ollama_running(timeout_sec: float = 3) -> bool:
    """Quick check if Ollama is already responding at localhost:11434. Does not start anything."""
    try:
        with urllib.request.urlopen("http://localhost:11434", timeout=int(timeout_sec)) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_ollama(timeout_sec: int = 60, check_interval: int = 2) -> bool:
    """Wait for Ollama server at localhost:11434. Returns True when ready."""
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


def check_model_available() -> bool:
    """Verify the review model exists. Returns True if OK."""
    try:
        models = ollama.list()
        model_list = models.get("models") if isinstance(models, dict) else getattr(models, "models", [])
        if not model_list:
            rprint("[yellow]No models listed (Ollama may still be loading).[/yellow]")
            return False
        # Ollama API can return "model" or "name" for the model name
        names = []
        for m in model_list:
            if isinstance(m, dict):
                names.append(m.get("model") or m.get("name", ""))
            else:
                names.append(getattr(m, "model", None) or getattr(m, "name", "") or str(m))
        names = [n for n in names if n]
        if MODEL_NAME in names:
            return True
        if any(str(n).split(":")[0] == MODEL_NAME for n in names):
            return True
        rprint(f"[bold red]Model '{MODEL_NAME}' not found.[/bold red]")
        rprint(f"[dim]Models currently available: {', '.join(names[:8])}{'...' if len(names) > 8 else ''}[/dim]")
        return False
    except Exception as e:
        rprint(f"[yellow]Could not list models: {e}[/yellow]")
        return False


# ==================================================================================
# ENVIRONMENT CLEANUP
# ==================================================================================

def cleanup_environment(do_cleanup: bool = True):
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
                subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                rprint("[dim]- Killed running ollama processes.[/dim]")
            except Exception as e:
                rprint(f"[dim red]- Failed to kill ollama: {e}[/dim red]")

            try:
                subprocess.run(f'taskkill /F /IM python.exe /FI "PID ne {current_pid}"',
                               shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                rprint("[dim]- Killed other python processes.[/dim]")
            except Exception as e:
                rprint(f"[dim red]- Failed to kill other python processes: {e}[/dim red]")

            rprint(f"[bold green]Using Ollama from: {OLLAMA_DIR} (IPEX-LLM)[/bold green]")
            try:
                ollama_path = OLLAMA_SERVE_PATH
                if os.path.exists(ollama_path):
                    # Run the .bat with its directory as cwd so it finds ollama.exe and libs (IPEX-LLM only; no system/PATH ollama)
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

        rprint("[bold yellow]Cleaning up old intermediate artifacts...[/bold yellow]")
        for fpath in glob.glob(os.path.join(OUTPUT_DIR, "Review_Step*.md")):
            try:
                os.remove(fpath)
                rprint(f"[dim]- Deleted old artifact: {fpath}[/dim]")
            except OSError as e:
                rprint(f"[dim red]- Error deleting {fpath}: {e}[/dim red]")
    else:
        rprint(f"[dim]Skipping cleanup (--no-cleanup-env). Waiting for Ollama at localhost:11434.[/dim]")
        rprint(f"[dim]If using IPEX-LLM, start it from: {OLLAMA_DIR}[/dim]")
        if not wait_for_ollama(timeout_sec=60):
            sys.exit(1)

# ==================================================================================
# REPORT GENERATION (unchanged logic)
# ==================================================================================

def generate_professional_report(results: List[Dict], base_name: str, filepath: str, blackboard: Optional[Blackboard] = None, out_dir: str = ".") -> str:
    """Produces the clean professional Markdown report with reasoning trails."""
    md_out = os.path.join(out_dir, f"{base_name}_ASAC_Final_Decision.md")

    s1 = results[0] if results else {}
    s2 = results[1] if len(results) > 1 else {}
    s3 = results[2] if len(results) > 2 else {}
    s4 = results[3] if len(results) > 3 else {}
    s5 = results[4] if len(results) > 4 else {}
    s6 = results[5] if len(results) > 5 else {}
    s7 = results[6] if len(results) > 6 else {}

    compliance_pass = s1.get("decision", "FAIL") == "PASS"
    if not compliance_pass:
        overall_rec = "**Return for Compliance Fix (Major Revision)**"
        quality_score = "2.5/5"
    else:
        overall_rec = s6.get("final_recommendation", "Major Revision")
        # Derive score from final_recommendation if impact_score not provided by model
        impact = s6.get("impact_score")
        if impact is not None and isinstance(impact, (int, float)):
            quality_score = f"{min(5, max(1, int(impact)))}/5"
        else:
            rec_to_score = {"ACCEPT": 5, "MINOR REVISE": 4, "MAJOR REVISE": 3, "REJECT": 2}
            quality_score = f"{rec_to_score.get(str(overall_rec).upper(), 3)}/5"

    with open(md_out, 'w', encoding='utf-8') as f:
        f.write(f"# ASAC 2026 Divisional Editor Report\n\n")
        f.write(f"**Manuscript ID**: {base_name}\n")
        f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y')}\n")
        f.write(f"**File**: {os.path.basename(filepath)}\n")
        f.write(f"**Overall Recommendation**: {overall_rec}\n")
        f.write(f"**Quality Score**: {quality_score}\n\n")

        structure_map = s1.get("paper_structure_map", "Not generated.")
        f.write(f"**Paper Structure Map** (created from actual headings)\n")
        if isinstance(structure_map, list):
            f.write(" -> ".join(str(i) for i in structure_map) + "\n\n")
        else:
            f.write(f"{structure_map}\n\n")

        f.write(f"**Executive Summary**\n{s6.get('impact_summary', 'Promising work requiring structural fixes.')}\n\n")

        f.write("**1. Compliance & Formatting (Desk Screen)**\n")
        f.write(f"• **Status**: {s1.get('decision', 'FAIL')}\n")
        if s1.get("violations"):
            f.write("• Issues (with exact evidence):\n")
            for v in s1["violations"]:
                f.write(f"  - {v.get('issue')}\n")
                if v.get('quote'):
                    f.write(f'    Quote: "{v.get("quote")[:150]}..."\n')
                f.write(f"    Fix: {v.get('fix','')}\n")
        f.write("\n")

        def write_section(title: str, stage: Dict):
            f.write(f"**{title}**\n")
            if stage.get("strengths"):
                f.write("• **Strengths**:\n")
                for s in stage.get("strengths", [])[:3]:
                    f.write(f"  • {s.get('point', '')}\n")
                    f.write(f'    **Quote**: "{s.get("quote", "")[:200]}..."\n')
                    f.write(f"    **Reasoning**: {s.get('reasoning', '')}\n")
                    if s.get('fix') and s.get('fix') != "None":
                        f.write(f"    **Fix**: {s.get('fix')}\n")
            issues = stage.get("major_issues", []) + stage.get("minor_issues", [])
            if issues:
                f.write("• **Issues**:\n")
                for issue in issues[:4]:
                    if isinstance(issue, dict):
                        f.write(f"  • {issue.get('point', '')}\n")
                        f.write(f'    **Quote**: "{issue.get("quote", "")[:200]}..."\n')
                        f.write(f"    **Reasoning**: {issue.get('reasoning', '')}\n")
                        f.write(f"    **Fix**: {issue.get('fix', '')}\n")
                    else:
                        f.write(f"  • {issue}\n")
            if stage.get("connection_analysis"):
                f.write(f"• **Connection**: {stage.get('connection_analysis')}\n")
            f.write("\n")

        write_section("2. Relevance & Theme Fit (Blue Sky Thinking)", s2)
        write_section("3. Conceptual & Theoretical Foundation", s3)
        write_section("4. Methodology & Analysis", s4)
        write_section("5. Readability & Writing Quality", s5)
        write_section("6. Implications & Overall Impact", s6)

        if s7:
            f.write("**7. Coherence & Cross-Section Consistency**\n")
            f.write(f"• **Coherence Score**: {s7.get('coherence_score', '—')}/10\n")
            if s7.get("structure_coherence"):
                f.write(f"• **Structure Coherence**: {s7['structure_coherence']}\n")
            if s7.get("abstract_body_alignment"):
                f.write(f"• **Abstract–Body Alignment**: {s7['abstract_body_alignment']}\n")
            if s7.get("contradictions"):
                f.write("• **Contradictions**:\n")
                for c in s7["contradictions"][:5]:
                    if isinstance(c, dict):
                        f.write(f"  - {c.get('issue', '')}\n")
                        if c.get('quote_a'):
                            f.write(f'    Quote A: "{str(c.get("quote_a"))[:150]}..."\n')
                        if c.get('quote_b'):
                            f.write(f'    Quote B: "{str(c.get("quote_b"))[:150]}..."\n')
                    else:
                        f.write(f"  - {c}\n")
            if s7.get("terminology_consistency"):
                f.write("• **Terminology**:\n")
                for t in s7["terminology_consistency"][:4]:
                    if isinstance(t, dict):
                        f.write(f"  - {t.get('term', '')}: {t.get('issue', '')}\n")
                    else:
                        f.write(f"  - {t}\n")
            if s7.get("actionable_recommendations"):
                f.write("• **Recommendations**: " + "; ".join(s7["actionable_recommendations"][:3]) + "\n")
            f.write("\n")

        f.write("**Prioritized Revision Plan** (do in this order)\n")
        f.write("1. **IMMEDIATE (7–10 days)**: Fix all compliance issues + relabel sections correctly.\n")
        for i, rec in enumerate(s3.get("actionable_recommendations", [])[:3], 2):
            f.write(f"{i}. {rec}\n")
        f.write("\n")

        f.write(f"**One-Sentence Summary for Author**\n")
        f.write(f'"{s6.get("impact_summary", "Strong potential – fix compliance and structure for fast-track consideration.")}"\n\n')

        if s6.get("student_developmental_feedback"):
            f.write("**Developmental Feedback (if student paper)**\n")
            f.write(s6["student_developmental_feedback"] + "\n\n")

        f.write("**Confidential Notes to Editor**\n")
        f.write(s6.get("confidential_comments_to_editor", "Worth saving after compliance fix.") + "\n")

        # Add reasoning trails section if blackboard is available
        if blackboard:
            f.write("\n---\n\n")
            f.write("## Reasoning Trails & Agent Thinking\n\n")
            f.write("This section shows the multi-pass thinking process used by each agent.\n\n")
            
            for idx, stage in enumerate(results[:7], 1):
                stage_name = stage.get("stage_name", f"Stage {idx}")
                f.write(f"### {stage_name}\n\n")
                
                # Reasoning notes
                reasoning_notes = blackboard.get_reasoning_notes(idx)
                if reasoning_notes:
                    f.write("**Reasoning Process:**\n")
                    for note in reasoning_notes:
                        f.write(f"- {note}\n")
                    f.write("\n")
                
                # Quote evidence
                quote_evidence = blackboard.get_quote_evidence(idx)
                if quote_evidence:
                    f.write("**Extracted Evidence:**\n")
                    for evidence in quote_evidence[:5]:  # Top 5 quotes
                        f.write(f"- **Quote**: \"{evidence.get('quote', '')[:200]}...\"\n")
                        f.write(f"  **Analysis**: {evidence.get('analysis', '')[:150]}...\n")
                        if evidence.get('location'):
                            f.write(f"  **Location**: {evidence.get('location')}\n")
                    f.write("\n")
                
                # Cross-stage observations
                cross_obs = blackboard.get_cross_stage_observations(idx)
                if cross_obs:
                    f.write("**Cross-Stage Connections:**\n")
                    for obs in cross_obs[:3]:  # Top 3 observations
                        f.write(f"- **Connection**: {obs.get('observation', '')[:200]}...\n")
                        if obs.get('evidence'):
                            f.write(f"  **Evidence**: {obs.get('evidence', '')[:150]}...\n")
                    f.write("\n")
                
                # Open questions
                open_questions = blackboard.get_open_questions(idx, unresolved_only=True)
                if open_questions:
                    f.write("**Open Questions:**\n")
                    for q in open_questions[:3]:  # Top 3 questions
                        f.write(f"- **{q.get('priority', 'medium').upper()}**: {q.get('question', '')}\n")
                        if q.get('context'):
                            f.write(f"  *Context*: {q.get('context', '')[:100]}...\n")
                    f.write("\n")
            
            # Critique results
            critique_results = blackboard.get("critique_results", [])
            if critique_results:
                f.write("### Inter-Agent Critique Results\n\n")
                for critique in critique_results[:5]:  # Top 5 critiques
                    target_stage = critique.get("target_stage_id", "?")
                    f.write(f"**Critique of Stage {target_stage}:**\n")
                    if critique.get("critique_summary"):
                        f.write(f"- {critique.get('critique_summary', '')[:300]}...\n")
                    if critique.get("blind_spots"):
                        f.write(f"- **Blind Spots**: {', '.join(critique.get('blind_spots', [])[:3])}\n")
                    f.write("\n")

    return md_out


def generate_evidence_trail_report(results: List[Dict], base_name: str, filepath: str, blackboard: Optional[Blackboard] = None, out_dir: str = ".") -> str:
    """
    Generate an editor-style report where each finding shows:
    - What the editors found
    - Direct quote evidence
    - Why this matters editorially (reasoning)
    - How it connects to other findings
    - Multi-pass reasoning trails
    """
    md_out = os.path.join(out_dir, f"{base_name}_ASAC_Evidence_Trail.md")
    stage_titles = [
        "Compliance & Formatting",
        "Relevance & Theme Fit",
        "Conceptual & Theoretical Foundation",
        "Methodology & Analysis",
        "Readability & Writing Quality",
        "Implications & Overall Impact",
        "Coherence & Cross-Section Consistency",
    ]
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# ASAC Evidence Trail — Editor Perspective\n\n")
        f.write(f"**Manuscript**: {os.path.basename(filepath)}\n")
        f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y')}\n\n")
        f.write("Every finding below is backed by a direct quote and editorial reasoning.\n\n")
        f.write("---\n\n")

        for idx, stage in enumerate(results):
            title = stage_titles[idx] if idx < len(stage_titles) else stage.get("stage_name", f"Stage {idx + 1}")
            f.write(f"## {idx + 1}. {title}\n\n")

            def write_finding(finding_type: str, items: List, severity: str = ""):
                if not items or not isinstance(items, list):
                    return
                for it in items:
                    if not isinstance(it, dict):
                        f.write(f"- **Finding**: {str(it)[:200]}\n\n")
                        continue
                    point = it.get("point") or it.get("issue") or ""
                    quote = it.get("quote") or ""
                    reasoning = it.get("reasoning") or it.get("editor_reasoning") or ""
                    fix = it.get("fix") or ""
                    f.write(f"**From an editor's perspective:** {point}\n\n")
                    if quote:
                        f.write(f"- **Direct quote**: \"{quote[:300]}{'...' if len(str(quote)) > 300 else ''}\"\n\n")
                    if reasoning:
                        f.write(f"- **Why this matters**: {reasoning}\n\n")
                    if severity:
                        f.write(f"- **Severity**: {severity}\n\n")
                    if fix and str(fix).strip() and str(fix).lower() != "n/a":
                        f.write(f"- **Fix**: {fix}\n\n")
                    f.write("\n")

            write_finding("Strengths", stage.get("strengths", []), "strength")
            write_finding("Violations", stage.get("violations", []), "major")
            write_finding("Major issues", stage.get("major_issues", []), "major")
            write_finding("Minor issues", stage.get("minor_issues", []), "minor")

            if stage.get("connection_analysis"):
                f.write(f"**Connection to other sections**: {stage['connection_analysis']}\n\n")

            if stage.get("contradictions"):
                f.write("**Contradictions (Stage 7)**\n\n")
                for c in stage.get("contradictions", []):
                    if isinstance(c, dict):
                        f.write(f"- {c.get('issue', '')}\n")
                        if c.get("quote_a"):
                            f.write(f'  Quote A: "{str(c["quote_a"])[:150]}..."\n')
                        if c.get("quote_b"):
                            f.write(f'  Quote B: "{str(c["quote_b"])[:150]}..."\n')
                    f.write("\n")
            
            # Add reasoning trail for this stage if available
            if blackboard:
                reasoning_notes = blackboard.get_reasoning_notes(idx + 1)
                if reasoning_notes:
                    f.write("**Agent Reasoning Process:**\n")
                    for note in reasoning_notes:
                        f.write(f"- {note}\n")
                    f.write("\n")
                
                quote_evidence = blackboard.get_quote_evidence(idx + 1)
                if quote_evidence:
                    f.write("**Additional Evidence Extracted:**\n")
                    for evidence in quote_evidence:
                        f.write(f"- \"{evidence.get('quote', '')[:250]}...\" → {evidence.get('analysis', '')[:150]}...\n")
                    f.write("\n")
            
            f.write("---\n\n")
        
        # Add summary of cross-stage observations and open questions
        if blackboard:
            f.write("## Cross-Stage Analysis Summary\n\n")
            
            all_cross_obs = blackboard.get_cross_stage_observations()
            if all_cross_obs:
                f.write("### Cross-Stage Observations\n\n")
                for obs in all_cross_obs[:10]:  # Top 10
                    f.write(f"- **Stages {obs.get('from_stage')} ↔ {obs.get('to_stage')}**: {obs.get('observation', '')[:200]}...\n")
                f.write("\n")
            
            all_open_questions = blackboard.get_open_questions(unresolved_only=True)
            if all_open_questions:
                f.write("### Unresolved Open Questions\n\n")
                for q in all_open_questions:
                    f.write(f"- **[{q.get('priority', 'medium').upper()}] Stage {q.get('stage_id')}**: {q.get('question', '')}\n")
                f.write("\n")

    return md_out


def run_fact_verification(results: List[Dict], full_text: str) -> Optional[Dict]:
    """
    Run fact verification pass: review each stage's findings against the full manuscript.
    Returns structured JSON with verified_findings, flagged_for_review, possible_misrepresentations, items_to_review.
    """
    max_manuscript_chars = 18000
    max_results_chars = 10000
    manuscript = full_text.strip()
    if len(manuscript) > max_manuscript_chars:
        manuscript = manuscript[:max_manuscript_chars] + "\n\n[... manuscript trimmed for context ...]"
    results_str = json.dumps(results, indent=0, ensure_ascii=False)
    if len(results_str) > max_results_chars:
        results_str = results_str[:max_results_chars] + "\n... [truncated]"

    user_prompt = f"""**FULL MANUSCRIPT** (use this to verify that stage findings are grounded in the whole article):

{manuscript}

---

**STAGE-BY-STAGE REVIEW FINDINGS** (check each finding's quote and claim against the full manuscript above):

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
                ctx = str(item.get('full_context', ''))
                f.write(f"- **Full context**: {ctx[:500] + '...' if len(ctx) > 500 else ctx}\n")
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

def main():
    parser = argparse.ArgumentParser(description="Journal Editor Coach — Agent-Based ASAC 6-Stage Review")
    parser.add_argument("filepath", help="Path to manuscript (PDF, DOCX, or TXT)")
    parser.add_argument("--no-cleanup-env", action="store_true",
                        help="Do not kill Ollama/Python or restart Ollama; only wait for existing server.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore saved progress and run a full new review (clear blackboard from last use).")
    args = parser.parse_args()

    progress_file = os.path.join(OUTPUT_DIR, "editor_progress.json")
    if getattr(args, "fresh", False):
        try:
            if os.path.exists(progress_file):
                os.remove(progress_file)
            for fpath in glob.glob(os.path.join(OUTPUT_DIR, "Review_Step*.md")):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
            rprint("[yellow]Cleared saved progress (--fresh). Starting full new review.[/yellow]\n")
        except OSError as e:
            rprint(f"[red]Could not remove progress file: {e}[/red]")

    cleanup_environment(do_cleanup=not args.no_cleanup_env)

    if not (args.filepath and args.filepath.strip()):
        rprint("[bold red]No manuscript file provided.[/bold red]")
        rprint("Drag and drop a PDF/DOCX onto the .bat file, or run: python journal_editor_coach.py \"path\\to\\manuscript.pdf\"")
        return
    if not os.path.exists(args.filepath):
        rprint(f"[bold red]File not found: {args.filepath}[/bold red]")
        return

    rprint("[dim]Checking that review model is available...[/dim]")
    if not check_model_available():
        sys.exit(1)
    rprint("[bold green]Model ready. Full review will take approximately 45–90 minutes.[/bold green]\n")

    rprint("[dim]Unloading model from memory for a fresh start...[/dim]")
    unload_model(MODEL_NAME)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rprint(f"[dim]Output folder: {os.path.abspath(OUTPUT_DIR)} (delete for fresh start)[/dim]\n")

    # --- 1. Ingest ---
    rprint("[bold green]Reading Manuscript...[/bold green]")
    full_text = extract_text(args.filepath)
    rprint(f"Loaded {len(full_text)} chars.")

    # --- 2. Section split ---
    sections_map = heuristic_section_split(full_text)

    # --- 3. Build Blackboard ---
    blackboard = Blackboard()
    blackboard.set("sections_map", sections_map)
    blackboard.set("output_dir", OUTPUT_DIR)

    # --- 4. Build Tool Registry ---
    registry = ToolRegistry()

    registry.register(Tool(
        name="get_section",
        description="Retrieve a manuscript section by key from the Blackboard.",
        fn=lambda key: blackboard.get("sections_map", {}).get(key, "")
    ))
    registry.register(Tool(
        name="get_stage_result",
        description="Retrieve a prior stage result from the Blackboard.",
        fn=lambda stage_id: blackboard.get_stage_result(stage_id) or {}
    ))
    registry.register(Tool(
        name="count_tokens",
        description="Estimate token count for a text string.",
        fn=lambda text: count_tokens(text)
    ))
    registry.register(Tool(
        name="trim_to_budget",
        description="Trim text to a max token budget.",
        fn=lambda text, max_tokens=2800: trim_to_budget(text, max_tokens)
    ))

    # --- 5. Build Agents ---
    stage_specs = build_stage_specs()

    progress_file = os.path.join(OUTPUT_DIR, "editor_progress.json")

    # Load prior progress if available (and clear if switching to a different article)
    current_filepath = os.path.abspath(args.filepath)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                saved_path = data.get("filepath")
                if saved_path and os.path.abspath(saved_path) != current_filepath:
                    os.remove(progress_file)
                    for fpath in glob.glob(os.path.join(OUTPUT_DIR, "Review_Step*.md")):
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass
                    rprint("[yellow]Different article detected. Cleared previous progress for a fresh review.[/yellow]")
                else:
                    saved_results = data["results"]
                    if len(saved_results) >= len(stage_specs):
                        rprint("[bold yellow]Previous run complete. Starting FRESH review...[/bold yellow]")
                        saved_results = []
                        try:
                            os.remove(progress_file)
                            for fpath in glob.glob(os.path.join(OUTPUT_DIR, "Review_Step*.md")):
                                try:
                                    os.remove(fpath)
                                except OSError:
                                    pass
                        except OSError:
                            pass
                    else:
                        blackboard.load_from_progress_list(saved_results)
                        rprint("[yellow]Found progress file. Resuming...[/yellow]")
            else:
                # Legacy format: list of results
                saved_results = data if isinstance(data, list) else []
                if len(saved_results) >= len(stage_specs):
                    rprint("[bold yellow]Previous run complete. Starting FRESH review...[/bold yellow]")
                    saved_results = []
                    try:
                        os.remove(progress_file)
                        for fpath in glob.glob(os.path.join(OUTPUT_DIR, "Review_Step*.md")):
                            try:
                                os.remove(fpath)
                            except OSError:
                                pass
                    except OSError:
                        pass
                else:
                    blackboard.load_from_progress_list(saved_results)
                    rprint("[yellow]Found progress file. Resuming...[/yellow]")
        except json.JSONDecodeError:
            rprint("[red]Corrupt progress file. Starting fresh.[/red]")
            try:
                os.remove(progress_file)
            except OSError:
                pass
            for fpath in glob.glob(os.path.join(OUTPUT_DIR, "Review_Step*.md")):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

    review_agents = [
        ReviewAgent(spec, blackboard, registry)
        for spec in stage_specs
    ]

    orchestrator = OrchestratorAgent(
        agents=review_agents,
        blackboard=blackboard,
        tools=registry,
        progress_file=progress_file,
        manuscript_filepath=args.filepath
    )

    # --- 6. Run ---
    results = orchestrator.act()

    # --- 7. Save final outputs ---
    base_name = os.path.splitext(os.path.basename(args.filepath))[0]

    with open(os.path.join(OUTPUT_DIR, f"{base_name}_ASAC_Report.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md_file = generate_professional_report(results, base_name, args.filepath, blackboard, out_dir=OUTPUT_DIR)
    evidence_file = generate_evidence_trail_report(results, base_name, args.filepath, blackboard, out_dir=OUTPUT_DIR)

    # Fact verification pass: review findings against full manuscript, flag items for human review
    verification = run_fact_verification(results, full_text)
    if verification is not None:
        with open(os.path.join(OUTPUT_DIR, f"{base_name}_fact_verification.json"), 'w', encoding='utf-8') as f:
            json.dump(verification, f, indent=2, ensure_ascii=False)
        generate_fact_verification_report(verification, os.path.join(OUTPUT_DIR, f"{base_name}_fact_verification.md"))
        n_verified = len(verification.get("verified_findings") or [])
        n_flagged = len(verification.get("flagged_for_review") or [])
        rprint(f"[bold green]Fact verification:[/bold green] {n_verified} verified, {n_flagged} flagged for your review. See {OUTPUT_DIR}/{base_name}_fact_verification.md")
    else:
        rprint("[dim]Fact verification skipped or failed.[/dim]")

    rprint(f"\n[bold green]✅ ASAC Agent-Based Review Complete![/bold green]")
    rprint(f"Final report: {md_file}")
    rprint(f"Evidence trail (quotes + reasoning): {evidence_file}")
    rprint(f"JSON data: {OUTPUT_DIR}/{base_name}_ASAC_Report.json")


if __name__ == "__main__":
    main()
