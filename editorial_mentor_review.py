"""
Editorial Mentor Review — Second-pass review with customizable questions.

Two modes:
  - Default: mentor answers each question in editorial_mentor_config.json.
  - --challenge: script prompts YOU with a question; you type your view; the mentor
    digs into the article and uses quotes to argue why your view might be wrong
    from a good editorial perspective (and convinces you with evidence).

Usage:
  python editorial_mentor_review.py "path/to/manuscript.pdf"
  python editorial_mentor_review.py "path/to/manuscript.pdf" --challenge
  python editorial_mentor_review.py "path/to/manuscript.pdf" --first-pass-report "BaseName_ASAC_Report.json"
  python editorial_mentor_review.py "path/to/manuscript.pdf" --config my_questions.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

# Reuse extraction and sectioning from main reviewer
try:
    from journal_editor_coach import (
        extract_text,
        heuristic_section_split,
        trim_to_budget,
        CONSOLE,
        rprint,
        OLLAMA_OPTIONS,
        MODEL_NAME,
        cleanup_environment,
    )
    import ollama
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError as e:
    print(f"Error: {e}. Run from the same folder as journal_editor_coach.py and install deps.")
    sys.exit(1)

# ==================================================================================
# CONFIG
# ==================================================================================

DEFAULT_CONFIG_PATH = "editorial_mentor_config.json"
MENTOR_INPUT_BUDGET = 3800  # tokens for manuscript excerpt in each Q
FIRST_PASS_SUMMARY_CHARS = 800  # max chars for minimal summary (fallback)
BLACKBOARD_CONTEXT_MAX_CHARS = 3500  # max chars for full blackboard (structure map + stage findings)

# ==================================================================================
# LOAD CONFIG
# ==================================================================================


def load_mentor_config(config_path: str, require_questions: bool = True) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if require_questions and ("questions" not in data or not data["questions"]):
        raise ValueError("Config must contain a non-empty 'questions' list (or use --challenge with challenge_question).")
    return data


def get_manuscript_excerpt(sections_map: Dict[str, str], focus: str, full_text: str) -> str:
    """Return text relevant to 'focus', trimmed to budget."""
    focus = (focus or "whole").lower().strip()
    if focus == "whole":
        return trim_to_budget(full_text, max_tokens=MENTOR_INPUT_BUDGET)
    # Map focus to section keys used in heuristic_section_split.
    # Methods/Results/Discussion accept variant names (Methodology, Findings, Implications).
    key_map = {
        "abstract": "Title & Abstract",
        "introduction": "Introduction",
        "methods": "Methods",
        "methodology": "Methods",
        "results": "Results",
        "findings": "Results",
        "discussion": "Discussion & Conclusions",
        "conclusions": "Discussion & Conclusions",
        "implications": "Discussion & Conclusions",
    }
    key = key_map.get(focus)
    if key and key in sections_map and sections_map[key].strip():
        return trim_to_budget(sections_map[key], max_tokens=MENTOR_INPUT_BUDGET)
    return trim_to_budget(full_text, max_tokens=MENTOR_INPUT_BUDGET)


def _load_first_pass_results(report_path: str) -> List[Dict]:
    """Load first-pass ASAC report JSON; return list of stage results or empty list."""
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def summarize_first_pass(report_path: str) -> str:
    """Produce a short text summary of the first-pass ASAC report for context."""
    results = _load_first_pass_results(report_path)
    if not results:
        return ""
    parts = []
    for i, stage in enumerate(results[:7], 1):
        name = stage.get("stage_name", f"Stage {i}")
        decision = stage.get("decision") or stage.get("final_recommendation") or stage.get("theoretical_contribution") or stage.get("method_rigor") or stage.get("readability_score") or "—"
        parts.append(f"{name}: {decision}")
    summary = "First-pass review outcomes: " + "; ".join(parts)
    if len(summary) > FIRST_PASS_SUMMARY_CHARS:
        summary = summary[:FIRST_PASS_SUMMARY_CHARS] + "..."
    return summary


def build_blackboard_context(report_path: str) -> str:
    """
    Build a blackboard-style summary from the first-pass ASAC report so the mentor
    can stay on track with what the first-step agents already told the user.
    Includes: paper structure map (from Stage 1) + key findings per stage.
    """
    results = _load_first_pass_results(report_path)
    if not results:
        return ""

    lines = ["=== FIRST-STEP BLACKBOARD (stay consistent with what the first pass told the user) ===\n"]

    # Stage 1: structure map
    s1 = results[0] if results else {}
    structure_map = s1.get("paper_structure_map", "")
    if structure_map:
        sm_str = structure_map if isinstance(structure_map, str) else " | ".join(str(x) for x in structure_map)
        lines.append("**Paper structure map (Stage 1):**")
        lines.append(sm_str[:1200] + ("..." if len(sm_str) > 1200 else ""))
        lines.append("")

    # Per-stage findings (what the first pass already said)
    for i, stage in enumerate(results[:7], 1):
        name = stage.get("stage_name", f"Stage {i}")
        decision = (
            stage.get("decision")
            or stage.get("final_recommendation")
            or stage.get("theoretical_contribution")
            or stage.get("method_rigor")
            or stage.get("readability_score")
            or (f"Coherence {stage.get('coherence_score')}/10" if stage.get("coherence_score") is not None else None)
            or "—"
        )
        lines.append(f"**{name}** — {decision}")

        def add_items(key: str, label: str, max_items: int = 3, quote_max: int = 80) -> None:
            items = stage.get(key)
            if not items or not isinstance(items, list):
                return
            for j, it in enumerate(items[:max_items]):
                if isinstance(it, dict):
                    point = it.get("point") or it.get("issue") or str(it.get("quote", ""))[:quote_max]
                    if it.get("quote"):
                        point += f' [quote: "{str(it["quote"])[:quote_max]}..."]' if len(str(it["quote"])) > quote_max else f' [quote: "{it["quote"]}"]'
                else:
                    point = str(it)[:120]
                lines.append(f"  - {point}")

        add_items("violations", "Violations")
        add_items("strengths", "Strengths")
        add_items("major_issues", "Major issues")
        add_items("actionable_recommendations", "Recommendations", max_items=2)
        if stage.get("coherence_score") is not None:
            lines.append(f"  Coherence score: {stage.get('coherence_score')}/10")
        if stage.get("contradictions"):
            for c in (stage["contradictions"] or [])[:2]:
                if isinstance(c, dict):
                    lines.append(f"  Contradiction: {c.get('issue', '')[:100]}")
        if stage.get("connection_analysis"):
            lines.append(f"  Connection: {str(stage['connection_analysis'])[:200]}")
        lines.append("")

    out = "\n".join(lines)
    if len(out) > BLACKBOARD_CONTEXT_MAX_CHARS:
        out = out[:BLACKBOARD_CONTEXT_MAX_CHARS] + "\n...[truncated]"
    return out


# ==================================================================================
# MENTOR Q&A
# ==================================================================================


def run_mentor_question(
    question_item: Dict,
    manuscript_excerpt: str,
    first_pass_summary: str,
    mentor_role: str,
    model_name: str,
) -> str:
    """Call the model as editorial mentor for one question. Returns raw answer text."""
    label = question_item.get("label", question_item.get("id", "Question"))
    question = question_item.get("question", "")
    if not question:
        return "[No question text provided.]"

    user_content = f"""You are answering as an editorial mentor. The author will see your feedback.

**Manuscript excerpt (for context):**
{manuscript_excerpt}
"""

    if first_pass_summary:
        user_content += f"""

**First-step blackboard (stay consistent with what the first pass already told the user):**
{first_pass_summary}
"""

    user_content += f"""

**Question for the author:**
{question}

Answer in one or two short paragraphs. Be specific and cite the manuscript with a short quote where it helps. Write directly to the author (use "you" / "your paper"). When a first-step blackboard is provided above, stay consistent with what the first pass already told the user. Output plain text only, no JSON."""

    messages = [
        {"role": "system", "content": mentor_role},
        {"role": "user", "content": user_content},
    ]
    response = ollama.chat(model=model_name, messages=messages, options=OLLAMA_OPTIONS)
    return (response.get("message") or {}).get("content", "").strip()


# ==================================================================================
# CHALLENGE MODE: prompt user, then argue against their view with quotes
# ==================================================================================

DEFAULT_CHALLENGE_QUESTION = (
    "What do you think is the main weakness of this paper (or the main thing the author should change)?"
)
DEFAULT_CHALLENGE_INSTRUCTION = """You are a senior editor. The user (a reviewer or author) gave their view about the manuscript. Your job is to dig into the manuscript and use DIRECT QUOTES as evidence to respond from a good editorial perspective. If the evidence in the text shows the user is WRONG or incomplete: argue why, with specific quotes that support a different or more nuanced reading. If the evidence shows the user is RIGHT: say so clearly and support their view with quotes from the manuscript—you can be convinced of the truth when they are correct. Either way, prove your case from the text. Write 2–4 short paragraphs, each anchored in at least one quote."""


def run_challenge_round(
    challenge_question: str,
    user_answer: str,
    manuscript_excerpt: str,
    first_pass_summary: str,
    challenge_instruction: str,
    model_name: str,
) -> str:
    """Use the manuscript to argue why the user's view might be wrong; return mentor response with quotes."""
    user_content = f"""**Manuscript excerpt:**
{manuscript_excerpt}
"""
    if first_pass_summary:
        user_content += f"""
**First-step blackboard (stay consistent with what the first pass already told the user):**
{first_pass_summary}
"""
    user_content += f"""
**Question the user was asked:**
{challenge_question}

**The user's view:**
{user_answer}

Your task: Using ONLY the manuscript above, respond from an editorial perspective. If the text supports a different view, argue why the user's view is wrong or incomplete, with quotes as proof. If the text supports the user's view, agree and back that up with quotes—you can be convinced when they are correct. Either way, use direct quotes from the manuscript. When a first-step blackboard is provided above, keep your response consistent with what the first pass already told the user; do not contradict it. Output plain text only."""

    messages = [
        {"role": "system", "content": challenge_instruction},
        {"role": "user", "content": user_content},
    ]
    response = ollama.chat(model=model_name, messages=messages, options=OLLAMA_OPTIONS)
    return (response.get("message") or {}).get("content", "").strip()


def write_challenge_report(
    challenge_question: str,
    user_answer: str,
    mentor_response: str,
    output_md: str,
    output_json: str,
    filepath: str,
) -> None:
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Editorial Mentor — Challenge Your View\n\n")
        f.write(f"**Manuscript**: {os.path.basename(filepath)}\n")
        f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y')}\n\n")
        f.write("---\n\n")
        f.write("## Question\n\n")
        f.write(f"{challenge_question}\n\n")
        f.write("## Your view\n\n")
        f.write(f"{user_answer}\n\n")
        f.write("## Mentor response (evidence from the manuscript — may agree or disagree)\n\n")
        f.write(mentor_response + "\n")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "challenge_question": challenge_question,
                "user_answer": user_answer,
                "mentor_response": mentor_response,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


# ==================================================================================
# REPORT
# ==================================================================================


def write_mentor_report(
    answers: List[Dict],
    output_md: str,
    output_json: str,
    filepath: str,
    config_path: str,
) -> None:
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Editorial Mentor Feedback\n\n")
        f.write(f"**Manuscript**: {os.path.basename(filepath)}\n")
        f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y')}\n")
        f.write(f"**Config**: {config_path}\n\n")
        f.write("---\n\n")
        for a in answers:
            f.write(f"## {a['label']}\n\n")
            f.write(f"*{a['question']}*\n\n")
            f.write(a["answer"] + "\n\n")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)


# ==================================================================================
# MAIN
# ==================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Editorial Mentor Review — customizable second-pass Q&A feedback."
    )
    parser.add_argument("filepath", help="Path to manuscript (PDF, DOCX, or TXT)")
    parser.add_argument(
        "--first-pass-report",
        default=None,
        help="Optional path to ASAC_Report.json from the first pass (adds context for mentor).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to JSON config with mentor questions (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Prompt you with a question; you answer; then the mentor uses the article (and quotes) to argue why your view might be wrong from an editorial perspective.",
    )
    parser.add_argument(
        "--no-cleanup-env",
        action="store_true",
        help="Do not kill/restart Ollama; only wait for existing server.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        rprint(f"[bold red]File not found: {args.filepath}[/bold red]")
        sys.exit(1)
    if not os.path.exists(args.config):
        rprint(f"[bold red]Config not found: {args.config}[/bold red]")
        sys.exit(1)

    cleanup_environment(do_cleanup=not args.no_cleanup_env)

    use_challenge = args.challenge

    if use_challenge:
        rprint(Panel(
            "[bold blue]Editorial Mentor — Challenge Your View[/bold blue]\n[dim]You answer a question; the mentor uses the article (and quotes) to respond: they may argue you're wrong or agree and back you up—whichever the text supports.[/dim]",
            expand=False,
        ))
        config = load_mentor_config(args.config, require_questions=False)
        challenge_question = config.get("challenge_question") or DEFAULT_CHALLENGE_QUESTION
        challenge_instruction = config.get("challenge_instruction") or DEFAULT_CHALLENGE_INSTRUCTION

        rprint("[green]Loading manuscript...[/green]")
        full_text = extract_text(args.filepath)
        sections_map = heuristic_section_split(full_text)
        manuscript_excerpt = get_manuscript_excerpt(sections_map, "whole", full_text)

        first_pass_report_path = args.first_pass_report
        if not first_pass_report_path:
            base_name = os.path.splitext(os.path.basename(args.filepath))[0]
            auto_path = f"{base_name}_ASAC_Report.json"
            if os.path.exists(auto_path):
                first_pass_report_path = auto_path
                rprint(f"[dim]Auto-detected first-pass report: {auto_path}[/dim]")
        first_pass_summary = ""
        if first_pass_report_path and os.path.exists(first_pass_report_path):
            first_pass_summary = build_blackboard_context(first_pass_report_path)
            rprint(f"[dim]Using first-step blackboard for context ({len(first_pass_summary)} chars).[/dim]")

        rprint("\n[bold cyan]Question:[/bold cyan]")
        rprint(challenge_question)
        rprint("")
        try:
            user_answer = input("Your view (then press Enter): ").strip()
        except EOFError:
            user_answer = ""
        if not user_answer:
            rprint("[yellow]No answer given. Exiting.[/yellow]")
            sys.exit(0)

        rprint("\n[dim]Mentor is reading the article to respond with evidence (may agree or disagree)...[/dim]")
        mentor_response = run_challenge_round(
            challenge_question,
            user_answer,
            manuscript_excerpt,
            first_pass_summary,
            challenge_instruction,
            MODEL_NAME,
        )

        rprint("\n[bold green]Mentor response:[/bold green]")
        rprint(mentor_response)

        base = os.path.splitext(os.path.basename(args.filepath))[0]
        output_md = f"{base}_Mentor_Challenge.md"
        output_json = f"{base}_Mentor_Challenge.json"
        write_challenge_report(
            challenge_question, user_answer, mentor_response,
            output_md, output_json, args.filepath,
        )
        rprint(f"\n[bold green]Saved:[/bold green] {output_md}  |  {output_json}")
        return

    # Original multi-question mentor flow
    rprint(Panel(
        "[bold blue]Editorial Mentor Review[/bold blue]\n[dim]Customizable questions · coaching feedback[/dim]",
        expand=False,
    ))

    config = load_mentor_config(args.config, require_questions=True)
    mentor_role = config.get("mentor_role") or (
        "You are an experienced editorial mentor. Be constructive, specific, and kind. "
        "Anchor advice in short quotes from the manuscript."
    )
    questions = config["questions"]

    rprint("[green]Loading manuscript...[/green]")
    full_text = extract_text(args.filepath)
    sections_map = heuristic_section_split(full_text)

    first_pass_report_path = args.first_pass_report
    if not first_pass_report_path:
        base_name = os.path.splitext(os.path.basename(args.filepath))[0]
        auto_path = f"{base_name}_ASAC_Report.json"
        if os.path.exists(auto_path):
            first_pass_report_path = auto_path
            rprint(f"[dim]Auto-detected first-pass report: {auto_path}[/dim]")
    first_pass_summary = ""
    if first_pass_report_path and os.path.exists(first_pass_report_path):
        first_pass_summary = build_blackboard_context(first_pass_report_path)
        rprint(f"[dim]Using first-step blackboard for context ({len(first_pass_summary)} chars).[/dim]")

    answers = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("[cyan]Mentor Q&A...[/cyan]", total=len(questions))
        for q in questions:
            progress.update(task, description=q.get("label", q.get("id", "Question"))[:40])
            excerpt = get_manuscript_excerpt(
                sections_map, q.get("focus", "whole"), full_text
            )
            answer = run_mentor_question(
                q, excerpt, first_pass_summary, mentor_role, MODEL_NAME
            )
            answers.append({
                "id": q.get("id", ""),
                "label": q.get("label", q.get("id", "Question")),
                "question": q.get("question", ""),
                "focus": q.get("focus", "whole"),
                "answer": answer,
            })
            progress.advance(task)

    base = os.path.splitext(os.path.basename(args.filepath))[0]
    output_md = f"{base}_Mentor_Feedback.md"
    output_json = f"{base}_Mentor_Feedback.json"
    write_mentor_report(answers, output_md, output_json, args.filepath, args.config)

    rprint(f"\n[bold green]Mentor feedback complete.[/bold green]")
    rprint(f"Report: {output_md}")
    rprint(f"Data:  {output_json}")


if __name__ == "__main__":
    main()
