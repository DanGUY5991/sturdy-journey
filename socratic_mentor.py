"""
Socratic Mentor — Multi-turn dialogue to learn editorial reasoning.

After the first-pass review, the mentor presents key findings and asks probing
questions. You respond; the mentor agrees, disagrees, or asks a follow-up.
Continues until topics are resolved, then produces a learning summary.

Usage:
  python socratic_mentor.py "path/to/manuscript.pdf"
  python socratic_mentor.py "path/to/manuscript.pdf" --first-pass-report "BaseName_ASAC_Report.json"
  python socratic_mentor.py "path/to/manuscript.pdf" --questions socratic_questions.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

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
    from editorial_mentor_review import (
        build_blackboard_context,
        _load_first_pass_results,
    )
    import ollama
    from rich.panel import Panel
    from rich.prompt import Prompt
except ImportError as e:
    print(f"Error: {e}. Run from the same folder as journal_editor_coach.py and editorial_mentor_review.py.")
    sys.exit(1)

# ==================================================================================
# CONFIG
# ==================================================================================

DEFAULT_QUESTIONS_PATH = "socratic_questions.json"
SOCRATIC_SYSTEM = """You are a Socratic editorial mentor. Your goal is to help the user learn to think like a great editor. You do this by asking probing questions and responding to their answers—agreeing when they are right, gently challenging when they are incomplete, and always grounding your feedback in the review findings and the manuscript. Be concise (2-4 sentences per turn). End with a clear question when you want the user to reflect further, or with a brief summary when the topic is resolved."""

# ==================================================================================
# LOAD CONFIG
# ==================================================================================


def load_socratic_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_stage_finding(stage: Dict, stage_id: int) -> str:
    """Turn one stage result into a short finding summary for a topic."""
    name = stage.get("stage_name", f"Stage {stage_id}")
    decision = (
        stage.get("decision")
        or stage.get("final_recommendation")
        or stage.get("theoretical_contribution")
        or stage.get("method_rigor")
        or stage.get("readability_score")
        or (f"{stage['coherence_score']}/10" if stage_id == 7 and stage.get("coherence_score") is not None else None)
        or "—"
    )
    parts = [f"{name}: {decision}"]
    for key in ("violations", "major_issues", "strengths"):
        items = stage.get(key)
        if items and isinstance(items, list):
            for it in items[:2]:
                if isinstance(it, dict):
                    parts.append(f"  - {it.get('point') or it.get('issue', '')} (quote: {str(it.get('quote', ''))[:80]}...)")
                else:
                    parts.append(f"  - {it}")
    return "\n".join(parts)


# ==================================================================================
# SOCRATIC DIALOGUE
# ==================================================================================


class SocraticDialogue:
    """
    Multi-turn Socratic dialogue. Extracts topics from the first-pass blackboard,
    presents findings and asks probing questions, processes user responses,
    and tracks when a topic is resolved.
    """

    def __init__(
        self,
        blackboard_context: str,
        questions_config: Dict[str, Any],
        stage_results: List[Dict],
        model_name: str,
    ):
        self.blackboard_context = blackboard_context
        self.questions_config = questions_config
        self.stage_results = stage_results or []
        self.model_name = model_name
        self.topics = self._extract_topics()
        self.current_topic_idx = 0
        self.dialogue_history: List[Dict[str, str]] = []
        self.learning_points: List[str] = []

    def _extract_topics(self) -> List[Dict[str, Any]]:
        """Build a list of topics from stage results and question config."""
        topics = []
        for i, stage in enumerate(self.stage_results[:7], 1):
            stage_key = f"stage_{i}"
            config = self.questions_config.get(stage_key, {})
            finding_text = _format_stage_finding(stage, i)
            template_questions = config.get("questions", [])
            topics.append({
                "stage_id": i,
                "stage_name": stage.get("stage_name", f"Stage {i}"),
                "finding": finding_text,
                "template_questions": template_questions,
                "discussion_done": False,
            })
        return topics

    def _call_mentor(self, user_message: str, system_extra: str = "") -> str:
        """One LLM call as the Socratic mentor."""
        system = SOCRATIC_SYSTEM
        if system_extra:
            system += "\n\n" + system_extra
        context = self.blackboard_context[:2000] if self.blackboard_context else ""
        prompt = f"""**Review context (first-pass findings):**
{context}

**Recent dialogue:**
{chr(10).join(f"Mentor: {h['mentor']}\nUser: {h['user']}" for h in self.dialogue_history[-4:])}

**User's latest response:**
{user_message}

Respond as the Socratic mentor. If the user has engaged well with the question, you may summarize and move on, or ask one more probing question. If they have not yet addressed the issue, ask a follow-up. Keep it short (2-4 sentences)."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        response = ollama.chat(model=self.model_name, messages=messages, options=OLLAMA_OPTIONS)
        return (response.get("message") or {}).get("content", "").strip()

    def next_question(self) -> str:
        """Generate the next probing question for the current topic."""
        if self.current_topic_idx >= len(self.topics):
            return ""
        topic = self.topics[self.current_topic_idx]
        finding = topic["finding"]
        templates = topic.get("template_questions", [])
        if templates:
            q = templates[0]
        else:
            q = "What do you think the author should do about this, and why?"
        context = self.blackboard_context[:1500] if self.blackboard_context else ""
        prompt = f"""**First-pass finding for this topic:**
{finding}

**Context:**
{context}

Generate ONE short Socratic question (1-2 sentences) to ask the user about this finding. The question should make them reflect on why this matters editorially or how they would address it. Do not answer the question yourself. Output only the question, no preamble."""
        messages = [
            {"role": "system", "content": "You are a Socratic mentor. Output only the probing question, nothing else."},
            {"role": "user", "content": prompt},
        ]
        response = ollama.chat(model=self.model_name, messages=messages, options=OLLAMA_OPTIONS)
        question = (response.get("message") or {}).get("content", "").strip()
        return question or q

    def process_user_response(self, response: str) -> str:
        """Evaluate user response and return mentor follow-up."""
        self.dialogue_history.append({"user": response})
        mentor_reply = self._call_mentor(response)
        self.dialogue_history[-1]["mentor"] = mentor_reply
        return mentor_reply

    def is_topic_resolved(self) -> bool:
        """Consider topic resolved after one exchange; can be refined later."""
        if self.current_topic_idx >= len(self.topics):
            return True
        return self.topics[self.current_topic_idx].get("discussion_done", False)

    def mark_topic_resolved(self):
        if self.current_topic_idx < len(self.topics):
            self.topics[self.current_topic_idx]["discussion_done"] = True
        self.current_topic_idx += 1

    def has_more_topics(self) -> bool:
        return self.current_topic_idx < len(self.topics)

    def get_learning_summary(self) -> str:
        """Generate a short 'what you learned' summary from the dialogue."""
        history_text = "\n".join(
            f"Mentor: {h.get('mentor', '')}\nUser: {h.get('user', '')}"
            for h in self.dialogue_history
        )
        prompt = f"""**Socratic dialogue (mentor and user):**
{history_text}

In 3-5 bullet points, summarize what the user learned or reflected on about being a good editor. Be specific to the topics discussed. Output only the bullet list."""
        messages = [
            {"role": "system", "content": "You summarize learning outcomes. Output only a bullet list."},
            {"role": "user", "content": prompt},
        ]
        response = ollama.chat(model=self.model_name, messages=messages, options=OLLAMA_OPTIONS)
        return (response.get("message") or {}).get("content", "").strip()


# ==================================================================================
# MAIN
# ==================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Socratic Mentor — learn editorial reasoning through dialogue.")
    parser.add_argument("filepath", help="Path to manuscript (PDF, DOCX, or TXT)")
    parser.add_argument("--first-pass-report", default=None, help="Path to ASAC_Report.json (auto-detected if same base name as manuscript)")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_PATH, help="Path to socratic_questions.json")
    parser.add_argument("--no-cleanup-env", action="store_true", help="Do not kill/restart Ollama.")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        rprint(f"[bold red]File not found: {args.filepath}[/bold red]")
        sys.exit(1)

    report_path = args.first_pass_report
    if not report_path:
        base = os.path.splitext(os.path.basename(args.filepath))[0]
        candidate = f"{base}_ASAC_Report.json"
        if os.path.exists(candidate):
            report_path = candidate
            rprint(f"[dim]Auto-detected report: {report_path}[/dim]")
    if not report_path or not os.path.exists(report_path):
        rprint("[bold red]No first-pass report found. Run the 6-stage review first, or pass --first-pass-report.[/bold red]")
        sys.exit(1)

    questions_path = args.questions
    if not os.path.exists(questions_path):
        rprint(f"[yellow]Questions file not found: {questions_path}. Using empty topic list.[/yellow]")
        questions_config = {}
    else:
        questions_config = load_socratic_config(questions_path)

    cleanup_environment(do_cleanup=not args.no_cleanup_env)

    rprint(Panel(
        "[bold blue]Socratic Mentor[/bold blue]\n[dim]Probing questions to help you learn editorial reasoning[/dim]",
        expand=False,
    ))

    blackboard_context = build_blackboard_context(report_path)
    stage_results = _load_first_pass_results(report_path)

    dialogue = SocraticDialogue(
        blackboard_context=blackboard_context,
        questions_config=questions_config,
        stage_results=stage_results,
        model_name=MODEL_NAME,
    )

    if not dialogue.topics:
        rprint("[yellow]No topics extracted from the report. Exiting.[/yellow]")
        sys.exit(0)

    rprint(f"[green]Loaded {len(dialogue.topics)} topics from the review.[/green]\n")

    while dialogue.has_more_topics():
        topic = dialogue.topics[dialogue.current_topic_idx]
        rprint(f"[bold cyan]Topic: {topic['stage_name']}[/bold cyan]")
        rprint(f"[dim]{topic['finding'][:300]}...[/dim]\n")

        question = dialogue.next_question()
        rprint(f"[bold]Mentor:[/bold] {question}\n")
        try:
            user_input = Prompt.ask("[bold]Your response[/bold]")
        except (EOFError, KeyboardInterrupt):
            rprint("\n[yellow]Exiting.[/yellow]")
            break
        if not user_input.strip():
            dialogue.mark_topic_resolved()
            continue

        mentor_reply = dialogue.process_user_response(user_input.strip())
        rprint(f"\n[bold]Mentor:[/bold] {mentor_reply}\n")
        dialogue.mark_topic_resolved()

    rprint("[bold green]Generating learning summary...[/bold green]")
    summary = dialogue.get_learning_summary()
    rprint("\n[bold]What you learned (editor perspective):[/bold]")
    rprint(summary)

    base = os.path.splitext(os.path.basename(args.filepath))[0]
    out_json = f"{base}_Socratic_Learning.json"
    out_md = f"{base}_Socratic_Learning.md"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "dialogue_history": dialogue.dialogue_history,
            "learning_summary": summary,
            "topics_covered": [t["stage_name"] for t in dialogue.topics],
        }, f, indent=2, ensure_ascii=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Socratic Learning Summary\n\n")
        f.write(f"**Manuscript**: {os.path.basename(args.filepath)}\n")
        f.write(f"**Date**: {datetime.now().strftime('%B %d, %Y')}\n\n")
        f.write("## What you learned (editor perspective)\n\n")
        f.write(summary + "\n\n")
        f.write("## Dialogue\n\n")
        for i, h in enumerate(dialogue.dialogue_history, 1):
            f.write(f"### Exchange {i}\n\n**User:** {h.get('user', '')}\n\n**Mentor:** {h.get('mentor', '')}\n\n")
    rprint(f"\n[bold green]Saved:[/bold green] {out_md}  |  {out_json}")


if __name__ == "__main__":
    main()
