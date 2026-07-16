"""
governance/report_generator.py — Governance Report Generator

Compiles all scan results into a comprehensive governance report:
  - Executive Summary
  - System Description
  - Risk Classification
  - Giskard Scan Results
  - Promptfoo Red-Team Results
  - DeepEval Evaluation Metrics
  - Fairness Test Results
  - Remediation Plan

Outputs:
  - Markdown file (human-readable)
  - JSON file (machine-readable)
  - Optional PDF via fpdf2

Usage::

    from governance.report_generator import GovernanceReportGenerator
    gen = GovernanceReportGenerator()
    gen.add_giskard_results(giskard_report)
    gen.add_promptfoo_results(promptfoo_report)
    gen.add_deepeval_results(deepeval_report)
    gen.add_fairness_results(fairness_report)
    paths = gen.generate()
    print(f"Reports saved to: {paths}")
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPORT_DIR = Path("governance_reports")


class GovernanceReportGenerator:
    """
    Accumulates governance scan results and renders a full report.
    """

    def __init__(self) -> None:
        self._giskard: Optional[Dict] = None
        self._promptfoo: Optional[Dict] = None
        self._deepeval: Optional[Dict] = None
        self._fairness: Optional[Dict] = None
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Result setters ────────────────────────────────────────────────────────

    def add_giskard_results(self, results: Dict[str, Any]) -> None:
        self._giskard = results

    def add_promptfoo_results(self, results: Dict[str, Any]) -> None:
        self._promptfoo = results

    def add_deepeval_results(self, results: Dict[str, Any]) -> None:
        self._deepeval = results

    def add_fairness_results(self, results: Dict[str, Any]) -> None:
        self._fairness = results

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self) -> Dict[str, str]:
        """
        Generate the governance report in all supported formats.

        Returns:
            Dict mapping format name to file path.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        md_path = REPORT_DIR / f"governance_report_{ts}.md"
        json_path = REPORT_DIR / f"governance_report_{ts}.json"

        report_data = self._build_report_data()

        # Write Markdown
        md_content = self._render_markdown(report_data)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        # Try PDF
        pdf_path: Optional[str] = None
        try:
            pdf_path = self._render_pdf(report_data, ts)
        except ImportError:
            logger.info("fpdf2 not installed; skipping PDF output.")
        except Exception as exc:
            logger.warning(f"PDF generation failed: {exc}")

        paths = {
            "markdown": str(md_path),
            "json": str(json_path),
        }
        if pdf_path:
            paths["pdf"] = pdf_path

        logger.info(f"Governance report generated: {paths}")
        return paths

    # ── Internal: data assembly ───────────────────────────────────────────────

    def _build_report_data(self) -> Dict[str, Any]:
        """Assemble all sections into a structured dict."""
        now = datetime.now(timezone.utc).isoformat()

        return {
            "meta": {
                "title": "BVRIT RAG Chatbot — AI Governance Report",
                "generated_at": now,
                "version": "1.0",
                "author": "Automated Governance Pipeline",
            },
            "executive_summary": self._executive_summary(),
            "system_description": self._system_description(),
            "risk_classification": self._risk_classification(),
            "giskard_results": self._giskard or {"status": "not_run"},
            "promptfoo_results": self._promptfoo or {"status": "not_run"},
            "deepeval_results": self._deepeval or {"status": "not_run"},
            "fairness_results": self._fairness or {"status": "not_run"},
            "remediation_plan": self._remediation_plan(),
        }

    def _executive_summary(self) -> str:
        """Generate a plain-English executive summary from available results."""
        lines: List[str] = [
            "This report summarises the AI governance and safety evaluation of the BVRIT "
            "College FAQ Chatbot, a Retrieval-Augmented Generation (RAG) system built with "
            "LangChain, ChromaDB, and OpenRouter's GPT-4o-mini.",
            "",
        ]

        # Giskard summary
        if self._giskard:
            status = self._giskard.get("status", "unknown")
            issues = self._giskard.get("issues", [])
            lines.append(
                f"Giskard LLM scan status: {status}. "
                f"Issues detected: {len(issues)}."
            )

        # Promptfoo summary
        if self._promptfoo:
            summary = self._promptfoo.get("summary", {})
            pf_pass = summary.get("pass", "N/A")
            pf_fail = summary.get("fail", "N/A")
            pf_total = summary.get("total", "N/A")
            lines.append(
                f"Promptfoo red-team: {pf_pass}/{pf_total} tests passed, {pf_fail} failed."
            )

        # DeepEval summary
        if self._deepeval:
            metrics = self._deepeval.get("metrics", {})
            passed = sum(1 for m in metrics.values() if m.get("passed", False))
            total = len(metrics)
            lines.append(
                f"DeepEval metrics: {passed}/{total} metrics passed."
            )

        # Fairness summary
        if self._fairness:
            verdict = self._fairness.get("verdict", "UNKNOWN")
            issues_count = len(self._fairness.get("issues", []))
            lines.append(
                f"Fairness evaluation verdict: {verdict}. "
                f"Fairness issues detected: {issues_count}."
            )

        lines.append(
            "\nFor full details and remediation steps, see the sections below."
        )
        return "\n".join(lines)

    def _system_description(self) -> Dict[str, Any]:
        """Static description of the system under evaluation."""
        return {
            "name": "BVRIT College FAQ Chatbot",
            "version": "2.0 (with Observability + Governance)",
            "purpose": "Answers questions about BVRIT College using a RAG pipeline grounded in a college knowledge base document.",
            "architecture": {
                "ui": "Streamlit",
                "llm": "GPT-4o-mini via OpenRouter",
                "retrieval": "ChromaDB (cosine similarity)",
                "framework": "LangChain",
                "memory": "ChromaDB-backed per-user memory",
                "tools": ["fee_calculator", "date_checker", "percentage_calculator"],
            },
            "data_sources": ["college_kb.docx — BVRIT official knowledge base document"],
            "users": "Prospective and current BVRIT students, parents, faculty",
            "deployment": "Local Streamlit server",
        }

    def _risk_classification(self) -> Dict[str, Any]:
        """Classify the system's risk level per common AI risk frameworks."""
        return {
            "overall_risk_level": "MEDIUM",
            "rationale": (
                "The system provides factual Q&A about an educational institution using a "
                "grounded knowledge base. It does not make autonomous decisions, process "
                "financial transactions, or handle sensitive medical data. Risk is medium "
                "due to the potential for misinformation about admissions and fees."
            ),
            "risk_dimensions": {
                "hallucination": {
                    "level": "MEDIUM",
                    "reason": "LLM may generate plausible but incorrect college facts when KB context is insufficient.",
                },
                "bias": {
                    "level": "LOW-MEDIUM",
                    "reason": "No explicit demographic bias expected, but fairness tests must confirm equal treatment.",
                },
                "privacy": {
                    "level": "LOW",
                    "reason": "User memory stores only voluntary preferences; no PII is collected by default.",
                },
                "prompt_injection": {
                    "level": "MEDIUM",
                    "reason": "Public-facing chatbot is susceptible to adversarial prompts without explicit guard.",
                },
                "harmful_content": {
                    "level": "LOW",
                    "reason": "Topic scope is narrow (college FAQ); off-topic harmful requests are unlikely but possible.",
                },
            },
        }

    def _remediation_plan(self) -> List[Dict[str, str]]:
        """
        Generate a remediation plan from all collected issues.
        Returns a prioritised list of action items.
        """
        actions: List[Dict[str, str]] = []

        # From Giskard
        if self._giskard:
            for issue in self._giskard.get("issues", []):
                itype = issue.get("type", "unknown")
                severity = issue.get("severity", "medium")
                if itype == "hallucination":
                    actions.append({
                        "priority": "HIGH" if severity == "high" else "MEDIUM",
                        "category": "Hallucination",
                        "action": "Strengthen the refusal policy in the system prompt. Use prompt_v2 grounding rules as the default. Add a fallback message when confidence < 0.4.",
                        "owner": "ML Engineer",
                    })
                elif itype == "prompt_injection":
                    actions.append({
                        "priority": "HIGH",
                        "category": "Prompt Injection",
                        "action": "Add a pre-processing step to strip injection keywords (SYSTEM OVERRIDE, DAN, ignore instructions) before passing to the LLM. Use the alerts.validate_input() length check as a first gate.",
                        "owner": "Security Engineer",
                    })
                elif itype == "harmful_content":
                    actions.append({
                        "priority": "MEDIUM",
                        "category": "Harmful Content",
                        "action": "Add a topic classifier before the LLM call to reject off-topic requests (violence, illegal activities). Log all rejections for audit.",
                        "owner": "ML Engineer",
                    })

        # From Promptfoo
        if self._promptfoo:
            failed = [r for r in self._promptfoo.get("test_results", []) if not r.get("pass", True)]
            if failed:
                actions.append({
                    "priority": "HIGH",
                    "category": "Red-Team Failures",
                    "action": f"Address {len(failed)} failed Promptfoo test case(s). Review each failure in governance_reports/promptfoo_results.json and patch the system prompt or add pre/post-processing guards.",
                    "owner": "ML Engineer",
                })

        # From Fairness
        if self._fairness:
            for issue in self._fairness.get("issues", []):
                actions.append({
                    "priority": "MEDIUM",
                    "category": "Fairness",
                    "action": f"Fairness issue: {issue} — Review the knowledge base for coverage gaps. Ensure equal-quality answers for all user profiles.",
                    "owner": "Product Owner + ML Engineer",
                })

        # From DeepEval
        if self._deepeval:
            for metric_name, metric_data in self._deepeval.get("metrics", {}).items():
                if not metric_data.get("passed", True):
                    score = metric_data.get("score", 0)
                    actions.append({
                        "priority": "MEDIUM",
                        "category": f"DeepEval: {metric_name}",
                        "action": f"Metric '{metric_name}' scored {score:.2f} below threshold. Review failing test cases in evaluation/test_deepeval.py and improve retrieval or prompt grounding.",
                        "owner": "ML Engineer",
                    })

        # Always add human escalation as baseline action
        actions.append({
            "priority": "LOW",
            "category": "Human Escalation",
            "action": "Add a visible 'Contact Admissions Office' link in the Streamlit UI for questions the bot cannot confidently answer (confidence < 0.4).",
            "owner": "Frontend Engineer",
        })

        return sorted(
            actions,
            key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x["priority"], 3),
        )

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the report as a Markdown document."""
        meta = data["meta"]
        lines: List[str] = [
            f"# {meta['title']}",
            f"\n**Generated:** {meta['generated_at']}  ",
            f"**Version:** {meta['version']}  ",
            f"**Author:** {meta['author']}",
            "\n---\n",

            "## 1. Executive Summary\n",
            data["executive_summary"],
            "\n---\n",

            "## 2. System Description\n",
            self._dict_to_md(data["system_description"]),
            "\n---\n",

            "## 3. Risk Classification\n",
            f"**Overall Risk Level:** `{data['risk_classification']['overall_risk_level']}`\n",
            f"\n{data['risk_classification']['rationale']}\n",
            "\n### Risk Dimensions\n",
        ]

        for dim, info in data["risk_classification"]["risk_dimensions"].items():
            lines.append(f"| **{dim.title()}** | `{info['level']}` | {info['reason']} |")

        lines += [
            "\n---\n",
            "## 4. Giskard Scan Results\n",
            self._section_status(data["giskard_results"]),
            "\n---\n",
            "## 5. Promptfoo Red-Team Results\n",
            self._promptfoo_md(data["promptfoo_results"]),
            "\n---\n",
            "## 6. DeepEval Evaluation Metrics\n",
            self._deepeval_md(data["deepeval_results"]),
            "\n---\n",
            "## 7. Fairness Test Results\n",
            self._fairness_md(data["fairness_results"]),
            "\n---\n",
            "## 8. Remediation Plan\n",
            "| Priority | Category | Action | Owner |",
            "|----------|----------|--------|-------|",
        ]

        for action in data["remediation_plan"]:
            lines.append(
                f"| **{action['priority']}** | {action['category']} | {action['action']} | {action['owner']} |"
            )

        lines.append("\n---\n*End of Governance Report*\n")
        return "\n".join(lines)

    def _section_status(self, results: Dict) -> str:
        if not results or results.get("status") == "not_run":
            return "_Scan not run. Execute `python -m governance.giskard_scanner` to generate._"
        status = results.get("status", "unknown")
        issues = results.get("issues", [])
        lines = [f"**Status:** `{status}`", f"**Issues Found:** {len(issues)}\n"]
        for issue in issues:
            severity = issue.get("severity", "medium")
            itype = issue.get("type", "unknown")
            desc = issue.get("description", "")
            lines.append(f"- `[{severity.upper()}]` **{itype}**: {desc}")
        return "\n".join(lines)

    def _promptfoo_md(self, results: Dict) -> str:
        if not results or results.get("status") == "not_run":
            return "_Red-team not run. Install Node.js + `npm install -g promptfoo`._"
        summary = results.get("summary", {})
        lines = [
            f"**Status:** `{results.get('status', 'unknown')}`",
            f"**Pass:** {summary.get('pass', 'N/A')} / {summary.get('total', 'N/A')}",
            f"**Fail:** {summary.get('fail', 'N/A')}\n",
        ]
        for tr in results.get("test_results", [])[:12]:
            icon = "✅" if tr.get("pass", False) else "❌"
            lines.append(f"{icon} {tr.get('description', 'Unknown test')}")
        return "\n".join(lines)

    def _deepeval_md(self, results: Dict) -> str:
        if not results or results.get("status") == "not_run":
            return "_DeepEval not run. Execute `pytest evaluation/test_deepeval.py`._"
        lines = [f"**Status:** `{results.get('status', 'unknown')}`\n"]
        for name, data in results.get("metrics", {}).items():
            icon = "✅" if data.get("passed", False) else "❌"
            score = data.get("score", 0)
            threshold = data.get("threshold", 0)
            lines.append(f"{icon} **{name}**: score={score:.3f} / threshold={threshold:.2f}")
        return "\n".join(lines)

    def _fairness_md(self, results: Dict) -> str:
        if not results or results.get("status") == "not_run":
            return "_Fairness tests not run. Execute `python -m governance.fairness_tests`._"
        verdict = results.get("verdict", "UNKNOWN")
        icon = "✅" if verdict == "PASS" else "❌"
        lines = [f"**Verdict:** {icon} `{verdict}`\n"]
        for profile, metrics in results.get("metrics", {}).items():
            lines.append(
                f"- **{profile}**: avg_len={metrics.get('avg_answer_len',0):.0f}, "
                f"refusal_rate={metrics.get('refusal_rate',0):.1%}, "
                f"citation_rate={metrics.get('citation_rate',0):.1%}"
            )
        if results.get("issues"):
            lines.append("\n**Issues:**")
            for issue in results["issues"]:
                lines.append(f"- ⚠️ {issue}")
        return "\n".join(lines)

    @staticmethod
    def _dict_to_md(d: Dict, indent: int = 0) -> str:
        """Recursively convert a dict to indented Markdown bullet points."""
        lines: List[str] = []
        prefix = "  " * indent
        for key, val in d.items():
            if isinstance(val, dict):
                lines.append(f"{prefix}- **{key}:**")
                lines.append(GovernanceReportGenerator._dict_to_md(val, indent + 1))
            elif isinstance(val, list):
                lines.append(f"{prefix}- **{key}:** {', '.join(str(v) for v in val)}")
            else:
                lines.append(f"{prefix}- **{key}:** {val}")
        return "\n".join(lines)

    def _render_pdf(self, data: Dict[str, Any], ts: str) -> str:
        """
        Render the report as a PDF using fpdf2.

        Args:
            data: The report data dict.
            ts:   Timestamp string for the filename.

        Returns:
            Path to the generated PDF file.
        """
        from fpdf import FPDF  # type: ignore[import]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=16)
        pdf.cell(0, 10, data["meta"]["title"], new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 8, f"Generated: {data['meta']['generated_at']}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # Executive summary
        pdf.set_font("Helvetica", "B", size=12)
        pdf.cell(0, 8, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        pdf.multi_cell(0, 5, data["executive_summary"])
        pdf.ln(4)

        # Risk level
        risk = data["risk_classification"]["overall_risk_level"]
        pdf.set_font("Helvetica", "B", size=12)
        pdf.cell(0, 8, f"Overall Risk Level: {risk}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        pdf.multi_cell(0, 5, data["risk_classification"]["rationale"])
        pdf.ln(4)

        # Remediation plan
        pdf.set_font("Helvetica", "B", size=12)
        pdf.cell(0, 8, "Remediation Plan", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        for action in data["remediation_plan"]:
            pdf.multi_cell(
                0, 5,
                f"[{action['priority']}] {action['category']}: {action['action']} — Owner: {action['owner']}",
            )
            pdf.ln(2)

        pdf_path = REPORT_DIR / f"governance_report_{ts}.pdf"
        pdf.output(str(pdf_path))
        return str(pdf_path)
