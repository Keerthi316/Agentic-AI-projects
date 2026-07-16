"""
governance/ — AI Governance & Safety for the BVRIT RAG Chatbot.

Modules:
    giskard_scanner   — Giskard LLM scan: hallucination, bias, prompt injection,
                        harmful content, data leakage, stereotypes
    promptfoo_runner  — Promptfoo red-team: jailbreaks, PII extraction, hijacking
    fairness_tests    — Fairness evaluation across user profiles
    report_generator  — Governance report builder (executive summary → remediation)
"""

from governance.giskard_scanner import GiskardScanner
from governance.promptfoo_runner import PromptfooRunner
from governance.fairness_tests import FairnessEvaluator
from governance.report_generator import GovernanceReportGenerator

__all__ = [
    "GiskardScanner",
    "PromptfooRunner",
    "FairnessEvaluator",
    "GovernanceReportGenerator",
]
