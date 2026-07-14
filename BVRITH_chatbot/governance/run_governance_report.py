"""
governance/run_governance_report.py — One-Shot Governance Report Runner

Runs every governance scan in sequence and compiles the full report:
  1. Giskard LLM scan
  2. Promptfoo red-team (config generation + optional CLI run)
  3. DeepEval evaluation metrics
  4. Fairness tests across user profiles
  5. Combined governance report (Markdown + JSON + optional PDF)

Usage (from the project root)::

    python -m governance.run_governance_report

Optional: pass --skip-live to skip scans requiring live API calls::

    python -m governance.run_governance_report --skip-live

Output: governance_reports/ directory
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# ── Add project root to path so local imports work ───────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("governance.runner")


def _build_chatbot_fn():
    """
    Build a lightweight chatbot callable for governance scans.

    Returns a function: (question: str) -> str
    Falls back to a mock function if the vector store cannot be initialised.
    """
    try:
        from vector_store import get_vector_store
        from chatbot import CollegeChatbot

        vector_store, status = get_vector_store()
        if vector_store is None:
            raise RuntimeError(f"Vector store unavailable: {status}")

        bot = CollegeChatbot(vector_store)
        logger.info("Live chatbot initialised for governance scans.")

        def chatbot_fn(question: str) -> str:
            result = bot.answer_question(question)
            return result["answer"]

        return chatbot_fn

    except Exception as exc:
        logger.warning(f"Could not init live chatbot ({exc}). Using mock answers.")

        def mock_fn(question: str) -> str:
            return (
                f"[MOCK ANSWER] Based on the BVRIT knowledge base, here is information "
                f"about your question regarding '{question[:60]}'. [Admissions]"
            )

        return mock_fn


def run(skip_live: bool = False) -> None:
    """
    Execute all governance scans and write the report.

    Args:
        skip_live: If True, skip scans that require live API calls.
    """
    from governance.giskard_scanner import GiskardScanner
    from governance.promptfoo_runner import PromptfooRunner
    from governance.fairness_tests import FairnessEvaluator
    from governance.report_generator import GovernanceReportGenerator
    from evaluation.deepeval_runner import DeepEvalRunner

    chatbot_fn = None if skip_live else _build_chatbot_fn()
    report_gen = GovernanceReportGenerator()

    # ── 1. Giskard scan ───────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 1/4 — Giskard LLM Scan")
    logger.info("=" * 50)
    giskard_results = GiskardScanner(
        chatbot_fn=chatbot_fn or (lambda q: "[MOCK]"),
        model_name="BVRIT-RAG-Chatbot",
    ).run()
    report_gen.add_giskard_results(giskard_results)
    logger.info(f"Giskard: status={giskard_results['status']}, issues={len(giskard_results.get('issues', []))}")

    # ── 2. Promptfoo config + optional CLI run ────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 2/4 — Promptfoo Red-Team")
    logger.info("=" * 50)
    pf_runner = PromptfooRunner(chatbot_fn=chatbot_fn)
    promptfoo_results = pf_runner.run()
    report_gen.add_promptfoo_results(promptfoo_results)
    logger.info(f"Promptfoo: status={promptfoo_results['status']}")
    logger.info(f"  Config written to: {promptfoo_results.get('config_path', 'N/A')}")

    # ── 3. DeepEval metrics ───────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 3/4 — DeepEval Evaluation Metrics")
    logger.info("=" * 50)
    deepeval_results = DeepEvalRunner(chatbot_fn=chatbot_fn).run()
    report_gen.add_deepeval_results(deepeval_results)
    logger.info(f"DeepEval: status={deepeval_results['status']}")
    for name, m in deepeval_results.get("metrics", {}).items():
        icon = "✓" if m.get("passed") else "✗"
        logger.info(f"  {icon} {name}: {m.get('score', 0):.3f} (threshold {m.get('threshold', 0):.2f})")

    # ── 4. Fairness tests ─────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 4/4 — Fairness Evaluation")
    logger.info("=" * 50)
    if chatbot_fn and not skip_live:
        fairness_results = FairnessEvaluator(chatbot_fn=chatbot_fn).run()
    else:
        fairness_results = {
            "status": "skipped",
            "verdict": "SKIPPED",
            "issues": [],
            "metrics": {},
            "profile_results": {},
        }
    report_gen.add_fairness_results(fairness_results)
    logger.info(f"Fairness verdict: {fairness_results.get('verdict', 'N/A')}")
    for issue in fairness_results.get("issues", []):
        logger.warning(f"  ⚠ {issue}")

    # ── 5. Generate report ────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Generating Governance Report")
    logger.info("=" * 50)
    paths = report_gen.generate()

    logger.info("\n✅ Governance Report Generated:")
    for fmt, path in paths.items():
        logger.info(f"  {fmt.upper()}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BVRIT RAG Chatbot — Governance Report Runner"
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip scans that require live API calls (use mock answers instead).",
    )
    args = parser.parse_args()
    run(skip_live=args.skip_live)
