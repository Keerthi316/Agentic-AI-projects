"""
Bug Report Triage – 4-Stage LLM Prompt Pipeline
================================================
Day 2 Homework: The Prompt Pipeline

Pipeline stages
---------------
1. Understand  – Role: Senior QA Engineer.       Structured JSON extraction.
2. Reason      – Role: Senior Debugging Engineer. Chain-of-thought root-cause analysis.
3. Produce     – Goal-oriented. Developer-ready fix recommendation.
4. Self-Check  – Critique all previous outputs for quality.

Every stage's input and output is printed so the pipeline is fully inspectable.
"""

import json
import os
import textwrap
import time

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"          # Change to any model available on your plan
MAX_RETRIES = 3                        # Maximum attempts for JSON parse retries
RETRY_DELAY = 2                        # Seconds to wait between API retries


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """
    Send a prompt to the OpenRouter API and return the raw text response.

    Raises RuntimeError if the API call fails after MAX_RETRIES attempts.
    """
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file before running the pipeline."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/bug-report-triage",   # optional but polite
        "X-Title": "Bug Report Triage Pipeline",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,   # Low temperature for deterministic, structured output
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as exc:
            print(f"  [API] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"API call failed after {MAX_RETRIES} attempts.") from exc


def parse_json(response_text: str, prompt_for_retry: str) -> dict:
    """
    Parse a JSON string returned by the LLM.

    If parsing fails the model is asked to regenerate *only* valid JSON.
    Retries up to MAX_RETRIES times before returning a best-effort error dict.

    Parameters
    ----------
    response_text   : Raw text from the LLM (may include markdown fences).
    prompt_for_retry: The original prompt, used to ask the model to fix its output.
    """
    current_text = response_text

    for attempt in range(1, MAX_RETRIES + 1):
        # Strip common markdown code-fences the model sometimes adds
        cleaned = current_text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])   # drop opening fence line
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])  # drop closing fence line
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            print(f"  [JSON] Parse attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                # Ask the model to fix its output
                fix_prompt = (
                    f"The following text was supposed to be valid JSON but failed to parse:\n\n"
                    f"{current_text}\n\n"
                    f"Original task was:\n{prompt_for_retry}\n\n"
                    f"Please return ONLY valid JSON with no explanation, no markdown fences, "
                    f"and no extra text."
                )
                print(f"  [JSON] Asking model to regenerate valid JSON (attempt {attempt + 1})…")
                current_text = call_llm(fix_prompt)
            else:
                # All retries exhausted – return a structured error so the pipeline continues
                print("  [JSON] All parse attempts failed. Returning error placeholder.")
                return {
                    "parse_error": True,
                    "raw_response": response_text[:500],
                    "error_message": str(exc),
                }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 70
SECTION  = "-" * 70

def print_stage_header(stage_num: int, title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  STAGE {stage_num}: {title}")
    print(DIVIDER)


def print_input(label: str, content: str) -> None:
    print(f"\n{'[ INPUT: ' + label + ' ]':^70}")
    print(SECTION)
    # Wrap long lines for readability
    for line in content.splitlines():
        print(textwrap.fill(line, width=70) if len(line) > 70 else line)
    print(SECTION)


def print_output(label: str, data: dict) -> None:
    print(f"\n{'[ OUTPUT: ' + label + ' ]':^70}")
    print(SECTION)
    print(json.dumps(data, indent=2))
    print(SECTION)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage1_understand(bug_report: str) -> dict:
    """
    Stage 1 – Understand
    Role  : Senior QA Engineer
    Goal  : Extract structured information from the raw bug report.
    Output: JSON with standardised QA fields.
    """
    print_stage_header(1, "UNDERSTAND  (Role: Senior QA Engineer)")

    prompt = f"""You are a Senior QA Engineer reviewing incoming bug reports.

Analyze the following bug report carefully and extract all relevant information.

BUG REPORT:
-----------
{bug_report}
-----------

Return ONLY a valid JSON object (no markdown, no extra text) with exactly these fields:
{{
  "bug_summary": "<one-sentence summary of the bug>",
  "steps_to_reproduce": ["<step 1>", "<step 2>", "..."],
  "expected_behavior": "<what should happen>",
  "actual_behavior": "<what actually happens>",
  "error_messages": ["<error message 1>", "..."],
  "affected_component": "<module / service / layer affected>",
  "missing_information": ["<info that would help debug but is absent>", "..."]
}}

If a field cannot be determined from the report, use null for strings or [] for arrays."""

    print_input("Raw Bug Report", bug_report)

    raw = call_llm(prompt)
    result = parse_json(raw, prompt)

    print_output("Stage 1 JSON", result)
    return result


def stage2_reason(stage1_output: dict) -> dict:
    """
    Stage 2 – Reason
    Role  : Senior Debugging Engineer
    Goal  : Root-cause analysis using chain-of-thought reasoning.
    Output: JSON with root cause, severity, confidence, and reasoning chain.
    """
    print_stage_header(2, "REASON  (Role: Senior Debugging Engineer, Chain-of-Thought)")

    stage1_json_str = json.dumps(stage1_output, indent=2)

    prompt = f"""You are a Senior Debugging Engineer.

You have received the following structured bug analysis from a QA Engineer:

{stage1_json_str}

Think step by step INTERNALLY to diagnose the root cause. Consider:
1. What could cause the actual_behavior to differ from expected_behavior?
2. What do the error_messages tell us?
3. What component is most likely at fault and why?
4. What is the severity given the affected_component and error context?

After your internal reasoning, return ONLY a valid JSON object (no markdown, no extra text):
{{
  "likely_root_cause": "<concise description of the most probable root cause>",
  "severity": "<one of: Critical | High | Medium | Low>",
  "confidence": "<percentage string, e.g. '75%'>",
  "reasoning": [
    "<step 1 of your chain-of-thought>",
    "<step 2 of your chain-of-thought>",
    "<step 3 ...>"
  ]
}}"""

    print_input("Stage 1 JSON (passed to Stage 2)", stage1_json_str)

    raw = call_llm(prompt)
    result = parse_json(raw, prompt)

    print_output("Stage 2 JSON", result)
    return result


def stage3_produce(stage1_output: dict, stage2_output: dict) -> dict:
    """
    Stage 3 – Produce
    Goal  : Generate an actionable, developer-ready triage ticket.
    Input : Stage 1 + Stage 2 outputs combined.
    Output: JSON with developer summary, recommended fix, and debugging steps.
    """
    print_stage_header(3, "PRODUCE  (Goal: Developer-Ready Fix Recommendation)")

    combined = {
        "qa_analysis": stage1_output,
        "root_cause_analysis": stage2_output,
    }
    combined_str = json.dumps(combined, indent=2)

    prompt = f"""You are a Principal Engineer creating a developer-ready bug triage document.

Use the following analysis to produce clear, actionable guidance:

{combined_str}

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "developer_summary": "<2-3 sentence summary a developer can act on immediately>",
  "recommended_fix": "<specific, technical recommendation for fixing the root cause>",
  "next_debugging_steps": [
    "<concrete step 1 a developer should take first>",
    "<concrete step 2>",
    "<concrete step 3>",
    "<add more as needed>"
  ]
}}"""

    print_input("Stage 1 + Stage 2 JSON (passed to Stage 3)", combined_str)

    raw = call_llm(prompt)
    result = parse_json(raw, prompt)

    print_output("Stage 3 JSON", result)
    return result


def stage4_selfcheck(
    bug_report: str,
    stage1_output: dict,
    stage2_output: dict,
    stage3_output: dict,
) -> dict:
    """
    Stage 4 – Self-Critique
    Goal  : Review all previous pipeline outputs for quality and consistency.
    Output: JSON with quality score, issues found, revision flag, and revised summary.
    """
    print_stage_header(4, "SELF-CHECK  (Self-Critique of Entire Pipeline)")

    all_outputs = {
        "original_bug_report": bug_report,
        "stage1_understand": stage1_output,
        "stage2_reason": stage2_output,
        "stage3_produce": stage3_output,
    }
    all_str = json.dumps(all_outputs, indent=2)

    prompt = f"""You are an AI Quality Auditor reviewing the outputs of a bug-triage pipeline.

Review all inputs and outputs below for accuracy, completeness, consistency, and usefulness:

{all_str}

Evaluate critically:
- Does the analysis accurately reflect the original bug report?
- Is the root cause plausible given the evidence?
- Are the fix recommendations specific and actionable?
- Is anything missing, contradictory, or vague?

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "quality_score": <integer 1-10>,
  "issues_found": [
    "<issue 1 if any, or 'None' if the output is good>",
    "..."
  ],
  "needs_revision": <true | false>,
  "revised_summary": "<an improved 2-3 sentence developer summary, or the original if no revision is needed>"
}}"""

    print_input("All Pipeline Outputs (passed to Stage 4)", all_str[:2000] + ("\n... [truncated for display]" if len(all_str) > 2000 else ""))

    raw = call_llm(prompt)
    result = parse_json(raw, prompt)

    print_output("Stage 4 JSON (Final Quality Report)", result)
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(bug_report: str, case_label: str) -> dict:
    """
    Run the full 4-stage triage pipeline on a single bug report.

    Returns a dict with all four stage outputs keyed by stage name.
    Gracefully handles failures at any stage so remaining stages still run.
    """
    print(f"\n{'#' * 70}")
    print(f"  TEST CASE: {case_label}")
    print(f"{'#' * 70}")

    results = {}

    # -- Stage 1 -------------------------------------------------------
    try:
        results["stage1"] = stage1_understand(bug_report)
    except Exception as exc:
        print(f"\n  [PIPELINE] Stage 1 failed: {exc}")
        results["stage1"] = {"error": str(exc)}

    # -- Stage 2 -------------------------------------------------------
    try:
        results["stage2"] = stage2_reason(results["stage1"])
    except Exception as exc:
        print(f"\n  [PIPELINE] Stage 2 failed: {exc}")
        results["stage2"] = {"error": str(exc)}

    # -- Stage 3 -------------------------------------------------------
    try:
        results["stage3"] = stage3_produce(results["stage1"], results["stage2"])
    except Exception as exc:
        print(f"\n  [PIPELINE] Stage 3 failed: {exc}")
        results["stage3"] = {"error": str(exc)}

    # -- Stage 4 -------------------------------------------------------
    try:
        results["stage4"] = stage4_selfcheck(
            bug_report,
            results["stage1"],
            results["stage2"],
            results["stage3"],
        )
    except Exception as exc:
        print(f"\n  [PIPELINE] Stage 4 failed: {exc}")
        results["stage4"] = {"error": str(exc)}

    print(f"\n{'#' * 70}")
    print(f"  PIPELINE COMPLETE: {case_label}")
    print(f"  Final quality score: {results['stage4'].get('quality_score', 'N/A')}/10")
    print(f"  Needs revision: {results['stage4'].get('needs_revision', 'N/A')}")
    print(f"{'#' * 70}")

    return results


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# -- Test Case 1: Normal bug report --------------------------------------
BUG_REPORT_1 = """
Title: Login button unresponsive on mobile Safari

Description:
When a user tries to log in using mobile Safari on iOS 17, tapping the
"Login" button does nothing. The button appears to register the tap
(briefly highlights) but the form is never submitted and the page does
not navigate. This works fine on Chrome for Android and desktop browsers.

Steps to reproduce:
1. Open https://app.example.com/login on an iPhone running iOS 17 with Safari.
2. Enter valid credentials.
3. Tap the "Login" button.

Expected: The user is authenticated and redirected to /dashboard.
Actual: Nothing happens. No network request is made (confirmed via proxy).

Environment: iOS 17.4, Safari 17, iPhone 14 Pro. App version 3.2.1.
"""

# -- Test Case 2: Complex bug with stack trace ----------------------------
BUG_REPORT_2 = """
Title: NullPointerException in PaymentService during checkout

Description:
Production is throwing a NullPointerException roughly 30% of the time
during the final step of the checkout flow. The cart service appears to
return a payment intent object that sometimes has a null `currency` field.
This happens exclusively for guest checkout users; logged-in users are
unaffected. Started occurring after the v4.1.0 deploy on 2026-06-28.

Stack Trace:
-----------
java.lang.NullPointerException: Cannot invoke "String.toLowerCase()" because "currency" is null
    at com.example.payment.PaymentService.normalizeCurrency(PaymentService.java:142)
    at com.example.payment.PaymentService.processPayment(PaymentService.java:87)
    at com.example.checkout.CheckoutController.completeOrder(CheckoutController.java:214)
    at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
    at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:897)

Frequency: ~30% of guest checkout attempts.
Deploy that introduced the issue: v4.1.0 (2026-06-28).
Related PR: #1847 – "Unify currency handling across cart and payment services".
"""

# -- Test Case 3: Deliberately broken / gibberish input ------------------
BUG_REPORT_3 = """
asdfghjkl thing broken pls fix urgent!!!
the widget thingy does not do the stuff it should when you click the
blue doohickey after logging in sometimes maybe on Tuesdays??
error says something like "bad" idk
"""


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

REFLECTION = """
=======================================================================
  PIPELINE REFLECTION  (150-200 words)
=======================================================================

The weakest stage in this pipeline is Stage 2 – Reason. Its core task
is root-cause analysis, yet it operates solely on information already
extracted by Stage 1. Because the LLM has no access to actual source
code, logs, metrics, deployment history, or runtime telemetry, its
chain-of-thought is entirely speculative. The model can identify
*plausible* hypotheses, but it cannot rule them in or out with evidence.
Confidence percentages are therefore self-reported guesses rather than
calibrated estimates, making them potentially misleading for engineering
teams under pressure.

RAG (Retrieval-Augmented Generation) could dramatically improve this
stage by injecting relevant snippets from the codebase, recent commit
diffs, and historical bug tickets directly into the prompt. External
tools such as a log-search API (e.g., CloudWatch Insights or Splunk)
or a test-results feed could supply runtime evidence the LLM otherwise
lacks. With that grounding, Stage 2 could move from educated guessing
to evidence-driven diagnosis, raising both accuracy and the practical
usefulness of the confidence scores it emits.
=======================================================================
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  BUG REPORT TRIAGE – 4-Stage LLM Prompt Pipeline")
    print("  Model:", MODEL)
    print("=" * 70)

    # Run all three test cases
    all_results = {}

    all_results["case1"] = run_pipeline(BUG_REPORT_1, "Case 1 – Normal Bug Report")
    all_results["case2"] = run_pipeline(BUG_REPORT_2, "Case 2 – Complex Bug with Stack Trace")
    all_results["case3"] = run_pipeline(BUG_REPORT_3, "Case 3 – Broken / Gibberish Input")

    # Print reflection
    print(REFLECTION)

    # Persist all results to a JSON file for reference
    output_path = "pipeline_results.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to: {output_path}")
    print("\n  Done.\n")
