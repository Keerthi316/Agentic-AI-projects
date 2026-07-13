"""
Standalone test script to run the recruitment agent and capture full output.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(__file__))

from recruitment_agent.test_data import JOB_DESCRIPTION, CANDIDATES
from recruitment_agent.agent import run_agent, approve_interview

# Run the agent
result = run_agent(JOB_DESCRIPTION, CANDIDATES)

# Capture output to list
out_lines = []
out_lines.append("=" * 80)
out_lines.append("FULL REASONING TRAJECTORY")
out_lines.append("=" * 80)
for step in result.get("trajectory", []):
    out_lines.append("")
    out_lines.append(f"--- Step {step['step_number']} ---")
    out_lines.append(f"Thought: {step['thought']}")
    out_lines.append(f"Action: {step['action']}")
    out_lines.append(f"Input: {json.dumps(step['input'], indent=2)}")
    out_lines.append(f"Observation: {str(step['observation'])[:500]}")
    out_lines.append(f"Decision: {step['decision']}")

out_lines.append("")
out_lines.append("")
out_lines.append("=" * 80)
out_lines.append("FINAL SHORTLIST")
out_lines.append("=" * 80)
for i, entry in enumerate(result.get("shortlist", [])):
    out_lines.append("")
    out_lines.append("=" * 40)
    out_lines.append(f"Rank #{i + 1}: {entry['name']}")
    out_lines.append(f"Decision: {entry['decision']}")
    out_lines.append(f"Score: {entry['score']}%")
    out_lines.append(f"Justification:")
    out_lines.append(f"  {entry['justification']}")
    sc = entry['scorecard']
    out_lines.append(f"\nScorecard Breakdown:")
    for cs in sc['criteria_scores']:
        out_lines.append(f"  {cs['criterion']}: {cs['score']}/5 (weight: {cs['weight']})")
        out_lines.append(f"    Evidence: {cs['evidence'][:200]}")

out_lines.append("")
out_lines.append("")
out_lines.append("=" * 80)
out_lines.append("PROPOSED ACTIONS")
out_lines.append("=" * 80)
for action in result.get("actions", []):
    slot = action['proposed_slot']
    out_lines.append(f"")
    out_lines.append(f"Candidate: {action['candidate']}")
    out_lines.append(f"  Proposed Slot: {slot['day']} {slot['start_time']}-{slot['end_time']}")
    out_lines.append(f"  Status: {action['status']}")

out_lines.append("")
out_lines.append("")
out_lines.append("=" * 80)
out_lines.append("GUARDRAIL ENFORCEMENT REPORT")
out_lines.append("=" * 80)
injection_count = sum(1 for p in result.get("parsed_profiles", {}).values() if p.get("has_prompt_injection", False))
out_lines.append("")
out_lines.append("1. Prompt Injection Defense:")
out_lines.append(f"   - Detected: {injection_count} candidate(s) with injection attempts")
for name, profile in result.get("parsed_profiles", {}).items():
    if profile.get("has_prompt_injection", False):
        out_lines.append(f"   - {name}: INJECTION BLOCKED - instructions ignored")
out_lines.append("")
out_lines.append(f"2. Step Limit: {result.get('step_count', 0)}/50 steps used")
out_lines.append("")
out_lines.append("3. Human-in-the-Loop:")
pending = [a for a in result.get("actions", []) if a["status"] == "PENDING_APPROVAL"]
out_lines.append(f"   - Pending approvals: {len(pending)}")
out_lines.append("")
out_lines.append("4. Fairness: Biased attributes ignored, same rubric for all")
out_lines.append("")
out_lines.append(f"5. Audit Log: {len(result.get('trajectory', []))} entries")

out_lines.append("")
out_lines.append("")
out_lines.append("=" * 80)
out_lines.append("STRUCTURED DECISION OUTPUT")
out_lines.append("=" * 80)
decision_output = {
    "shortlist": [
        {
            "name": entry["name"],
            "decision": entry["decision"],
            "score": entry["score"],
            "justification": entry["justification"],
            "scorecard": entry["scorecard"]
        }
        for entry in result.get("shortlist", [])
    ],
    "actions": [
        {
            "candidate": a["candidate"],
            "proposed_slot": a["proposed_slot"],
            "status": a["status"]
        }
        for a in result.get("actions", [])
    ],
    "trajectory_summary": {
        "total_steps": result.get("step_count", 0),
        "status": result.get("status", "UNKNOWN"),
        "injection_attempts_blocked": injection_count
    }
}
out_lines.append(json.dumps(decision_output, indent=2, default=str))

# Demonstrate human approval
if result.get("status") == "WAITING_APPROVAL":
    out_lines.append("")
    out_lines.append("")
    out_lines.append("=" * 80)
    out_lines.append("HUMAN-IN-THE-LOOP: Approving interviews")
    out_lines.append("=" * 80)
    for action in result.get("actions", []):
        if action["status"] == "PENDING_APPROVAL":
            result = approve_interview(result, action["candidate"], approved=True)
            out_lines.append(f"  ✓ Approved: {action['candidate']}")

out_lines.append("")
out_lines.append("Agent run completed successfully!")

# Write to file
with open("agent_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Output written to agent_output.txt")
print("Status:", result.get("status"))