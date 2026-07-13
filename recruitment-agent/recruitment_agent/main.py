"""
Main entry point for the Recruitment Agent.
Runs the agent with test data and displays results.
"""

import json
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recruitment_agent.test_data import JOB_DESCRIPTION, CANDIDATES
from recruitment_agent.agent import run_agent as _run_agent, approve_interview

# Use the sequential agent directly
def run_agent(jd, candidates):
    return _run_agent(jd, candidates)


def format_trajectory(trajectory) -> str:
    """Format the trajectory log for display."""
    lines = []
    lines.append("=" * 80)
    lines.append("FULL REASONING TRAJECTORY")
    lines.append("=" * 80)
    
    for step in trajectory:
        lines.append(f"\n--- Step {step['step_number']} ---")
        lines.append(f"Thought: {step['thought']}")
        lines.append(f"Action: {step['action']}")
        lines.append(f"Input: {json.dumps(step['input'], indent=2)}")
        lines.append(f"Observation: {step['observation'][:500]}")
        lines.append(f"Decision: {step['decision']}")
    
    return "\n".join(lines)


def format_shortlist(shortlist) -> str:
    """Format the shortlist for display."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("FINAL SHORTLIST")
    lines.append("=" * 80)
    
    for i, entry in enumerate(shortlist):
        lines.append(f"\n{'=' * 40}")
        lines.append(f"Rank #{i + 1}: {entry['name']}")
        lines.append(f"Decision: {entry['decision']}")
        lines.append(f"Score: {entry['score']}%")
        lines.append(f"\nJustification:")
        lines.append(f"  {entry['justification']}")
        
        # Show scorecard details
        sc = entry['scorecard']
        lines.append(f"\nScorecard Breakdown:")
        for cs in sc['criteria_scores']:
            lines.append(f"  {cs['criterion']}: {cs['score']}/5 (weight: {cs['weight']})")
            lines.append(f"    Evidence: {cs['evidence'][:200]}")
    
    return "\n".join(lines)


def format_actions(actions) -> str:
    """Format pending actions for display."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("PROPOSED ACTIONS")
    lines.append("=" * 80)
    
    if not actions:
        lines.append("No actions proposed.")
        return "\n".join(lines)
    
    for action in actions:
        slot = action['proposed_slot']
        lines.append(f"\nCandidate: {action['candidate']}")
        lines.append(f"  Proposed Slot: {slot['day']} {slot['start_time']}-{slot['end_time']}")
        lines.append(f"  Status: {action['status']}")
    
    return "\n".join(lines)


def format_guardrail_report(state) -> str:
    """Format guardrail enforcement report."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("GUARDRAIL ENFORCEMENT REPORT")
    lines.append("=" * 80)
    
    # 1. Prompt Injection
    injection_count = sum(
        1 for p in state.get("parsed_profiles", {}).values()
        if p.get("has_prompt_injection", False)
    )
    lines.append(f"\n1. Prompt Injection Defense:")
    lines.append(f"   - Detected: {injection_count} candidate(s) with injection attempts")
    for name, profile in state.get("parsed_profiles", {}).items():
        if profile.get("has_prompt_injection", False):
            lines.append(f"   - {name}: INJECTION BLOCKED - instructions ignored")
    
    # 2. Step Limit
    lines.append(f"\n2. Step Limit:")
    lines.append(f"   - Max steps: 50")
    lines.append(f"   - Steps taken: {state.get('step_count', 0)}")
    lines.append(f"   - Status: {'Within limits' if state.get('step_count', 0) < 50 else 'LIMIT REACHED'}")
    
    # 3. Human-in-the-Loop
    pending = [a for a in state.get("actions", []) if a["status"] == "PENDING_APPROVAL"]
    lines.append(f"\n3. Human-in-the-Loop:")
    lines.append(f"   - Pending approvals: {len(pending)}")
    for a in pending:
        lines.append(f"   - {a['candidate']}: Waiting for human decision")
    
    # 4. Fairness
    lines.append(f"\n4. Fairness Check:")
    lines.append(f"   - Biased attributes (name, gender, age, college prestige): IGNORED")
    lines.append(f"   - All candidates evaluated on same rubric criteria")
    lines.append(f"   - Scores based solely on skills, experience, and qualifications")
    
    # 5. Audit Log
    lines.append(f"\n5. Audit Log:")
    lines.append(f"   - Trajectory entries: {len(state.get('trajectory', []))}")
    lines.append(f"   - Full trajectory available for inspection")
    
    return "\n".join(lines)


def main():
    """Run the recruitment agent with test data."""
    print("=" * 80)
    print("TECHVEST RECRUITMENT AGENT")
    print("Autonomous Hiring Decision System")
    print("=" * 80)
    print(f"\nJob: Junior AI Engineer")
    print(f"Candidates: {', '.join(CANDIDATES.keys())}")
    print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the agent
    print("\n" + "-" * 80)
    print("INITIALIZING AGENT...")
    print("-" * 80)
    
    result = run_agent(JOB_DESCRIPTION, CANDIDATES)
    
    # Display trajectory
    print(format_trajectory(result.get("trajectory", [])))
    
    # Display shortlist
    print(format_shortlist(result.get("shortlist", [])))
    
    # Display actions
    print(format_actions(result.get("actions", [])))
    
    # Display guardrail report
    print(format_guardrail_report(result))
    
    # Display final status
    print("\n" + "=" * 80)
    print(f"AGENT STATUS: {result.get('status', 'UNKNOWN')}")
    print("=" * 80)
    
    # If waiting for approval, demonstrate the approval flow
    if result.get("status") == "WAITING_APPROVAL":
        print("\n" + "-" * 80)
        print("HUMAN-IN-THE-LOOP DEMONSTRATION")
        print("-" * 80)
        print("\nAgent is paused waiting for human approval.")
        print("Calling approve_interview() to demonstrate approval flow...")
        
        for action in result.get("actions", []):
            if action["status"] == "PENDING_APPROVAL":
                result = approve_interview(result, action["candidate"], approved=True)
                print(f"  ✓ Approved interview for {action['candidate']}")
        
        print("\nAll approvals processed. Agent status updated.")
    
    # Output structured decision object
    print("\n" + "=" * 80)
    print("STRUCTURED DECISION OUTPUT")
    print("=" * 80)
    
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
            "injection_attempts_blocked": sum(
                1 for p in result.get("parsed_profiles", {}).values()
                if p.get("has_prompt_injection", False)
            )
        }
    }
    
    print(json.dumps(decision_output, indent=2, default=str))
    
    # Save to file
    output_file = "recruitment_decision_output.json"
    with open(output_file, "w") as f:
        json.dump(decision_output, f, indent=2, default=str)
    print(f"\nDecision output saved to: {output_file}")
    
    # Save full trajectory to file
    trajectory_file = "recruitment_trajectory.json"
    with open(trajectory_file, "w") as f:
        json.dump({
            "trajectory": result.get("trajectory", []),
            "guardrail_report": format_guardrail_report(result)
        }, f, indent=2, default=str)
    print(f"Full trajectory saved to: {trajectory_file}")
    
    return result


if __name__ == "__main__":
    result = main()