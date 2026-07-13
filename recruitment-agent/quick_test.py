"""
Quick test to verify the agent works and write results to a file.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from recruitment_agent.test_data import JOB_DESCRIPTION, CANDIDATES
from recruitment_agent.agent import run_agent
from recruitment_agent.tools import parse_resume, build_rubric, score_candidate

# Test scoring first
r = build_rubric('')
results = []
for name, text in CANDIDATES.items():
    p = parse_resume(text)
    sc = score_candidate(p, r)
    results.append({
        "name": name,
        "score": sc["normalized_score"],
        "injection": p.get("has_prompt_injection", False)
    })

with open("scoring_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Run full agent
result = run_agent(JOB_DESCRIPTION, CANDIDATES)

# Save full state
with open("full_state.json", "w") as f:
    # Convert to serializable dict
    state_dict = {}
    for k, v in result.items():
        try:
            json.dumps({k: v}, default=str)
            state_dict[k] = v
        except:
            state_dict[k] = str(v)
    json.dump(state_dict, f, indent=2, default=str)

# Save just the important parts
output = {
    "status": result.get("status"),
    "steps": result.get("step_count"),
    "shortlist": [
        {
            "name": e["name"],
            "decision": e["decision"],
            "score": e["score"],
            "justification": e["justification"]
        }
        for e in result.get("shortlist", [])
    ],
    "actions": [
        {
            "candidate": a["candidate"],
            "slot": a["proposed_slot"],
            "status": a["status"]
        }
        for a in result.get("actions", [])
    ],
    "trajectory": [
        {
            "step": s["step_number"],
            "thought": s["thought"],
            "action": s["action"],
            "decision": s["decision"]
        }
        for s in result.get("trajectory", [])
    ],
    "injection_blocked": sum(
        1 for p in result.get("parsed_profiles", {}).values()
        if p.get("has_prompt_injection", False)
    )
}

with open("agent_output.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("SUCCESS - output written to agent_output.json")
print("Status:", result.get("status"))
print("Candidates evaluated:", len(result.get("shortlist", [])))