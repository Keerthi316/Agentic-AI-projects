"""Debug script to test agent execution and save results to file."""
import sys, os, json, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recruitment_agent.test_data import JOB_DESCRIPTION, CANDIDATES
from recruitment_agent.agent import _run_sequential

try:
    result = _run_sequential(JOB_DESCRIPTION, CANDIDATES)
    
    debug_info = {
        "status": result.get("status"),
        "step_count": result.get("step_count"),
        "error": result.get("error"),
        "shortlist_count": len(result.get("shortlist", [])),
        "actions_count": len(result.get("actions", [])),
        "trajectory_count": len(result.get("trajectory", [])),
        "shortlist": [
            {
                "name": e["name"],
                "decision": e["decision"],
                "score": e["score"],
                "justification": e["justification"]
            }
            for e in result.get("shortlist", [])
        ],
        "trajectory": [
            {
                "step": s["step_number"],
                "thought": s["thought"],
                "action": s["action"],
                "decision": s["decision"],
                "observation": s["observation"][:200]
            }
            for s in result.get("trajectory", [])
        ],
        "parsed_profiles": {
            name: {
                "years_exp": p.get("years_of_experience", 0),
                "has_injection": p.get("has_prompt_injection", False)
            }
            for name, p in result.get("parsed_profiles", {}).items()
        }
    }
    
    with open("debug_result.json", "w") as f:
        json.dump(debug_info, f, indent=2, default=str)
    
    print("SUCCESS - debug_result.json written")
    
except Exception as e:
    error_info = {
        "error": str(e),
        "traceback": traceback.format_exc()
    }
    with open("debug_error.json", "w") as f:
        json.dump(error_info, f, indent=2)
    print("ERROR - debug_error.json written")