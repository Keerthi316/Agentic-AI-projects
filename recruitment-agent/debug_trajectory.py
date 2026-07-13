import sys, os, json, traceback
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()
from graph.graph import run_agent
from recruitment_agent.test_data import JOB_DESCRIPTION, CANDIDATES

try:
    r = run_agent(JOB_DESCRIPTION, CANDIDATES)
    print("Status:", r.status)
    print("Error:", r.error)
    print("Steps:", r.step_count)
    for s in r.trajectory:
        print(f"  Step {s['step_number']}: {s['tool']} -> {s['decision']}")
        print(f"    Thought: {s['thought'][:80]}")
    print("Scorecards:", list(r.scorecards.keys()))
    print("Shortlist:", r.shortlist)
    print("Actions:", r.actions)
except Exception as e:
    print("EXCEPTION:", e)
    traceback.print_exc()