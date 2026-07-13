from recruitment_agent.tools import parse_resume, build_rubric, score_candidate
from recruitment_agent.test_data import JOB_DESCRIPTION, CANDIDATES

r = build_rubric(JOB_DESCRIPTION)
for name, text in CANDIDATES.items():
    p = parse_resume(text)
    sc = score_candidate(p, r)
    ns = sc["normalized_score"]
    print(f"{name}: {ns}% - {sc['total_weighted_score']}/{sc['max_possible_score']}")
    for cs in sc["criteria_scores"]:
        print(f"  {cs['criterion']}: {cs['score']}/5 (w:{cs['weight']})")
    print()