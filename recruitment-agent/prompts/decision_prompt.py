DECISION_PROMPT = """You are the final Hiring Decision Agent.

Using the candidate scorecards, produce a ranked shortlist with scores normalized to 0-100.

### DECISION RULES (strictly enforced):
- score < 50 → decision = "Reject"
- 50 <= score <= 70 → decision = "Hold"
- score > 70 → decision = "Interview"

The score must be a number between 0 and 100, calculated as a weighted average of all criteria scores converted to percentage.

Return JSON
{{
"ranking":[
{{
"candidate":"",
"rank":1,
"decision":"",
"score":0,
"summary":"",
"evidence":[],
"interview_focus":[],
"slot":""
}}
]
}}

Scorecards:
{scorecards}
"""
