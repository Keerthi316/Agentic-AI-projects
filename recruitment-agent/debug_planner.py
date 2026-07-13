import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from prompts.planner_prompt import PLANNER_PROMPT

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("MODEL", "openai/gpt-4o-mini")
llm = ChatOpenAI(model=model, openai_api_key=api_key, openai_api_base="https://openrouter.ai/api/v1", temperature=0.1)

# Simulate state after scoring Priya
jd_summary = '{"job_title": "Junior AI Engineer", "required_skills": ["Python", "Machine Learning", "SQL"]}'
rubric_summary = '{"criteria": [{"name": "Python", "weight": 25}]}'
remaining = "['Rahul Verma', 'Alex Johnson']"
processed = "['Priya Sharma']"
shortlist = "Not yet ranked"

prompt = PLANNER_PROMPT.format(
    jd_summary=jd_summary,
    rubric_summary=rubric_summary,
    remaining_candidates=remaining,
    processed_candidates=processed,
    shortlist_summary=shortlist
)

response = llm.invoke([HumanMessage(content=prompt)])
content = response.content.strip()
print("RAW RESPONSE:")
print(content)
print()
if content.startswith("```"):
    content = content.split("\n", 1)[1]
    content = content.rsplit("\n", 1)[0]
    if content.endswith("```"):
        content = content[:-3]
try:
    data = json.loads(content)
    print("PARSED:")
    print(json.dumps(data, indent=2))
except:
    print("FAILED TO PARSE")