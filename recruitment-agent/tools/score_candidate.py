"""
LLM-based candidate scoring tool.
Uses the scorer prompt to evaluate candidates against rubric.
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from prompts.scorer_prompt import SCORER_PROMPT
from models.schemas import ScoreCard, ParsedResume


def score_candidate(profile: ParsedResume, job_req: str, rubric: str, llm: ChatOpenAI) -> ScoreCard:
    """
    Score a candidate against the job requirements and rubric using an LLM.
    """
    prompt = SCORER_PROMPT.format(
        profile=profile.model_dump_json(indent=2),
        job_req=job_req,
        rubric=rubric
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("\n", 1)[0]
        if content.endswith("```"):
            content = content[:-3]

    data = json.loads(content)
    return ScoreCard(**data)