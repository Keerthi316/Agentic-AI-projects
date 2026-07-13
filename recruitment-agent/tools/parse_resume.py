"""
LLM-based resume parsing tool.
Uses the planner prompt to extract structured data from resume text.
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from prompts.parser_prompt import RESUME_PARSER_PROMPT
from models.schemas import ParsedResume


def parse_resume(resume_text: str, llm: ChatOpenAI) -> ParsedResume:
    """
    Parse a resume using an LLM to extract structured data.
    The prompt includes injection defense instructions.
    """
    prompt = RESUME_PARSER_PROMPT.format(resume=resume_text)
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    
    # Clean markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("\n", 1)[0]
        if content.endswith("```"):
            content = content[:-3]
    
    data = json.loads(content)
    return ParsedResume(**data)