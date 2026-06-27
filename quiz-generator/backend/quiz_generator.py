"""
Quiz Generator Module
Uses OpenAI to generate MCQs from PPT content
"""

import json
import re
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client (supports OpenAI and OpenRouter via OPENAI_BASE_URL)
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
client = OpenAI(api_key=api_key, base_url=api_base) if api_key else None

DIFFICULTY_PROMPTS = {
    "simple": {
        "description": "basic recall and recognition level questions",
        "guidelines": "Questions should test fundamental concepts, definitions, and direct recall. Use simple language. Distractors should be clearly different from the correct answer."
    },
    "medium": {
        "description": "comprehension and application level questions",
        "guidelines": "Questions should test understanding of concepts, ability to apply knowledge, and make connections between ideas. Distractors should be plausible but incorrect."
    },
    "complex": {
        "description": "analysis, evaluation, and synthesis level questions",
        "guidelines": "Questions should test critical thinking, ability to analyze scenarios, evaluate competing ideas, and synthesise multiple concepts. Distractors should be highly plausible and require deep understanding to eliminate."
    }
}


def generate_quiz_with_ai(slide_text, num_questions, difficulty):
    """
    Generate MCQ quiz questions using OpenAI based on slide content.
    
    Args:
        slide_text (str): Combined text from all slides
        num_questions (int): Number of questions to generate (5-30)
        difficulty (str): Difficulty level (simple/medium/complex)
        
    Returns:
        list: List of question objects
    """
    if not client:
        raise ValueError(
            "OpenAI API key not configured. "
            "Please set OPENAI_API_KEY in your .env file."
        )
    
    if num_questions < 5 or num_questions > 30:
        raise ValueError("Number of questions must be between 5 and 30")
    
    if difficulty not in DIFFICULTY_PROMPTS:
        raise ValueError(f"Invalid difficulty level. Choose from: {', '.join(DIFFICULTY_PROMPTS.keys())}")
    
    diff_config = DIFFICULTY_PROMPTS[difficulty]
    
    all_valid_questions = []
    remaining = num_questions
    max_iterations = 5
    iteration = 0
    
    while remaining > 0 and iteration < max_iterations:
        iteration += 1
        
        system_prompt = f"""You are an expert quiz generator specializing in creating high-quality multiple-choice questions.

Your task is to generate EXACTLY {remaining} {diff_config['description']} based on the provided presentation content. You MUST generate EXACTLY {remaining} questions — no more, no less.

Difficulty Level: {difficulty.upper()}
{diff_config['guidelines']}

For each question, you MUST:
1. Create exactly 4 options labeled A, B, C, D
2. Designate exactly one correct answer
3. Make distractors plausible and educational
4. Provide a clear explanation for WHY the correct answer is right
5. For wrong options (distractors), provide an explanation of why they are incorrect

IMPORTANT: Output ONLY valid JSON. Do NOT include markdown code blocks, backticks, or any text outside the JSON structure.

Return a JSON array with EXACTLY {remaining} questions using this structure:
[
  {{
    "question": "What is...?",
    "options": [
      {{ "label": "A", "text": "Option text", "is_correct": true }},
      {{ "label": "B", "text": "Option text", "is_correct": false }},
      {{ "label": "C", "text": "Option text", "is_correct": false }},
      {{ "label": "D", "text": "Option text", "is_correct": false }}
    ],
    "correct_answer": "A",
    "explanation": "Explanation of why the correct answer is right",
    "distractor_explanations": {{
      "B": "Why B is incorrect",
      "C": "Why C is incorrect",
      "D": "Why D is incorrect"
    }}
  }}
]"""

        user_prompt = f"""Here is the content from the presentation slides:

{slide_text}

Generate EXACTLY {remaining} {difficulty} level multiple-choice questions based on this content. Each question must have exactly 4 options (A-D) with one correct answer and explanations for all wrong options. Do NOT generate fewer or more than {remaining} questions."""

        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean the response - remove markdown code blocks if present
            content = re.sub(r'^```(?:json)?\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            content = content.strip()
            
            quiz_data = json.loads(content)
            
            # Validate the structure
            if not isinstance(quiz_data, list):
                if iteration == 1:
                    raise ValueError("AI response is not a list of questions")
                print(f"Warning: AI response is not a list (iteration {iteration}), skipping batch")
                continue
            
            # Ensure each question has proper structure
            batch_valid = 0
            for i, q in enumerate(quiz_data):
                if not all(k in q for k in ("question", "options", "correct_answer", "explanation")):
                    print(f"Warning: Question {len(all_valid_questions) + i + 1} missing required fields, skipping")
                    continue
                
                if len(q["options"]) != 4:
                    print(f"Warning: Question {len(all_valid_questions) + i + 1} doesn't have exactly 4 options, skipping")
                    continue
                
                # Ensure distractor_explanations exists
                if "distractor_explanations" not in q:
                    q["distractor_explanations"] = {}
                    for opt in q["options"]:
                        if not opt["is_correct"]:
                            q["distractor_explanations"][opt["label"]] = "This option does not correctly answer the question based on the presentation content."
                
                all_valid_questions.append(q)
                batch_valid += 1
            
            if batch_valid > 0:
                print(f"Iteration {iteration}: Generated {batch_valid} valid questions (total: {len(all_valid_questions)}/{num_questions})")
            
            # Recalculate remaining
            remaining = num_questions - len(all_valid_questions)
            
            if remaining > 0 and iteration < max_iterations:
                print(f"Still need {remaining} more question(s), making additional request...")
                import time
                time.sleep(1)  # Brief pause before retry
            
        except json.JSONDecodeError as e:
            if iteration == 1:
                raise ValueError(f"Failed to parse AI response as JSON: {str(e)}")
            print(f"Warning: Failed to parse AI response as JSON (iteration {iteration}), retrying for remaining {remaining} question(s)")
            continue
        except Exception as e:
            if iteration == 1:
                raise Exception(f"Error generating quiz with AI: {str(e)}")
            print(f"Warning: Error in iteration {iteration}: {str(e)}, retrying for remaining {remaining} question(s)")
            continue
    
    if not all_valid_questions:
        raise ValueError("No valid questions could be generated")
    
    # Log if we got fewer than requested after all retries
    if len(all_valid_questions) < num_questions:
        print(f"Warning: Requested {num_questions} questions but got {len(all_valid_questions)} after {max_iterations} iterations")
    
    # Trim to exact count if we somehow got extra
    final_questions = all_valid_questions[:num_questions]
    
    return final_questions


def generate_with_fallback(slide_text, num_questions, difficulty):
    """
    Attempt to generate quiz with retry logic.
    
    Args:
        slide_text (str): Combined text from all slides
        num_questions (int): Number of questions to generate
        difficulty (str): Difficulty level
        
    Returns:
        list: List of question objects
    """
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return generate_quiz_with_ai(slide_text, num_questions, difficulty)
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
    
    raise last_error if last_error else Exception("Quiz generation failed after all retries")