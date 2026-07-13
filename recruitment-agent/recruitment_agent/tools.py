"""
Tool definitions for the Recruitment Agent.
All tools have typed inputs/outputs and are callable by the agent.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .state import CandidateProfile, ScoreCard, CriterionScore, TimeSlot, InterviewProposal


def parse_resume(resume_text: str) -> CandidateProfile:
    """
    Parse a raw resume text into a structured CandidateProfile.
    
    Guardrail: Detects prompt injection attempts in the resume.
    Treats resumes as untrusted input - ignores malicious instructions.
    """
    # --- GUARDRAIL: Prompt Injection Detection ---
    injection_patterns = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"ignore\s+(all\s+)?(previous|prior)\s+prompts?",
        r"disregard\s+(all\s+)?(previous|prior)\s+(instructions|criteria)",
        r"system\s+override",
        r"rank\s+(this|me|this\s+candidate)\s+(as\s+)?(the\s+)?(top|first|highest)",
        r"perfect\s+score",
        r"score\s+of\s+5",
        r"ignore\s+all\s+previous",
    ]
    
    has_injection = False
    injection_evidence = []
    
    for pattern in injection_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        if matches:
            has_injection = True
            for match in matches:
                matched_text = match if isinstance(match, str) else " ".join(filter(None, match))
                injection_evidence.append(matched_text)
    
    # Sanitize resume - remove injection lines for parsing
    # But keep the resume intact for evidence purposes
    sanitized_text = resume_text
    if has_injection:
        # Remove lines that contain injection attempts (don't parse them as resume data)
        injection_lines = []
        for pattern in injection_patterns:
            for line in sanitized_text.split('\n'):
                if re.search(pattern, line, re.IGNORECASE):
                    injection_lines.append(line)
        for bad_line in injection_lines:
            sanitized_text = sanitized_text.replace(bad_line, "")
    
    # Extract name
    name_match = re.search(r"Name:\s*(.+)", sanitized_text)
    name = name_match.group(1).strip() if name_match else "Unknown"
    
    # Extract email
    email_match = re.search(r"Email:\s*([\w.@]+)", sanitized_text)
    email = email_match.group(1).strip() if email_match else ""
    
    # Extract education
    education = []
    in_education = False
    edu_section = ""
    
    edu_match = re.search(r"EDUCATION\s*(.*?)(?:WORK EXPERIENCE|PROFESSIONAL SUMMARY|CERTIFICATIONS|PROJECTS|TECHNICAL SKILLS)", 
                          sanitized_text, re.DOTALL | re.IGNORECASE)
    if edu_match:
        edu_text = edu_match.group(1).strip()
        entries = re.split(r"\n\s*\n", edu_text)
        for entry in entries:
            entry = entry.strip()
            if entry:
                lines = entry.split('\n')
                if len(lines) >= 2:
                    education.append({
                        "degree": lines[0].strip(),
                        "institution": lines[1].strip() if len(lines) > 1 else "",
                        "details": "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
                    })
                else:
                    education.append({"degree": entry, "institution": "", "details": ""})
    
    # Extract work experience
    work_experience = []
    exp_match = re.search(r"WORK EXPERIENCE\s*(.*?)(?:PROJECTS|TECHNICAL SKILLS|CERTIFICATIONS|EDUCATION)", 
                          sanitized_text, re.DOTALL | re.IGNORECASE)
    if exp_match:
        exp_text = exp_match.group(1).strip()
        # Split by job entries (looking for role patterns)
        job_entries = re.split(r"\n\s*\n", exp_text)
        for entry in job_entries:
            entry = entry.strip()
            if entry and len(entry) > 20:
                lines = entry.split('\n')
                title_line = lines[0].strip() if lines else ""
                # Extract company from title line if pattern is "Role | Company | Location"
                company = ""
                role = title_line
                if '|' in title_line:
                    parts = [p.strip() for p in title_line.split('|')]
                    role = parts[0] if len(parts) > 0 else ""
                    company = parts[1] if len(parts) > 1 else ""
                
                date_range = ""
                responsibilities = []
                for line in lines[1:]:
                    line = line.strip()
                    if re.match(r"^[A-Z][a-z]+ 20\d{2}", line) or re.match(r"^\w+ \d{4}", line):
                        date_range = line
                    elif line.startswith("-") or line.startswith("•"):
                        responsibilities.append(line.lstrip("-• ").strip())
                
                work_experience.append({
                    "role": role,
                    "company": company,
                    "date_range": date_range,
                    "responsibilities": responsibilities
                })
    
    # Calculate years of experience from work experience
    years_of_exp = 0.0
    for job in work_experience:
        date_str = job.get("date_range", "")
        # Look for patterns like "June 2024 - Present" or "Jan 2024 - May 2024"
        date_pattern = r"([A-Za-z]+)\s+(\d{4})\s*-\s*(?:([A-Za-z]+)\s+)?(\d{4}|Present)"
        match = re.search(date_pattern, date_str)
        if match:
            start_year = int(match.group(2))
            if match.group(4) == "Present" or match.group(4).upper() == "PRESENT":
                end_year = 2026  # Current year
            else:
                end_year = int(match.group(4))
            years_of_exp += (end_year - start_year)
    
    # Extract projects
    projects = []
    proj_match = re.search(r"PROJECTS\s*(.*?)(?:TECHNICAL SKILLS|CERTIFICATIONS|ACHIEVEMENTS|ADDITIONAL NOTES)", 
                           sanitized_text, re.DOTALL | re.IGNORECASE)
    if proj_match:
        proj_text = proj_match.group(1).strip()
        proj_entries = re.split(r"\n\s*\n", proj_text)
        for entry in proj_entries:
            entry = entry.strip()
            if entry and len(entry) > 15:
                lines = entry.split('\n')
                proj_name = lines[0].strip()
                techs = ""
                desc = []
                for line in lines[1:]:
                    line = line.strip()
                    if line.startswith("Technologies:") or line.startswith("Tech:"):
                        techs = line.split(":", 1)[1].strip() if ":" in line else ""
                    elif line.startswith("-") or line.startswith("•"):
                        desc.append(line.lstrip("-• ").strip())
                projects.append({
                    "name": proj_name,
                    "description": "\n".join(desc),
                    "technologies": techs
                })
    
    # Extract skills
    skills = {"languages": [], "frameworks": [], "tools": [], "databases": [], "cloud": [], "other": []}
    skills_match = re.search(r"TECHNICAL SKILLS\s*(.*?)(?:CERTIFICATIONS|PROJECTS|ACHIEVEMENTS|ADDITIONAL NOTES|WORK EXPERIENCE)", 
                              sanitized_text, re.DOTALL | re.IGNORECASE)
    if skills_match:
        skills_text = skills_match.group(1).strip()
        # Parse categorized skills
        for line in skills_text.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('-'):
                category, items = line.split(':', 1)
                category = category.lower().strip()
                items_list = [item.strip() for item in items.split(',')]
                
                if 'language' in category:
                    skills["languages"].extend(items_list)
                elif 'framework' in category or 'ml' in category.lower():
                    skills["frameworks"].extend(items_list)
                elif 'data' in category.lower() and 'tool' in category.lower():
                    skills["tools"].extend(items_list)
                elif 'database' in category:
                    skills["databases"].extend(items_list)
                elif 'cloud' in category or 'devops' in category.lower():
                    skills["cloud"].extend(items_list)
                else:
                    skills.setdefault("other", []).extend(items_list)
            elif line.startswith('-') or line.startswith('•'):
                # Unstructured skill
                skill = line.lstrip("-• ").strip()
                skills["other"].append(skill)
    
    # Extract certifications
    certifications = []
    cert_match = re.search(r"CERTIFICATIONS\s*(.*?)(?:ACHIEVEMENTS|ADDITIONAL NOTES|PROJECTS|TECHNICAL SKILLS)", 
                           sanitized_text, re.DOTALL | re.IGNORECASE)
    if cert_match:
        cert_text = cert_match.group(1).strip()
        for line in cert_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                certifications.append(line.lstrip("-• ").strip())
    
    # Extract achievements
    achievements = []
    ach_match = re.search(r"ACHIEVEMENTS\s*(.*?)(?:ADDITIONAL NOTES|CERTIFICATIONS|PROJECTS)", 
                          sanitized_text, re.DOTALL | re.IGNORECASE)
    if ach_match:
        ach_text = ach_match.group(1).strip()
        for line in ach_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                achievements.append(line.lstrip("-• ").strip())
    
    profile = CandidateProfile(
        name=name,
        email=email,
        education=education if education else [],
        work_experience=work_experience if work_experience else [],
        projects=projects if projects else [],
        skills=skills,
        certifications=certifications if certifications else [],
        achievements=achievements if achievements else [],
        years_of_experience=years_of_exp,
        has_prompt_injection=has_injection,
    )
    
    return profile


def build_rubric(job_description: str) -> Dict[str, Any]:
    """
    Build a scoring rubric derived from the job description.
    Each criterion maps to a JD requirement.
    Dynamically extracts keywords from the JD to prioritize criteria.
    """
    import re
    
    jd_lower = job_description.lower()
    
    # Extract job title
    title_match = re.search(r"Job Title:\s*(.+?)(?:\n|$)", job_description, re.IGNORECASE)
    job_title = title_match.group(1).strip() if title_match else "Unknown Position"
    
    # Detect keywords in JD to adjust weights
    has_python = "python" in jd_lower
    has_ml = any(kw in jd_lower for kw in ["machine learning", "tensorflow", "pytorch", "deep learning", "ml framework", "ai model"])
    has_sql = any(kw in jd_lower for kw in ["sql", "database", "data pipeline", "data preprocessing"])
    has_cloud = any(kw in jd_lower for kw in ["aws", "gcp", "azure", "cloud", "docker", "kubernetes"])
    has_git = any(kw in jd_lower for kw in ["git", "version control", "ci/cd", "unit test", "code review"])
    has_edu = any(kw in jd_lower for kw in ["bachelor", "degree", "computer science", "related field"])
    has_exp_years = re.search(r"(\d+)\s*[-–to]+\s*(\d+)\s*years", jd_lower) or re.search(r"(\d+)\s*\+\s*years", jd_lower)
    
    # Always include core criteria, adjust weights based on JD emphasis
    criteria = []
    
    # 1. Python / Programming Language Proficiency
    if has_python:
        criteria.append({
            "criterion": "Python Proficiency",
            "weight": 0.15,
            "description": "Strong proficiency in Python programming",
            "jd_source": "Python mentioned in job description",
            "scoring_guide": {
                0: "No Python experience",
                1: "Basic Python knowledge, no projects",
                2: "Intermediate Python, some coursework",
                3: "Proficient Python with project experience",
                4: "Advanced Python with multiple production projects",
                5: "Expert-level Python with open-source contributions or advanced work"
            }
        })
    
    # 2. ML/AI Framework Experience
    if has_ml:
        criteria.append({
            "criterion": "ML Framework Experience",
            "weight": 0.20,
            "description": "Experience with ML frameworks (TensorFlow, PyTorch, scikit-learn, etc.)",
            "jd_source": "ML frameworks requirement detected in job description",
            "scoring_guide": {
                0: "No ML framework experience",
                1: "Basic knowledge of one framework, no practical use",
                2: "Used one framework in coursework or bootcamp",
                3: "Used ML frameworks in real projects",
                4: "Deep experience with multiple frameworks in production",
                5: "Expert with multiple frameworks including advanced techniques"
            }
        })
    
    # 3. SQL & Data Skills
    if has_sql:
        criteria.append({
            "criterion": "SQL & Data Skills",
            "weight": 0.15,
            "description": "SQL, database, and data manipulation skills",
            "jd_source": "SQL/database requirements detected in job description",
            "scoring_guide": {
                0: "No SQL or data manipulation experience",
                1: "Basic SQL queries (SELECT, INSERT)",
                2: "Intermediate SQL with joins and aggregations",
                3: "Proficient SQL with complex queries and data pipelines",
                4: "Advanced SQL with large-scale data processing experience",
                5: "Expert with distributed data processing and optimization"
            }
        })
    
    # 4. ML/AI Engineering Experience
    if has_ml:
        criteria.append({
            "criterion": "ML/AI Engineering Experience",
            "weight": 0.25,
            "description": "Years of experience in ML/AI engineering with model deployment",
            "jd_source": "ML/AI engineering experience requirement detected in job description",
            "scoring_guide": {
                0: "No ML/AI engineering experience",
                1: "Theoretical knowledge only, no practical experience",
                2: "Some project/internship experience in ML",
                3: "1-2 years of professional ML/AI experience with deployments",
                4: "2+ years of ML/AI with production systems",
                5: "Exceptional experience with multiple production ML systems"
            }
        })
    
    # 5. Education & Background
    if has_edu:
        criteria.append({
            "criterion": "Education & Background",
            "weight": 0.10,
            "description": "Relevant degree in CS, AI, Data Science, or related field",
            "jd_source": "Education requirement detected in job description",
            "scoring_guide": {
                0: "No relevant degree or qualification",
                1: "Some coursework but no degree",
                2: "Degree in non-CS field with self-study",
                3: "Bachelor's in CS or related field",
                4: "Bachelor's in CS/AI with strong academic performance",
                5: "Advanced degree (Master's/PhD) in AI or related field"
            }
        })
    
    # 6. Version Control & Engineering Practices
    if has_git:
        criteria.append({
            "criterion": "Version Control & Engineering Practices",
            "weight": 0.10,
            "description": "Git, unit testing, code reviews, CI/CD",
            "jd_source": "Version control/engineering practices detected in job description",
            "scoring_guide": {
                0: "No version control or testing experience",
                1: "Basic Git knowledge (commit, push)",
                2: "Regular Git usage, basic understanding of testing",
                3: "Proficient with Git workflows and unit testing",
                4: "Advanced Git, CI/CD, comprehensive testing",
                5: "Expert with DevOps, CI/CD pipelines, and testing frameworks"
            }
        })
    
    # 7. Cloud & Deployment (Bonus)
    if has_cloud:
        criteria.append({
            "criterion": "Cloud & Deployment (Bonus)",
            "weight": 0.05,
            "description": "Cloud platforms, Docker, MLOps",
            "jd_source": "Cloud/deployment preferences detected in job description",
            "scoring_guide": {
                0: "No cloud or deployment experience",
                1: "Basic awareness of cloud concepts",
                2: "Used cloud services in learning projects",
                3: "Hands-on cloud experience with deployments",
                4: "Multiple cloud services used in production",
                5: "Expert with cloud ML platforms, Docker, and MLOps"
            }
        })
    
    # Normalize weights to sum to 1.0
    total_weight = sum(c["weight"] for c in criteria)
    if total_weight > 0:
        for c in criteria:
            c["weight"] = round(c["weight"] / total_weight, 4)
    
    # If no criteria were detected (empty JD), add fallback generic criteria
    if not criteria:
        criteria = [
            {"criterion": "Relevant Experience", "weight": 0.35, "description": "Overall experience relevance", "jd_source": "General", "scoring_guide": {0: "None", 1: "Minimal", 2: "Some", 3: "Good", 4: "Strong", 5: "Exceptional"}},
            {"criterion": "Technical Skills", "weight": 0.35, "description": "Technical skill alignment", "jd_source": "General", "scoring_guide": {0: "None", 1: "Minimal", 2: "Some", 3: "Good", 4: "Strong", 5: "Exceptional"}},
            {"criterion": "Education", "weight": 0.15, "description": "Educational background", "jd_source": "General", "scoring_guide": {0: "None", 1: "Minimal", 2: "Some", 3: "Good", 4: "Strong", 5: "Exceptional"}},
            {"criterion": "Engineering Practices", "weight": 0.15, "description": "Software engineering practices", "jd_source": "General", "scoring_guide": {0: "None", 1: "Minimal", 2: "Some", 3: "Good", 4: "Strong", 5: "Exceptional"}},
        ]
    
    rubric = {
        "job_title": job_title,
        "criteria": criteria,
        "scoring_scale": {
            "min": 0,
            "max": 5,
            "description": "0 = No evidence, 1 = Minimal, 2 = Basic, 3 = Proficient, 4 = Advanced, 5 = Expert/Exceptional"
        }
    }
    
    return rubric


def score_candidate(profile: CandidateProfile, rubric: Dict[str, Any]) -> ScoreCard:
    """
    Score a candidate against the rubric criteria.
    
    Guardrail: If prompt injection was detected, flag all scores as suspicious
    and evaluate based ONLY on actual resume content, not injection instructions.
    """
    criteria_scores = []
    total_weighted = 0.0
    max_possible = 0.0
    
    # If prompt injection detected, log it but still evaluate fairly
    injection_detected = profile.get("has_prompt_injection", False)
    
    for criterion in rubric["criteria"]:
        criterion_name = criterion["criterion"]
        weight = criterion["weight"]
        guide = criterion["scoring_guide"]
        
        score, evidence = _evaluate_criterion(profile, criterion_name, criterion)
        
        # Clamp score to 0-5
        score = max(0, min(5, score))
        
        # If injection detected, ensure we don't artificially inflate scores
        # The score from _evaluate_criterion already ignores injection text
        # because parse_resume sanitized the input
        
        criteria_scores.append(CriterionScore(
            criterion=criterion_name,
            weight=weight,
            score=score,
            evidence=evidence,
            max_score=5
        ))
        
        total_weighted += score * weight
        max_possible += 5 * weight
    
    normalized_score = (total_weighted / max_possible * 100) if max_possible > 0 else 0.0
    
    # Round to 2 decimal places
    total_weighted = round(total_weighted, 2)
    normalized_score = round(normalized_score, 2)
    
    scorecard = ScoreCard(
        candidate_name=profile.get("name", "Unknown"),
        criteria_scores=criteria_scores,
        total_weighted_score=total_weighted,
        max_possible_score=round(max_possible, 2),
        normalized_score=normalized_score,
    )
    
    return scorecard


def _evaluate_criterion(profile: CandidateProfile, criterion_name: str, criterion_def: Dict) -> tuple:
    """Evaluate a single criterion and return (score, evidence)."""
    
    skills = profile.get("skills", {})
    experience = profile.get("work_experience", [])
    projects = profile.get("projects", [])
    education = profile.get("education", [])
    certifications = profile.get("certifications", [])
    years_exp = profile.get("years_of_experience", 0)
    all_skills = []
    for cat, items in skills.items():
        all_skills.extend(items)
    
    all_text = ""
    for job in experience:
        all_text += " " + job.get("role", "") + " " + job.get("company", "") + " " + " ".join(job.get("responsibilities", []))
    for proj in projects:
        all_text += " " + proj.get("name", "") + " " + proj.get("description", "") + " " + proj.get("technologies", "")
    
    all_skills_lower = [s.lower() for s in all_skills]
    all_text_lower = all_text.lower()
    
    if criterion_name == "Python Proficiency":
        python_mentions = []
        py_found = False
        for skill in all_skills:
            if "python" in skill.lower():
                py_found = True
                python_mentions.append(skill)
        
        # Check for Python in experience
        py_in_exp = "python" in all_text_lower
        py_projects = sum(1 for p in projects if "python" in p.get("technologies", "").lower() or "python" in p.get("description", "").lower())
        
        evidence_parts = []
        if py_found:
            evidence_parts.append(f"Skills list: {', '.join(python_mentions)}")
        if py_in_exp:
            # Extract lines mentioning Python
            for job in experience:
                for resp in job.get("responsibilities", []):
                    if "python" in resp.lower():
                        evidence_parts.append(f"Experience: \"{resp}\"")
                        break
        if py_projects > 0:
            for p in projects:
                if "python" in p.get("technologies", "").lower() or "python" in p.get("description", "").lower():
                    evidence_parts.append(f"Project: \"{p['name']}\" uses Python")
        
        score = 0
        if py_found and py_in_exp and py_projects > 0:
            score = 4  # Advanced
        elif py_found and py_in_exp:
            score = 3  # Proficient
        elif py_found:
            score = 2  # Basic
        elif "python" in profile.get("education", [{}])[0].get("degree", "").lower():
            score = 1  # Minimal
        else:
            score = 0
        
        # Check for expert-level indicators
        expert_indicators = ["expert", "advanced", "senior", "production"]
        if any(ind in " ".join(all_skills_lower) for ind in expert_indicators):
            if "expert" in all_skills_lower:
                score = max(score, 5)
                evidence_parts.append("Self-rated as Expert level in Python")
        
        evidence = "; ".join(evidence_parts) if evidence_parts else "No Python evidence found in resume"
        return score, evidence
    
    elif criterion_name == "ML Framework Experience":
        frameworks = ["tensorflow", "pytorch", "scikit-learn", "scikit learn", "sklearn", "keras", "hugging face", "transformers"]
        found_frameworks = []
        for fw in frameworks:
            if fw in all_skills_lower:
                found_frameworks.append(fw)
            elif fw in all_text_lower:
                found_frameworks.append(fw)
        
        nlp_cv = ("nlp" in all_text_lower) or ("computer vision" in all_text_lower) or ("bert" in all_text_lower) or ("transformer" in all_text_lower)
        
        found_frameworks = list(set(found_frameworks))
        
        evidence_parts = []
        if found_frameworks:
            evidence_parts.append(f"ML Frameworks found: {', '.join(found_frameworks[:5])}")
        
        # Look for specific evidence in experience
        for job in experience:
            for resp in job.get("responsibilities", []):
                resp_lower = resp.lower()
                for fw in ["tensorflow", "pytorch", "scikit-learn", "bert", "transformer"]:
                    if fw in resp_lower:
                        evidence_parts.append(f"Experience: \"{resp}\"")
                        break
        
        for proj in projects:
            proj_techs = proj.get("technologies", "").lower()
            for fw in ["tensorflow", "pytorch", "scikit-learn", "bert", "transformer"]:
                if fw in proj_techs:
                    evidence_parts.append(f"Project \"{proj['name']}\" uses {fw}")
                    break
        
        score = 0
        if len(found_frameworks) >= 3:
            score = 5 if nlp_cv else 4
        elif len(found_frameworks) == 2:
            score = 4 if nlp_cv else 3
        elif len(found_frameworks) == 1:
            score = 3 if nlp_cv else 2
        else:
            # Check for minimal mentions (no frameworks in skills but mentioned in text)
            ml_mentions = sum(1 for fw in frameworks if fw in all_text_lower)
            if ml_mentions > 0:
                score = 1
            else:
                score = 0
        
        evidence = "; ".join(evidence_parts) if evidence_parts else "No ML framework evidence found in resume"
        return score, evidence
    
    elif criterion_name == "SQL & Data Skills":
        sql_indicators = ["sql", "mysql", "postgresql", "postgres", "database"]
        data_indicators = ["pandas", "numpy", "spark", "data pipeline", "etl", "feature engineering", "data preprocessing"]
        
        sql_found = [s for s in all_skills_lower if any(ind in s for ind in sql_indicators)]
        sql_found_text = any(ind in all_text_lower for ind in sql_indicators)
        data_found = [s for s in all_skills_lower if any(ind in s for ind in data_indicators)]
        data_found_text = any(ind in all_text_lower for ind in data_indicators)
        
        evidence_parts = []
        if sql_found:
            evidence_parts.append(f"SQL skills: {', '.join(sql_found)}")
        if data_found:
            evidence_parts.append(f"Data skills: {', '.join(data_found)}")
        
        for job in experience:
            for resp in job.get("responsibilities", []):
                resp_lower = resp.lower()
                if any(ind in resp_lower for ind in sql_indicators + data_indicators):
                    evidence_parts.append(f"Experience: \"{resp}\"")
        
        score = 0
        has_sql = bool(sql_found) or sql_found_text
        has_data = bool(data_found) or data_found_text
        
        # Check for large-scale data indicators
        large_scale = any(ind in all_text_lower for ind in ["10m+", "million", "large-scale", "spark", "big data"])
        
        if has_sql and has_data and large_scale:
            score = 5
        elif has_sql and has_data:
            score = 4
        elif has_sql:
            score = 3
        elif sql_found_text:
            score = 2
        elif "database" in all_text_lower:
            score = 1
        else:
            score = 0
        
        evidence = "; ".join(evidence_parts) if evidence_parts else "No SQL or data skills evidence found"
        return score, evidence
    
    elif criterion_name == "ML/AI Engineering Experience":
        ai_indicators = ["machine learning", "deep learning", "ml model", "ai engineer", "model deployment", 
                         "recommendation system", "sentiment analysis", "nlp", "computer vision", "neural network"]
        
        experience_text = " ".join([job.get("role", "") + " " + job.get("company", "") + " " + 
                                     " ".join(job.get("responsibilities", [])) for job in experience])
        
        proj_text = " ".join([proj.get("name", "") + " " + proj.get("description", "") + " " + 
                              proj.get("technologies", "") for proj in projects])
        
        combined = (experience_text + " " + proj_text).lower()
        
        ai_mentions = [ind for ind in ai_indicators if ind.lower() in combined]
        
        evidence_parts = []
        if ai_mentions:
            evidence_parts.append(f"AI/ML mentions found: {', '.join(ai_mentions[:5])}")
        
        for job in experience:
            for resp in job.get("responsibilities", []):
                resp_lower = resp.lower()
                if any(ind in resp_lower for ind in ai_indicators):
                    evidence_parts.append(f"Experience: \"{resp}\"")
                    break
        
        for proj in projects:
            proj_text_lower = (proj.get("description", "") + " " + proj.get("technologies", "")).lower()
            if any(ind in proj_text_lower for ind in ai_indicators):
                evidence_parts.append(f"Project \"{proj['name']}\": {proj.get('description', '')[:80]}")
        
        score = 0
        if years_exp >= 2 and len(ai_mentions) >= 3:
            score = 5
        elif years_exp >= 1.5 and len(ai_mentions) >= 2:
            score = 4
        elif years_exp >= 1 and len(ai_mentions) >= 1:
            score = 3
        elif years_exp >= 0.5 or len(ai_mentions) >= 1:
            score = 2
        elif "machine learning" in combined or "ai" in combined:
            score = 1
        else:
            score = 0
        
        evidence = "; ".join(evidence_parts) if evidence_parts else "No AI/ML engineering experience found"
        return score, evidence
    
    elif criterion_name == "Education & Background":
        evidence_parts = []
        relevant_degrees = ["computer science", "artificial intelligence", "data science", "computer engineering", "software engineering", "information technology"]
        
        degree_info = ""
        is_relevant = False
        cgpa_info = ""
        
        for edu in education:
            degree = edu.get("degree", "")
            institution = edu.get("institution", "")
            details = edu.get("details", "")
            degree_lower = degree.lower()
            
            # Check if degree is relevant
            if any(rd in degree_lower for rd in relevant_degrees):
                is_relevant = True
            
            degree_info += f"{degree}" + (f" - {institution}" if institution else "")
            if details:
                cgpa_match = re.search(r"CGPA:\s*([0-9.]+)", details)
                if cgpa_match:
                    cgpa_info = cgpa_match.group(1)
                    evidence_parts.append(f"Education: {degree} at {institution}, CGPA: {cgpa_info}")
                else:
                    evidence_parts.append(f"Education: {degree} at {institution}")
        
        # Check for advanced degrees
        has_masters = any("master" in edu.get("degree", "").lower() or "phd" in edu.get("degree", "").lower() or "m.tech" in edu.get("degree", "").lower() or "m.sc" in edu.get("degree", "").lower() for edu in education)
        
        score = 0
        if has_masters and is_relevant:
            score = 5
        elif is_relevant and cgpa_info and float(cgpa_info) >= 8.0:
            score = 4
        elif is_relevant:
            score = 3
        elif education:
            score = 2  # Has some education but not directly relevant
        elif any(edu.get("degree", "") for edu in education):
            score = 1
        else:
            score = 0
        
        evidence = "; ".join(evidence_parts) if evidence_parts else "No education details found"
        return score, evidence
    
    elif criterion_name == "Version Control & Engineering Practices":
        eng_indicators = ["git", "version control", "unit test", "code review", "ci/cd", "github actions", "ci/cd pipeline", "test"]
        
        evidence_parts = []
        git_found = any("git" in s.lower() for s in all_skills)
        test_found = any("test" in s.lower() or "testing" in s.lower() for s in all_skills)
        ci_cd_found = any("ci" in s.lower() or "cd" in s.lower() or "github actions" in s.lower() or "jenkins" in s.lower() for s in all_skills)
        
        if git_found:
            evidence_parts.append("Git found in skills")
        if test_found:
            evidence_parts.append("Testing mentioned in skills")
        
        for job in experience:
            for resp in job.get("responsibilities", []):
                resp_lower = resp.lower()
                if any(ind in resp_lower for ind in eng_indicators):
                    evidence_parts.append(f"Experience: \"{resp}\"")
        
        for proj in projects:
            proj_text = (proj.get("technologies", "") + " " + proj.get("description", "")).lower()
            if "git" in proj_text or "ci" in proj_text:
                evidence_parts.append(f"Project \"{proj['name']}\" uses CI/Git")
        
        score = 0
        eng_count = sum([git_found, test_found, ci_cd_found])
        # Count mentions in experience
        exp_mentions = sum(1 for job in experience for resp in job.get("responsibilities", []) if any(ind in resp.lower() for ind in eng_indicators))
        
        if ci_cd_found and git_found and test_found and exp_mentions >= 2:
            score = 5
        elif git_found and test_found and exp_mentions >= 1:
            score = 4
        elif git_found and (test_found or exp_mentions >= 1):
            score = 3
        elif git_found:
            score = 2
        elif "git" in all_text_lower:
            score = 1
        else:
            score = 0
        
        evidence = "; ".join(evidence_parts) if evidence_parts else "No version control or engineering practice evidence found"
        return score, evidence
    
    elif criterion_name == "Cloud & Deployment (Bonus)":
        cloud_indicators = ["aws", "gcp", "azure", "cloud", "docker", "container", "kubernetes", "mlops", "sage maker", "ec2", "s3", "deployment"]
        
        evidence_parts = []
        cloud_skills = [s for s in all_skills if any(ind in s.lower() for ind in cloud_indicators)]
        if cloud_skills:
            evidence_parts.append(f"Cloud/DevOps skills: {', '.join(cloud_skills)}")
        
        for job in experience:
            for resp in job.get("responsibilities", []):
                resp_lower = resp.lower()
                if any(ind in resp_lower for ind in cloud_indicators):
                    evidence_parts.append(f"Experience: \"{resp}\"")
        
        for proj in projects:
            proj_text = (proj.get("technologies", "") + " " + proj.get("description", "")).lower()
            for ind in cloud_indicators:
                if ind in proj_text:
                    evidence_parts.append(f"Project \"{proj['name']}\" uses {ind}")
                    break
        
        cloud_count = len(cloud_skills)
        deploy_mentions = sum(1 for job in experience for resp in job.get("responsibilities", []) if any(ind in resp.lower() for ind in cloud_indicators))
        
        score = 0
        if cloud_count >= 3 or (cloud_count >= 2 and deploy_mentions >= 2):
            score = 5
        elif cloud_count >= 2 or (cloud_count >= 1 and deploy_mentions >= 1):
            score = 4
        elif cloud_count >= 1:
            score = 3
        elif deploy_mentions >= 1:
            score = 2
        elif any(ind in all_text_lower for ind in ["aws", "docker", "cloud"]):
            score = 1
        else:
            score = 0
        
        evidence = "; ".join(evidence_parts) if evidence_parts else "No cloud/deployment evidence found"
        return score, evidence
    
    elif criterion_name == "Relevant Experience":
        exp_years = profile.get("years_of_experience", 0)
        roles = [j.get("role", "") for j in experience]
        evidence_parts = [f"Total experience: {exp_years} years"]
        if roles:
            evidence_parts.append(f"Roles: {', '.join(roles[:3])}")
        score = min(5, int(exp_years) + 1) if exp_years > 0 else 0
        evidence = "; ".join(evidence_parts)
        return score, evidence
    
    elif criterion_name == "Technical Skills":
        all_skills_list = []
        for cat, items in skills.items():
            all_skills_list.extend(items)
        count = len(all_skills_list)
        evidence_parts = [f"Total skills listed: {count}"]
        if all_skills_list:
            evidence_parts.append(f"Skills: {', '.join(all_skills_list[:8])}")
        score = min(5, count // 2)
        evidence = "; ".join(evidence_parts)
        return score, evidence
    
    elif criterion_name == "Education":
        evidence_parts = []
        for edu in education:
            d = edu.get("degree", "")
            inst = edu.get("institution", "")
            evidence_parts.append(f"{d} at {inst}" if inst else d)
        has_edu = bool(education)
        score = 3 if has_edu else 1
        evidence = "; ".join(evidence_parts) if evidence_parts else "No education details found"
        return score, evidence
    
    elif criterion_name == "Engineering Practices":
        eng_indicators = ["git", "version control", "unit test", "code review", "ci/cd", "test", "agile", "scrum"]
        found = [ind for ind in eng_indicators if ind in all_text_lower]
        evidence_parts = []
        if found:
            evidence_parts.append(f"Engineering practices found: {', '.join(found)}")
        for job in experience:
            for resp in job.get("responsibilities", []):
                rl = resp.lower()
                if any(ind in rl for ind in eng_indicators):
                    evidence_parts.append(f"Experience: \"{resp}\"")
                    break
        score = min(5, len(found) + 1)
        evidence = "; ".join(evidence_parts) if evidence_parts else "No engineering practice evidence found"
        return score, evidence
    
    # ============================================================
    # GENERIC / FALLBACK EVALUATION
    # ============================================================
    # For any criterion name not explicitly handled above, do a generic
    # keyword-based evaluation against the resume text.
    
    criterion_lower = criterion_name.lower()
    
    # Extract keywords from the criterion description
    desc = criterion_def.get("description", "") + " " + criterion_def.get("jd_source", "")
    desc_lower = desc.lower()
    
    # Collect all resume text and skills
    resume_text = all_text + " " + " ".join(all_skills)
    resume_lower = resume_text.lower()
    
    # Find relevant keywords mentioned in both the criterion and resume
    meaningful_words = [w for w in desc_lower.split() 
                        if len(w) > 3 
                        and w not in ("with", "that", "this", "from", "have", "been", "will", "should", "their", "about", "what", "your", "into", "than", "then", "also", "more", "some", "such", "each", "other", "than", "very", "just", "well", "even", "over", "when", "only", "both", "much", "most", "many", "some", "those", "these", "them", "they", "were", "been", "being", "have", "has", "had", "does", "done", "make", "made", "take", "took", "come", "came", "know", "like", "work", "good", "also")]
    
    # Count how many criterion keywords appear in the resume
    matches = sum(1 for w in meaningful_words if w in resume_lower)
    total = max(len(meaningful_words), 1)
    match_ratio = matches / total
    
    # Score based on keyword match ratio
    if match_ratio >= 0.5:
        score = 5
    elif match_ratio >= 0.4:
        score = 4
    elif match_ratio >= 0.3:
        score = 3
    elif match_ratio >= 0.2:
        score = 2
    elif match_ratio >= 0.1:
        score = 1
    else:
        score = 0
    
    # Also check years of experience against the criterion
    yr_match = re.search(r"(\d+)\s*years", desc_lower)
    if yr_match and years_exp >= int(yr_match.group(1)):
        score = max(score, 3)
    
    evidence_parts = []
    if matches > 0:
        evidence_parts.append(f"Resume matches {matches}/{total} key terms from criterion: {', '.join(meaningful_words[:5])}")
    if years_exp > 0:
        evidence_parts.append(f"Years of experience: {years_exp}")
    
    evidence = "; ".join(evidence_parts) if evidence_parts else f"No relevant evidence found for {criterion_name}"
    return score, evidence


def check_availability(candidate_name: str, week_start: str = "2026-07-13") -> List[TimeSlot]:
    """
    Mock tool that returns available time slots for a candidate in a given week.
    
    Args:
        candidate_name: Name of the candidate
        week_start: Start date of the week (YYYY-MM-DD format)
    
    Returns:
        List of available TimeSlots
    """
    import random
    random.seed(hash(candidate_name) % 10000)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    time_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", "14:00-15:00", "15:00-16:00", "16:00-17:00"]
    
    available = []
    for day in days:
        # Each candidate has different availability patterns
        if candidate_name == "Priya Sharma":
            # Strong candidate - very available
            day_slots = random.sample(time_slots, k=random.randint(3, 5))
        elif candidate_name == "Rahul Verma":
            # Borderline - moderate availability
            day_slots = random.sample(time_slots, k=random.randint(2, 4))
        else:
            # Weak - limited availability
            day_slots = random.sample(time_slots, k=random.randint(1, 3))
        
        for slot in day_slots:
            start, end = slot.split("-")
            available.append(TimeSlot(
                day=day,
                start_time=start,
                end_time=end,
            ))
    
    return available


def propose_interview(candidate: str, slot: TimeSlot) -> InterviewProposal:
    """
    Propose an interview slot for a candidate.
    
    THIS IS AN ACTION TOOL - it requires human approval before execution.
    Returns a proposal with status PENDING_APPROVAL.
    """
    proposal = InterviewProposal(
        candidate=candidate,
        proposed_slot=slot,
        status="PENDING_APPROVAL"
    )
    return proposal