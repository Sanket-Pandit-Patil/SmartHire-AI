import json
from pathlib import Path


def load_skills(file_path: str = "data/skills.json") -> list:
    """
    Load skills from a JSON file.

    Args:
        file_path: path to skills JSON file

    Returns:
        list of skills
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Skills file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        skills = json.load(f)

    return skills


def extract_skills(text: str, skills_list: list) -> list:
    """
    Extract matching skills from text using a predefined skills list.

    Args:
        text: cleaned text
        skills_list: list of known skills

    Returns:
        sorted list of detected skills
    """
    found_skills = set()

    for skill in skills_list:
        if skill.lower() in text:
            found_skills.add(skill)

    return sorted(found_skills)


def get_skill_analysis(resume_skills: list, jd_skills: list) -> dict:
    """
    Compare resume skills against JD skills.

    Args:
        resume_skills: skills found in resume
        jd_skills: skills found in job description

    Returns:
        dictionary with matched, missing, and extra skills
    """
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched_skills = sorted(resume_set.intersection(jd_set))
    missing_skills = sorted(jd_set - resume_set)
    extra_skills = sorted(resume_set - jd_set)

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills
    }