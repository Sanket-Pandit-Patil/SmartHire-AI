from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """
    Calculate TF-IDF cosine similarity between resume text and job description.

    Args:
        resume_text: cleaned resume text
        jd_text: cleaned job description text

    Returns:
        similarity score as percentage
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)


def calculate_skill_match_score(resume_skills: list, jd_skills: list) -> float:
    """
    Calculate skill match percentage.

    Args:
        resume_skills: list of skills found in resume
        jd_skills: list of skills found in job description

    Returns:
        skill match score as percentage
    """
    if not jd_skills:
        return 0.0

    matched_count = len(set(resume_skills).intersection(set(jd_skills)))
    total_jd_skills = len(set(jd_skills))

    return round((matched_count / total_jd_skills) * 100, 2)


def calculate_final_score(text_score: float, skill_score: float) -> float:
    """
    Calculate final score using weighted average.

    Weights:
    - text similarity: 50%
    - skill match: 50%

    Args:
        text_score: TF-IDF similarity score
        skill_score: skill match score

    Returns:
        final combined score
    """
    final_score = (0.5 * text_score) + (0.5 * skill_score)
    return round(final_score, 2)


def get_score_label(score: float) -> str:
    """
    Return qualitative label for final score.
    """
    if score >= 80:
        return "Excellent Match"
    elif score >= 65:
        return "Good Match"
    elif score >= 45:
        return "Moderate Match"
    else:
        return "Low Match"