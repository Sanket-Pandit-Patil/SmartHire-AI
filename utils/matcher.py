from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.embedding_model import model

# Load embedding model once
#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """
    Calculate TF-IDF cosine similarity between resume text and job description.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)


def calculate_semantic_similarity(resume_text: str, jd_text: str) -> float:
    """
    Calculate semantic similarity using sentence embeddings.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    embeddings = model.encode([resume_text, jd_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return round(similarity * 100, 2)


def calculate_skill_match_score(resume_skills: list, jd_skills: list) -> float:
    """
    Calculate skill match percentage.
    """
    if not jd_skills:
        return 0.0

    matched_count = len(set(resume_skills).intersection(set(jd_skills)))
    total_jd_skills = len(set(jd_skills))

    return round((matched_count / total_jd_skills) * 100, 2)


def calculate_final_score(tfidf_score: float, semantic_score: float, skill_score: float) -> float:
    """
    Calculate final score using weighted average.

    Weights:
    - TF-IDF score: 30%
    - Semantic similarity: 40%
    - Skill match score: 30%
    """
    final_score = (0.3 * tfidf_score) + (0.4 * semantic_score) + (0.3 * skill_score)
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