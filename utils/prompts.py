def build_feedback_prompt(
    resume_skills,
    jd_skills,
    matched_skills,
    missing_skills,
    final_score,
    tfidf_score,
    semantic_score,
    skill_match_score
):
    return f"""
You are an AI career assistant.

Analyze the candidate's resume fit for a job description using only the structured information below.

Resume Skills:
{resume_skills}

Job Description Skills:
{jd_skills}

Matched Skills:
{matched_skills}

Missing Skills:
{missing_skills}

Scores:
- TF-IDF Score: {tfidf_score}%
- Semantic Score: {semantic_score}%
- Skill Match Score: {skill_match_score}%
- Final Match Score: {final_score}%

Tasks:
1. Write a short overall evaluation of the candidate's fit.
2. List 3 strengths of the resume for this role.
3. List 3 weaknesses or gaps.
4. Suggest 5 specific improvements to increase the match.
5. Keep the answer practical, concise, and grounded only in the information above.

Return the answer in this format:

Overall Evaluation:
...

Strengths:
1. ...
2. ...
3. ...

Weaknesses:
1. ...
2. ...
3. ...

Suggestions:
1. ...
2. ...
3. ...
4. ...
5. ...
"""


def build_interview_prompt(
    retrieved_context,
    job_description,
    missing_skills
):
    return f"""
You are a technical interviewer.

Based only on the relevant resume context from the candidate and job description below, generate relevant interview questions.

Relevant Resume Context:
{retrieved_context}

Job Description:
{job_description[:3000]}

Missing Skills:
{missing_skills}

Tasks:
1. Generate 5 technical interview questions.
2. Generate 3 project-based questions.
3. Generate 3 HR/behavioral questions.
4. Generate 3 improvement-focused questions based on missing skills.

Return the answer in this format:

Technical Questions:
1. ...
2. ...
3. ...
4. ...
5. ...

Project Questions:
1. ...
2. ...
3. ...

HR Questions:
1. ...
2. ...
3. ...

Improvement Questions:
1. ...
2. ...
3. ...
"""

def build_chat_prompt(retrieved_context, chat_history, user_question):
    # Format chat history to a string
    history_str = ""
    for msg in chat_history[-5:]: # Keep last 5 messages for context limit
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
        
    return f"""
You are a helpful hiring assistant answering a recruiter's or candidate's questions about a specific resume.

Base your answer ONLY on the relevant resume context chunked below. If the answer is not in the context, say "I don't have enough information in the resume to answer that."

Relevant Resume Context:
{retrieved_context}

Recent Chat History:
{history_str}

User Question:
{user_question}
"""