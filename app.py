import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import streamlit as st
import pandas as pd

from utils.parser import extract_text_from_pdf
from utils.preprocess import clean_text
from utils.skill_extractor import load_skills, extract_skills, get_skill_analysis
from utils.matcher import (
    calculate_tfidf_similarity,
    calculate_semantic_similarity,
    calculate_final_score,
    get_score_label
)
from utils.prompts import build_feedback_prompt, build_interview_prompt, build_chat_prompt
from utils.llm_engine import generate_llm_response
from utils.rag import chunk_text, create_vector_store, retrieve_context

st.set_page_config(
    page_title="SmartHire AI | Executive ATS",
    page_icon="💎",
    layout="wide"
)

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize Session State
if "candidates" not in st.session_state:
    st.session_state["candidates"] = {}
if "leaderboard_df" not in st.session_state:
    st.session_state["leaderboard_df"] = None

st.markdown('<h1 class="main-header">SmartHire AI 💎</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #94a3b8; font-size: 1.2rem; font-weight: 300; margin-bottom: 30px;">Next-Generation Enterprise Talent Acquisition & RAG Analysis</p>', unsafe_allow_html=True)


# Sidebar
st.sidebar.header("Job Configuration")
job_description = st.sidebar.text_area(
    "Paste Job Description",
    height=250,
    placeholder="Paste the full job description here..."
)

uploaded_resumes = st.sidebar.file_uploader(
    "Upload Resumes (PDF)", 
    type=["pdf"], 
    accept_multiple_files=True
)

analyze_button = st.sidebar.button("Analyze Candidates", type="primary")

if analyze_button:
    if not uploaded_resumes:
        st.sidebar.error("Please upload at least one resume.")
    elif not job_description.strip():
        st.sidebar.error("Please paste a job description.")
    else:
        st.session_state["candidates"] = {}  # reset
        
        skills_list = load_skills()
        cleaned_jd = clean_text(job_description)
        jd_skills = extract_skills(cleaned_jd, skills_list)

        leaderboard_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, resume_file in enumerate(uploaded_resumes):
            status_text.text(f"Processing candidate {i+1} of {len(uploaded_resumes)}: {resume_file.name}")
            
            try:
                resume_text = extract_text_from_pdf(resume_file)
                if not resume_text.strip():
                    continue

                cleaned_resume = clean_text(resume_text)
                resume_skills = extract_skills(cleaned_resume, skills_list)
                analysis = get_skill_analysis(resume_skills, jd_skills)

                # Skill Score
                matched_count = len(analysis["matched_skills"])
                total_jd_skills = len(set(jd_skills))
                skill_match_score = round((matched_count / total_jd_skills) * 100, 2) if total_jd_skills > 0 else 0.0

                # RAG Vector Store
                resume_chunks = chunk_text(cleaned_resume, chunk_size=600, overlap=100)
                faiss_index = create_vector_store(resume_chunks)
                
                # We do a base retrieval for general context
                retrieved_chunks = retrieve_context(cleaned_jd, faiss_index, resume_chunks, top_k=5)
                retrieved_context_str = "\n\n---\n\n".join(retrieved_chunks)

                tfidf_score = calculate_tfidf_similarity(cleaned_resume, cleaned_jd)
                semantic_score = calculate_semantic_similarity(cleaned_resume, cleaned_jd)
                final_score = calculate_final_score(tfidf_score, semantic_score, skill_match_score)
                score_label = get_score_label(final_score)

                # Generate AI Feedback
                feedback_prompt = build_feedback_prompt(
                    resume_skills, jd_skills, analysis["matched_skills"], analysis["missing_skills"],
                    final_score, tfidf_score, semantic_score, skill_match_score
                )
                feedback_response = generate_llm_response(feedback_prompt)

                interview_prompt = build_interview_prompt(
                    retrieved_context_str, job_description, analysis["missing_skills"]
                )
                interview_response = generate_llm_response(interview_prompt)

                # Save candidate data
                candidate_id = resume_file.name
                st.session_state["candidates"][candidate_id] = {
                    "name": resume_file.name,
                    "final_score": final_score,
                    "label": score_label,
                    "tfidf": tfidf_score,
                    "semantic": semantic_score,
                    "skill_score": skill_match_score,
                    "matched_skills": analysis["matched_skills"],
                    "missing_skills": analysis["missing_skills"],
                    "feedback": feedback_response,
                    "interview": interview_response,
                    "raw_text": resume_text,
                    "faiss_index": faiss_index,
                    "chunks": resume_chunks
                }

                leaderboard_data.append({
                    "Candidate": resume_file.name,
                    "Match Score": final_score,
                    "Label": score_label,
                    "Skills Match": skill_match_score,
                    "Semantic": semantic_score
                })

            except Exception as e:
                st.error(f"Error processing {resume_file.name}: {e}")

            progress_bar.progress((i + 1) / len(uploaded_resumes))

        progress_bar.empty()
        status_text.empty()
        
        if leaderboard_data:
            df = pd.DataFrame(leaderboard_data)
            df = df.sort_values(by="Match Score", ascending=False).reset_index(drop=True)
            st.session_state["leaderboard_df"] = df
            st.success("Analysis complete!")

# Leaderboard Section
if st.session_state.get("leaderboard_df") is not None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏆 Global Candidate Ranking")
    st.dataframe(
        st.session_state["leaderboard_df"].style.background_gradient(cmap='Greens', subset=['Match Score']),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔎 Deep Dive & Intel")
    
    candidate_names = list(st.session_state["candidates"].keys())
    selected_candidate = st.selectbox("Select Candidate Profile:", candidate_names)
    st.markdown('</div>', unsafe_allow_html=True)


    if selected_candidate:
        data = st.session_state["candidates"][selected_candidate]
        
        # Dashboard Overview Metrics
        st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin: 0; color: #f8fafc;">{data['name']}</h2>
                        <span style="color: #6366f1; font-weight: 600;">{data['label']} Integration Analyst</span>
                    </div>
                    <div style="text-align: right;">
                        <h1 style="margin: 0; color: #10b981; font-size: 3.5rem;">{data['final_score']}%</h1>
                        <span style="color: #94a3b8; font-size: 0.9rem;">TOTAL MATCH SCORE</span>
                    </div>
                </div>
                <div class="score-container">
                    <div class="score-card">
                        <div class="score-value">{data['skill_score']}%</div>
                        <div class="score-label">Technical Skills</div>
                    </div>
                    <div class="score-card">
                        <div class="score-value">{data['semantic']}%</div>
                        <div class="score-label">Semantic Alignment</div>
                    </div>
                    <div class="score-card">
                        <div class="score-value">{data['tfidf']}%</div>
                        <div class="score-label">Keyword Match</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

            
        tabs = st.tabs(["📊 Evaluation", "💬 Ask AI (Chat)", "📄 Raw Extraction"])
        
        with tabs[0]:
            col_match1, col_match2 = st.columns(2)
            with col_match1:
                st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown("#### ✅ Expert Skills Matched")
                if data["matched_skills"]:
                    badges = "".join([f'<span class="skill-badge skill-match">{s}</span>' for s in data["matched_skills"]])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.warning("No significant matches found.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_match2:
                st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown("#### ❌ Critical Skills Missing")
                if data["missing_skills"]:
                    badges = "".join([f'<span class="skill-badge skill-missing">{s}</span>' for s in data["missing_skills"]])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.success("Perfect alignment! No skills missing.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🧠 AI Strategic Feedback")
            st.markdown(f'<div style="color: #cbd5e1; line-height: 1.6;">{data["feedback"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🎤 Recommended Interview Track")
            st.markdown(f'<div style="color: #cbd5e1; line-height: 1.6;">{data["interview"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            
        with tabs[1]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 💬 Interactive AI Dossier")
            st.write("Inquire about specific experiences, project details, or cloud competency.")
            st.markdown('</div>', unsafe_allow_html=True)

            
            chat_key = f"chat_{selected_candidate}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = []
            
            # Display chat history
            for message in st.session_state[chat_key]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("E.g., Does this candidate have cloud experience?"):
                # Append user message
                st.session_state[chat_key].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Searching resume and thinking..."):
                        # RAG step: retrieve context based on the specific question asked
                        chat_chunks = retrieve_context(prompt, data["faiss_index"], data["chunks"], top_k=3)
                        chat_context_str = "\n".join(chat_chunks)
                        
                        llm_chat_prompt = build_chat_prompt(
                            retrieved_context=chat_context_str,
                            chat_history=st.session_state[chat_key],
                            user_question=prompt
                        )
                        
                        response = generate_llm_response(llm_chat_prompt)
                        st.markdown(response)
                        with st.expander("Show RAG Context Used"):
                            st.write(chat_context_str)
                            
                st.session_state[chat_key].append({"role": "assistant", "content": response})

        with tabs[2]:
            st.text_area("Extracted PDF Text", data["raw_text"], height=400)