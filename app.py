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
    page_title="SmartHire AI System",
    page_icon="📄",
    layout="wide"
)

# Initialize Session State
if "candidates" not in st.session_state:
    st.session_state["candidates"] = {}
if "leaderboard_df" not in st.session_state:
    st.session_state["leaderboard_df"] = None

st.title("📄 SmartHire AI: Enterprise ATS")
st.subheader("Phase 6: Multi-Resume Ranking & Interactive Chatbot")

st.write(
    "Upload multiple resume PDFs and a target Job Description. The system will rank all candidates "
    "on a leaderboard. Select a candidate to see deeper insights and chat with the AI about their resume."
)

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

# UI Display
if st.session_state.get("leaderboard_df") is not None:
    st.markdown("---")
    st.markdown("## 🏆 Candidate Leaderboard")
    st.dataframe(
        st.session_state["leaderboard_df"].style.background_gradient(cmap='Greens', subset=['Match Score']),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("## 🔎 Deep Dive & Candidate Chat")
    
    candidate_names = list(st.session_state["candidates"].keys())
    selected_candidate = st.selectbox("Select a candidate to review:", candidate_names)

    if selected_candidate:
        data = st.session_state["candidates"][selected_candidate]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### {data['name']} - {data['final_score']}% ({data['label']})")
        with col2:
            st.metric("Skill Match", f"{data['skill_score']}%")
            
        tabs = st.tabs(["📊 Evaluation", "💬 Ask AI (Chat)", "📄 Raw Extraction"])
        
        with tabs[0]:
            st.markdown("#### Matched Skills")
            st.success(", ".join(data["matched_skills"]) if data["matched_skills"] else "None")
            st.markdown("#### Missing Skills")
            st.error(", ".join(data["missing_skills"]) if data["missing_skills"] else "None")
            
            st.markdown("#### AI Feedback")
            st.info(data["feedback"])
            
            st.markdown("#### Suggested Interview Questions")
            st.warning(data["interview"])
            
        with tabs[1]:
            st.markdown("#### Chat with the Candidate's Resume (RAG)")
            st.write("Ask anything about this candidate. The AI will search their resume chunks to answer.")
            
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