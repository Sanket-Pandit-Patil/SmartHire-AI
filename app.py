import streamlit as st

from utils.parser import extract_text_from_pdf
from utils.preprocess import clean_text
from utils.skill_extractor import load_skills, extract_skills, get_skill_analysis


st.set_page_config(
    page_title="SmartHire AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 SmartHire AI")
st.subheader("Phase 2: Resume Parsing + Skill Extraction")

st.write(
    "Upload a resume PDF and paste a job description. "
    "This phase extracts text, cleans it, and compares skills."
)

# Sidebar
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload your resume in PDF format.")
st.sidebar.write("2. Paste the target job description.")
st.sidebar.write("3. Click analyze to extract and compare skills.")

# Inputs
uploaded_resume = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
job_description = st.text_area(
    "Paste Job Description",
    height=250,
    placeholder="Paste the full job description here..."
)

if uploaded_resume is not None:
    st.success(f"Uploaded file: {uploaded_resume.name}")

analyze_button = st.button("Analyze Resume")

if analyze_button:
    if uploaded_resume is None:
        st.error("Please upload a resume PDF.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        try:
            with st.spinner("Processing resume and job description..."):
                # Step 1: Extract raw resume text
                resume_text = extract_text_from_pdf(uploaded_resume)

                if not resume_text.strip():
                    st.warning("No text could be extracted from the uploaded PDF.")
                else:
                    # Step 2: Clean texts
                    cleaned_resume = clean_text(resume_text)
                    cleaned_jd = clean_text(job_description)

                    # Step 3: Load skills
                    skills_list = load_skills()

                    # Step 4: Extract skills
                    resume_skills = extract_skills(cleaned_resume, skills_list)
                    jd_skills = extract_skills(cleaned_jd, skills_list)

                    # Step 5: Compare skills
                    analysis = get_skill_analysis(resume_skills, jd_skills)

                    st.success("Analysis completed successfully!")

                    # ---------------------------------------------------
                    # Text Preview Section
                    # ---------------------------------------------------
                    st.markdown("## Extracted Content")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Resume Text")
                        st.text_area(
                            "Resume Text",
                            value=resume_text,
                            height=300
                        )

                    with col2:
                        st.markdown("### Job Description")
                        st.text_area(
                            "Job Description Text",
                            value=job_description,
                            height=300
                        )

                    # ---------------------------------------------------
                    # Skills Found Section
                    # ---------------------------------------------------
                    st.markdown("## Skills Found")

                    col3, col4 = st.columns(2)

                    with col3:
                        st.markdown("### Resume Skills")
                        if resume_skills:
                            st.write(resume_skills)
                        else:
                            st.info("No known skills found in resume.")

                    with col4:
                        st.markdown("### JD Skills")
                        if jd_skills:
                            st.write(jd_skills)
                        else:
                            st.info("No known skills found in job description.")

                    # ---------------------------------------------------
                    # Skill Comparison Section
                    # ---------------------------------------------------
                    st.markdown("## Skill Match Results")

                    col5, col6, col7 = st.columns(3)

                    with col5:
                        st.markdown("### Matched Skills")
                        if analysis["matched_skills"]:
                            st.success(", ".join(analysis["matched_skills"]))
                        else:
                            st.warning("No matched skills found.")

                    with col6:
                        st.markdown("### Missing Skills")
                        if analysis["missing_skills"]:
                            st.error(", ".join(analysis["missing_skills"]))
                        else:
                            st.success("No missing skills!")

                    with col7:
                        st.markdown("### Extra Skills")
                        if analysis["extra_skills"]:
                            st.info(", ".join(analysis["extra_skills"]))
                        else:
                            st.info("No extra skills found.")

                    # ---------------------------------------------------
                    # Metrics Section
                    # ---------------------------------------------------
                    st.markdown("## Quick Stats")

                    col8, col9, col10 = st.columns(3)

                    with col8:
                        st.metric("Resume Skills", len(resume_skills))

                    with col9:
                        st.metric("JD Skills", len(jd_skills))

                    with col10:
                        st.metric("Matched Skills", len(analysis["matched_skills"]))

        except Exception as e:
            st.error(f"Something went wrong: {e}")