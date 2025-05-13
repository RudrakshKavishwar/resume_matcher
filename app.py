import streamlit as st
import pandas as pd
from matcher_backend import (
    extract_text_from_pdf,
    extract_skills,
    clean_text,
    prepare_jobs_df,
    match_resume_to_jobs
)

st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("🤖 AI Resume Matcher")

with st.expander("📤 Upload Your Resume & Job Dataset"):
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    uploaded_csv = st.file_uploader("Upload Job Dataset (CSV)", type=["csv"])

if uploaded_resume and uploaded_csv:
    try:
        resume_text = clean_text(extract_text_from_pdf(uploaded_resume))
        resume_skills = extract_skills(resume_text)

        df = pd.read_csv(uploaded_csv)
        jobs_df = prepare_jobs_df(df)

        if jobs_df is not None:
            with st.spinner("Matching your resume with job descriptions..."):
                matches = match_resume_to_jobs(resume_text, resume_skills, jobs_df)
                st.success("✅ Top Matching Jobs Found!")
                st.dataframe(matches)

                csv = matches.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", csv, "job_matches.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("Please upload both a resume PDF and a job dataset CSV.")
