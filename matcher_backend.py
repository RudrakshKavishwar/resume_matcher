import re
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# Try loading en_core_web_sm if available
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Use blank English pipeline
    nlp = spacy.blank("en")

    # Add tagger component manually if not present
    if "tagger" not in nlp.pipe_names:
        nlp.add_pipe("tagger")
        nlp.initialize()  # ✅ This is required to prevent the E109 error

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "".join([page.extract_text() for page in reader.pages])

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^\w\s.,]", "", text)
    return text.strip()

def extract_skills(text):
    doc = nlp(text)
    return list(set([chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]))

def prepare_jobs_df(df):
    df.columns = [col.strip().lower() for col in df.columns]
    title_col = next((col for col in df.columns if 'title' in col), None)
    desc_col = next((col for col in df.columns if 'description' in col), None)
    
    if not title_col or not desc_col:
        raise ValueError("Missing Job Title or Job Description column.")
    
    df.rename(columns={title_col: 'Job Title', desc_col: 'Job Description'}, inplace=True)
    df = df[['Job Title', 'Job Description']].dropna()
    df = df[df['Job Description'].str.len() > 50]
    df.drop_duplicates(subset='Job Description', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def match_resume_to_jobs(resume_text, resume_skills, jobs_df, top_k=10):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    results = []

    for _, row in jobs_df.iterrows():
        job_title = row['Job Title']
        job_desc = row['Job Description']

        job_embedding = model.encode(job_desc, convert_to_tensor=True)
        semantic_score = util.cos_sim(resume_embedding, job_embedding).item()

        job_keywords = set([chunk.text.lower() for chunk in nlp(job_desc).noun_chunks])
        skill_score = len(set(resume_skills).intersection(job_keywords)) / len(job_keywords) if job_keywords else 0

        final_score = 0.6 * semantic_score + 0.4 * skill_score

        results.append({
            'Job Title': job_title,
            'Similarity Score': round(semantic_score, 3),
            'Skill Match %': round(skill_score * 100, 2),
            'Final Score': round(final_score, 3)
        })

    result_df = pd.DataFrame(results).sort_values(by='Final Score', ascending=False).head(top_k)
    return result_df
