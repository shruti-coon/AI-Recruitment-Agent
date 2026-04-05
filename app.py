import streamlit as st
import PyPDF2
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Load NLP
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI Recruitment Agent", layout="wide")

st.title("🤖 AI Recruitment Agent")
st.markdown("### Smart Resume Screening & Candidate Ranking System")

# Sidebar
st.sidebar.header("📌 Instructions")
st.sidebar.info("""
1. Upload multiple resumes  
2. Enter job description  
3. View rankings & insights  
4. Download report  
""")

# Skills list
SKILLS = [
    "python", "java", "react", "django", "machine learning",
    "data science", "sql", "html", "css", "javascript"
]

# -------- FUNCTIONS -------- #

def extract_text(file):
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    except:
        return ""

def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS if skill in text]

def tfidf_score(resume, job_desc):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, job_desc])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

def skill_score(candidate_skills, job_desc):
    job_skills = extract_skills(job_desc)
    match = set(candidate_skills) & set(job_skills)
    return len(match) / len(job_skills) if job_skills else 0

def final_score(tfidf, skill):
    return (0.6 * skill) + (0.4 * tfidf)

# -------- UI -------- #

uploaded_files = st.file_uploader(
    "📄 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True
)

job_desc = st.text_area("📝 Enter Job Description")

# -------- PROCESS -------- #

if uploaded_files and job_desc:

    results = []
    progress = st.progress(0)

    for i, file in enumerate(uploaded_files):
        text = extract_text(file)

        if not text:
            st.warning(f"⚠️ Could not read {file.name}")
            continue

        skills = extract_skills(text)
        tfidf = tfidf_score(text, job_desc)
        skill = skill_score(skills, job_desc)
        score = final_score(tfidf, skill)

        results.append({
            "Candidate": file.name,
            "Score (%)": round(score * 100, 2),
            "Skills": ", ".join(skills)
        })

        progress.progress((i + 1) / len(uploaded_files))

    if results:
        df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)

        st.success("✅ Analysis Complete!")

        # Top candidate
        st.subheader("🏆 Top Candidate")
        st.metric(label=df.iloc[0]["Candidate"], value=f"{df.iloc[0]['Score (%)']}%")

        # Ranking
        st.subheader("📊 Candidate Ranking")
        st.dataframe(df, use_container_width=True)

        # Chart
        st.subheader("📈 Score Visualization (Pie Chart)")

        fig, ax = plt.subplots()

        labels = df["Candidate"]
        sizes = df["Score (%)"]

        
        colors = [
       "#A8DADC",  
       "#457B9D", 
       "#F1FAEE",  
       "#E63946",  
       "#FFB4A2",  
       "#BDB2FF",  
       ]

        ax.pie(
        sizes,
        labels=labels,
        colors=colors[:len(labels)],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white'}
        ) 

        ax.axis('equal')

        st.pyplot(fig)    
    
        # Skill Analysis
        st.subheader("📌 Skill Analysis")
        job_skills = extract_skills(job_desc)

        for _, row in df.iterrows():
            candidate_skills = row["Skills"].split(", ") if row["Skills"] else []
            matched = set(candidate_skills) & set(job_skills)
            missing = set(job_skills) - set(candidate_skills)

            with st.expander(f"🔍 {row['Candidate']} Analysis"):
                st.write("✅ Matched Skills:", list(matched))
                st.write("❌ Missing Skills:", list(missing))

                if missing:
                    st.info(f"💡 Improve: {', '.join(missing)}")

        # -------- DOWNLOAD REPORT -------- #
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Report (CSV)",
            data=csv,
            file_name="recruitment_report.csv",
            mime='text/csv'
        )

    else:
        st.error("❌ No valid resumes processed")