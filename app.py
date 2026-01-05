import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load environment first
load_dotenv()

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import re
from langchain_groq import ChatGroq

# Page config
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📋",
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""

# Load models (cached for performance)
@st.cache_resource
def load_models():
    """Load spaCy and SentenceTransformer models"""
    try:
        nlp = spacy.load("en_core_web_md")
    except:
        st.error("Please run: python -m spacy download en_core_web_md")
        st.stop()
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, embedder

nlp, embedder = load_models()

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        text = ""
        with open(tmp_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        os.unlink(tmp_path)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return ""

def extract_skills(text, nlp):
    """Extract skills and keywords using spaCy"""
    doc = nlp(text.lower())
    
    # Common tech skills (you can expand this list)
    tech_skills = {
        'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'mongodb',
        'aws', 'azure', 'docker', 'kubernetes', 'machine learning', 'deep learning',
        'nlp', 'computer vision', 'tensorflow', 'pytorch', 'scikit-learn',
        'git', 'agile', 'scrum', 'rest api', 'graphql', 'html', 'css',
        'c++', 'c#', 'golang', 'rust', 'ruby', 'php', 'swift', 'kotlin',
        'pandas', 'numpy', 'data analysis', 'data science', 'analytics',
        'excel', 'powerbi', 'tableau', 'spark', 'hadoop', 'etl'
    }
    
    found_skills = set()
    text_lower = text.lower()
    
    # Find tech skills
    for skill in tech_skills:
        if skill in text_lower:
            found_skills.add(skill)
    
    # Extract entities (organizations, dates, etc.)
    entities = {
        'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
        'dates': [ent.text for ent in doc.ents if ent.label_ == 'DATE'],
        'education': []
    }
    
    # Extract education keywords
    education_keywords = ['bachelor', 'master', 'phd', 'university', 'college', 'degree', 'b.tech', 'm.tech', 'mba', 'b.s.', 'm.s.']
    for keyword in education_keywords:
        if keyword in text_lower:
            entities['education'].append(keyword)
    
    return list(found_skills), entities

def extract_contact_info(text):
    """Extract email and phone from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\d{10}\b'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    
    return {
        'email': emails[0] if emails else "Not found",
        'phone': phones[0] if phones else "Not found"
    }

def calculate_match_score(job_desc, resume_text, job_embedding, resume_embedding):
    """Calculate match score between job and resume"""
    # Semantic similarity using embeddings
    semantic_score = cosine_similarity([job_embedding], [resume_embedding])[0][0]
    
    # Keyword matching
    job_words = set(job_desc.lower().split())
    resume_words = set(resume_text.lower().split())
    keyword_overlap = len(job_words & resume_words) / len(job_words) if job_words else 0
    
    # Combined score (70% semantic, 30% keyword)
    final_score = (semantic_score * 0.7 + keyword_overlap * 0.3) * 100
    
    return round(final_score, 2)

def analyze_skill_gap(job_skills, candidate_skills):
    """Find missing skills"""
    job_skills_set = set(skill.lower() for skill in job_skills)
    candidate_skills_set = set(skill.lower() for skill in candidate_skills)
    
    missing_skills = job_skills_set - candidate_skills_set
    matching_skills = job_skills_set & candidate_skills_set
    
    return list(matching_skills), list(missing_skills)

def generate_interview_questions(job_desc, candidate_skills, missing_skills):
    """Generate interview questions using LLM"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ["Please set GROQ_API_KEY to generate questions"]
    
    try:
        llm = ChatGroq(
            temperature=0.7,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=api_key
        )
        
        prompt = f"""Based on this job description and candidate profile, generate 5 relevant interview questions.

Job Description: {job_desc[:500]}

Candidate Skills: {', '.join(candidate_skills[:10])}
Missing Skills: {', '.join(missing_skills[:5])}

Generate 5 interview questions that:
1. Test the candidate's existing skills
2. Explore their potential to learn missing skills
3. Assess cultural fit

Format as a numbered list."""

        response = llm.invoke(prompt)
        return response.content
    except:
        return ["Error generating questions. Check API key."]

# Main UI
st.title("📋 AI-Powered Resume Screener & Job Matcher")
st.markdown("Upload a job description and multiple resumes to get instant rankings, skill analysis, and interview questions!")

# Two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=300,
        placeholder="Enter the job requirements, skills, and qualifications...",
        value=st.session_state.job_description
    )
    st.session_state.job_description = job_description

with col2:
    st.header("📄 Upload Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload candidate resumes (PDF)",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_resumes:
        st.success(f"✅ {len(uploaded_resumes)} resume(s) uploaded")

# Process button
if st.button("🚀 Analyze Candidates", type="primary", disabled=not (job_description and uploaded_resumes)):
    with st.spinner("Analyzing resumes... This may take a moment"):
        try:
            # Extract job skills
            job_skills, _ = extract_skills(job_description, nlp)
            job_embedding = embedder.encode(job_description)
            
            results = []
            
            # Process each resume
            for resume_file in uploaded_resumes:
                # Extract text
                resume_text = extract_text_from_pdf(resume_file)
                
                if not resume_text:
                    continue
                
                # Extract information
                candidate_skills, entities = extract_skills(resume_text, nlp)
                contact_info = extract_contact_info(resume_text)
                
                # Calculate match score
                resume_embedding = embedder.encode(resume_text)
                match_score = calculate_match_score(
                    job_description, 
                    resume_text, 
                    job_embedding, 
                    resume_embedding
                )
                
                # Analyze skill gap
                matching_skills, missing_skills = analyze_skill_gap(job_skills, candidate_skills)
                
                results.append({
                    'name': resume_file.name,
                    'score': match_score,
                    'skills': candidate_skills,
                    'matching_skills': matching_skills,
                    'missing_skills': missing_skills,
                    'organizations': entities['organizations'],
                    'education': entities['education'],
                    'contact': contact_info,
                    'resume_text': resume_text
                })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            st.session_state.results = results
            
            st.success(f"✅ Analysis complete! Processed {len(results)} candidates")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Display results
if st.session_state.results:
    st.divider()
    st.header("📊 Results Dashboard")
    
    results = st.session_state.results
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", len(results))
    with col2:
        st.metric("Avg Match Score", f"{sum(r['score'] for r in results) / len(results):.1f}%")
    with col3:
        top_score = results[0]['score'] if results else 0
        st.metric("Top Match", f"{top_score:.1f}%")
    with col4:
        qualified = sum(1 for r in results if r['score'] >= 70)
        st.metric("Qualified (>70%)", qualified)
    
    # Score distribution chart
    st.subheader("📈 Score Distribution")
    df = pd.DataFrame(results)
    fig = px.bar(
        df, 
        x='name', 
        y='score',
        title='Candidate Match Scores',
        labels={'name': 'Candidate', 'score': 'Match Score (%)'},
        color='score',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(xaxis_tickangle=0)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed candidate cards
    st.subheader("🎯 Candidate Details")
    
    for i, candidate in enumerate(results, 1):
        with st.expander(f"#{i} - {candidate['name']} - Score: {candidate['score']}%", expanded=(i==1)):
            
            # Score indicator
            score = candidate['score']
            if score >= 80:
                color = "🟢"
                status = "Excellent Match"
            elif score >= 70:
                color = "🟡"
                status = "Good Match"
            elif score >= 60:
                color = "🟠"
                status = "Moderate Match"
            else:
                color = "🔴"
                status = "Weak Match"
            
            st.markdown(f"### {color} {status} - {score}%")
            
            # Two columns for details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📧 Contact Information**")
                st.write(f"Email: {candidate['contact']['email']}")
                st.write(f"Phone: {candidate['contact']['phone']}")
                
                st.markdown("**🎓 Education**")
                if candidate['education']:
                    for edu in candidate['education'][:3]:
                        st.write(f"• {edu}")
                else:
                    st.write("Not detected")
                
                st.markdown("**🏢 Organizations**")
                if candidate['organizations']:
                    for org in candidate['organizations'][:5]:
                        st.write(f"• {org}")
                else:
                    st.write("Not detected")
            
            with col2:
                st.markdown("**✅ Matching Skills**")
                if candidate['matching_skills']:
                    st.write(", ".join(candidate['matching_skills'][:10]))
                else:
                    st.write("No direct matches")
                
                st.markdown("**❌ Missing Skills (Gaps)**")
                if candidate['missing_skills']:
                    st.write(", ".join(candidate['missing_skills'][:10]))
                else:
                    st.write("No significant gaps!")
                
                st.markdown("**💼 All Detected Skills**")
                if candidate['skills']:
                    st.write(", ".join(candidate['skills'][:15]))
                else:
                    st.write("Skills not clearly detected")
            
            # Generate interview questions
            if st.button(f"Generate Interview Questions for {candidate['name']}", key=f"btn_{i}"):
                with st.spinner("Generating questions..."):
                    questions = generate_interview_questions(
                        st.session_state.job_description,
                        candidate['skills'],
                        candidate['missing_skills']
                    )
                    st.markdown("**🎤 Suggested Interview Questions:**")
                    st.markdown(questions)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Built with spaCy 🔤 | Sentence-Transformers 🧠 | Scikit-learn 📊 | Groq ⚡ | Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)