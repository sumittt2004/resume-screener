# 📋 AI-Powered Resume Screener & Job Matcher

An intelligent resume screening system that automatically ranks candidates, extracts key information, analyzes skill gaps, and generates personalized interview questions using NLP and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![spaCy](https://img.shields.io/badge/spaCy-3.7+-green.svg)

## 🎯 Problem Statement

**The Challenge**: Recruiters receive 100+ resumes per job posting and spend hours manually screening them.

**The Solution**: Automate initial screening using AI to:
- Rank candidates by match percentage
- Extract skills, experience, and education automatically
- Identify skill gaps
- Generate relevant interview questions

**Time Saved**: Reduces screening time from hours to minutes!

## 📸 Screenshots

### Front
![Front Page](screenshots/Front.png)

### Result Dashboard
![Dashboard](screenshots/Result%20dashboard.png)

### Candidate Detail
![Candidate Detail](screenshots/Candidate%20Detail.png)

### Ai Interview
![Ai Interview](screenshots/Ai%20interview.png)


## ✨ Features

### Core Functionality
✅ **Semantic Matching** - Understands meaning, not just keywords (e.g., "ML Engineer" matches "Machine Learning Specialist")  
✅ **Multi-Resume Processing** - Upload and analyze multiple candidates simultaneously  
✅ **Skill Extraction** - Automatically identifies technical skills using NLP  
✅ **Contact Information** - Extracts email and phone numbers  
✅ **Education Detection** - Finds degrees and universities  
✅ **Company History** - Identifies previous employers  
✅ **Skill Gap Analysis** - Shows what skills candidates are missing  
✅ **Interview Questions** - AI-generated questions tailored to each candidate  
✅ **Visual Dashboard** - Interactive charts showing score distribution  

### Smart Scoring Algorithm
- **70% Semantic Similarity**: Uses sentence embeddings to compare job description with resume meaning
- **30% Keyword Overlap**: Ensures important keywords are present
- **Final Score**: 0-100% match percentage

## 🏗️ Architecture

```
┌─────────────────┐
│ Job Description │
│   + Resumes     │
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│  PDF Text Extractor│
│     (PyPDF)        │
└────────┬───────────┘
         │
         ▼
┌────────────────────────┐
│   NLP Processing       │
│   (spaCy + Regex)      │
│ • Skill Extraction     │
│ • Entity Recognition   │
│ • Contact Extraction   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Embedding Generation   │
│ (SentenceTransformers) │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Similarity Scoring    │
│  (Cosine Similarity)   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│   Ranking & Analysis   │
│ • Match Scores         │
│ • Skill Gaps           │
│ • Recommendations      │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Interview Questions   │
│    (Groq LLM)          │
└────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API Key (free at [console.groq.com](https://console.groq.com/))

### Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/resume-screener.git
   cd resume-screener
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_md
   ```

5. **Set up environment variables**
   
   Create `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. **Open browser** at `http://localhost:8501`

## 💡 Usage

### Step 1: Prepare Job Description
Paste the job posting including:
- Required skills
- Qualifications
- Responsibilities
- Nice-to-have skills

### Step 2: Upload Resumes
- Click "Browse files"
- Select multiple PDF resumes
- Wait for upload confirmation

### Step 3: Analyze
- Click "Analyze Candidates"
- Wait 10-30 seconds (depending on number of resumes)

### Step 4: Review Results
- See overall metrics (average score, qualified candidates)
- View score distribution chart
- Expand each candidate for detailed analysis
- Generate interview questions for top candidates

### Example Output

```
Candidate Rankings:
#1 - john_smith.pdf - 87.5% 🟢 Excellent Match
    ✅ Matching Skills: Python, Machine Learning, TensorFlow, AWS
    ❌ Missing Skills: Kubernetes, Go
    📧 john.smith@email.com | 📱 555-1234
    
#2 - jane_doe.pdf - 76.2% 🟡 Good Match
    ✅ Matching Skills: Python, Data Science, SQL
    ❌ Missing Skills: Deep Learning, Docker, AWS
    📧 jane.doe@email.com | 📱 555-5678
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **NLP** | spaCy | Named Entity Recognition (skills, companies, dates) |
| **Embeddings** | Sentence-Transformers | Convert text to semantic vectors |
| **Similarity** | Scikit-learn | Cosine similarity for matching |
| **PDF Parsing** | PyPDF | Extract text from resume PDFs |
| **LLM** | Groq (Llama 3.3) | Generate interview questions |
| **Visualization** | Plotly | Interactive charts |
| **UI** | Streamlit | Web interface |
| **Data** | Pandas | Data manipulation |

## 🧠 How It Works

### 1. Text Extraction
- Loads PDF resumes using PyPDF
- Extracts text from all pages
- Cleans and normalizes text

### 2. Skill Extraction
```python
# Predefined skill dictionary
tech_skills = ['python', 'java', 'machine learning', 'aws', ...]

# NLP entity recognition
entities = nlp(text)
organizations = [e for e in entities if e.label_ == 'ORG']
```

### 3. Semantic Matching
```python
# Convert to embeddings
job_vector = embedder.encode(job_description)
resume_vector = embedder.encode(resume_text)

# Calculate similarity
score = cosine_similarity(job_vector, resume_vector)
```

### 4. Scoring Formula
```
Final Score = (Semantic Similarity × 0.7) + (Keyword Overlap × 0.3)
```

**Why this formula?**
- Semantic (70%): Understands meaning and context
- Keywords (30%): Ensures important terms are present

### 5. Skill Gap Analysis
```python
missing_skills = job_required_skills - candidate_skills
matching_skills = job_required_skills ∩ candidate_skills
```

## 📊 Scoring Interpretation

| Score | Rating | Recommendation |
|-------|--------|----------------|
| 80-100% | 🟢 Excellent | Definitely interview |
| 70-79% | 🟡 Good | Strong candidate, interview |
| 60-69% | 🟠 Moderate | Consider if skills can be trained |
| <60% | 🔴 Weak | Likely not a good fit |

## 🎓 Key Concepts Demonstrated

- **Named Entity Recognition (NER)**: Extracting structured info from unstructured text
- **Semantic Search**: Finding similar documents by meaning
- **Vector Embeddings**: Numerical representation of text
- **Cosine Similarity**: Measuring document similarity
- **Natural Language Processing**: Understanding human language
- **PDF Processing**: Extracting text from documents
- **Data Visualization**: Creating interactive charts

## 📈 Future Enhancements

- [ ] Support for Word documents (.docx)
- [ ] LinkedIn profile integration
- [ ] Bulk email invitations to top candidates
- [ ] Custom skill taxonomy per industry
- [ ] Diversity & inclusion metrics
- [ ] ATS (Applicant Tracking System) integration
- [ ] Video interview scheduling
- [ ] Candidate database with search
- [ ] Multi-language support
- [ ] Export results to Excel/PDF

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Add `GROQ_API_KEY` in Secrets
5. Deploy!

### Docker (Coming Soon)
```bash
docker build -t resume-screener .
docker run -p 8501:8501 resume-screener
```

## 📝 Project Structure

```
resume-screener/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (not committed)
├── .gitignore            # Git ignore rules
├── README.md             # Documentation
├── resumes/              # Sample resumes (optional)
└── venv/                 # Virtual environment (not committed)
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

<<<<<<< HEAD

=======
>>>>>>> 8963eeebbf40158f33569cedb9af92e474f23560

## 🙏 Acknowledgments

- [spaCy](https://spacy.io/) for industrial-strength NLP
- [Sentence-Transformers](https://www.sbert.net/) for state-of-the-art embeddings
- [Groq](https://groq.com/) for fast LLM inference
- [Streamlit](https://streamlit.io/) for rapid prototyping

## 👤 Author

**Sumit Mishra**
- GitHub: [@sumittt2004](https://github.com/sumittt2004)
- LinkedIn: [Sumit Mishr](https://www.linkedin.com/in/mishra-sumit-/)

---

⭐ **Star this repo if it helped you!**

