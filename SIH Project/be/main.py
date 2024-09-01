from flask import Flask, render_template, request
import spacy
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Initialize the Matcher
matcher = Matcher(nlp.vocab)

# Define patterns for extracting sections
patterns = {
    "Objective": [{"LOWER": "objective"}],
    "Education": [{"LOWER": "education"}],
    "Skills": [{"LOWER": "skills"}],
    "Experience": [{"LOWER": "experience"}],
    "Projects": [{"LOWER": "projects"}],
    "Certifications": [{"LOWER": "certifications"}],
    "Awards": [{"LOWER": "awards"}],
}

# Add patterns to Matcher
for section, pattern in patterns.items():
    matcher.add(section, [pattern])

def extract_sections(text):
    doc = nlp(text)
    matches = matcher(doc)
    
    sections = {key: "" for key in patterns.keys()}
    current_section = None
    start_pos = 0
    
    for match_id, start, end in matches:
        section_name = nlp.vocab.strings[match_id]
        if current_section:
            sections[current_section] = text[start_pos:doc[start].idx].strip()
        current_section = section_name
        start_pos = doc[start].idx
    
    if current_section:
        sections[current_section] = text[start_pos:].strip()
    
    return sections

def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1] * 100

def match_relevancy(candidate_sections, expert_sections):
    relevancy_scores = {}
    
    for section in candidate_sections.keys():
        candidate_text = candidate_sections[section]
        expert_text = expert_sections[section]
        
        if candidate_text and expert_text:
            score = calculate_similarity(candidate_text, expert_text)
            relevancy_scores[section] = score
        else:
            relevancy_scores[section] = 0.0
    
    overall_relevancy = sum(relevancy_scores.values()) / len(relevancy_scores)
    relevancy_scores["Overall Relevancy"] = overall_relevancy
    
    return relevancy_scores

@app.route('/')
def index():
    candidate_sections = {}  # Dummy data for testing
    expert_sections = {}     # Dummy data for testing
    relevancy_scores = {}    # Dummy data for testing
    
    print("Candidate Sections:", candidate_sections)
    print("Expert Sections:", expert_sections)
    print("Relevancy Scores:", relevancy_scores)
    
    return render_template('index.html', candidate_sections=candidate_sections,
                           expert_sections=expert_sections, relevancy_scores=relevancy_scores)
@app.route('/relevancy', methods=['POST'])
def relevancy():
    resume1 = request.form.get('resume1', '')
    resume2 = request.form.get('resume2', '')
    
    if resume1 and resume2:
        candidate_sections = extract_sections(resume1)
        expert_sections = extract_sections(resume2)
        
        relevancy_scores = match_relevancy(candidate_sections, expert_sections)
        
        # Debugging output
        print("Candidate Sections:", candidate_sections)
        print("Expert Sections:", expert_sections)
        print("Relevancy Scores:", relevancy_scores)
        
        return render_template('result.html', candidate_sections=candidate_sections,
                               expert_sections=expert_sections, relevancy_scores=relevancy_scores)
    else:
        return "Please provide both resumes."


if __name__ == "__main__":
    app.run(debug=True)

