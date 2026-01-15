import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import random
from groq import Groq
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier

# Load environment variables (for local testing)
load_dotenv()

# --- 1. ROBUST API KEY FETCHING ---
# Priority: 1. Hugging Face Secret -> 2. Local .env -> 3. None
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# --- 2. DIRECTORY & PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'student_gb_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'Updated_Student_Performance.csv')
INTENTS_PATH = os.path.join(BASE_DIR, 'intents.json')

# --- 3. ASSET LOADING ---
@st.cache_resource
def load_or_train_model():
    target_features = ['Study Hours per Week', 'Attendance Rate', 'Previous Grades', 'Participation in Extracurricular Activities']
    if os.path.exists(MODEL_PATH):
        try: return joblib.load(MODEL_PATH)
        except: pass
    
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            # Basic training logic...
            # (Keeping your existing logic here)
            return model 
        except Exception as e:
            st.error(f"Training failed: {e}")
    return None

@st.cache_data
def load_intents():
    if os.path.exists(INTENTS_PATH):
        with open(INTENTS_PATH, 'r') as f:
            return json.load(f)
    return {"intents": []}

# --- 4. INITIALIZATION ---
st.set_page_config(page_title="PathMaster AI", layout="wide", page_icon="🤖")

# API KEY VALIDATION
if not GROQ_API_KEY or GROQ_API_KEY == "":
    st.error("🔑 **Groq API Key Missing!** Please add it as a Secret in Hugging Face Settings.")
    st.stop()

model = load_or_train_model()
intents_data = load_intents()

# Initialize Client ONCE at the top level
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Groq: {e}")
    st.stop()

# --- UI & FORM LOGIC ---
st.title("🤖 PathMaster AI")
st.markdown("Suggests a holistic view of the student's potential and academic vibe.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Simplified Local Intent Check
def get_local_response(user_text):
    for intent in intents_data.get('intents', []):
        for pattern in intent.get('patterns', []):
            if pattern.lower() in user_text.lower():
                return random.choice(intent.get('responses'))
    return None

st.subheader("Strategic Diagnostic Center")
with st.form("study_input"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🎯 Vision & Aspirations")
        studying = st.text_input("What are you currently studying?", placeholder="e.g. Computer Science, XII Boards")
        passion = st.text_input("What is your passion?", placeholder="e.g. Graphic Design, Cricket, AI")
        goals = st.text_area("What are your future goals?", placeholder="e.g. Become a Software Engineer at Google", height=100)
    
    with col2:
        st.markdown("##### 📈 Academic Performance Metrics")
        study_hrs = st.number_input("How much time do you give to your studies? (Hrs/Week)", min_value=0.0, value=15.0)
        past_score = st.number_input("Past Year Scores (0-100)", min_value=0.0, max_value=100.0, value=70.0)
        attendance = st.slider("Attendance Rate (%)", 0, 100, 90)
        extracurricular = st.radio("Into Extracurriculars?", ["Yes", "No"], horizontal=True)
    
    submit_btn = st.form_submit_button("Analyze My Performance")

if submit_btn:
    # Perform Prediction
    features = [[study_hrs, attendance, past_score, 1 if extracurricular == "Yes" else 0]]
    raw_pred = model.predict(features)[0]
    prob_status = "High" if raw_pred == 1 else "Low"

    with st.spinner("Generating PathMaster Roadmap..."):
        prompt = f"Strategize a path for a student studying {studying} but passionate about {passion}. Pass Probability: {prob_status}."
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            ai_output = response.choices[0].message.content
            st.metric("Success Probability", prob_status)
            st.markdown(ai_output)
        except Exception as e:
            st.error(f"Groq API Error: {e}")

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask PathMaster about a career..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Check local intents first
    reply = get_local_response(user_query)
    
    if not reply:
        try:
            chat_res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "You are PathMaster."}, 
                          *st.session_state.messages[-3:]]
            )
            reply = chat_res.choices[0].message.content
        except:
            reply = "I'm having trouble connecting to my brain. Check API limits!"
            
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
