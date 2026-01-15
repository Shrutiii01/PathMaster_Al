import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import random
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1. SETTINGS & ASSET LOADING ---
st.set_page_config(page_title="PathMaster AI", layout="wide", page_icon="🤖")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_data
def load_intents():
    try:
        with open('intents.json', 'r') as f:
            return json.load(f)
    except Exception:
        return {"intents": []}

@st.cache_resource
def load_model():
    # Only loading the model file created by 'create_brain.py'
    if os.path.exists('student_gb_model.pkl'):
        return joblib.load('student_gb_model.pkl')
    return None

model = load_model()
intents_data = load_intents()

if model is None:
    st.error("Model file (student_gb_model.pkl) not found! Please run 'create_brain.py' first.")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = "No analysis performed yet."

def get_local_response(user_text):
    for intent in intents_data.get('intents', []):
        for pattern in intent.get('patterns', []):
            if pattern.lower() in user_text.lower():
                return random.choice(intent.get('responses'))
    return None

# --- 2. UI: STUDENT PROFILE HUB ---
st.title("🤖 PathMaster AI")
st.markdown("Holistic academic forecasting and career alignment strategy.")

with st.form("study_input"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🎯 Vision & Aspirations")
        studying = st.text_input("Current Course", placeholder="e.g. Computer Science, XII Boards")
        passion = st.text_input("Your Passion", placeholder="e.g. Graphic Design, AI")
        goals = st.text_area("Future Goals", placeholder="e.g. Software Engineer at Google", height=100)
    
    with col2:
        st.markdown("##### 📈 Academic Metrics")
        study_hrs = st.number_input("Study Hours/Week", min_value=0.0, value=15.0)
        past_score = st.number_input("Past Year Score (0-100)", min_value=0.0, max_value=100.0, value=70.0)
        attendance = st.slider("Attendance Rate (%)", 0, 100, 90)
        extracurricular = st.radio("Extracurriculars?", ["Yes", "No"], horizontal=True)
    
    submit_btn = st.form_submit_button("Run Strategic Analysis")

# --- 3. INTEGRATED LOGIC ---
if submit_btn:
    # Feature names must match 'create_brain.py' exactly
    features = ['Study Hours per Week', 'Attendance Rate', 'Previous Grades', 'Participation in Extracurricular Activities']
    
    input_df = pd.DataFrame([[
        study_hrs, 
        attendance, 
        past_score, 
        1 if extracurricular == "Yes" else 0
    ]], columns=features)
    
    # ML Prediction
    raw_pred = model.predict(input_df)[0]
    prob_status = "Positive" if raw_pred == 1 else "Critical"

    # AI Strategy Analysis
    with st.spinner("Analyzing alignment..."):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            prompt = f"""
            System: You are an Expert Career Strategist. 
            Profile: Studies: {studying}, Passion: {passion}, Goals: {goals}.
            Stats: {study_hrs}hrs/wk, {attendance}% attendance, {past_score}% score.
            ML Forecast: {prob_status}
            
            Task:
            1. Calculate ALIGNMENT SCORE (0-100).
            2. Provide a VERDICT and 3-step ROADMAP.
            3. Guardrail: Only discuss education/career. If irrelevant, refuse politely.
            """
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            ai_output = response.choices[0].message.content
            
            # Extract Score
            score_val = 50
            for line in ai_output.split('\n'):
                if "SCORE:" in line:
                    try: score_val = int(''.join(filter(str.isdigit, line)))
                    except: pass

            # --- 4. VISUAL RESULTS ---
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Goal Alignment", f"{score_val}%")
            m2.metric("Academic Effort", f"{study_hrs} hrs/wk")
            m3.metric("Brain Forecast", prob_status)

            st.progress(score_val / 100)
            st.info("### 📋 Strategic Roadmap")
            st.markdown(ai_output)
            st.session_state.current_prediction = ai_output

        except Exception as e:
            st.error(f"AI Error: {e}")

# --- 5. CHATBOX ---
st.divider()
st.subheader("💬 Strategic Advisory Session")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about your roadmap..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    buddy_reply = get_local_response(user_query)
    
    if not buddy_reply:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            # Shorten context to avoid token bloat
            context = f"Forecast: {st.session_state.current_prediction[:500]}..."
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": f"You are a friendly Study Buddy. Context: {context}"},
                    *st.session_state.messages[-5:] # Send last 5 messages for speed/context
                ]
            )
            buddy_reply = response.choices[0].message.content
        except Exception as e:
            buddy_reply = f"Chat Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(buddy_reply)
    st.session_state.messages.append({"role": "assistant", "content": buddy_reply})