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
def load_model_and_train_if_missing():
    """
    Checks for the model file. If missing, it trains a new one using the CSV.
    """
    model_path = 'student_gb_model.pkl'
    data_path = 'Updated_Student_Performance.csv'
    features = ['Study Hours per Week', 'Attendance Rate', 'Previous Grades', 'Participation in Extracurricular Activities']

    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    elif os.path.exists(data_path):
        # Auto-train logic if model is missing during deployment
        df = pd.read_csv(data_path)
        X = df[features].copy()
        y = df['Passed'].map({'Yes': 1, 'No': 0}).fillna(0)

        # Preprocessing
        for col in ['Study Hours per Week', 'Attendance Rate', 'Previous Grades']:
            X[col] = X[col].fillna(X[col].median())
        X['Participation in Extracurricular Activities'] = X['Participation in Extracurricular Activities'].map({'Yes': 1, 'No': 0}).fillna(0)

        # Training
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save so we don't have to train again
        joblib.dump(model, model_path)
        return model
    return None

model = load_model_and_train_if_missing()
intents_data = load_intents()

if model is None:
    st.error("Critical Error: CSV data file missing. Cannot train model.")
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
        studying = st.text_input("Current Course", placeholder="e.g. Computer Science")
        passion = st.text_input("Your Passion", placeholder="e.g. Graphic Design")
        goals = st.text_area("Future Goals", placeholder="e.g. Software Engineer", height=100)
    with col2:
        st.markdown("##### 📈 Academic Metrics")
        study_hrs = st.number_input("Study Hours/Week", min_value=0.0, value=15.0)
        past_score = st.number_input("Past Year Score (0-100)", min_value=0.0, max_value=100.0, value=70.0)
        attendance = st.slider("Attendance Rate (%)", 0, 100, 90)
        extracurricular = st.radio("Extracurriculars?", ["Yes", "No"], horizontal=True)
    submit_btn = st.form_submit_button("Run Strategic Analysis")

# --- 3. INTEGRATED LOGIC ---
if submit_btn:
    features_list = ['Study Hours per Week', 'Attendance Rate', 'Previous Grades', 'Participation in Extracurricular Activities']
    input_df = pd.DataFrame([[
        study_hrs, attendance, past_score, 1 if extracurricular == "Yes" else 0
    ]], columns=features_list)
    
    raw_pred = model.predict(input_df)[0]
    prob_status = "Positive" if raw_pred == 1 else "Critical"

    with st.spinner("Analyzing alignment..."):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            prompt = f"""Expert Strategist: Studies: {studying}, Passion: {passion}, Goals: {goals}. ML Forecast: {prob_status}."""
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            ai_output = response.choices[0].message.content
            
            st.divider()
            st.metric("Brain Forecast", prob_status)
            st.info("### 📋 Strategic Roadmap")
            st.markdown(ai_output)
            st.session_state.current_prediction = ai_output
        except Exception as e:
            st.error(f"AI Error: {e}")

# --- 4. CHATBOX ---
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
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "You are a Study Buddy."}, *st.session_state.messages[-5:]]
            )
            buddy_reply = response.choices[0].message.content
        except:
            buddy_reply = "Service busy, try again!"
    with st.chat_message("assistant"):
        st.markdown(buddy_reply)
    st.session_state.messages.append({"role": "assistant", "content": buddy_reply})