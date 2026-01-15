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

# --- 1. DIRECTORY & PATH SETUP ---
# This ensures the app finds files correctly on Linux deployment servers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'student_gb_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'Updated_Student_Performance.csv')
INTENTS_PATH = os.path.join(BASE_DIR, 'intents.json')

# --- 2. ROBUST ASSET LOADING & AUTO-TRAINING ---
@st.cache_resource
def load_or_train_model():
    """
    Finds the model or trains it automatically from the CSV.
    Uses case-insensitive column matching for maximum compatibility.
    """
    target_features = [
        'Study Hours per Week', 
        'Attendance Rate', 
        'Previous Grades', 
        'Participation in Extracurricular Activities'
    ]

    # Step A: Try loading existing model
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Corrupted model file detected. Retraining... Error: {e}")
    
    # Step B: Train from CSV if model is missing or broken
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            
            # Case-insensitive column matching
            actual_cols = {col.lower().strip(): col for col in df.columns}
            synced_features = []
            
            for feat in target_features:
                if feat.lower() in actual_cols:
                    synced_features.append(actual_cols[feat.lower()])
                else:
                    st.error(f"Required column '{feat}' missing from CSV.")
                    return None

            # Prepare Data
            X = df[synced_features].copy()
            
            # Find and map 'Passed' column
            passed_col = actual_cols.get('passed')
            if not passed_col:
                st.error("Target column 'Passed' not found in CSV.")
                return None
            
            y = df[passed_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0)

            # Preprocessing
            for col in synced_features:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
            
            # Force binary for extracurriculars
            extra_col = synced_features[3]
            X[extra_col] = df[extra_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0)

            # Train the Brain
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Save for future use
            joblib.dump(model, MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Auto-training logic failed: {e}")
            return None
    
    return None

@st.cache_data
def load_intents():
    if os.path.exists(INTENTS_PATH):
        try:
            with open(INTENTS_PATH, 'r') as f:
                return json.load(f)
        except:
            return {"intents": []}
    return {"intents": []}

# --- 3. INITIALIZATION ---
st.set_page_config(page_title="PathMaster AI", layout="wide", page_icon="🤖")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = load_or_train_model()
intents_data = load_intents()

if model is None:
    st.error(f"🚨 **Critical Files Missing!** Ensure 'Updated_Student_Performance.csv' is in your GitHub repository.")
    st.info(f"App is looking at: {DATA_PATH}")
    st.stop()

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

# --- 4. UI: STUDENT PROFILE HUB ---
st.title("🤖 PathMaster AI")
st.markdown("Your strategic diagnostic center for academic and career alignment.")

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

# --- 5. LOGIC & AI FORECASTING ---
if submit_btn:
    features_list = ['Study Hours per Week', 'Attendance Rate', 'Previous Grades', 'Participation in Extracurricular Activities']
    input_df = pd.DataFrame([[
        study_hrs, attendance, past_score, 1 if extracurricular == "Yes" else 0
    ]], columns=features_list)
    
    raw_pred = model.predict(input_df)[0]
    prob_status = "Positive" if raw_pred == 1 else "Critical"

    with st.spinner("Calculating strategy..."):
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

# --- 6. CHATBOT INTERFACE ---
st.divider()
st.subheader("💬 Strategic Advisory Session")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about your roadmap or pivot options..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    buddy_reply = get_local_response(user_query)
    
    if not buddy_reply:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "You are a helpful Study Buddy Advisor."}, *st.session_state.messages[-5:]]
            )
            buddy_reply = response.choices[0].message.content
        except:
            buddy_reply = "The advisor is currently busy. Please try again in a moment."
            
    with st.chat_message("assistant"):
        st.markdown(buddy_reply)
    st.session_state.messages.append({"role": "assistant", "content": buddy_reply})