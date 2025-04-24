import streamlit as st
import torch
from PIL import Image
import pandas as pd
import numpy as np
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr

def text_to_speech(text):
    from gtts import gTTS
    import tempfile
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def compute_bmi(weight_kg, height_cm):
    h = height_cm / 100.0
    return None if h <= 0 else weight_kg / (h * h)

def generate_grocery_list(ingredients_str):
    items = [i.strip() for i in ingredients_str.split(',')]
    return sorted(set(items))

# Load model & data 

model = YOLO('ingredient_model.pt')
CONF_THRESHOLD = 0.25

df = pd.read_csv('recipe_dataset.csv')
df['ingredients']   = df['ingredients'].astype(str).str.lower()
df['recipe_type']   = df['recipe_type'].astype(str).str.lower()

tfidf        = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_matrix = tfidf.fit_transform(df['ingredients'].tolist())

# Detection & Recommendation 

def detect_ingredients_yolo(img: Image.Image):
    results = model(img)
    names   = model.names
    dets    = []
    for box in results[0].boxes.data:
        cls, conf = int(box[5]), float(box[4])
        if conf >= CONF_THRESHOLD:
            dets.append((names[cls], conf * 100))
    return dets

def recommend_recipes(detected, recipe_type=None, top_n=50):
    if not detected:
        return pd.DataFrame()
    q    = " ".join(name for name, _ in detected)
    vec  = tfidf.transform([q])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    scores = pd.DataFrame({'idx': np.arange(len(sims)), 'score': sims})
    if recipe_type and recipe_type != '— Any —':
        mask   = df['recipe_type'].str.contains(recipe_type)
        scores = scores[mask.values]
    top = scores.nlargest(top_n, 'score')
    return df.loc[top['idx']].assign(similarity=top['score'].values)

# Sidebar

st.sidebar.title("🏥 Health & Shopping")

st.sidebar.header("BMI Calculator")
weight = st.sidebar.number_input("Weight (kg)", 0.0, 300.0, 70.0, 0.1)
height = st.sidebar.number_input("Height (cm)", 0.0, 250.0, 170.0, 0.1)
if st.sidebar.button("Compute BMI"):
    bmi = compute_bmi(weight, height)
    if bmi:
        st.sidebar.success(f"Your BMI: {bmi:.1f}")
        if bmi < 18.5:    st.sidebar.info("Underweight")
        elif bmi < 25:    st.sidebar.info("Normal weight")
        elif bmi < 30:    st.sidebar.info("Overweight")
        else:             st.sidebar.info("Obese")
    else:
        st.sidebar.error("Enter valid height")

# Main App 
st.title("🍲 Ingredient Detection & Recipe Recommendations 🍽️")
st.write("Choose an input method, then provide ingredients!")

# Initialize session state for input mode
if 'mode' not in st.session_state:
    st.session_state.mode = None

# Buttons for selecting mode
col_a, col_b, col_c, col_d = st.columns(4)
if col_a.button("📷 Photo"):
    st.session_state.mode = 'photo'
if col_b.button("⬆️ Upload"):
    st.session_state.mode = 'upload'
if col_c.button("✍️ Type"):
    st.session_state.mode = 'type'
if col_d.button("🎤 Speak"):
    st.session_state.mode = 'speak'

mode = st.session_state.mode

# Based on mode, show widget
if mode == 'photo':
    photo = st.camera_input("Take a photo")
    if photo:
        img = Image.open(photo).convert('RGB')

elif mode == 'upload':
    up = st.file_uploader("Choose an image", ["jpg","jpeg","png"])
    if up:
        img = Image.open(up).convert('RGB')

elif mode == 'type':
    txt = st.text_area("Enter ingredients, comma-separated")
    if txt:
        detected = [(ing.strip(),None) for ing in txt.split(",")]
        last_detected = detected

elif mode == 'speak':
    recognizer, mic = sr.Recognizer(), sr.Microphone()
    st.write("Click to speak your ingredients")
    if st.button("Start Listening"):
        with mic as src:
            recognizer.adjust_for_ambient_noise(src)
            audio = recognizer.listen(src)
        try:
            spoken = recognizer.recognize_google(audio)
            st.write("Heard:", spoken)
            detected = [(ing.strip(),None) for ing in spoken.split(",")]
            last_detected = detected
        except:
            st.write("Couldn't recognize speech.")
            last_detected = []

else:
    st.info("Select an input method above.")

# Image-based detection 
if 'img' in locals():
    st.image(img, use_container_width=True)
    st.write("Detecting ingredients…")
    detected = detect_ingredients_yolo(img)
    if detected:
        for name,conf in detected:
            st.write(f"• {name} — {conf:.1f}%")
        last_detected = detected
    else:
        st.write("No ingredients found.")
        last_detected = []

# Detection
if 'img' in locals():
    st.image(img, use_container_width=True)
    st.write("Detecting ingredients…")
    detected = detect_ingredients_yolo(img)
    if detected:
        for name, conf in detected:
            st.write(f"• {name} — {conf:.1f}%")
        last_detected = detected
    else:
        st.write("No ingredients found.")
        last_detected = []

# Recommendations
recs = pd.DataFrame()
if 'last_detected' in locals() and last_detected:
    types = ['— Any —'] + sorted(df['recipe_type'].unique().tolist())
    types = [t.title() for t in types]  
    rtype = st.selectbox("Filter by recipe type", types)
    recs  = recommend_recipes(last_detected, rtype)
    recs  = recs.drop_duplicates(subset=['title'])

    st.write("### Recommended Recipes")
    shown = set()
    for _, r in recs.iterrows():
        title       = r['title']
        if title in shown:
            continue
        shown.add(title)

        prep_time   = r.get('prep', '')
        servings    = r.get('servings', '')
        total_time  = r.get('total', '')
        rtype_text  = r.get('recipe_type', '')
        img_url     = r['image'] if pd.notna(r['image']) else "https://via.placeholder.com/200"
        ingredients = r.get('ingredients', '')
        directions  = r.get('directions', '')
        link        = r.get('url', '')

        col1, col2 = st.columns([3, 1])
        with col2:
            st.image(img_url, caption=title, use_container_width=True)
        with col1:
            st.subheader(title)
            st.write(f"🕒 Prep Time: {prep_time}")
            st.write(f"⏳ Total Time: {total_time}")
            st.write(f"🍽  Servings: {servings}")
            st.write(f"📖 {rtype_text.title()}")
            st.markdown(f"""
                <a href="{link}" target="_blank"
                   style="
                     display:block; width:180px; text-align:center;
                     background:#007bff; color:#fff; padding:8px;
                     border-radius:4px; text-decoration:none; font-weight:bold;
                     transition: .3s;">
                  View Recipe
                </a>
                <style>a:hover{{background:rgba(0,87,179,0.8);}}</style>
            """, unsafe_allow_html=True)

            if st.button(f"🔊 Listen to {title} ingredients", key=f"ing-{title}"):
                st.audio(text_to_speech(ingredients), format='audio/mp3')
            if st.button(f"🔊 Listen to {title} instructions", key=f"dir-{title}"):
                st.audio(text_to_speech(directions), format='audio/mp3')
        st.write("---")

# Sidebar: Grocery List Generator
if not recs.empty:
    st.sidebar.header("🛒 Grocery List Generator")
    opts = ["— none —"] + recs['title'].tolist()
    chosen = st.sidebar.selectbox("Select a recommended recipe", opts)
    if st.sidebar.button("Generate Grocery List"):
        if chosen != "— none —":
            row = recs[recs['title'] == chosen].iloc[0]
            for item in generate_grocery_list(row['ingredients']):
                st.sidebar.write(f"- {item}")
        else:
            st.sidebar.warning("Please select a recipe first.")

