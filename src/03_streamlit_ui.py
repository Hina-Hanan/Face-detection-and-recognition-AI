# src/03_streamlit_ui.py

import os
# Set TensorFlow environment variables BEFORE importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
from deepface import DeepFace
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

# Configure TensorFlow
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError:
    pass

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

TRAIN_DB = "data/train_db"
EMBEDDINGS_CACHE = "data/train_embeddings_cache.pkl"
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
THRESHOLD = 0.7

@st.cache_data
def load_embeddings():
    """Load pre-computed embeddings cache"""
    if os.path.exists(EMBEDDINGS_CACHE):
        with open(EMBEDDINGS_CACHE, 'rb') as f:
            return pickle.load(f)
    return {}

def find_best_match(test_embedding, db_embeddings, threshold=0.7):
    """Find best match using pre-computed embeddings"""
    if len(db_embeddings) == 0:
        return None, 1.0, None
    
    db_emb_list = []
    db_paths = []
    db_persons = []
    
    for img_path, data in db_embeddings.items():
        db_emb_list.append(data['embedding'])
        db_paths.append(img_path)
        db_persons.append(data['person'])
    
    db_emb_matrix = np.array(db_emb_list)
    test_emb = np.array(test_embedding).reshape(1, -1)
    
    similarities = cosine_similarity(test_emb, db_emb_matrix)[0]
    best_idx = np.argmax(similarities)
    best_distance = 1 - similarities[best_idx]
    
    if best_distance <= threshold:
        return db_persons[best_idx], best_distance, db_paths[best_idx]
    else:
        return None, best_distance, None

st.set_page_config(page_title="Face Recognition System", layout="wide")

st.title("🔍 Face Recognition System")
st.markdown("**Upload an image to recognize faces from the database**")

# Sidebar
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", ["Facenet", "ArcFace", "VGG-Face"], index=0)
detector = st.sidebar.selectbox("Detector", ["opencv", "retinaface", "mtcnn"], index=0)
threshold = st.sidebar.slider("Recognition Threshold", 0.0, 1.0, 0.7, 0.05)

# Load embeddings
db_embeddings = load_embeddings()

if len(db_embeddings) == 0:
    st.warning("⚠️ No embeddings cache found. Please run `python src/02_evaluate.py` first to build the cache.")
else:
    st.sidebar.success(f"✅ Loaded {len(db_embeddings)} cached embeddings")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Uploaded Image")
        st.image(image, caption="Your uploaded image", use_column_width=True)
    
    # Save temp file
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    with st.spinner("🔍 Recognizing face..."):
        try:
            # Get embedding for uploaded image
            embedding_result = DeepFace.represent(
                img_path=temp_path,
                model_name=model_name,
                detector_backend=detector,
                enforce_detection=False,
                align=True
            )
            
            if isinstance(embedding_result, list) and len(embedding_result) > 0:
                embedding = embedding_result[0]['embedding']
            elif isinstance(embedding_result, dict):
                embedding = embedding_result['embedding']
            else:
                st.error("❌ Could not extract face embedding")
                st.stop()
            
            # Find best match
            person_name, distance, match_path = find_best_match(embedding, db_embeddings, threshold)
            
            with col2:
                st.subheader("🎯 Recognition Result")
                
                if person_name:
                    st.success(f"✅ **Recognized: {person_name}**")
                    confidence = (1 - distance) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.metric("Distance", f"{distance:.3f}")
                    
                    if match_path and os.path.exists(match_path):
                        match_img = Image.open(match_path)
                        st.image(match_img, caption=f"Best match from database", use_column_width=True)
                else:
                    st.warning("❌ **No match found**")
                    st.info(f"Distance to closest match: {distance:.3f}")
                    st.info("The person might not be in the database or the image quality is too low.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure the image contains a clear face.")

# Show accuracy scores
st.sidebar.markdown("---")
if st.sidebar.checkbox("📊 Show Evaluation Results"):
    try:
        pred_df = pd.read_csv("plots/predictions_facenet.csv")
        
        st.sidebar.subheader("Test Set Accuracy")
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(pred_df['true'], pred_df['pred'])
        st.sidebar.metric("Overall Accuracy", f"{acc*100:.1f}%")
        st.sidebar.metric("Total Test Images", len(pred_df))
        
        # Show detailed results
        st.subheader("📊 Detailed Evaluation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Predictions Table**")
            st.dataframe(pred_df.head(20))
        
        with col2:
            # Accuracy by person
            person_acc = pred_df.groupby('true').apply(
                lambda x: accuracy_score(x['true'], x['pred'])
            ).sort_values(ascending=False)
            
            st.write("**Accuracy by Person**")
            st.dataframe(person_acc.to_frame('Accuracy').head(20))
        
        # Confusion matrix if available
        if os.path.exists("plots/confusion_matrix_facenet.png"):
            st.subheader("Confusion Matrix")
            st.image("plots/confusion_matrix_facenet.png")
    
    except FileNotFoundError:
        st.info("💡 Run `python src/02_evaluate.py` first to generate evaluation results")
    except Exception as e:
        st.error(f"Error loading results: {e}")

# Footer
st.markdown("---")
st.markdown("**💡 Tips:**")
st.markdown("- Use clear, front-facing photos for best results")
st.markdown("- Adjust the threshold slider if you get too many/few matches")
st.markdown("- The system uses pre-computed embeddings for fast recognition")