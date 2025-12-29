# src/02_evaluate.py

import os
# CRITICAL: Set TensorFlow environment variables BEFORE importing TensorFlow/DeepFace
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import time
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError:
    pass

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from deepface import DeepFace

# Paths
TEST_METADATA = "data/test_metadata.csv"
TRAIN_DB = "data/train_db"
EMBEDDINGS_CACHE = "data/train_embeddings_cache.pkl"

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

def build_database_embeddings():
    """
    Pre-compute embeddings for all images in training database.
    This is done ONCE and cached, making evaluation much faster.
    Progress is saved incrementally so you can resume if interrupted.
    """
    # Try to load existing cache
    db_embeddings = {}
    if os.path.exists(EMBEDDINGS_CACHE):
        print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}...")
        try:
            with open(EMBEDDINGS_CACHE, 'rb') as f:
                db_embeddings = pickle.load(f)
            print(f"✅ Loaded {len(db_embeddings)} cached embeddings")
        except Exception as e:
            print(f"⚠️  Cache file corrupted, will rebuild: {e}")
            db_embeddings = {}
    
    # Get all images from training database
    db_paths = []
    for person_folder in os.listdir(TRAIN_DB):
        person_path = os.path.join(TRAIN_DB, person_folder)
        if not os.path.isdir(person_path):
            continue
        
        for img_file in os.listdir(person_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_path, img_file)
                db_paths.append((img_path, person_folder))
    
    # Filter out already processed images
    remaining_paths = [(path, person) for path, person in db_paths if path not in db_embeddings]
    
    if len(remaining_paths) == 0:
        print(f"✅ All {len(db_paths)} images already processed!")
        return db_embeddings
    
    print(f"Found {len(db_paths)} total images in training database")
    print(f"  - Already cached: {len(db_embeddings)}")
    print(f"  - Remaining to process: {len(remaining_paths)}")
    print(f"Building embeddings for remaining images (this may take a while)...")
    
    # Compute embeddings for remaining images
    for idx, (img_path, person_folder) in enumerate(remaining_paths):
        try:
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )
            
            # DeepFace.represent returns a list, get first face
            if isinstance(embedding, list) and len(embedding) > 0:
                embedding = embedding[0]['embedding']
            elif isinstance(embedding, dict):
                embedding = embedding['embedding']
            
            db_embeddings[img_path] = {
                'embedding': np.array(embedding),
                'person': person_folder
            }
            
            # Save progress every 50 images (incremental saves)
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(remaining_paths)} remaining images...")
                # Save incrementally
                os.makedirs(os.path.dirname(EMBEDDINGS_CACHE), exist_ok=True)
                with open(EMBEDDINGS_CACHE, 'wb') as f:
                    pickle.dump(db_embeddings, f)
                print(f"  💾 Progress saved ({len(db_embeddings)}/{len(db_paths)} total)")
                
        except Exception as e:
            print(f"  Skipping {img_path}: {str(e)[:60]}")
            continue
    
    # Final save
    print(f"Saving final embeddings cache to {EMBEDDINGS_CACHE}...")
    os.makedirs(os.path.dirname(EMBEDDINGS_CACHE), exist_ok=True)
    with open(EMBEDDINGS_CACHE, 'wb') as f:
        pickle.dump(db_embeddings, f)
    
    print(f"✅ Database embeddings cached ({len(db_embeddings)}/{len(db_paths)} images)")
    return db_embeddings


def find_best_match(test_embedding, db_embeddings, threshold=0.7):
    """
    Find best match in database using pre-computed embeddings.
    Much faster than DeepFace.find() which scans database each time.
    """
    if len(db_embeddings) == 0:
        return None, 1.0
    
    # Extract embeddings and paths
    db_emb_list = []
    db_paths = []
    db_persons = []
    
    for img_path, data in db_embeddings.items():
        db_emb_list.append(data['embedding'])
        db_paths.append(img_path)
        db_persons.append(data['person'])
    
    db_emb_matrix = np.array(db_emb_list)
    test_emb = np.array(test_embedding).reshape(1, -1)
    
    # Compute cosine similarities
    similarities = cosine_similarity(test_emb, db_emb_matrix)[0]
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_distance = 1 - similarities[best_idx]  # Convert similarity to distance
    
    if best_distance <= threshold:
        return db_persons[best_idx], best_distance
    else:
        return None, best_distance


def safe_recognize(img_path: str, db_embeddings: dict) -> str:
    """
    Recognize face using pre-computed database embeddings.
    """
    try:
        # Get embedding for test image
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )
        
        if isinstance(embedding, list) and len(embedding) > 0:
            embedding = embedding[0]['embedding']
        elif isinstance(embedding, dict):
            embedding = embedding['embedding']
        
        # Find best match
        pred_person, distance = find_best_match(embedding, db_embeddings, threshold=0.7)
        
        if pred_person is None:
            return "Unknown"
        
        return pred_person
        
    except Exception as e:
        print(f"[ERROR] {os.path.basename(img_path)} -> {str(e)[:80]}")
        return "Error"


def evaluate(max_samples: int = 200):
    """
    Evaluate DeepFace recognition using pre-computed embeddings.
    """
    if not os.path.exists(TEST_METADATA):
        raise FileNotFoundError(f"{TEST_METADATA} not found.")
    if not os.path.isdir(TRAIN_DB):
        raise FileNotFoundError(f"{TRAIN_DB} not found.")

    # Pre-compute database embeddings (cached after first run)
    db_embeddings = build_database_embeddings()
    
    test_df = pd.read_csv(TEST_METADATA)
    test_df = test_df.head(max_samples)
    print(f"\nEvaluating {len(test_df)} test images with {MODEL_NAME} + {DETECTOR_BACKEND}...")

    y_true = []
    y_pred = []
    start_time = time.time()
    consecutive_errors = 0
    max_consecutive_errors = 10

    for i, row in test_df.iterrows():
        img_path = row["img_path"]
        true_person = row["person"]

        pred_person = safe_recognize(img_path, db_embeddings)

        if pred_person == "Error":
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"\n⚠️  Stopping: {consecutive_errors} consecutive errors.")
                break
        else:
            consecutive_errors = 0

        y_true.append(true_person)
        y_pred.append(pred_person)

        if (i + 1) % 20 == 0 or (i + 1) == len(test_df):
            elapsed = time.time() - start_time
            acc_so_far = accuracy_score(y_true, y_pred)
            error_count = y_pred.count("Error")
            print(
                f"  {i+1}/{len(test_df)} done | "
                f"acc: {acc_so_far:.3f} | "
                f"errors: {error_count} | "
                f"{elapsed/(i+1):.2f} sec/img"
            )
            tf.keras.backend.clear_session()

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nFinal accuracy: {overall_acc:.3f}")

    os.makedirs("plots", exist_ok=True)
    pred_df = pd.DataFrame({"true": y_true, "pred": y_pred})
    pred_df.to_csv("plots/predictions_facenet.csv", index=False)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    mask_known = ~pd.Series(y_pred).isin(["Unknown", "Error"])
    y_true_known = pd.Series(y_true)[mask_known]
    y_pred_known = pd.Series(y_pred)[mask_known]

    if len(y_true_known) > 0:
        labels = sorted(list(set(y_true_known) | set(y_pred_known)))
        cm = confusion_matrix(y_true_known, y_pred_known, labels=labels)

        plt.figure(figsize=(max(6, len(labels) * 0.4), 6))
        sns.heatmap(cm, annot=False, cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix - {MODEL_NAME}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig("plots/confusion_matrix_facenet.png", dpi=300)
        plt.close()

    plt.figure(figsize=(4, 4))
    plt.bar([MODEL_NAME], [overall_acc])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Face Recognition Accuracy")
    plt.savefig("plots/accuracy_facenet.png", dpi=300)
    plt.close()

    return overall_acc


if __name__ == "__main__":
    acc = evaluate(max_samples=200)
    print(f"\n✅ Evaluation complete. Accuracy: {acc:.3f}")
