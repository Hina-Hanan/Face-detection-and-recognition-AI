# Face Detection and Face Recognition Project

A complete face detection and recognition system that identifies individuals from images using deep learning. This project implements a full pipeline from dataset preparation to interactive recognition through a web interface.

## 📋 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Algorithms & Methods](#algorithms-methods)
- [Performance](#performance)
- [Results](#results)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)

<a name="introduction"></a>
## 🎯 Introduction

This project implements a face detection and recognition system that:

1. **Processes** a dataset of face images organized by person
2. **Builds** a training database with data augmentation
3. **Generates** face embeddings using deep learning models
4. **Matches** test images against the database to identify people
5. **Provides** a web interface for interactive recognition

### Face Detection vs Face Recognition

- **Face Detection**: Locates faces in images using OpenCV's detector backend. Implemented via DeepFace with `detector_backend="opencv"` and `enforce_detection=False` (allows processing to continue even if no face is detected).

- **Face Recognition**: Identifies who the face belongs to. Uses the Facenet model to generate 128-dimensional embeddings, then matches them using cosine similarity against pre-computed database embeddings.

<a name="features"></a>
## ✨ Features

- ✅ Complete dataset preparation pipeline with filtering and augmentation
- ✅ Pre-computed embedding cache for fast recognition
- ✅ Batch evaluation with comprehensive metrics
- ✅ Interactive web interface using Streamlit
- ✅ Robust error handling for edge cases
- ✅ Support for multiple face recognition models
- ✅ Visualization of results (confusion matrix, accuracy charts)

<a name="technologies-used"></a>
## 🛠 Technologies Used

### Core Deep Learning & Face Recognition
- **TensorFlow** (>=2.20.0) - Backend for DeepFace models
- **tf_keras** (>=2.20.0) - Legacy Keras support in TensorFlow 2.20.0
- **DeepFace** (0.0.91) - Face recognition library providing models and detection

### Computer Vision & Image Processing
- **OpenCV** (via DeepFace) - Face detection using Haar cascades
- **Pillow** (>=10.4.0) - Image loading and manipulation

### Data Processing & Analysis
- **Pandas** (>=2.2.0) - Dataset metadata and CSV handling
- **NumPy** (>=2.1.0,<2.4.0) - Array operations for embeddings and similarity
- **scikit-learn** (>=1.5.0) - Train/test split, metrics, cosine similarity

### Visualization
- **Matplotlib** (>=3.9.0) - Plotting accuracy and confusion matrix
- **Seaborn** (>=0.13.0) - Confusion matrix heatmap visualization

### Web Interface
- **Streamlit** (>=1.38.0) - Web UI for image upload and recognition

<a name="project-structure"></a>
## 📁 Project Structure

```
face_recognition/
├── data/
│   ├── raw/                          # Original dataset (person folders)
│   │   └── {person_name}/            # Each person has a folder
│   │       └── *.jpg                 # Multiple images per person
│   ├── train_db/                     # Training database (augmented)
│   │   └── {person_name}/            # Person folders with augmented images
│   ├── test_set/                     # Test images (20% split)
│   │   └── {person_name}/            # Test images only
│   ├── train_metadata.csv            # Training set metadata
│   ├── test_metadata.csv             # Test set metadata
│   └── train_embeddings_cache.pkl    # Pre-computed embeddings cache
├── src/
│   ├── 00_filter_dataset.py          # Dataset filtering script
│   ├── 01_dataset_prep.py            # Dataset preparation & augmentation
│   ├── 02_evaluate.py                # Evaluation & embedding generation
│   └── 03_streamlit_ui.py            # Web interface
├── plots/                            # Generated evaluation results
│   ├── predictions_facenet.csv      # Prediction results
│   ├── confusion_matrix_facenet.png # Confusion matrix visualization
│   └── accuracy_facenet.png          # Accuracy bar chart
├── requirements.txt                  # Python dependencies
├── packages.txt                      # System packages
└── README.md                         # This file
```

<a name="installation"></a>
## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- System libraries (for OpenCV): `libgl1` (Linux) or equivalent

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face_recognition
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv fvenv
   # On Windows
   fvenv\Scripts\activate
   # On Linux/Mac
   source fvenv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system packages** (if needed)
   - For Linux/Streamlit Cloud: Packages listed in `packages.txt` will be installed automatically
   - For local development: Ensure OpenCV dependencies are available

<a name="usage"></a>
## 📖 Usage

### Step 1: Prepare Your Dataset

Organize your images in the following structure:
```
data/raw/
├── Person_Name_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Person_Name_2/
│   ├── image1.jpg
│   └── ...
```

### Step 2: Filter Dataset (Optional)

Filter the dataset to keep only persons with 2+ images and select a subset:
```bash
python src/00_filter_dataset.py
```

This script will:
- Keep only persons with 2+ images
- Randomly select 25% of those persons
- Delete unselected persons
- Clean up processed data

### Step 3: Prepare Dataset

Split the dataset into train/test sets and augment training data:
```bash
python src/01_dataset_prep.py
```

This creates:
- `data/train_db/` - Training images with augmentation
- `data/test_set/` - Test images (20% split)
- `data/train_metadata.csv` - Training metadata
- `data/test_metadata.csv` - Test metadata

### Step 4: Build Embeddings and Evaluate

Generate embeddings and evaluate the recognition system:
```bash
python src/02_evaluate.py
```

This will:
- Build embeddings cache for all training images
- Evaluate on test set
- Generate accuracy metrics and visualizations
- Save results to `plots/` directory

**Note**: The first run will take longer as it builds the embeddings cache. Subsequent runs will be faster.

### Step 5: Launch Web Interface

Start the interactive Streamlit web application:
```bash
streamlit run src/03_streamlit_ui.py
```

The web interface allows you to:
- Upload images for recognition
- Adjust recognition threshold
- Switch between different models and detectors
- View evaluation results

<a name="system-architecture"></a>
## 🏗 System Architecture

### Data Flow

```
Raw Data (data/raw/)
    ↓
[00_filter_dataset.py] → Filter & Select
    ↓
[01_dataset_prep.py] → Split & Augment
    ↓
Training DB (data/train_db/) + Test Set (data/test_set/)
    ↓
[02_evaluate.py] → Build Embeddings Cache
    ↓
Embeddings Cache (data/train_embeddings_cache.pkl)
    ↓
[03_streamlit_ui.py] → Interactive Recognition
```

### Component Interactions

- **00_filter_dataset.py** → Prepares raw data by filtering and selecting persons
- **01_dataset_prep.py** → Reads `data/raw/`, creates `data/train_db/` and `data/test_set/`
- **02_evaluate.py** → Reads `data/train_db/` and `data/test_metadata.csv`, generates `data/train_embeddings_cache.pkl` and `plots/`
- **03_streamlit_ui.py** → Reads `data/train_embeddings_cache.pkl` for fast recognition

<a name="algorithms-methods"></a>
## 🔬 Algorithms & Methods

### 1. Facenet Model
- **Where**: Used in `02_evaluate.py` and `03_streamlit_ui.py` via `DeepFace.represent(model_name="Facenet")`
- **Purpose**: Generates 128-dimensional face embeddings
- **Advantages**: Good accuracy, efficient embeddings
- **Limitations**: Requires aligned faces, sensitive to extreme angles/lighting

### 2. OpenCV Face Detector
- **Where**: `detector_backend="opencv"` in all DeepFace calls
- **Purpose**: Fast face detection using Haar cascades
- **Advantages**: Fast, lightweight, works well for frontal faces
- **Limitations**: Less robust to extreme poses than deep learning detectors

### 3. Cosine Similarity
- **Where**: `find_best_match()` function in both evaluation and UI
- **Purpose**: Measures similarity between face embeddings
- **Advantages**: Normalized metric, works well for high-dimensional embeddings
- **Limitations**: Assumes normalized embeddings, requires threshold tuning

### 4. Data Augmentation
- **Where**: `augment_train()` in `01_dataset_prep.py`
- **Methods**:
  - Horizontal flip
  - Rotation (+5 degrees)
  - Brightness adjustment
- **Purpose**: Increases training data diversity
- **Advantages**: Improves model robustness to variations

<a name="performance"></a>
## ⚡ Performance

### Optimization Strategies

1. **Embedding Caching**: Pre-computes all training embeddings once, stored in pickle file
2. **Incremental Saves**: Progress saved every 50 images during cache building
3. **TensorFlow Optimization**: Thread limits, memory growth, log suppression
4. **Streamlit Caching**: `@st.cache_data` decorator on embedding loading
5. **Session Clearing**: Clears TensorFlow session every 20 images in evaluation

### Time Complexity

- **Embedding Generation**: O(1) per image (deep learning inference)
- **Database Building**: O(N) where N = number of training images
- **Recognition (with cache)**: O(1) embedding + O(M) similarity where M = database size
- **Recognition (without cache)**: O(1) embedding + O(M) similarity + O(N) embedding generation

<a name="results"></a>
## 📊 Results

### Evaluation Outputs

1. **Console Output**: Progress updates, accuracy metrics, classification report
2. **CSV File**: `plots/predictions_facenet.csv` with true vs predicted labels
3. **Confusion Matrix**: `plots/confusion_matrix_facenet.png` - Heatmap visualization
4. **Accuracy Chart**: `plots/accuracy_facenet.png` - Overall accuracy bar chart

### Recognition Results Format

- **Person Name**: Matched person's folder name
- **Confidence**: `(1 - distance) * 100` percentage
- **Distance**: Cosine distance (0.0 = identical, 1.0 = completely different)
- **Threshold**: 0.7 (configurable in UI)

<a name="limitations"></a>
## ⚠️ Limitations

1. **Single Face Processing**: Only processes first detected face in multi-face images
2. **Fixed Threshold**: Hardcoded 0.7 threshold in evaluation (configurable in UI)
3. **Limited Augmentation**: Only flip, rotation, brightness (no scale, noise, etc.)
4. **OpenCV Detector**: Less robust to extreme poses than deep learning detectors
5. **No Face Alignment Visualization**: Cannot see detected face regions
6. **Sequential Processing**: No parallelization in embedding generation
7. **Memory**: All embeddings loaded into memory (may be limiting for very large databases)
8. **No Batch Processing**: Processes images one at a time
9. **Error Recovery**: Stops evaluation after 10 consecutive errors
10. **Single Model**: Only Facenet evaluated (other models available but not tested)

<a name="future-enhancements"></a>
## 🔮 Future Enhancements

1. **Multi-face Detection**: Process all faces in an image, not just the first
2. **Adaptive Threshold**: Learn optimal threshold per person or dataset
3. **Advanced Augmentation**: Scale, noise, blur, color jitter
4. **Alternative Detectors**: RetinaFace, MTCNN for better detection
5. **Batch Processing**: Process multiple images in parallel
6. **Database Management**: Add/remove persons without rebuilding entire cache
7. **Real-time Video**: Process video streams for live recognition
8. **Model Comparison**: Evaluate multiple models (ArcFace, VGG-Face) side-by-side
9. **Face Alignment Visualization**: Show detected face regions in UI
10. **Performance Metrics**: Track inference time, memory usage
11. **Database Search**: Find similar faces (not just best match)
12. **Confidence Calibration**: Better confidence score interpretation

## 📝 Key Functions

### Dataset Preparation
- `scan_and_filter_dataset()` - Scans and filters dataset to persons with 2+ images
- `split_dataset()` - Splits dataset into 80/20 train/test
- `augment_train()` - Applies data augmentation to training images

### Recognition
- `build_database_embeddings()` - Pre-computes embeddings for all training images
- `find_best_match()` - Finds best matching person using cosine similarity
- `safe_recognize()` - Recognizes face in image with error handling
- `evaluate()` - Evaluates recognition system on test set

## 🐛 Error Handling

The system handles various edge cases:

- **No Face Detected**: `enforce_detection=False` allows processing to continue
- **Multiple Faces**: Only first detected face is processed
- **Unknown Users**: Returns "Unknown" if distance > threshold
- **Image Loading Errors**: Skips problematic images, logs errors
- **Consecutive Errors**: Stops evaluation after 10 consecutive errors
- **Cache Corruption**: Automatically rebuilds cache if corrupted

## 📄 License

[Add your license information here]

## 👤 Author

[Add your name/contact information here]

## 🙏 Acknowledgments

- DeepFace library by Sefik Ilkin Serengil
- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools

---

**Note**: This project is fully implemented and working. Follow the usage steps above to run the complete pipeline from dataset preparation to interactive recognition.

