# Speech Emotion Recognition (SER)

A professional Speech Emotion Recognition project utilizing a deep learning model that combines convolutional and sequence networks: **CNN2D + BiLSTM + Attention Mechanism**. 

---

## Project Description

This project is designed to classify core human emotions through audio data (`.wav` files). The system supports recognizing **7 basic emotion labels**:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

### Architecture
The system utilizes one of the most advanced architectures for time-series signals:
1. **Mel-Spectrogram**: Used as the input audio feature (extracted using `librosa`).
2. **2D CNN (2D Convolutional Neural Network)**: Extracts spatial features from the Spectrogram.
3. **BiLSTM (Bidirectional LSTM)**: Learns temporal dependencies from the output of the CNN.
4. **Attention Mechanism**: Helps the model focus on the time frames that carry the clearest emotional signals.

### Key Techniques
- **Data Augmentation**: Adding white noise, SpecAugment (Time mask & Freq mask) to increase data diversity and avoid overfitting.
- **Stratified K-Fold Cross Validation**: Splits the dataset into multiple balanced folds to ensure the most generalized model evaluation.
- **Class Weighting**: Automatically adjusts weights to improve recognition on imbalanced data.

---

## Directory Structure

```text
Speech-Emotion-Recognition/
│
├── code/
│   ├── extract_features.py # Script to extract and transform features from audio (Creates X.npy & y.npy)
│   ├── train.py            # Trains the CNN+BiLSTM+Attention model and saves the best model
│   └── predict.py          # Script to recognize emotion from any audio file
│
├── dataset/                # (To be prepared) Directory containing audio data, organized by emotion folders
├── ser_best.keras          # Model Weights file (Pre-trained)
├── labels.npy              # Dictionary file storing emotion labels
├── f1_scores.png           # F1 Score evaluation chart of the model
├── ketqua.txt              # Detailed training results
├── Report.docx             # Project report
└── README.md               # This project documentation
```

---

## Requirements & Installation

The project requires **Python 3.8+**. To install the necessary libraries, you can run the following command:

```bash
pip install numpy librosa tensorflow scikit-learn matplotlib seaborn
```

---

## Usage Guide

### 1. Feature Extraction
The `extract_features.py` script will iterate through directories by emotion labels, process padding, apply Data Augmentation, and extract Mel-Spectrogram features into Tensors. You need to adjust `ROOT_PATH` in the code to properly point to your Dataset directory before running.
```bash
python code/extract_features.py
```
*Result:* Generates 2 datasets `X.npy` (audio features) and `y.npy` (corresponding labels) in the current directory.

### 2. Training
Use the `train.py` script to train the model based on the generated `*.npy` files. 
```bash
python code/train.py
```
*Result:* The model will go through the K-Fold process, print out the Confusion Matrix, and save the best model weights as `ser_best.keras` and labels as `labels.npy`.

### 3. Prediction / Inference
Use the `predict.py` script and pass the path of a `.wav` test audio file you want to check. 
```bash
python code/predict.py path/to/your_test_audio.wav
```
*Example Terminal Output:*
```
====================
 Emotion : happy
 Confidence : 96.85%
====================

 All emotions:
angry     : 0.15%
disgust   : 0.10%
fear      : 1.00%
happy     : 96.85%
neutral   : 0.50%
sad       : 1.20%
surprise  : 0.20%
```

---

## Evaluation
Through Stratified K-Fold Cross Validation, the model outputs detailed Precision, Recall, and F1-Score metrics for each label. The system also flexibly applies automatic adjustment mechanisms including lowering the Learning Rate (`ReduceLROnPlateau`) and early stopping (`EarlyStopping`) to prevent overfitting and minimize inefficient training time.

*Implemented as part of a scientific research / academic project.*
