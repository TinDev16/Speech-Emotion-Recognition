# Speech Emotion Recognition

## Introduction
- This repository handles building and training Speech Emotion Recognition System.
- The basic idea behind this tool is to build and train/test a suited machine learning ( as well as deep learning ) algorithm that could recognize and detects human emotions from speech.
- This is useful for many industry fields such as making product recommendations, affective computing, etc.

## Requirements
- **Python 3.6+**

### Python Packages
- **librosa>=0.10.0**
- **numpy>=1.23**
- **scikit-learn>=1.2**
- **tensorflow>=2.12**
- **joblib>=1.2**
- **soundfile>=0.12.1**
- **h5py>=3.8**

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/TinDev16/Speech-Emotion-Recognition
```

2. Install lib
```bash
pip install -r requirements.txt
```

3. Run
```bash
python train.py
```

### Dataset
This repository used 4 datasets (including this repo's custom dataset) which are downloaded and formatted already in `data` folder:
- **RAVDESS** : The **R**yson **A**udio-**V**isual **D**atabase of **E**motional **S**peech and **S**ong that contains 24 actors (12 male, 12 female), vocalizing two lexically-matched statements in a neutral North American accent.
- **TESS**: **T**oronto **E**motional **S**peech **S**et that was modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966). A set of 200 target words were spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years).
- **SAVEE**
- **CREMA**

### Emotions available
There are 9 emotions available: : 
o 01: Neutral
o	02: Calm
o	03: Happy
o	04: Sad
o	05: Angry
o	06: Fear
o	07: Disgust
o	08: Surprise
