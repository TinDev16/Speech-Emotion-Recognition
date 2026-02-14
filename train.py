# =====================================================
# SPEECH EMOTION RECOGNITION – MULTI DATASET (8 LABELS)
# =====================================================

import os
import numpy as np
import librosa
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# =====================
# PATHS
# =====================
ROOT = "data"
TRAINING = os.path.join(ROOT, "training")
VALIDATION = os.path.join(ROOT, "validation")
EMODB = os.path.join(ROOT, "emodb", "wav")
CUSTOM = [os.path.join(ROOT, "train-custom")]

# =====================
# AUDIO PARAMS
# =====================
SR = 22050
DURATION = 3
OFFSET = 0.5
MAX_LEN = 130

# =====================
# EMO MAP
# =====================
VALID_LABELS = [
    "angry", "calm", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

emodb_map = {
    "W": "angry",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral"
}

# =====================
# FEATURE EXTRACTION
# =====================
def extract_features(path):
    y, sr = librosa.load(path, sr=SR, duration=DURATION, offset=OFFSET)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    feat = np.vstack([mfcc, chroma, mel]).T

    if feat.shape[0] < MAX_LEN:
        feat = np.pad(feat, ((0, MAX_LEN - feat.shape[0]), (0, 0)))
    else:
        feat = feat[:MAX_LEN]
    return feat

# =====================
# LOAD DATA
# =====================
X, y = [], []

def load_folder(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.endswith(".wav"):
                continue
            label = None

            # RAVDESS (label text)
            if "_" in f and f.split("_")[-1].replace(".wav", "") in VALID_LABELS:
                label = f.split("_")[-1].replace(".wav", "")

            if label not in VALID_LABELS:
                continue

            try:
                X.append(extract_features(os.path.join(root, f)))
                y.append(label)
            except:
                pass

# training + validation
load_folder(TRAINING)
load_folder(VALIDATION)

# custom
for c in CUSTOM:
    load_folder(c)

# EMO-DB
for f in os.listdir(EMODB):
    if not f.endswith(".wav"):
        continue
    code = f[-6].upper()
    label = emodb_map.get(code)
    if label:
        X.append(extract_features(os.path.join(EMODB, f)))
        y.append(label)

X = np.array(X)
y = np.array(y)

print("X:", X.shape)
print("Labels:", np.unique(y))

# =====================
# ENCODE + SCALE
# =====================
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(len(X), -1)).reshape(X.shape)

joblib.dump(scaler, "scaler.save")

# =====================
# SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y_enc, random_state=42
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_enc),
    y=y_enc
)
class_weights = dict(enumerate(class_weights))

# =====================
# MODEL
# =====================
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True),
                  input_shape=X_train.shape[1:]),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(le.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================
# TRAIN
# =====================
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[ReduceLROnPlateau(patience=3, factor=0.5)]
)

# =====================
# EVAL
# =====================
pred = np.argmax(model.predict(X_test), axis=1)
true = np.argmax(y_test, axis=1)

print(classification_report(true, pred, target_names=le.classes_))
print(confusion_matrix(true, pred))

model.save("ser_final.keras")
np.save("labels.npy", le.classes_)
print("✅ DONE")
