# =====================================================
# SPEECH EMOTION RECOGNITION - BiLSTM (8 LABELS)
# One-file version - FIXED SCALER
# =====================================================

# =============================
# 1️⃣ IMPORT & CONFIG
# =============================
import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

import joblib   # ✅ THÊM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


# =============================
# 2️⃣ PARAMETERS
# =============================
ROOT_PATH = "speech-emotion-recognition-en"

SAMPLE_RATE = 22050
DURATION = 3
OFFSET = 0.5
MAX_LEN = 130


# =============================
# 3️⃣ EMOTION MAP (8 LABELS)
# =============================
ravdess_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

crema_map = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "FEA": "fear",
    "DIS": "disgust",
    "NEU": "neutral"
}

savee_map = {
    "a": "angry",
    "h": "happy",
    "s": "sad",
    "f": "fear",
    "d": "disgust",
    "n": "neutral"
}


# =============================
# 4️⃣ FEATURE EXTRACTION
# =============================
def extract_features(path):
    y, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        duration=DURATION,
        offset=OFFSET
    )

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)

    features = np.vstack([mfcc, chroma, mel]).T  # (time, features)

    if features.shape[0] < MAX_LEN:
        features = np.pad(
            features,
            ((0, MAX_LEN - features.shape[0]), (0, 0))
        )
    else:
        features = features[:MAX_LEN, :]

    return features


# =============================
# 5️⃣ MAIN
# =============================
if __name__ == "__main__":

    print("📂 Loading dataset...")
    X, y = [], []

    for root, _, files in os.walk(ROOT_PATH):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue

            path = os.path.join(root, file)
            emotion = None
            root_lower = root.lower()

            if "ravdess" in root_lower:
                parts = file.split("-")
                if len(parts) > 2:
                    emotion = ravdess_map.get(parts[2])

            elif "crema" in root_lower:
                emotion = crema_map.get(file.split("_")[2])

            elif "savee" in root_lower:
                emotion = savee_map.get(
                    file.split("_")[1][0].lower()
                )

            elif "tess" in root_lower:
                folder = os.path.basename(root).lower()
                emotion = folder.split("_")[1]
                if emotion == "ps":
                    emotion = "surprise"

            if emotion is None:
                continue

            try:
                X.append(extract_features(path))
                y.append(emotion)
            except:
                pass

    X = np.array(X)
    y = np.array(y)

    print("✅ Data loaded")
    print("X shape:", X.shape)
    print("Labels:", np.unique(y))


    # =============================
    # 6️⃣ ENCODE + NORMALIZE
    # =============================
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    scaler = StandardScaler()
    X = scaler.fit_transform(
        X.reshape(len(X), -1)
    ).reshape(X.shape)

    # ✅ LƯU SCALER
    joblib.dump(scaler, "scaler.save")
    print("💾 Scaler saved: scaler.save")


    # =============================
    # 7️⃣ SPLIT + CLASS WEIGHT
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_enc),
        y=y_enc
    )
    class_weights = dict(enumerate(class_weights))


    # =============================
    # 8️⃣ BiLSTM MODEL
    # =============================
    model = Sequential([
        Bidirectional(
            LSTM(128, return_sequences=True),
            input_shape=X_train.shape[1:]
        ),
        Dropout(0.3),

        Bidirectional(LSTM(64)),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(y_cat.shape[1], activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()


    # =============================
    # 9️⃣ TRAIN
    # =============================
    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5)
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights
    )


    # =============================
    # 🔟 EVALUATION
    # =============================
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n📊 Classification Report")
    print(classification_report(
        y_true, y_pred, target_names=le.classes_
    ))

    print("📉 Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))


    # =============================
    # 1️⃣1️⃣ SAVE MODEL
    # =============================
    model.save("ser_bilstm_improved.h5")
    np.save("labels.npy", le.classes_)

    print("\n✅ TRAINING DONE")
    print("🎯 Labels:", le.classes_)
