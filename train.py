# =====================================================
# SER FINAL - CNN2D + BiLSTM + ATTENTION + MIXUP
# =====================================================

import os
import numpy as np
import librosa
import random
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


# =============================
# CONFIG
# =============================
ROOT_PATH = "Data"
SAMPLE_RATE = 22050
DURATION = 4
OFFSET = 0.5
MAX_LEN = 130

N_SPLITS = 5
BATCH_SIZE = 32

VALID_EMOTIONS = ["angry","happy","sad","fear","disgust","neutral"]


# =============================
# LABEL MAP
# =============================
ravdess_map = {
    "01": "neutral","03": "happy","04": "sad",
    "05": "angry","06": "fear","07": "disgust"
}

crema_map = {
    "ANG": "angry","HAP": "happy","SAD": "sad",
    "FEA": "fear","DIS": "disgust","NEU": "neutral"
}

savee_map = {
    "a": "angry","h": "happy","s": "sad",
    "f": "fear","d": "disgust","n": "neutral"
}


# =============================
# GET LABEL
# =============================
def get_emotion(path, file):
    p = path.lower()

    if "ravdess" in p:
        return ravdess_map.get(file.split("-")[2])

    elif "crema" in p:
        return crema_map.get(file.split("_")[2])

    elif "savee" in p:
        return savee_map.get(file.split("_")[1][0])

    elif "tess" in p:
        emotion = os.path.basename(path).split("_")[1]
        if emotion == "ps":
            emotion = "happy"
        return emotion

    return None


# =============================
# AUGMENT
# =============================
def add_noise(y):
    noise = np.random.randn(len(y))
    return y + 0.005 * noise


# =============================
# FEATURE
# =============================
def extract_features(path, augment=False):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE,
                             duration=DURATION, offset=OFFSET)
    except:
        return None

    if augment:
        y = add_noise(y)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0,0),(0, MAX_LEN-mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]

    return mel.T   # (time, freq)


# =============================
# LOAD DATA
# =============================
X, y = [], []

for root, _, files in os.walk(ROOT_PATH):
    for file in files:
        if not file.endswith(".wav"):
            continue

        path = os.path.join(root, file)

        emotion = get_emotion(root, file)
        if emotion is None:
            continue

        emotion = emotion.lower()
        if emotion not in VALID_EMOTIONS:
            continue

        feat = extract_features(path)
        if feat is None:
            continue

        X.append(feat)
        y.append(emotion)

        # augment
        aug = extract_features(path, augment=True)
        if aug is not None:
            X.append(aug)
            y.append(emotion)


X = np.array(X)
y = np.array(y)

print("Data:", X.shape)
print("Labels:", np.unique(y))


# =============================
# ENCODE
# =============================
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)


# =============================
# MIXUP
# =============================
def mixup(X, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = np.random.permutation(len(X))

    X_mix = lam * X + (1 - lam) * X[index]
    y_mix = lam * y + (1 - lam) * y[index]

    return X_mix, y_mix


def data_generator(X, y, batch_size):
    while True:
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]

        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            if np.random.rand() < 0.7:
                X_batch, y_batch = mixup(X_batch, y_batch)

            yield X_batch, y_batch


# =============================
# MODEL
# =============================
def build_model(input_shape, n_classes):
    inp = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Reshape((-1, x.shape[-1]))(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)

    # ATTENTION
    att = Dense(1, activation='tanh')(x)
    att = Flatten()(att)
    att = Activation('softmax')(att)
    att = RepeatVector(x.shape[-1])(att)
    att = Permute([2,1])(att)

    x = Multiply()([x, att])
    x = Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inp, out)

    model.compile(
        optimizer=Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    return model


# =============================
# TRAIN
# =============================
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_preds, all_true = [], []

for fold, (tr, val) in enumerate(kf.split(X, y_enc), 1):
    print(f"\nOLD {fold}")

    X_train, X_val = X[tr], X[val]
    y_train, y_val = y_cat[tr], y_cat[val]

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    model = build_model(X_train.shape[1:], y_cat.shape[1])

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=2)
    ]

    train_gen = data_generator(X_train, y_train, BATCH_SIZE)

    model.fit(
        train_gen,
        steps_per_epoch=len(X_train)//BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=40,
        callbacks=callbacks,
        verbose=1
    )

    pred = np.argmax(model.predict(X_val), axis=1)
    true = np.argmax(y_val, axis=1)

    all_preds.extend(pred)
    all_true.extend(true)


# =============================
# REPORT
# =============================
print("\nFINAL REPORT")
print(classification_report(all_true, all_preds, target_names=le.classes_))


cm = confusion_matrix(all_true, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)

plt.title("Confusion Matrix")
plt.show()


# =============================
# SAVE
# =============================
model.save("ser_best.keras")
np.save("labels.npy", le.classes_)

print("DONE")