# =====================================================
# SER FINAL UPGRADE - CNN2D + BiLSTM + ATTENTION + MIXUP
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

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# =============================
# CONFIG
# =============================
ROOT_PATH = r"C:\Users\Chien\Desktop\SER\merged_ser\merged_ser_dataset"

SAMPLE_RATE = 22050
DURATION = 4
OFFSET = 0.5
MAX_LEN = 130

N_SPLITS = 3

# merged dataset của bạn có 7 emotion
VALID_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# =============================
# GET LABEL
# =============================
# merged_ser_dataset/
# ├── angry/
# ├── disgust/
# ├── fear/
# ...

def get_emotion(path, file):
    emotion = os.path.basename(path).lower()

    if emotion in VALID_EMOTIONS:
        return emotion

    return None

# =============================
# AUGMENT
# =============================
def add_noise(y):
    noise = np.random.randn(len(y))
    return y + 0.005 * noise

def spec_augment(spec):
    spec = spec.copy()

    # time mask
    t = random.randint(5, 20)
    t0 = random.randint(0, spec.shape[0]-t)
    spec[t0:t0+t, :] = 0

    # freq mask
    f = random.randint(5, 15)
    f0 = random.randint(0, spec.shape[1]-f)
    spec[:, f0:f0+f] = 0

    return spec


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
# LOAD PRE-EXTRACTED FEATURES
# =============================
X = np.load("X.npy")
y = np.load("y.npy")

print("Data:", X.shape)
print("Labels:", np.unique(y))


# =============================
# ENCODE
# =============================
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)


# =============================
# MODEL (CNN2D + BiLSTM)
# =============================
def build_model(input_shape, n_classes):

    inp = Input(shape=input_shape)

    # CNN BLOCK 1
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # CNN BLOCK 2
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # CNN BLOCK 3
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # reshape cho LSTM
    shape = x.shape

    x = Reshape(
        (shape[1] * shape[2], shape[3])
    )(x)

    # BiLSTM
    x = Bidirectional(
        LSTM(128, return_sequences=False)
    )(x)

    # Dense
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inp, out)

    model.compile(
        optimizer=Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        ),
        metrics=["accuracy"]
    )

    return model


# =============================
# TRAIN 1 TIME ONLY
# =============================

X_train, X_val, y_train_enc, y_val_enc = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)

# one-hot
y_train = to_categorical(y_train_enc)
y_val = to_categorical(y_val_enc)

# reshape cho CNN2D
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# class weight
cw = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_enc),
    y=y_train_enc
)

cw = dict(enumerate(cw))

# build model
model = build_model(
    X_train.shape[1:],
    y_cat.shape[1]
)

callbacks = [
    EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        patience=2
    )
]

# train
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    class_weight=cw,
    verbose=1
)

# predict
all_preds = np.argmax(
    model.predict(X_val),
    axis=1
)

all_true = np.argmax(
    y_val,
    axis=1
)

# =============================
# REPORT
# =============================
print("\n FINAL REPORT")
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

print(" DONE")