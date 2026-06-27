import os
import librosa
import numpy as np
import time

ROOT_PATH = os.path.abspath('/content/merged_ser_dataset/merged_ser_dataset (2)/merged_ser_dataset')

SAMPLE_RATE = 22050
DURATION = 4
OFFSET = 0.5
MAX_LEN = 130

VALID_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

if not os.path.exists(ROOT_PATH):
    raise FileNotFoundError(f"Dataset folder not found: {ROOT_PATH}. Run the unzip cell first or update ROOT_PATH.")

X, y = [], []
total_files = 13695

# =============================
# AUGMENT
# =============================
def add_noise(y_audio):
    return y_audio + 0.005 * np.random.randn(len(y_audio))
# =============================
# FEATURE EXTRACTION
# =============================
def extract_features(path, augment=False):
    y_audio, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, offset=OFFSET)
    if augment: y_audio = add_noise(y_audio)
    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)
    if mel.shape[1] < MAX_LEN: mel = np.pad(mel, ((0, 0), (0, MAX_LEN - mel.shape[1])), mode= "constant")
    else: mel = mel[:, :MAX_LEN]
    # Normalize to [0, 1]
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    mel = np.expand_dims(mel, axis=-1)

    return mel

# =============================
# PROCESS
# =============================
processed = 0
start_time = time.time()

for emotion in os.listdir(ROOT_PATH):

    emotion_path = os.path.join(ROOT_PATH, emotion)

    if emotion not in VALID_EMOTIONS:
        continue

    print(f"\n Processing emotion: {emotion}")

    for file in os.listdir(emotion_path):

        if not file.endswith(".wav"):
            continue

        path = os.path.join(emotion_path, file)

        try:
            # original
            feat = extract_features(path)

            X.append(feat)
            y.append(emotion)

            # augment
            aug = extract_features(path, augment=True)

            X.append(aug)
            y.append(emotion)

            processed += 1

            # progress
            if processed % 100 == 0:

                elapsed = time.time() - start_time

                percent = (processed / total_files) * 100

                speed = elapsed / processed

                remaining = (total_files - processed) * speed

                print(
                    f" {processed}/{total_files} "
                    f"({percent:.2f}%) | "
                    f"ETA: {remaining/60:.1f} min"
                )

        except Exception as e:
            print(" Error:", file, e)
# =============================
# SAVE
# =============================
X, y = np.array(X), np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)
np.save("X.npy", X)
np.save("y.npy", y)



