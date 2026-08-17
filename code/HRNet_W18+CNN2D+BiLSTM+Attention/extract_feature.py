import os
import shutil
import zipfile
import librosa
import numpy as np
import time
from google.colab import drive

# ==========================================================
# MOUNT GOOGLE DRIVE
# ==========================================================
drive.mount('/content/drive')

# ==========================================================
# CONFIG - DATASET ZIP ON DRIVE
# ==========================================================
DRIVE_ZIP_PATH = "/content/drive/MyDrive/SER_dataset/merged_ser.zip"  # sửa đúng path zip thật
LOCAL_ZIP_PATH = "/content/merged_ser.zip"
EXTRACT_DIR = "/content/merged_ser_dataset"

VALID_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


if not os.path.exists(LOCAL_ZIP_PATH):
    print(f"Dang copy zip tu Drive ve local: {DRIVE_ZIP_PATH}")
    shutil.copy(DRIVE_ZIP_PATH, LOCAL_ZIP_PATH)
    print("Copy xong.")
else:
    print("File zip da co san tren local, bo qua buoc copy.")


if not os.path.exists(EXTRACT_DIR):
    print(f"Dang giai nen: {LOCAL_ZIP_PATH}")
    with zipfile.ZipFile(LOCAL_ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print("Giai nen xong.")
else:
    print("Da giai nen truoc do, bo qua.")

def find_root_path(base_dir, valid_emotions):
    for root, dirs, files in os.walk(base_dir):
        if any(d in valid_emotions for d in dirs):
            return root
    return None

ROOT_PATH = find_root_path(EXTRACT_DIR, VALID_EMOTIONS)

if ROOT_PATH is None:
    raise FileNotFoundError(
        f"Khong tim thay thu muc chua cac emotion folder trong {EXTRACT_DIR}. "
        f"Kiem tra lai cau truc: {os.listdir(EXTRACT_DIR)}"
    )

print("ROOT_PATH tu dong tim duoc:", ROOT_PATH)
print("Noi dung ROOT_PATH:", os.listdir(ROOT_PATH))

# ==========================================================
# FEATURE EXTRACTION CONFIG
# ==========================================================
SAMPLE_RATE = 22050
DURATION = 4
OFFSET = 0.5
MAX_LEN = 130

X = []
y = []

# ==========================================================
# COUNT TOTAL FILES (đếm động, không hardcode)
# ==========================================================
total_files = 0

for emotion in os.listdir(ROOT_PATH):
    emotion_path = os.path.join(ROOT_PATH, emotion)

    if emotion not in VALID_EMOTIONS or not os.path.isdir(emotion_path):
        continue

    total_files += len([
        f for f in os.listdir(emotion_path)
        if f.endswith(".wav")
    ])

print("=" * 50)
print(" SER FEATURE EXTRACTION")
print("=" * 50)
print("Total wav files:", total_files)
print("Augment x2 enabled")
print("=" * 50)

if total_files == 0:
    raise RuntimeError(
        "Total_files = 0. ROOT_PATH van chua dung, kiem tra lai cau truc thu muc "
        "truoc khi chay tiep (tranh lang phi thoi gian extract ra mang rong)."
    )

# ==========================================================
# AUGMENT
# ==========================================================
def add_noise(y_audio):
    noise = np.random.randn(len(y_audio))
    return y_audio + 0.005 * noise

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_features(path, augment=False):

    y_audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        duration=DURATION,
        offset=OFFSET
    )

    if augment:
        y_audio = add_noise(y_audio)

    mel = librosa.feature.melspectrogram(
        y=y_audio,
        sr=sr,
        n_mels=128
    )

    mel = librosa.power_to_db(mel)

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(
            mel,
            ((0, 0), (0, MAX_LEN - mel.shape[1]))
        )
    else:
        mel = mel[:, :MAX_LEN]

    return mel.T

# ==========================================================
# PROCESS
# ==========================================================
processed = 0
start_time = time.time()

for emotion in os.listdir(ROOT_PATH):
    emotion_path = os.path.join(ROOT_PATH, emotion)

    if emotion not in VALID_EMOTIONS or not os.path.isdir(emotion_path):
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

# ==========================================================
# SAVE LOCALLY
# ==========================================================
X = np.array(X)
y = np.array(y)

print("\n" + "=" * 50)
print(" Saving numpy files...")
print("X shape:", X.shape)
print("y shape:", y.shape)

if X.shape[0] == 0:
    raise RuntimeError("X rong sau khi extract - khong upload len Drive de tranh ghi de file loi.")

np.save("X.npy", X)
np.save("y.npy", y)
print(" Saved X.npy and y.npy locally")
print("=" * 50)

# ==========================================================
# CREATE ZIP FILES
# ==========================================================
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/SER_dataset_1"
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)

def zip_single_file(file_path, zip_name):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=os.path.basename(file_path))
    print(f"Da nen: {zip_name}")

zip_single_file("X.npy", "X.zip")
zip_single_file("y.npy", "y.zip")

# ==========================================================
# UPLOAD 2 FILE ZIP TO DRIVE
# ==========================================================
shutil.copy("X.zip", os.path.join(DRIVE_OUTPUT_DIR, "X.zip"))
shutil.copy("y.zip", os.path.join(DRIVE_OUTPUT_DIR, "y.zip"))

print(f"Da upload X.zip va y.zip len: {DRIVE_OUTPUT_DIR}")
print("HOAN TAT.")
