# =========================
# MOUNT DRIVE
# =========================
from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================
# SAVE DIR
# =========================
SAVE_DIR = "/content/drive/MyDrive/SER_HRNet_final"
os.makedirs(SAVE_DIR, exist_ok=True)
CKPT_PATH = os.path.join(SAVE_DIR, "checkpoint.pth")
BEST_PATH = os.path.join(SAVE_DIR, "best_model.pth")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# LOAD DATA
# =========================
X = np.load("/content/X.npy")
y = np.load("/content/y.npy")

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(le.classes_)

print("Classes:", le.classes_)

# =========================
# SPLIT: TRAIN / VAL / TEST
# =========================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.2,
    random_state=42,
    stratify=y_temp
)

print("Train:", len(X_train))
print("Val:", len(X_val))
print("Test:", len(X_test))

# =========================
# CLASS WEIGHT
# =========================
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
weights = torch.tensor(weights, dtype=torch.float).to(device)

# =========================
# DATASET
# =========================
class SERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float)
        x = x.permute(2, 0, 1)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

train_loader = DataLoader(SERDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(SERDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader  = DataLoader(SERDataset(X_test, y_test), batch_size=32, shuffle=False)

# =========================
# MODEL
# =========================
class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        w = torch.softmax(self.att(x), dim=1)
        return torch.sum(w * x, dim=1)


class SERModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.hrnet = timm.create_model(
            "hrnet_w18",
            pretrained=False,
            in_chans=1,
            features_only=True
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(1024, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=128 * 2,
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )

        self.att = Attention(64)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = F.interpolate(x, size=(128,128), mode="bilinear", align_corners=False)

        x = self.hrnet(x)[-1]
        x = self.cnn(x)

        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, W, C * H)

        x, _ = self.lstm(x)
        x = self.att(x)

        return self.classifier(x)


model = SERModel(num_classes).to(device)

# =========================
# LOSS / OPTIM / SCHEDULER
# =========================
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3
)

# =========================
# LOGS (FOR PLOT)
# =========================
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}



# =========================
# RESUME CHECKPOINT
# =========================
start_epoch = 0
best_val = float("inf")
counter = 0
patience = 7

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = ckpt["epoch"] + 1
    best_val = ckpt["best_val"]
    counter = ckpt["counter"]

    print("Resumed from epoch", start_epoch)

# =========================
# TRAIN LOOP
# =========================
EPOCHS = 40

for epoch in range(start_epoch, EPOCHS):

    # TRAIN
    model.train()
    train_loss, correct, total = 0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for x, y in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = out.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        loop.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)
    train_acc = correct / total

    # VALIDATION
    model.eval()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    scheduler.step(val_loss)

       # ===== SAVE LOG =====
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)


    print("\nEpoch", epoch+1)
    print("Train:", train_loss, train_acc)
    print("Val:", val_loss, val_acc)




    # CHECKPOINT SAVE
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val,
        "counter": counter
    }, CKPT_PATH)
    
    
    # optional: Save each epoch's model state_dict for later analysis or ensemble methods
    torch.save(
        model.state_dict(),
        os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth")
    )


    # BEST MODEL
    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), BEST_PATH)
        print("!!! Best model saved")
    else:
        counter += 1
        if counter >= patience:
            print("!!! Early stopping")
            break

# =========================
# FINAL TEST
# =========================
print("\n===== TESTING BEST MODEL =====")

model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model.eval()

test_loss, correct, total = 0, 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        test_loss += loss.item()
        pred = out.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

test_loss /= len(test_loader)
test_acc = correct / total

print("Test Loss:", test_loss)
print("Test Acc:", test_acc)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


# =========================
# METRICS & CLASSIFICATION REPORT
# =========================
print("\n===== CLASSIFICATION REPORT =====")s
print(classification_report(
    all_labels,
    all_preds,
    target_names=le.classes_,
    digits=4
))

precision = precision_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")

print("\n===== MACRO METRICS =====")
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)
