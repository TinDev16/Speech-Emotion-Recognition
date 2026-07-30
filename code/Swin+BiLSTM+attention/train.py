import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ================== CONFIG ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 40
N_SPLITS = 3
LR = 1e-4
PATIENCE = 5
REDUCE_PATIENCE = 2

# ================== CHECK DATA ==================
if not os.path.exists("X.npy") or not os.path.exists("y.npy"):
    raise FileNotFoundError("Không tìm thấy X.npy hoặc y.npy")

print("Loading data...")
X = np.load("X.npy")
y = np.load("y.npy")

print("X:", X.shape, "y:", y.shape)

le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
print("Classes:", le.classes_)

# ================== DATASET ==================
class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]

        img = torch.tensor(img).unsqueeze(0)

        img = nn.functional.interpolate(
            img.unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        img = img.repeat(3, 1, 1)

        return img, self.y[idx]


def get_loader(X_tr, y_tr, X_val, y_val):
    train_ds = SpectrogramDataset(X_tr, y_tr)
    val_ds = SpectrogramDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader

# ================== MODEL ==================
class SwinBiLSTMAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        for name, p in self.swin.named_parameters():
            if "layers.0" in name or "layers.1" in name:
                p.requires_grad = False

        self.proj = nn.Linear(768, 256)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.attn = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f = self.swin.forward_features(x)

        if f.dim() == 4:
            B, H, W, C = f.shape
            f = f.view(B, H * W, C)

        f = self.proj(f)

        out, _ = self.lstm(f)

        w = self.attn(out)
        w = torch.softmax(w, dim=1)

        out = torch.sum(out * w, dim=1)

        return self.fc(out)

# ================== TRAIN ==================
def train_one_fold(model, train_loader, val_loader, fold):
    model.to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=REDUCE_PATIENCE
    )

    scaler = GradScaler()

    best = 0
    bad = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}")

        for x, yb in loop:
            x, yb = x.to(DEVICE), yb.to(DEVICE)

            opt.zero_grad()

            with autocast():
                out = model(x)
                loss = nn.CrossEntropyLoss()(out, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item() * x.size(0)
            pred = out.argmax(1)

            correct += (pred == yb).sum().item()
            total += yb.size(0)

            loop.set_postfix(acc=correct / total, loss=loss.item())

        train_acc = correct / total
        train_loss = total_loss / len(train_loader.dataset)

        # ===== validation =====
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0

        with torch.no_grad():
            for x, yb in val_loader:
                x, yb = x.to(DEVICE), yb.to(DEVICE)

                with autocast():
                    out = model(x)
                    loss = nn.CrossEntropyLoss()(out, yb)

                v_loss += loss.item() * x.size(0)
                pred = out.argmax(1)

                v_correct += (pred == yb).sum().item()
                v_total += yb.size(0)

        val_acc = v_correct / v_total
        val_loss = v_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        sched.step(val_loss)

        if val_acc > best:
            best = val_acc
            bad = 0
            torch.save(model.state_dict(), f"best_fold_{fold}.pth")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping")
                break

    return best

# ================== MAIN ==================
if __name__ == "__main__":

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    scores = []
    all_true, all_pred = [], []

    for fold, (tr, va) in enumerate(skf.split(X, y_enc), 1):

        print(f"\n===== FOLD {fold} =====")

        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y_enc[tr], y_enc[va]

        train_loader, val_loader = get_loader(X_tr, y_tr, X_va, y_va)

        model = SwinBiLSTMAttention(num_classes)

        best = train_one_fold(model, train_loader, val_loader, fold)
        scores.append(best)

        model.load_state_dict(torch.load(f"best_fold_{fold}.pth"))
        model.eval()

        y_true, y_pred = [], []

        with torch.no_grad():
            for x, yb in val_loader:
                x = x.to(DEVICE)
                out = model(x)
                pred = out.argmax(1).cpu().numpy()

                y_pred.extend(pred)
                y_true.extend(yb.numpy())

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        print(classification_report(y_true, y_pred, target_names=le.classes_))

    print("\nAVG ACC:", np.mean(scores))

    cm = confusion_matrix(all_true, all_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.show()

    np.save("classes.npy", le.classes_)