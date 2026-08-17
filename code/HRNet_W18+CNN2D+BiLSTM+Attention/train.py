# ==========================================================
# DOWNLOAD DATA FROM GOOGLE DRIVE (Kaggle-compatible)
# ==========================================================
# Replace the two IDs below with your own Google Drive file IDs.
# Get them by right-clicking each file in Drive -> Share -> "Anyone
# with the link" -> Viewer, then copy the ID between /d/ and /view
# in the shared link:
#   https://drive.google.com/file/d/<THIS_IS_THE_ID>/view?usp=sharing

# ==========================================================
# IMPORT
# ==========================================================
import gc
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import Dataset, DataLoader
# ==========================================================
# FREEZE CONFIG
# ==========================================================
# Cac stage cuoi cua HRNet se duoc FINE-TUNE (khong freeze).
# Cac stage con lai (conv1, layer1, stage2...) se bi DONG BANG.
FREEZE_TRAINABLE_KEYWORDS = ("stage4.2",)
# ==========================================================
# FREEZE CONFIG
# ==========================================================
# Cac stage cuoi cua HRNet se duoc FINE-TUNE (khong freeze).
# Cac stage con lai (conv1, layer1, stage2...) se bi DONG BANG.


# ==========================================================
# SAVE DIRECTORY
# ==========================================================
# NOTE: Kaggle has no Google Drive mount, so results are saved to
# /kaggle/working/, which is downloadable from the notebook's
# Output tab, and persists across reruns only if you Save/commit
# the notebook version.
SAVE_DIR = "/content/drive/MyDrive/SER_HRNet_3Fold_v7_nospecagu"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================================
# DEVICE
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

# ==========================================================
# LOAD DATA
# ==========================================================
X = np.load("/content/X.npy")
y = np.load("/content/y.npy")

print("X shape :", X.shape, flush=True)
print("y shape :", y.shape, flush=True)

# ==========================================================
# LABEL ENCODING
# ==========================================================
le = LabelEncoder()
y = le.fit_transform(y)

num_classes = len(le.classes_)

print("Classes :", le.classes_, flush=True)




# ==========================================================
# DATASET
# ==========================================================
class SERDataset(Dataset):

    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_np = self.X[idx]


        x = torch.tensor(
            x_np,
            dtype=torch.float32
        )

        # (H,W) -> (1,H,W)
        x = x.unsqueeze(0)

        label = torch.tensor(
            self.y[idx],
            dtype=torch.long
        )

        return x, label


# ==========================================================
# ATTENTION
# ==========================================================
class Attention(nn.Module):

    def __init__(self, hidden_size, dropout=0.2):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        weight = torch.softmax(
            self.attention(x),
            dim=1
        )

        output = torch.sum(
            weight * x,
            dim=1
        )

        return output


# ==========================================================
# MODEL
# ==========================================================
class SERModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # -----------------------------
        # HRNet
        # -----------------------------
        self.hrnet = timm.create_model(
            "hrnet_w18",
            pretrained=True,
            in_chans=1,
            drop_path_rate=0.3,
            features_only=True,
        )

        # -----------------------------
        # CNN
        # -----------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(1984,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 8)),
            nn.Dropout2d(0.3)
        )

        # -----------------------------
        # BiLSTM
        # -----------------------------
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )

        # Dropout applied to LSTM output (nn.LSTM's own `dropout` arg
        # only regularizes BETWEEN stacked layers, not the final output)
        self.lstm_dropout = nn.Dropout(0.4)

        # -----------------------------
        # Attention
        # -----------------------------
        self.att = Attention(256, dropout=0.4)

        # -----------------------------
        # Classifier
        # -----------------------------
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # Resize
        x = F.interpolate(
            x,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )

        # HRNet
        y_list = self.hrnet(x)
        target_size = y_list[0].shape[-2:]
        y_list = [
            F.interpolate(y, size=target_size, mode='bilinear', align_corners=False)
            if y.shape[-2:] != target_size else y
            for y in y_list
        ]
        x = torch.cat(y_list, dim=1)  # (batch, 1984, H, W)

        # CNN
        x = self.cnn(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch, 32, 8, 64)
        x = x.view(x.size(0), x.size(1), -1)    # (batch, 32, 512)

        # LSTM
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)

        # Attention
        x = self.att(x)

        # FC
        x = self.classifier(x)

        return x


def freeze_hrnet_backbone(model, trainable_keywords=FREEZE_TRAINABLE_KEYWORDS):
    """
    Dong bang toan bo HRNet backbone, chi mo (fine-tune) cac layer
    co ten chua 1 trong cac tu khoa trong trainable_keywords.

    Ly do: cac stage dau (conv1, layer1, stage2...) hoc dac trung
    tong quat (canh, texture) -> nen giu nguyen pretrained.
    Cac stage cuoi (stage3, stage4) hoc dac trung dac thu theo
    ImageNet -> nen fine-tune lai cho phu hop voi anh spectrogram.
    """
    n_frozen = 0
    n_trainable = 0

    for name, param in model.hrnet.named_parameters():
        if any(keyword in name for keyword in trainable_keywords):
            param.requires_grad = True
            n_trainable += 1
        else:
            param.requires_grad = False
            n_frozen += 1

    print(f"[Freeze HRNet] So tensor freeze: {n_frozen} | So tensor fine-tune: {n_trainable}", flush=True)


def set_hrnet_bn_eval_for_frozen(model, trainable_keywords=FREEZE_TRAINABLE_KEYWORDS):
    """
    Goi ham nay MOI LAN sau model.train() trong training loop.

    Vi sao can: model.train() se tu dong bat train-mode cho TAT CA
    submodule (ke ca cac layer da bi freeze), khien BatchNorm o
    cac layer freeze van bi cap nhat running_mean/running_var du
    weight/bias khong doi. Ham nay ep cac BatchNorm thuoc phan
    freeze quay ve eval-mode, chi cho phep BatchNorm o stage3/stage4
    (phan dang fine-tune) o train-mode.
    """
    for name, module in model.hrnet.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if any(keyword in name for keyword in trainable_keywords):
                module.train()   # phan fine-tune -> cho update running stats
            else:
                module.eval()    # phan freeze -> giu nguyen running stats


# ==========================================================
# HELPER: EVALUATE MODEL ON A DATALOADER
# ==========================================================
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# ==========================================================
# HELPER: PLOT & SAVE CONFUSION MATRIX
# ==========================================================
def plot_confusion_matrix(cm, class_names, title, save_path, normalize=False):
    """
    cm            : confusion matrix (raw counts)
    class_names   : list of class labels
    title         : plot title
    save_path     : path to save the PNG
    normalize     : if True, show row-normalized (%) values on the plot
                     (raw cm is still shown in save_path + '_raw' as well)
    """

    if normalize:
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # in case a class has 0 support
        display_matrix = cm_norm
        fmt_values = [[f"{v*100:.1f}%" for v in row] for row in cm_norm]
    else:
        display_matrix = cm
        fmt_values = [[f"{v:d}" for v in row] for row in cm]

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.1),
                                     max(5, len(class_names) * 1.0)))

    im = ax.imshow(display_matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # annotate each cell
    thresh = display_matrix.max() / 2.0 if display_matrix.max() > 0 else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, fmt_values[i][j],
                ha="center", va="center",
                color="white" if display_matrix[i, j] > thresh else "black",
                fontsize=9
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================================
# STRATIFIED 3-FOLD CROSS VALIDATION
# ==========================================================
skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

EPOCHS = 60
PATIENCE = 5

# ==========================================================
# SAVE METRICS OF EACH FOLD (macro-avg)
# ==========================================================
fold_acc = []
fold_precision = []
fold_recall = []
fold_f1 = []
fold_reports = {}  # per-class report dict for every fold
fold_confusion_matrices = {}  # raw confusion matrix (np.array) for every fold

# ==========================================================
# START CROSS VALIDATION
# ==========================================================
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):

    print("\n" + "=" * 60, flush=True)
    print(f"                FOLD {fold}/3", flush=True)
    print("=" * 60, flush=True)

    # ------------------------------------------------------
    # Save path of current fold (define early, needed for the
    # "already evaluated" shortcut below)
    # ------------------------------------------------------
    BEST_PATH = os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth")
    CKPT_PATH = os.path.join(SAVE_DIR, f"checkpoint_fold{fold}.pth")
    report_txt_path = os.path.join(SAVE_DIR, f"classification_report_fold{fold}.txt")
    report_json_path = os.path.join(SAVE_DIR, f"classification_report_fold{fold}.json")
    cm_csv_path = os.path.join(SAVE_DIR, f"confusion_matrix_fold{fold}.csv")

    # ========================================================
    # SKIP FOLD ENTIRELY IF ALREADY FULLY EVALUATED
    # ========================================================
    if os.path.exists(report_json_path) and os.path.exists(cm_csv_path):

        print(f"Fold {fold}: da co ket qua evaluate tu truoc -> BO QUA, nhay sang fold ke tiep.", flush=True)

        with open(report_json_path, "r") as f:
            report_dict = json.load(f)

        fold_reports[f"fold{fold}"] = report_dict

        fold_acc.append(report_dict["accuracy"])
        fold_precision.append(report_dict["macro avg"]["precision"])
        fold_recall.append(report_dict["macro avg"]["recall"])
        fold_f1.append(report_dict["macro avg"]["f1-score"])

        cm_loaded = np.loadtxt(cm_csv_path, delimiter=",", skiprows=1)
        cm_loaded = np.atleast_2d(cm_loaded).astype(np.int64)
        fold_confusion_matrices[f"fold{fold}"] = cm_loaded

        print(
            f"Fold {fold} (loaded) -> Acc: {fold_acc[-1]:.4f} | "
            f"Precision(macro): {fold_precision[-1]:.4f} | "
            f"Recall(macro): {fold_recall[-1]:.4f} | "
            f"F1(macro): {fold_f1[-1]:.4f}",
            flush=True
        )

        continue
    # ========================================================
    # END SKIP BLOCK
    # ========================================================

    # ------------------------------------------------------
    # Split data
    # ------------------------------------------------------
    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    print("Train:", X_train.shape, flush=True)
    print("Validation:", X_val.shape, flush=True)

    # ------------------------------------------------------
    # Class Weight
    # ------------------------------------------------------
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    ).to(device)

    # ------------------------------------------------------
    # Dataset / DataLoader
    # (augment=True only for train -> SpecAugment never touches val)
    # ------------------------------------------------------
    train_dataset = SERDataset(X_train, y_train, augment=True)
    val_dataset = SERDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------
    model = SERModel(num_classes).to(device)

    # ------------------------------------------------------
    # Freeze HRNet: giu nguyen cac stage dau (dac trung tong quat),
    # chi fine-tune stage3/stage4 (dac trung dac thu)
    # ------------------------------------------------------
    freeze_hrnet_backbone(model, trainable_keywords=FREEZE_TRAINABLE_KEYWORDS)

    # ------------------------------------------------------
    # Optimizer
    # AdamW (decoupled weight decay) thay cho Adam, them weight_decay
    # de tang regularization cho tung nhom tham so.
    # ------------------------------------------------------
    optimizer = torch.optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.hrnet.parameters()),
         'lr': 1e-5, 'weight_decay': 0.01},
        {'params': model.cnn.parameters(),
         'lr': 1e-4, 'weight_decay': 0.01},
        {'params': model.lstm.parameters(),
         'lr': 1e-4, 'weight_decay': 0.01},
        {'params': model.att.parameters(),
         'lr': 1e-4, 'weight_decay': 0.01},
        {'params': model.classifier.parameters(),
         'lr': 1e-4, 'weight_decay': 0.01}
    ], betas=(0.9, 0.999))

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        factor=0.5,
    )

    # ------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------
    best_val_loss = np.inf
    early_stop_counter = 0

    # ------------------------------------------------------
    # History
    # ------------------------------------------------------
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # ======================================================
    # RESUME CHECKPOINT
    # ======================================================
    start_epoch = 0

    if os.path.exists(CKPT_PATH):

        print(f"\nLoading checkpoint: {CKPT_PATH}", flush=True)

        checkpoint = torch.load(CKPT_PATH, map_location=device)

        model.load_state_dict(checkpoint["model"])

        # Re-apply freeze mask after loading weights (load_state_dict
        # does not touch requires_grad, but we redo it here to be
        # explicit and safe against any future refactor).


        optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            print("[WARN] Checkpoint cu khong co scheduler state -> dung mac dinh.", flush=True)

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        history = checkpoint["history"]
        early_stop_counter = checkpoint["early_stop_counter"]

        print(f"Resume Fold {fold} from Epoch {start_epoch}", flush=True)
        print(f"early_stop_counter da luu = {early_stop_counter}/{PATIENCE}", flush=True)

    skip_training = False

    if early_stop_counter >= PATIENCE:
        print(f"Fold {fold}: checkpoint cho thay da Early Stopping roi (chi con thieu buoc evaluate).", flush=True)
        skip_training = True

    elif start_epoch >= EPOCHS:
        print(f"Fold {fold} already finished.", flush=True)
        skip_training = True

    if not skip_training:
        print("\nStart Training...", flush=True)

        # ==================================================
        # TRAINING LOOP
        # ==================================================
        for epoch in range(start_epoch, EPOCHS):

            print("\n", flush=True)
            print("-" * 60, flush=True)
            print(f"Fold {fold} - Epoch {epoch + 1}/{EPOCHS}", flush=True)
            print("-" * 60, flush=True)

            # -----------------------------
            # TRAIN
            # -----------------------------
            model.train()
            # Ep BatchNorm o phan HRNet dang bi freeze quay ve eval-mode,
            # de running_mean/running_var cua chung khong bi thay doi.
            set_hrnet_bn_eval_for_frozen(model, trainable_keywords=FREEZE_TRAINABLE_KEYWORDS)

            running_loss = 0.0
            correct = 0
            total = 0

            train_bar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch + 1}")

            for images, labels in train_bar:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / len(train_loader)
            train_acc = correct / total

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            print(f"Train Loss : {train_loss:.4f}", flush=True)
            print(f"Train Acc  : {train_acc:.4f}", flush=True)

            # =====================================================
            # VALIDATION
            # =====================================================
            model.eval()

            val_running_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            val_loss = val_running_loss / len(val_loader)
            val_acc = correct / total

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            scheduler.step(val_loss)

            print(f"Val Loss   : {val_loss:.4f}", flush=True)
            print(f"Val Acc    : {val_acc:.4f}", flush=True)

            # =====================================================
            # SAVE BEST MODEL
            # =====================================================
            if val_loss < best_val_loss:

                best_val_loss = val_loss
                early_stop_counter = 0

                torch.save(model.state_dict(), BEST_PATH)

                print(">>> Best model saved.", flush=True)

            else:
                early_stop_counter += 1
                print(f"EarlyStopping Counter: {early_stop_counter}/{PATIENCE}", flush=True)

            # =====================================================
            # SAVE CHECKPOINT
            # =====================================================
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "early_stop_counter": early_stop_counter
                },
                CKPT_PATH
            )

            # =====================================================
            # EARLY STOPPING
            # =====================================================
            if early_stop_counter >= PATIENCE:
                print(f"\nEarly stopping at Epoch {epoch + 1}", flush=True)
                break

    # ==========================================================
    # EVALUATE BEST MODEL OF THIS FOLD (per-class metrics)
    # ==========================================================
    if not os.path.exists(BEST_PATH):
        print(f"[WARNING] No best model found for fold {fold}, skipping evaluation.", flush=True)
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        continue

    print(f"\nEvaluating BEST model of Fold {fold}...", flush=True)

    model.load_state_dict(torch.load(BEST_PATH, map_location=device))

    y_true, y_pred = evaluate(model, val_loader, device)

    # ------------------------------------------------------
    # Overall (macro-avg) metrics
    # ------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    fold_acc.append(acc)
    fold_precision.append(precision_macro)
    fold_recall.append(recall_macro)
    fold_f1.append(f1_macro)

    print(
        f"Fold {fold} -> Acc: {acc:.4f} | "
        f"Precision(macro): {precision_macro:.4f} | "
        f"Recall(macro): {recall_macro:.4f} | "
        f"F1(macro): {f1_macro:.4f}",
        flush=True
    )

    # ------------------------------------------------------
    # Per-class report
    # ------------------------------------------------------
    report_dict = classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    report_str = classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        zero_division=0
    )

    print(f"\nClassification Report - Fold {fold}:\n{report_str}", flush=True)

    fold_reports[f"fold{fold}"] = report_dict

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    fold_confusion_matrices[f"fold{fold}"] = cm

    # ------------------------------------------------------
    # Save everything
    # ------------------------------------------------------
    try:
        with open(report_txt_path, "w") as f:
            f.write(f"Fold {fold} Classification Report\n")
            f.write("=" * 60 + "\n")
            f.write(report_str)

        with open(report_json_path, "w") as f:
            json.dump(report_dict, f, indent=4)

        np.savetxt(
            cm_csv_path,
            cm,
            delimiter=",",
            fmt="%d",
            header=",".join(le.classes_),
            comments=""
        )

        plot_confusion_matrix(
            cm,
            class_names=le.classes_,
            title=f"Confusion Matrix - Fold {fold} (counts)",
            save_path=os.path.join(SAVE_DIR, f"confusion_matrix_fold{fold}_counts.png"),
            normalize=False
        )

        plot_confusion_matrix(
            cm,
            class_names=le.classes_,
            title=f"Confusion Matrix - Fold {fold} (normalized)",
            save_path=os.path.join(SAVE_DIR, f"confusion_matrix_fold{fold}_normalized.png"),
            normalize=True
        )

        print(f"Da luu ket qua fold {fold} vao: {cm_csv_path}", flush=True)

    except Exception as e:
        print(f"[WARNING] Loi khi ghi file cho fold {fold}: {e}", flush=True)
        print("Ket qua van co trong RAM, nhung report_json/cm_csv chua duoc luu -> lan resume sau se KHONG bi bo qua fold nay.", flush=True)

    # ==========================================================
    # Giai phong GPU/RAM truoc khi sang fold tiep theo
    # ==========================================================
    del model, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Da giai phong GPU/RAM sau fold {fold}.\n", flush=True)

# ==========================================================
# SUMMARY ACROSS ALL FOLDS
# ==========================================================
if len(fold_acc) > 0:

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY - 3-FOLD CROSS VALIDATION (macro-avg)", flush=True)
    print("=" * 60, flush=True)

    print(f"Accuracy         : {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}", flush=True)
    print(f"Precision(macro) : {np.mean(fold_precision):.4f} ± {np.std(fold_precision):.4f}", flush=True)
    print(f"Recall(macro)    : {np.mean(fold_recall):.4f} ± {np.std(fold_recall):.4f}", flush=True)
    print(f"F1(macro)        : {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}", flush=True)

    # ------------------------------------------------------
    # Per-class average across folds
    # ------------------------------------------------------
    per_class_avg = {}
    for cls in le.classes_:
        p_list = [fold_reports[f"fold{i}"][cls]["precision"] for i in range(1, 4) if f"fold{i}" in fold_reports]
        r_list = [fold_reports[f"fold{i}"][cls]["recall"] for i in range(1, 4) if f"fold{i}" in fold_reports]
        f_list = [fold_reports[f"fold{i}"][cls]["f1-score"] for i in range(1,4) if f"fold{i}" in fold_reports]

        per_class_avg[cls] = {
            "precision_mean": float(np.mean(p_list)),
            "precision_std": float(np.std(p_list)),
            "recall_mean": float(np.mean(r_list)),
            "recall_std": float(np.std(r_list)),
            "f1_mean": float(np.mean(f_list)),
            "f1_std": float(np.std(f_list)),
        }

    print("\nPer-class average across folds:", flush=True)
    for cls, m in per_class_avg.items():
        print(
            f"  {cls:12s} | "
            f"P: {m['precision_mean']:.4f} ± {m['precision_std']:.4f} | "
            f"R: {m['recall_mean']:.4f} ± {m['recall_std']:.4f} | "
            f"F1: {m['f1_mean']:.4f} ± {m['f1_std']:.4f}",
            flush=True
        )

    # ------------------------------------------------------
    # Aggregated confusion matrix (sum over all folds)
    # ------------------------------------------------------
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    for cm in fold_confusion_matrices.values():
        cm_total += cm

    cm_total_csv_path = os.path.join(SAVE_DIR, "confusion_matrix_total_3fold.csv")
    np.savetxt(
        cm_total_csv_path,
        cm_total,
        delimiter=",",
        fmt="%d",
        header=",".join(le.classes_),
        comments=""
    )

    plot_confusion_matrix(
        cm_total,
        class_names=le.classes_,
        title="Confusion Matrix - Sum over 3 Folds (counts)",
        save_path=os.path.join(SAVE_DIR, "confusion_matrix_total_3fold_counts.png"),
        normalize=False
    )

    plot_confusion_matrix(
        cm_total,
        class_names=le.classes_,
        title="Confusion Matrix - Sum over 3 Folds (normalized)",
        save_path=os.path.join(SAVE_DIR, "confusion_matrix_total_3fold_normalized.png"),
        normalize=True
    )

    print(f"\nAggregated confusion matrix (3 folds) saved to: {cm_total_csv_path}", flush=True)

    summary = {
        "fold_acc": fold_acc,
        "fold_precision": fold_precision,
        "fold_recall": fold_recall,
        "fold_f1": fold_f1,
        "mean_acc": float(np.mean(fold_acc)),
        "mean_precision": float(np.mean(fold_precision)),
        "mean_recall": float(np.mean(fold_recall)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_acc": float(np.std(fold_acc)),
        "std_precision": float(np.std(fold_precision)),
        "std_recall": float(np.std(fold_recall)),
        "std_f1": float(np.std(fold_f1)),
        "per_class_avg": per_class_avg,
        "per_fold_reports": fold_reports,
        "confusion_matrix_total": cm_total.tolist(),
        "class_names": le.classes_.tolist(),
    }

    summary_path = os.path.join(SAVE_DIR, "summary_3fold.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSummary saved to: {summary_path}", flush=True)

else:
    print("No fold was evaluated (no best model found in any fold).", flush=True)
