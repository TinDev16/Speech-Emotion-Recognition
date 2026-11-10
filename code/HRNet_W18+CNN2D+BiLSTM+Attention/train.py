import os

os.makedirs("/kaggle/working/data", exist_ok=True)

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

from tqdm import tqdm

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
# RUN CONFIG
#   log-mel -> z-score -> HRNet -> MultiScaleNeck -> BiLSTM -> Attention -> classifier
#
#   3 SPLITS:
#     TRAIN : fit the weights
#     VALID : early stopping, LR scheduler, best-checkpoint selection
#     TEST  : final evaluation ONLY (report + confusion matrix), never used for selection
# ==========================================================
SEEDS = [42]
EPOCHS = 100

PATIENCE = 15                 # early stopping, in epochs
SCHED_PATIENCE = 3            # ReduceLROnPlateau
SCHED_FACTOR = 0.5
LABEL_SMOOTHING = 0           # 0 = off
SELECT_METRIC = "f1"          # "f1" (val macro-F1) or "loss" (val loss)

WARMUP_EPOCHS = 5

INPUT_SIZE = 224
DROP_PATH = 0.3

# Multi-scale neck
T_STEPS = 32
F_BINS = 8
MID = 32

# LSTM
LSTM_LAYERS = 2
LSTM_HIDDEN = 256

# Learning rates
LR_HRNET = 3e-4
LR_NECK = 3e-4
LR_LSTM = 3e-4
LR_ATT = 1e-4
LR_CLS = 1e-4

# SpecAugment (train only): masked regions are filled with the sample mean
SPEC_N_MASKS = 2
SPEC_MIN_RATIO = 0.1
SPEC_MAX_RATIO = 0.12

# HRNet: only these parts are trainable; the rest is frozen.
HRNET_TRAINABLE_KEYWORDS = ("stage3", "stage4", "transition3", "incre_module",)

SEQ_MODE = "multiscale"

# ==========================================================
# SAVE DIRECTORY (new folder: results here are evaluated on TEST,
# old folders were evaluated on VALID and must not be reused)
# ==========================================================
SAVE_DIR = "/kaggle/working/SER_HRNet_multiscale"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================================
# DEVICE
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

# ==========================================================
# LOAD DATA (already split, leakage-free: train / valid / test)
# ==========================================================
DATA_DIR = r"/kaggle/input/datasets/giathnhnguyn1203/ravdess-v2/RavTess/features_leakage_free"

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train_raw = np.load(os.path.join(DATA_DIR, "y_train.npy"), allow_pickle=True)

X_valid = np.load(os.path.join(DATA_DIR, "X_valid.npy"))
y_valid_raw = np.load(os.path.join(DATA_DIR, "y_valid.npy"), allow_pickle=True)

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test_raw = np.load(os.path.join(DATA_DIR, "y_test.npy"), allow_pickle=True)

print("X_train shape :", X_train.shape, flush=True)
print("y_train shape :", y_train_raw.shape, flush=True)
print("X_valid shape :", X_valid.shape, flush=True)
print("y_valid shape :", y_valid_raw.shape, flush=True)
print("X_test  shape :", X_test.shape, flush=True)
print("y_test  shape :", y_test_raw.shape, flush=True)

# ==========================================================
# SANITY CHECK
# ==========================================================
if len(X_valid) == 0:
    raise RuntimeError("X_valid is empty - check extract_feature.py output.")
if len(X_test) == 0:
    raise RuntimeError("X_test is empty - re-run extract_feature.py (3-split version).")

if X_train.shape[1:] != X_valid.shape[1:] or X_train.shape[1:] != X_test.shape[1:]:
    raise RuntimeError(
        f"Feature shape mismatch: train {X_train.shape[1:]} | "
        f"valid {X_valid.shape[1:]} | test {X_test.shape[1:]}"
    )

if (len(X_train) != len(y_train_raw) or len(X_valid) != len(y_valid_raw)
        or len(X_test) != len(y_test_raw)):
    raise RuntimeError("X / y length mismatch.")

# ==========================================================
# NORMALIZATION STATS (computed from TRAIN only, used for train, valid AND test)
# ==========================================================
if X_train.ndim == 4:                       # (N, time, mel, C): log-Mel + delta + delta-delta
    IN_CH = X_train.shape[-1]
    NORM_MEAN = X_train.mean(axis=(0, 1, 2), dtype=np.float64)          # one value per channel
    NORM_STD = X_train.std(axis=(0, 1, 2), dtype=np.float64) + 1e-8
else:                                       # (N, time, mel): log-Mel only
    IN_CH = 1
    NORM_MEAN = np.array([X_train.mean(dtype=np.float64)])
    NORM_STD = np.array([X_train.std(dtype=np.float64) + 1e-8])
print(f"Input channels: {IN_CH}", flush=True)
print(f"Z-score stats (train only, per channel): mean={NORM_MEAN} | std={NORM_STD}", flush=True)

# ==========================================================
# LABEL ENCODING (fit on TRAIN only)
# ==========================================================
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_valid = le.transform(y_valid_raw)
y_test = le.transform(y_test_raw)

num_classes = len(le.classes_)
print("Classes :", le.classes_, flush=True)


# ==========================================================
# DATASET (SpecAugment for train only, then z-score)
# ==========================================================
class SERDataset(Dataset):

    def __init__(self, X, y, norm_mean, norm_std, augment=False, n_masks=2,
                 min_ratio=0.1, max_ratio=0.12):
        self.X = X
        self.y = y
        # per-channel stats, shape (C, 1, 1)
        self.norm_mean = torch.as_tensor(norm_mean, dtype=torch.float32).view(-1, 1, 1)
        self.norm_std = torch.as_tensor(norm_std, dtype=torch.float32).view(-1, 1, 1)
        self.augment = augment
        self.n_masks = n_masks
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __len__(self):
        return len(self.X)

    def _mask_axis(self, x, axis, fill):
        """x is (C, time, mel). Mask n_masks contiguous bands along `axis`
        (1 = time, 2 = mel) in ALL channels at once, with the per-channel `fill`."""
        L = x.shape[axis]
        w_min = max(1, int(L * self.min_ratio))
        w_max = max(w_min, int(L * self.max_ratio))
        for _ in range(self.n_masks):
            w = int(torch.randint(w_min, w_max + 1, (1,)))
            w = min(w, L)
            start = int(torch.randint(0, L - w + 1, (1,)))
            if axis == 1:
                x[:, start:start + w, :] = fill
            else:
                x[:, :, start:start + w] = fill
        return x

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)   # (time, mel) or (time, mel, C)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x = x.permute(2, 0, 1).contiguous()                  # (C, time, mel)

        if self.augment:
            fill = x.mean(dim=(1, 2), keepdim=True)           # per-channel mean BEFORE masking
            x = self._mask_axis(x, axis=1, fill=fill)         # Time Masking
            x = self._mask_axis(x, axis=2, fill=fill)         # Frequency Masking

        # per-channel z-score AFTER SpecAugment, same stats for train / valid / test
        x = (x - self.norm_mean) / self.norm_std

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return x, label


# ==========================================================
# ATTENTION
# ==========================================================
class Attention(nn.Module):

    def __init__(self, in_dim=512, attn_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(attn_dim, 1)
        )

    def forward(self, x):                                   # (B, L, in_dim)
        weight = torch.softmax(self.attention(x), dim=1)    # (B, L, 1)
        return torch.sum(weight * x, dim=1)                 # (B, in_dim)


# ==========================================================
# MODEL
# ==========================================================
class MultiScaleNeck(nn.Module):
    """All HRNet maps -> per-position projection (1x1) -> common (time, freq)
    grid -> concat -> fold freq into features -> Linear -> (B, T, out_dim)."""

    def __init__(self, in_chs, mid=32, t_steps=32, f_bins=8, out_dim=256):
        super().__init__()
        self.t_steps = t_steps
        self.f_bins = f_bins
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, mid, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )
            for c in in_chs
        ])
        self.out = nn.Linear(mid * len(in_chs) * f_bins, out_dim)

    def _resize(self, f):
        T, Fb = self.t_steps, self.f_bins
        # time axis (H)
        if f.shape[-2] > T:
            f = F.adaptive_avg_pool2d(f, (T, f.shape[-1]))
        elif f.shape[-2] < T:
            f = F.interpolate(f, size=(T, f.shape[-1]), mode="bilinear", align_corners=False)
        # frequency axis (W)
        if f.shape[-1] > Fb:
            f = F.adaptive_avg_pool2d(f, (f.shape[-2], Fb))
        elif f.shape[-1] < Fb:
            f = F.interpolate(f, size=(f.shape[-2], Fb), mode="bilinear", align_corners=False)
        return f

    def forward(self, feats):
        xs = [self._resize(p(f)) for p, f in zip(self.proj, feats)]
        x = torch.cat(xs, dim=1)                          # (B, mid*n, T, Fb)
        B, C, T, Fb = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, T, Fb * C)   # (B, T, Fb*C)
        return self.out(x)                                # (B, T, out_dim)


class HRNetSER(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.hrnet = timm.create_model(
            "hrnet_w18",
            pretrained=True,
            in_chans=IN_CH,
            drop_path_rate=DROP_PATH,
            features_only=True,
        )

        chs = tuple(self.hrnet.feature_info.channels())
        print("HRNet feature channels:", chs, flush=True)

        self.proj = MultiScaleNeck(chs, mid=MID, t_steps=T_STEPS,
                                   f_bins=F_BINS, out_dim=256)

        lstm_out = LSTM_HIDDEN * 2                       # bidirectional
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if LSTM_LAYERS > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.3)
        self.att = Attention(in_dim=lstm_out, attn_dim=256)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Linear(lstm_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE),
                          mode="bilinear", align_corners=False)

        feats = self.hrnet(x)          # 5 maps: stem + 4 parallel branches
        x = self.proj(feats)           # (B, T_STEPS, 256)

        x, _ = self.lstm(x)            # (B, T_STEPS, 2*LSTM_HIDDEN)
        x = self.att(x)                # (B, 2*LSTM_HIDDEN)
        return self.classifier(x)


def freeze_hrnet_backbone(model, trainable_keywords=HRNET_TRAINABLE_KEYWORDS):
    n_frozen = 0
    n_trainable = 0
    for name, param in model.hrnet.named_parameters():
        if any(k in name for k in trainable_keywords):
            param.requires_grad = True
            n_trainable += 1
        else:
            param.requires_grad = False
            n_frozen += 1
    print(f"[Freeze HRNet] Frozen: {n_frozen} | Trainable: {n_trainable}", flush=True)

    names = [n for n, _ in model.hrnet.named_parameters()]
    for k in trainable_keywords:
        n_match = sum(k in n for n in names)
        print(f"[Freeze HRNet] keyword '{k}' matches {n_match} parameter tensors", flush=True)
        if n_match == 0:
            print(f"[WARNING] keyword '{k}' matches nothing "
                  f"(top-level: {sorted({n.split('.')[0] for n in names})})", flush=True)


def set_hrnet_bn_eval_for_frozen(model, trainable_keywords=HRNET_TRAINABLE_KEYWORDS):
    for name, module in model.hrnet.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if any(k in name for k in trainable_keywords):
                module.train()
            else:
                module.eval()


def print_param_counts(model):
    for name in ["hrnet", "proj", "lstm", "att", "classifier"]:
        m = getattr(model, name)
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:10s} total={total/1e6:.3f}M | trainable={trainable/1e6:.3f}M", flush=True)


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


def run_validation(model, loader, criterion, device):
    """returns val loss, accuracy and macro-F1 in a single pass."""
    model.eval()
    loss_sum = 0.0
    preds, labs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item()
            preds.extend(outputs.argmax(1).cpu().numpy())
            labs.extend(labels.cpu().numpy())
    preds, labs = np.array(preds), np.array(labs)
    return (loss_sum / len(loader),
            accuracy_score(labs, preds),
            f1_score(labs, preds, average="macro", zero_division=0))


def plot_confusion_matrix(cm, class_names, title, save_path, normalize=False):
    if normalize:
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
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
# DATASETS / DATALOADERS
# ==========================================================
val_dataset = SERDataset(X_valid, y_valid, NORM_MEAN, NORM_STD, augment=False)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False,
    num_workers=2, pin_memory=True,
)

test_dataset = SERDataset(X_test, y_test, NORM_MEAN, NORM_STD, augment=False)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False,
    num_workers=2, pin_memory=True,
)

train_dataset = SERDataset(
    X_train, y_train, NORM_MEAN, NORM_STD, augment=True,
    n_masks=SPEC_N_MASKS, min_ratio=SPEC_MIN_RATIO, max_ratio=SPEC_MAX_RATIO,
)

# ---- sanity check: input fed to HRNet must be ~ N(0, 1) ----
_xb, _ = next(iter(DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)))
_xv, _ = next(iter(DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)))
_xt, _ = next(iter(DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)))
print(f"[Check] train batch: mean={_xb.mean():.3f} std={_xb.std():.3f} "
      f"min={_xb.min():.2f} max={_xb.max():.2f}", flush=True)
print(f"[Check] valid batch: mean={_xv.mean():.3f} std={_xv.std():.3f} "
      f"min={_xv.min():.2f} max={_xv.max():.2f}", flush=True)
print(f"[Check] test  batch: mean={_xt.mean():.3f} std={_xt.std():.3f} "
      f"min={_xt.min():.2f} max={_xt.max():.2f}", flush=True)
del _xb, _xv, _xt

# ==========================================================
# CLASS WEIGHTS (computed from TRAIN only)
# ==========================================================
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
weights = torch.tensor(weights, dtype=torch.float32).to(device)

# ==========================================================
# METRICS ACROSS RUNS (different seeds, SAME fixed split) - all on TEST
# ==========================================================
run_acc = []
run_precision = []
run_recall = []
run_f1 = []
run_reports = {}
run_confusion_matrices = {}

for run_id, seed in enumerate(SEEDS, start=1):

    print("\n" + "=" * 60, flush=True)
    print(f"                RUN {run_id}/{len(SEEDS)} (seed={seed}, seq_mode={SEQ_MODE})", flush=True)
    print("=" * 60, flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    BEST_PATH = os.path.join(SAVE_DIR, f"best_model_run{run_id}.pth")
    CKPT_PATH = os.path.join(SAVE_DIR, f"checkpoint_run{run_id}.pth")
    report_txt_path = os.path.join(SAVE_DIR, f"test_classification_report_run{run_id}.txt")
    report_json_path = os.path.join(SAVE_DIR, f"test_classification_report_run{run_id}.json")
    cm_csv_path = os.path.join(SAVE_DIR, f"test_confusion_matrix_run{run_id}.csv")

    if os.path.exists(report_json_path) and os.path.exists(cm_csv_path):
        print(f"Run {run_id}: results already exist -> SKIP.", flush=True)

        with open(report_json_path, "r") as f:
            report_dict = json.load(f)
        run_reports[f"run{run_id}"] = report_dict

        run_acc.append(report_dict["accuracy"])
        run_precision.append(report_dict["macro avg"]["precision"])
        run_recall.append(report_dict["macro avg"]["recall"])
        run_f1.append(report_dict["macro avg"]["f1-score"])

        cm_loaded = np.loadtxt(cm_csv_path, delimiter=",", skiprows=1)
        cm_loaded = np.atleast_2d(cm_loaded).astype(np.int64)
        run_confusion_matrices[f"run{run_id}"] = cm_loaded

        continue

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=2, pin_memory=True,
    )

    model = HRNetSER(num_classes).to(device)
    freeze_hrnet_backbone(model, trainable_keywords=HRNET_TRAINABLE_KEYWORDS)
    print("Parameter counts:", flush=True)
    print_param_counts(model)

    optimizer = torch.optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, model.hrnet.parameters()),
         'lr': LR_HRNET, 'weight_decay': 0.02},
        {'params': model.proj.parameters(), 'lr': LR_NECK, 'weight_decay': 0.05},
        {'params': model.lstm.parameters(), 'lr': LR_LSTM, 'weight_decay': 0.05},
        {'params': model.att.parameters(), 'lr': LR_ATT, 'weight_decay': 0.05},
        {'params': model.classifier.parameters(), 'lr': LR_CLS, 'weight_decay': 0.01}
    ], betas=(0.9, 0.999))

    base_lrs = [g["lr"] for g in optimizer.param_groups]
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    print(f"Warmup: {WARMUP_EPOCHS} epochs = {warmup_steps} steps | base lrs = {base_lrs}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    # steps on VALID loss (never on test)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHED_PATIENCE, factor=SCHED_FACTOR,
    )

    # best_score is "higher is better": val macro-F1, or -val_loss
    best_score = -np.inf
    early_stop_counter = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    start_epoch = 0

    if os.path.exists(CKPT_PATH):
        print(f"\nLoading checkpoint: {CKPT_PATH}", flush=True)
        checkpoint = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        history = checkpoint["history"]
        early_stop_counter = checkpoint["early_stop_counter"]
        print(f"Resume Run {run_id} from Epoch {start_epoch}", flush=True)

    skip_training = False
    if early_stop_counter >= PATIENCE:
        print(f"Run {run_id}: checkpoint shows early stopping already triggered.", flush=True)
        skip_training = True
    elif start_epoch >= EPOCHS:
        print(f"Run {run_id} already finished.", flush=True)
        skip_training = True

    if not skip_training:
        print("\nStart Training...", flush=True)

        for epoch in range(start_epoch, EPOCHS):

            print(f"\nRun {run_id} - Epoch {epoch + 1}/{EPOCHS}", flush=True)

            model.train()
            set_hrnet_bn_eval_for_frozen(model, trainable_keywords=HRNET_TRAINABLE_KEYWORDS)

            running_loss = 0.0
            correct = 0
            total = 0

            train_bar = tqdm(train_loader, desc=f"Run {run_id} Epoch {epoch + 1}")

            for step_in_epoch, (images, labels) in enumerate(train_bar):
                images = images.to(device)
                labels = labels.to(device)

                # ---- linear warmup (per step) ----
                global_step = epoch * len(train_loader) + step_in_epoch
                if global_step < warmup_steps:
                    scale = (global_step + 1) / warmup_steps
                    for g, base in zip(optimizer.param_groups, base_lrs):
                        g["lr"] = base * scale

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
            print(f"Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f}", flush=True)

            # -------------------- VALIDATION (X_valid, no augmentation) --------------------
            val_loss, val_acc, val_f1 = run_validation(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            if epoch >= WARMUP_EPOCHS:
                scheduler.step(val_loss)

            cur_lrs = [g["lr"] for g in optimizer.param_groups]
            print(f"Val Loss   : {val_loss:.4f} | Val Acc    : {val_acc:.4f} | Val F1 : {val_f1:.4f} | "
                  f"lr(hrnet)={cur_lrs[0]:.2e} lr(lstm)={cur_lrs[2]:.2e}", flush=True)

            score = val_f1 if SELECT_METRIC == "f1" else -val_loss
            if score > best_score:
                best_score = score
                early_stop_counter = 0
                torch.save(model.state_dict(), BEST_PATH)
                print(f">>> Best model saved (by val {SELECT_METRIC}).", flush=True)
            else:
                early_stop_counter += 1
                print(f"EarlyStopping Counter: {early_stop_counter}/{PATIENCE}", flush=True)

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_score": best_score,
                    "history": history,
                    "early_stop_counter": early_stop_counter
                },
                CKPT_PATH
            )

            if early_stop_counter >= PATIENCE:
                print(f"\nEarly stopping at Epoch {epoch + 1}", flush=True)
                break

    # ---------------- EVALUATE BEST MODEL OF THIS RUN ON TEST ----------------
    if not os.path.exists(BEST_PATH):
        print(f"[WARNING] No best model found for run {run_id}, skipping evaluation.", flush=True)
        del model, optimizer, scheduler, train_loader
        gc.collect()
        torch.cuda.empty_cache()
        continue

    print(f"\nEvaluating BEST model of Run {run_id} on TEST set...", flush=True)
    model.load_state_dict(torch.load(BEST_PATH, map_location=device))

    y_true, y_pred = evaluate(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    run_acc.append(acc)
    run_precision.append(precision_macro)
    run_recall.append(recall_macro)
    run_f1.append(f1_macro)

    print(
        f"Run {run_id} [TEST] -> Acc: {acc:.4f} | Precision(macro): {precision_macro:.4f} | "
        f"Recall(macro): {recall_macro:.4f} | F1(macro): {f1_macro:.4f}",
        flush=True
    )

    report_dict = classification_report(
        y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, target_names=le.classes_, zero_division=0
    )
    print(f"\nClassification Report (TEST) - Run {run_id}:\n{report_str}", flush=True)

    run_reports[f"run{run_id}"] = report_dict

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    run_confusion_matrices[f"run{run_id}"] = cm

    try:
        with open(report_txt_path, "w") as f:
            f.write(f"Run {run_id} Classification Report (TEST)\n")
            f.write("=" * 60 + "\n")
            f.write(report_str)

        with open(report_json_path, "w") as f:
            json.dump(report_dict, f, indent=4)

        np.savetxt(
            cm_csv_path, cm, delimiter=",", fmt="%d",
            header=",".join(le.classes_), comments=""
        )

        # training curves (train/val) for later inspection
        with open(os.path.join(SAVE_DIR, f"history_run{run_id}.json"), "w") as f:
            json.dump(history, f, indent=4)

        # confusion matrix: TEST only
        plot_confusion_matrix(
            cm, class_names=le.classes_,
            title=f"Confusion Matrix (TEST) - Run {run_id} (counts)",
            save_path=os.path.join(SAVE_DIR, f"test_confusion_matrix_run{run_id}_counts.png"),
            normalize=False
        )
        plot_confusion_matrix(
            cm, class_names=le.classes_,
            title=f"Confusion Matrix (TEST) - Run {run_id} (normalized)",
            save_path=os.path.join(SAVE_DIR, f"test_confusion_matrix_run{run_id}_normalized.png"),
            normalize=True
        )

        print(f"Saved TEST results of run {run_id} to: {cm_csv_path}", flush=True)

    except Exception as e:
        print(f"[WARNING] Failed to write files for run {run_id}: {e}", flush=True)

    del model, optimizer, scheduler, train_loader
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Freed GPU/RAM after run {run_id}.\n", flush=True)

# ==========================================================
# SUMMARY ACROSS ALL RUNS (same fixed X_test every time)
# ==========================================================
if len(run_acc) > 0:

    DDOF = 1 if len(run_acc) > 1 else 0

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY - MULTI-SEED RUNS on fixed X_test", flush=True)
    print("=" * 60, flush=True)

    print(f"Accuracy         : {np.mean(run_acc):.4f} ± {np.std(run_acc, ddof=DDOF):.4f}", flush=True)
    print(f"Precision(macro) : {np.mean(run_precision):.4f} ± {np.std(run_precision, ddof=DDOF):.4f}", flush=True)
    print(f"Recall(macro)    : {np.mean(run_recall):.4f} ± {np.std(run_recall, ddof=DDOF):.4f}", flush=True)
    print(f"F1(macro)        : {np.mean(run_f1):.4f} ± {np.std(run_f1, ddof=DDOF):.4f}", flush=True)

    per_class_avg = {}
    for cls in le.classes_:
        p_list = [run_reports[f"run{i}"][cls]["precision"] for i in range(1, len(SEEDS) + 1) if f"run{i}" in run_reports]
        r_list = [run_reports[f"run{i}"][cls]["recall"] for i in range(1, len(SEEDS) + 1) if f"run{i}" in run_reports]
        f_list = [run_reports[f"run{i}"][cls]["f1-score"] for i in range(1, len(SEEDS) + 1) if f"run{i}" in run_reports]

        per_class_avg[cls] = {
            "precision_mean": float(np.mean(p_list)),
            "precision_std": float(np.std(p_list, ddof=DDOF)),
            "recall_mean": float(np.mean(r_list)),
            "recall_std": float(np.std(r_list, ddof=DDOF)),
            "f1_mean": float(np.mean(f_list)),
            "f1_std": float(np.std(f_list, ddof=DDOF)),
        }

    print("\nPer-class average across runs (TEST):", flush=True)
    for cls, m in per_class_avg.items():
        print(
            f"  {cls:12s} | P: {m['precision_mean']:.4f} ± {m['precision_std']:.4f} | "
            f"R: {m['recall_mean']:.4f} ± {m['recall_std']:.4f} | "
            f"F1: {m['f1_mean']:.4f} ± {m['f1_std']:.4f}",
            flush=True
        )

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    for cm in run_confusion_matrices.values():
        cm_total += cm

    cm_total_csv_path = os.path.join(SAVE_DIR, "test_confusion_matrix_total.csv")
    np.savetxt(
        cm_total_csv_path, cm_total, delimiter=",", fmt="%d",
        header=",".join(le.classes_), comments=""
    )

    plot_confusion_matrix(
        cm_total, class_names=le.classes_,
        title="Confusion Matrix (TEST) - Sum over runs (counts)",
        save_path=os.path.join(SAVE_DIR, "test_confusion_matrix_total_counts.png"),
        normalize=False
    )
    plot_confusion_matrix(
        cm_total, class_names=le.classes_,
        title="Confusion Matrix (TEST) - Sum over runs (normalized)",
        save_path=os.path.join(SAVE_DIR, "test_confusion_matrix_total_normalized.png"),
        normalize=True
    )

    summary = {
        "evaluated_on": "test",
        "run_acc": run_acc,
        "run_precision": run_precision,
        "run_recall": run_recall,
        "run_f1": run_f1,
        "mean_acc": float(np.mean(run_acc)),
        "mean_precision": float(np.mean(run_precision)),
        "mean_recall": float(np.mean(run_recall)),
        "mean_f1": float(np.mean(run_f1)),
        "std_acc": float(np.std(run_acc, ddof=DDOF)),
        "std_precision": float(np.std(run_precision, ddof=DDOF)),
        "std_recall": float(np.std(run_recall, ddof=DDOF)),
        "std_f1": float(np.std(run_f1, ddof=DDOF)),
        "std_ddof": DDOF,
        "warmup_epochs": WARMUP_EPOCHS,
        "per_class_avg": per_class_avg,
        "per_run_reports": run_reports,
        "confusion_matrix_total": cm_total.tolist(),
        "class_names": le.classes_.tolist(),
        "seq_mode": SEQ_MODE,
        "config": {
            "norm_mean": NORM_MEAN.tolist(),
            "norm_std": NORM_STD.tolist(),
            "in_channels": IN_CH,
            "label_smoothing": LABEL_SMOOTHING,
            "select_metric": SELECT_METRIC,
            "patience": PATIENCE,
            "sched_patience": SCHED_PATIENCE,
            "t_steps": T_STEPS,
            "lstm_layers": LSTM_LAYERS,
            "lstm_hidden": LSTM_HIDDEN,
            "lr_cls": LR_CLS,
        },
    }

    summary_path = os.path.join(SAVE_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSummary saved to: {summary_path}", flush=True)

else:
    print("No run was evaluated (no best model found in any run).", flush=True)



