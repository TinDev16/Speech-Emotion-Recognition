#Log-Mel -> (local, global) -> CNN-BiLSTM 

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D,
    Concatenate, Dense, LSTM, Bidirectional, Dropout,
    Multiply, Add, Lambda, GlobalAveragePooling1D,
    Activation, LayerNormalization, MultiHeadAttention, Layer,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Config ──────────────────────────────────────────────────────────────────

CFG = dict(
    # ── Experiment
    attention     = 'se',   # 'se' | 'cbam'
    n_splits      = 3,
    epochs        = 50,
    batch_size    = 32,
    lr            = 1e-4,
    label_smooth  = 0.1,
    # ── Architecture
    cnn_filters   = [32, 64],  
    proj_channels = 64,         
    lstm_units    = 128,       
    attn_heads    = 4,
    attn_key_dim  = 64,
    dense_units   = 128,
    # ── Dropout
    dropout_freq  = 0.3,    
    dropout_rnn   = 0.3,      
    dropout_out   = 0.4,     
    # ── SE / CBAM
    se_ratio      = 8,
    # ── SpecAugment  (applied inside model, train-only)
    time_mask_W   = 30,         # max time-mask width (frames)
    freq_mask_F   = 13,         # max freq-mask width (mel bins)
    n_time_masks  = 2,
    n_freq_masks  = 2,
    # ── Mixup
    mixup_alpha   = 0.3,        # Beta(α,α) coefficient;  0 = disabled
)

# =============================================================================
#  CUSTOM LAYERS
# =============================================================================

class SpecAugment(Layer):
    """
    SpecAugment: Time Masking + Frequency Masking (Park et al., 2019).
    ─ Active ONLY when training=True (transparent at inference).
    ─ Input shape : (batch, time, mel, 1)

    Time mask  : zeros out t consecutive frames  (t ∼ U[0, W])
    Freq mask  : zeros out f consecutive mel bins (f ∼ U[0, F])
    Applied n_*_masks times each.
    """
    def __init__(self, time_mask_W=30, freq_mask_F=13,
                 n_time_masks=2, n_freq_masks=2, **kwargs):
        super().__init__(**kwargs)
        self.time_mask_W  = time_mask_W
        self.freq_mask_F  = freq_mask_F
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks

    def _apply(self, x):
        T = tf.shape(x)[1]   # time frames  (dynamic)
        M = tf.shape(x)[2]   # mel bins     (dynamic)

        # ── Time masks ─────────────────────────────────────────────────────
        for _ in range(self.n_time_masks):
            t  = tf.random.uniform([], 0, self.time_mask_W + 1, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, tf.maximum(T - t, 1),  dtype=tf.int32)
            # Boolean mask of shape (T,):  True in the zeroed region
            mask = 1.0 - tf.cast(
                tf.logical_and(tf.range(T) >= t0, tf.range(T) < t0 + t),
                tf.float32,
            )                                              # (T,)
            x = x * mask[tf.newaxis, :, tf.newaxis, tf.newaxis]

        # ── Frequency masks ─────────────────────────────────────────────────
        for _ in range(self.n_freq_masks):
            f  = tf.random.uniform([], 0, self.freq_mask_F + 1, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, tf.maximum(M - f, 1),  dtype=tf.int32)
            mask = 1.0 - tf.cast(
                tf.logical_and(tf.range(M) >= f0, tf.range(M) < f0 + f),
                tf.float32,
            )                                              # (M,)
            x = x * mask[tf.newaxis, tf.newaxis, :, tf.newaxis]

        return x

    def call(self, x, training=False):
        # Python-bool branch (eager / fit); tensor branch (tf.function / graph)
        if isinstance(training, bool):
            return self._apply(x) if training else x
        return tf.cond(tf.cast(training, tf.bool), lambda: self._apply(x), lambda: x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            time_mask_W=self.time_mask_W,
            freq_mask_F=self.freq_mask_F,
            n_time_masks=self.n_time_masks,
            n_freq_masks=self.n_freq_masks,
        ))
        return cfg


class PositionalEncoding(Layer):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017).
    ─ PE is precomputed in build() and stored as a non-trainable weight.
    ─ Input  shape: (batch, time, d_model)
    ─ Output shape: (batch, time, d_model)   →  x + PE[:seq_len]

    Even indices → sin, odd indices → cos.
    """
    def __init__(self, max_len: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = int(input_shape[-1])      # known statically after BiLSTM
        pe = self._sinusoidal_pe(self.max_len, d_model)   # (1, max_len, d_model)
        self.pe = self.add_weight(
            name='sinusoidal_pe',
            shape=pe.shape,
            initializer=tf.keras.initializers.Constant(pe),
            trainable=False,
        )
        super().build(input_shape)

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> np.ndarray:
        pos    = np.arange(max_len, dtype=np.float32)[:, np.newaxis]
        dim    = np.arange(d_model, dtype=np.float32)[np.newaxis, :]
        angles = pos / np.power(10_000.0, (2 * (dim // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])   # even → sin
        angles[:, 1::2] = np.cos(angles[:, 1::2])   # odd  → cos
        return angles[np.newaxis, :, :]              # (1, max_len, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + tf.cast(self.pe[:, :seq_len, :], x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg['max_len'] = self.max_len
        return cfg


# =============================================================================
#  ATTENTION BLOCKS  (functional helpers)
# =============================================================================

def se_block(x, ratio: int = 8, name: str = 'se'):
    """
    Squeeze-and-Excitation Block.
    Input : (batch, time, mel, channels)
    ─ Squeeze  : global-average over (time, mel)  →  (batch, 1, 1, C)
    ─ Excitation: FC(C//r, relu) → FC(C, sigmoid) →  per-channel scale
    """
    c = x.shape[-1]
    r = max(c // ratio, 4)
    sq = Lambda(lambda t: tf.reduce_mean(t, axis=[1, 2], keepdims=True),
                name=f'{name}_squeeze')(x)                   # (b,1,1,c)
    ex = Dense(r, activation='relu',    use_bias=False, name=f'{name}_fc1')(sq)
    ex = Dense(c, activation='sigmoid', use_bias=False, name=f'{name}_fc2')(ex)
    return Multiply(name=f'{name}_scale')([x, ex])


def cbam_block(x, ratio: int = 8, name: str = 'cbam'):
    """
    Convolutional Block Attention Module (CBAM).
    Input : (batch, time, mel, channels)
    Step 1 – Channel Attention : shared-MLP on avg+max pooled features.
    Step 2 – Spatial Attention : 7×7 conv on avg+max over channel axis.
    """
    c = x.shape[-1]
    r = max(c // ratio, 4)

    # ── Channel Attention ──────────────────────────────────────────────────
    avg_c = Lambda(lambda t: tf.reduce_mean(t, axis=[1, 2], keepdims=True),
                   name=f'{name}_ch_avg')(x)
    max_c = Lambda(lambda t: tf.reduce_max(t, axis=[1, 2], keepdims=True),
                   name=f'{name}_ch_max')(x)
    # Shared MLP applied independently to avg and max
    mlp_r  = lambda t, sfx: Dense(c, use_bias=False, name=f'{name}_fc2{sfx}')(
                              Dense(r, activation='relu', use_bias=False,
                                    name=f'{name}_fc1{sfx}')(t))
    ch_att = Activation('sigmoid', name=f'{name}_ch_sig')(
                Add(name=f'{name}_ch_add')([mlp_r(avg_c, 'a'), mlp_r(max_c, 'm')]))
    x = Multiply(name=f'{name}_ch_scale')([x, ch_att])

    # ── Spatial Attention ──────────────────────────────────────────────────
    avg_s = Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                   name=f'{name}_sp_avg')(x)
    max_s = Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                   name=f'{name}_sp_max')(x)
    sp = Concatenate(axis=-1, name=f'{name}_sp_cat')([avg_s, max_s])   # (b,t,m,2)
    sp = Conv2D(1, (7, 7), padding='same', activation='sigmoid',
                use_bias=False, name=f'{name}_sp_conv')(sp)
    return Multiply(name=f'{name}_sp_scale')([x, sp])


# =============================================================================
#  RESIDUAL BLOCK
# =============================================================================

def residual_block(x, filters: int, kernel_size=(3, 3), name: str = 'res'):
    """
    Standard Residual Block  (He et al., 2016).
    ─ Conv → BN → ReLU → Conv → BN → Add(skip) → ReLU
    ─ 1×1 projection on the skip path if channel dims differ.
    """
    skip = x

    # Project skip connection if in/out channels differ
    if x.shape[-1] != filters:
        skip = Conv2D(filters, (1, 1), padding='same',
                      use_bias=False, name=f'{name}_proj')(x)
        skip = BatchNormalization(name=f'{name}_proj_bn')(skip)

    # Main path
    x = Conv2D(filters, kernel_size, padding='same',
               use_bias=False, name=f'{name}_c1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('relu', name=f'{name}_r1')(x)
    x = Conv2D(filters, kernel_size, padding='same',
               use_bias=False, name=f'{name}_c2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)

    # Merge & activate
    x = Add(name=f'{name}_add')([x, skip])
    x = Activation('relu', name=f'{name}_r2')(x)
    return x


# =============================================================================
#  MODEL
# =============================================================================

def build_ser_model(input_shape, n_classes, cfg=CFG):
    """
    Build the full SER model.

    Parameters
    ----------
    input_shape : tuple – (time_frames, n_mels, 1)
    n_classes   : int
    cfg         : dict  – CFG hyperparameter dict

    Returns
    -------
    Compiled tf.keras.Model
    """
    f1, f2 = cfg['cnn_filters']
    inp = Input(shape=input_shape, name='input_logmel')

    # ── ① SpecAugment (train-only) ────────────────────────────────────────
    x_a = SpecAugment(
        time_mask_W=cfg['time_mask_W'],  freq_mask_F=cfg['freq_mask_F'],
        n_time_masks=cfg['n_time_masks'], n_freq_masks=cfg['n_freq_masks'],
        name='spec_augment',
    )(inp)

    # ── ② Local CNN branch  (3×3 – fine spectro-temporal detail) ──────────
    L = Conv2D(f1, (3, 3), padding='same', use_bias=False, name='L_stem')(x_a)
    L = BatchNormalization(name='L_stem_bn')(L)
    L = Activation('relu')(L)
    L = residual_block(L, f1, (3, 3), name='L_res1')
    L = residual_block(L, f2, (3, 3), name='L_res2')
    L = MaxPooling2D((1, 2), name='L_pool')(L)          # (T, mel//2, f2)

    # ── ③ Global CNN branch (5×5 – broad spectro-temporal context) ────────
    G = Conv2D(f1, (5, 5), padding='same', use_bias=False, name='G_stem')(x_a)
    G = BatchNormalization(name='G_stem_bn')(G)
    G = Activation('relu')(G)
    G = residual_block(G, f1, (5, 5), name='G_res1')
    G = residual_block(G, f2, (5, 5), name='G_res2')
    G = MaxPooling2D((1, 2), name='G_pool')(G)          # (T, mel//2, f2)

    # ── ④ Concatenate  →  (T, mel//2, 2*f2=128) ──────────────────────────
    x = Concatenate(axis=-1, name='concat_LG')([L, G])

    # ── ⑤ SE or CBAM ──────────────────────────────────────────────────────
    if cfg['attention'] == 'cbam':
        x = cbam_block(x, ratio=cfg['se_ratio'], name='cbam')
    else:
        x = se_block(x, ratio=cfg['se_ratio'], name='se')

    # ── ⑥ 1×1 Conv  +  FreqPool  (parameter-efficient sequence prep) ──────
    #   Old:  TD(Flatten)  → (T, mel//2 * 128)  e.g. (T, 8192) if mel=128
    #         TD(Dense256) → ~2 M params
    #   New:  Conv2D(1×1)  → (T, mel//2, 64)
    #         mean over mel → (T, 64)                 → ~8 K params total ✓
    x = Conv2D(cfg['proj_channels'], (1, 1), padding='same',
               activation='relu', use_bias=False, name='proj_1x1')(x)
    x = BatchNormalization(name='proj_bn')(x)
    # Average pooling over the mel/frequency axis: (T, mel//2, 64) → (T, 64)
    x = Lambda(lambda t: tf.reduce_mean(t, axis=2), name='freq_avg_pool')(x)
    x = Dropout(cfg['dropout_freq'], name='freq_drop')(x)

    # ── ⑦ BiLSTM  →  (T, lstm_units*2 = 256) ─────────────────────────────
    x = Bidirectional(
        LSTM(cfg['lstm_units'], return_sequences=True),
        name='bilstm',
    )(x)
    x = Dropout(cfg['dropout_rnn'], name='bilstm_drop')(x)

    # ── ⑧ Positional Encoding  +  Multi-Head Self-Attention ───────────────
    #   PE is injected HERE (after BiLSTM) so the attention layer knows
    #   the absolute position of each time step, not just relative order.
    x = PositionalEncoding(max_len=1024, name='pos_enc')(x)
    attn = MultiHeadAttention(
        num_heads=cfg['attn_heads'],
        key_dim=cfg['attn_key_dim'],
        name='mh_self_attn',
    )(x, x)                                       # query=key=value=x
    x = Add(name='attn_residual')([x, attn])      # residual connection
    x = LayerNormalization(name='attn_ln')(x)

    # ── ⑨ Temporal pooling  →  (256,) ─────────────────────────────────────
    x = GlobalAveragePooling1D(name='gap')(x)

    # ── ⑩ Dense head  →  Softmax ──────────────────────────────────────────
    x = Dense(cfg['dense_units'], activation='relu', name='dense_head')(x)
    x = Dropout(cfg['dropout_out'], name='out_drop')(x)
    out = Dense(n_classes, activation='softmax', name='softmax')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['lr']),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg['label_smooth']
        ),
        metrics=['accuracy'],
    )
    return model


# =============================================================================
#  MIXUP  GENERATOR
# =============================================================================

def mixup_generator(X, y, sample_weights, batch_size: int,
                    alpha: float = 0.3, shuffle: bool = True):
    """
    Infinite generator that yields Mixup-blended mini-batches.

    Each batch is a 3-tuple  (X_mix, y_mix, sw_mix)  so Keras can apply
    per-sample weighting derived from class_weight, even with soft labels.

    Parameters
    ----------
    X, y          : ndarray  – full training set (after to_categorical)
    sample_weights: ndarray  – per-sample weight from class_weight
    alpha         : float    – Beta(α,α) mixup coefficient; 0 = no mixing
    """
    n = len(X)
    while True:
        order = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n - batch_size + 1, batch_size):
            idx  = order[start : start + batch_size]
            lam  = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
            perm = np.random.permutation(len(idx))

            X_mix  = lam * X[idx]              + (1 - lam) * X[idx[perm]]
            y_mix  = lam * y[idx]              + (1 - lam) * y[idx[perm]]
            sw_mix = lam * sample_weights[idx] + (1 - lam) * sample_weights[idx[perm]]
            yield X_mix, y_mix, sw_mix


# =============================================================================
#  DATA LOADING & PREPROCESSING
# =============================================================================

X = np.load("X.npy")   # expected shape: (N, time_frames, n_mels)
y = np.load("y.npy")   # expected shape: (N,)  –  string or int labels

# Add channel dim for Conv2D:  (N, T, M)  →  (N, T, M, 1)
if X.ndim == 3:
    X = X[..., np.newaxis]

print(f"X : {X.shape}   (N, time_frames, n_mels, channel)")
print(f"y : {y.shape}")

# Global z-score normalisation across all training samples
mu    = X.mean(axis=(0, 1, 2), keepdims=True)
sigma = X.std(axis=(0, 1, 2),  keepdims=True) + 1e-8
X     = (X - mu) / sigma

# Encode string/int labels  →  0 … n_classes-1
le        = LabelEncoder()
y_enc     = le.fit_transform(y)
n_classes = len(le.classes_)
print(f"Classes : {le.classes_}  (n={n_classes})")


# =============================================================================
#  K-FOLD CROSS-VALIDATION
# =============================================================================

skf = StratifiedKFold(n_splits=CFG['n_splits'], shuffle=True, random_state=42)

histories      = []
all_fold_preds = []
all_fold_true  = []

# ── Global-best tracking ────────────────────────────────────────────────────
best_val_acc    = -np.inf
BEST_MODEL_PATH = 'ser_best_overall.keras'

print(f"\n{'═'*62}")
print(f"  {CFG['n_splits']}-Fold CV  │  attention={CFG['attention'].upper()}"
      f"  │  input={X.shape[1:]}")
print(f"  classes={n_classes}  │  mixup_alpha={CFG['mixup_alpha']}")
print(f"{'═'*62}\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc), 1):

    print(f"{'─'*62}")
    print(f"  FOLD {fold} / {CFG['n_splits']}")
    print(f"{'─'*62}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_cat    = to_categorical(y_enc[train_idx], n_classes)
    y_val_cat      = to_categorical(y_enc[val_idx],   n_classes)

    # ── Per-sample weights from balanced class weights ──────────────────────
    cw_arr = compute_class_weight(
        'balanced', classes=np.unique(y_enc), y=y_enc[train_idx]
    )
    cw_dict   = dict(enumerate(cw_arr))
    sw_train  = np.array([cw_dict[c] for c in y_enc[train_idx]], dtype=np.float32)

    # ── Build fresh model ───────────────────────────────────────────────────
    model = build_ser_model(X.shape[1:], n_classes, CFG)
    if fold == 1:
        model.summary()

    # ── Callbacks ───────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=8,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1,
        ),
        ModelCheckpoint(
            f'best_fold{fold}.keras',
            monitor='val_accuracy', save_best_only=True, verbose=0,
        ),
    ]

    # ── Train with Mixup generator ──────────────────────────────────────────
    #   Generator yields 3-tuples (X, y, sample_weight).
    #   Do NOT pass class_weight separately — sample_weight handles balancing.
    steps = max(len(X_train) // CFG['batch_size'], 1)

    history = model.fit(
        mixup_generator(
            X_train, y_train_cat, sw_train,
            batch_size=CFG['batch_size'],
            alpha=CFG['mixup_alpha'],
        ),
        steps_per_epoch=steps,
        validation_data=(X_val, y_val_cat),
        epochs=CFG['epochs'],
        callbacks=callbacks,
        verbose=1,
    )
    histories.append(history)

    # ── Fold evaluation ─────────────────────────────────────────────────────
    preds       = model.predict(X_val, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    all_fold_preds.append(pred_labels)
    all_fold_true.append(y_enc[val_idx])

    fold_acc = float(np.mean(pred_labels == y_enc[val_idx]))
    print(f"\n  ✓  Fold {fold}  Val Accuracy : {fold_acc*100:.2f}%")

    # ── Save globally best model ─────────────────────────────────────────────
    #   Compare fold_acc (computed on current best-weight model) with running
    #   global best.  EarlyStopping(restore_best_weights=True) guarantees the
    #   model already holds the best weights from this fold.
    if fold_acc > best_val_acc:
        best_val_acc = fold_acc
        model.save(BEST_MODEL_PATH)
        print(f"  ★  New global best — saved to '{BEST_MODEL_PATH}' "
              f"(val_acc={fold_acc*100:.2f}%)")
    print()


# =============================================================================
#  PERSIST ARTEFACTS
# =============================================================================

np.save("classes.npy", le.classes_)
np.save("norm_stats.npy", np.array([mu, sigma]))   # needed at inference time

print(f"\nSaved:")
print(f"  Best model  : {BEST_MODEL_PATH}  (val_acc={best_val_acc*100:.2f}%)")
print(f"  Classes     : classes.npy")
print(f"  Norm stats  : norm_stats.npy  (apply same normalisation at inference!)")


# =============================================================================
#  AGGREGATE EVALUATION
# =============================================================================

all_preds = np.concatenate(all_fold_preds)
all_true  = np.concatenate(all_fold_true)

overall_acc = np.mean(all_preds == all_true)

print(f"\n{'═'*62}")
print(f"  Overall CV Accuracy : {overall_acc*100:.2f}%")
print(f"{'═'*62}")
print(classification_report(all_true, all_preds, target_names=le.classes_))


# =============================================================================
#  PLOTS
# =============================================================================

def plot_confusion_matrix(true, pred, classes, save='confusion_matrix.png'):
    cm      = confusion_matrix(true, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2f'],
        ['Counts', 'Normalised'],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    linewidths=0.4, linecolor='white')
        ax.set_title(f'Confusion Matrix ({title})', fontsize=12)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)

    plt.suptitle(f'All Folds Combined  (n={len(true)})', fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save}")


def plot_histories(histories, n_splits, save='training_history.png'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    palette   = plt.cm.tab10.colors

    for i, h in enumerate(histories):
        c  = palette[i % len(palette)]
        ep = range(1, len(h.history['loss']) + 1)

        for ax, key_tr, key_val, label in zip(
            [axes[0], axes[1]],
            ['loss', 'accuracy'],
            ['val_loss', 'val_accuracy'],
            ['Loss', 'Accuracy'],
        ):
            ax.plot(ep, h.history[key_tr],  '--', color=c, alpha=0.6,
                    label=f'F{i+1} Train')
            ax.plot(ep, h.history[key_val], '-',  color=c,
                    label=f'F{i+1} Val')
            ax.set_title(label);  ax.set_xlabel('Epoch')
            ax.legend(fontsize=7, ncol=2);  ax.grid(alpha=0.3)

    plt.suptitle(
        f'Training History – {n_splits}-Fold CV  '
        f'[attention={CFG["attention"].upper()}  |  '
        f'mixup_alpha={CFG["mixup_alpha"]}]',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save}")


plot_confusion_matrix(all_true, all_preds, le.classes_)
plot_histories(histories, CFG['n_splits'])

print("\nAll done ✓")
