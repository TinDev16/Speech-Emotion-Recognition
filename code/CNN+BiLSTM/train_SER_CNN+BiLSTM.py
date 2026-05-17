import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# K-Fold Configuration
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Build Model (CNN + BiLSTM)
def build_ser_model(input_shape, n_classes):
    inp = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Bidirectional(LSTM(128, return_sequences=False))(x)

    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"])
    return model

# Training with K-Fold
all_fold_preds = []
all_fold_true = []
histories = []

print(f"Starting {n_splits}-Fold Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc), 1):
    print(f"\n--- FOLD {fold} ---")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = to_categorical(y_enc[train_idx]), to_categorical(y_enc[val_idx])
    # class-weight
    cw = compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc[train_idx])
    cw = dict(enumerate(cw))

    model = build_ser_model(X_train.shape[1:], len(le.classes_))

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=2, factor=0.5, monitor='val_loss')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        callbacks=callbacks,
        class_weight=cw,
        verbose=1
    )
    histories.append(history)
    
model.save("ser_model_kfold.keras")
np.save("classes.npy", le.classes_)
print("Done!")