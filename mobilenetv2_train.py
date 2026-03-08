import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ==========================================
# CONFIG
# ==========================================
DATA_DIR = "dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 25
FINE_TUNE_EPOCHS = 15 
FINE_TUNE_AT = 150 
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 3e-6

os.makedirs("models", exist_ok=True)

# ── Save plots here so Flask can serve them ──────────────────────────────────
PLOTS_DIR = os.path.join("web", "static", "plots") 
os.makedirs(PLOTS_DIR, exist_ok=True)

def savefig(name):
    """Tight-save and close the current figure."""
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {os.path.join(PLOTS_DIR, name)}")

# ==========================================
# LOAD DATASETS
# ==========================================

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=True, seed=123
)
val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
)
test_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
)

class_names = train_ds_raw.class_names
NUM_CLASSES = len(class_names)
print("Class order:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

# ==========================================
# CLASS WEIGHTS
# ==========================================
train_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
train_labels = np.argmax(train_labels, axis=1)
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ==========================================
# DATA AUGMENTATION
# ==========================================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2, 0.2),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.3),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomWidth(0.1),
    layers.RandomHeight(0.1),
])

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                       kernel_regularizer=regularizers.l2(1e-5))(x)
model = keras.Model(inputs, outputs)

# ==========================================
# CUSTOM CALLBACK
# ==========================================
class PrecisionRecallCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data
        self.precisions, self.recalls, self.f1_scores = [], [], []

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for images, labels in self.val_data:
            preds = self.model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        self.precisions.append(p); self.recalls.append(r); self.f1_scores.append(f)
        print(f"\nEpoch {epoch+1} – Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")


def make_callbacks(patience=7):
    return [
        ModelCheckpoint("models/best_model.keras", save_best_only=True,
                        monitor="val_loss", mode="min", verbose=1),
        EarlyStopping(patience=patience,  
                      restore_best_weights=True, 
                      monitor="val_loss", mode="min"),
        ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.4, 
                          patience=4, min_lr=1e-7, verbose=1),
        PrecisionRecallCallback(val_ds)
    ]


# ==========================================
# STAGE 1: TRAIN HEAD
# ==========================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
    loss="categorical_crossentropy", metrics=["accuracy"]
)
print("=== Training Head ===")
callbacks_head = make_callbacks(patience=5)
history1 = model.fit(
    train_ds, epochs=INITIAL_EPOCHS, validation_data=val_ds,
    callbacks=callbacks_head, class_weight=class_weights, verbose=1
)


# ==========================================
# STAGE 2: FINE-TUNE
# ==========================================
print("=== Fine-tuning ===")
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
    loss="categorical_crossentropy", metrics=["accuracy"]
)
callbacks_finetune = make_callbacks(patience=5)
history2 = model.fit(
    train_ds, epochs=FINE_TUNE_EPOCHS, validation_data=val_ds,
    callbacks=callbacks_finetune, class_weight=class_weights,
    initial_epoch=len(history1.history['accuracy']), verbose=1
)


model.save("models/final_model.keras")
model.save("models/best_model.keras")

print("\n=== Final Test Evaluation ===")
test_results = model.evaluate(test_ds, verbose=0)
print(f"Test Loss: {test_results[0]:.4f}, Test Accuracy: {test_results[1]:.4f}")

# ==========================================
# PREDICTIONS
# ==========================================
print("\nGenerating predictions for evaluation...")
y_true, y_pred, y_score = [], [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_score.extend(preds)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true  = np.array(y_true)
y_pred  = np.array(y_pred)
y_score = np.array(y_score)

precision_val = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall_val    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1_val        = f1_score(y_true, y_pred, average='weighted', zero_division=0)

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
precision_per_class = [report[c]['precision'] for c in class_names]
recall_per_class    = [report[c]['recall']    for c in class_names]
f1_per_class        = [report[c]['f1-score']  for c in class_names]

# ==========================================
# PLOT 1 – Training History
# ==========================================
def concat_history(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

train_acc  = concat_history(history1, history2, 'accuracy')
val_acc    = concat_history(history1, history2, 'val_accuracy')
train_loss = concat_history(history1, history2, 'loss')
val_loss   = concat_history(history1, history2, 'val_loss')
epochs_range = range(1, len(train_acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Training Loss',      linewidth=2)
plt.plot(epochs_range, val_loss,   label='Validation Loss',    linewidth=2)
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, label='Training Accuracy',   linewidth=2)
plt.plot(epochs_range, val_acc,   label='Validation Accuracy', linewidth=2)
plt.xlabel('Epochs'); plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(); plt.grid(True, alpha=0.3)
savefig("training_history.png")

# ==========================================
# PLOT 2 – Confusion Matrix
# ==========================================
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label"); plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.xticks(rotation=45); plt.yticks(rotation=0)
savefig("confusion_matrix.png")

# ==========================================
# PLOT 3 – Per-Class Metrics
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, data, name in zip(axes,
                           [precision_per_class, recall_per_class, f1_per_class],
                           ['Precision', 'Recall', 'F1-Score']):
    ax.bar(class_names, data, alpha=0.7, color='skyblue', edgecolor='navy')
    ax.set_title(f'{name} per Class')
    ax.set_ylabel(name); ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
savefig("per_class_metrics.png")

# ==========================================
# PLOT 4 – Support Distribution
# ==========================================
support = np.bincount(y_true)
plt.figure(figsize=(10, 6))
plt.bar(class_names, support, alpha=0.7, color='lightcoral', edgecolor='darkred')
plt.title("Support (Number of Test Samples per Class)")
plt.xlabel("Classes"); plt.ylabel("Number of Images")
plt.xticks(rotation=45); plt.grid(axis='y', alpha=0.3)
savefig("support_distribution.png")

# ==========================================
# PLOT 5 – Precision / Recall / F1 Over Epochs
# ==========================================
pr_head = callbacks_head[-1]
pr_fine = callbacks_finetune[-1]

all_precisions = pr_head.precisions + pr_fine.precisions
all_recalls    = pr_head.recalls    + pr_fine.recalls
all_f1s        = pr_head.f1_scores  + pr_fine.f1_scores

if len(all_precisions) > 0:
    er = range(1, len(all_precisions) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(er, all_precisions, label="Precision", linewidth=2, marker='o')
    plt.plot(er, all_recalls,    label="Recall",    linewidth=2, marker='s')
    plt.plot(er, all_f1s,        label="F1-Score",  linewidth=2, marker='^')
    plt.xlabel("Epochs"); plt.ylabel("Score")
    plt.title("Precision, Recall, and F1-Score Over Epochs")
    plt.legend(); plt.grid(True, alpha=0.3)
    savefig("prf1_over_epochs.png")

# ==========================================
# PLOT 6 – ROC Curve
# ==========================================
plt.figure(figsize=(8, 6))
if NUM_CLASSES > 2:
    fpr_d, tpr_d, roc_auc_d = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr_d[i], tpr_d[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc_d[i] = auc(fpr_d[i], tpr_d[i])
    all_fpr  = np.unique(np.concatenate([fpr_d[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr_d[i], tpr_d[i])
    mean_tpr /= NUM_CLASSES
    roc_auc_mean = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr,
             label=f'Micro-average ROC (AUC = {roc_auc_mean:.4f})', linewidth=2)
else:
    fpr_b, tpr_b, _ = roc_curve(y_true, y_score[:, 1])
    roc_auc_mean = auc(fpr_b, tpr_b)
    plt.plot(fpr_b, tpr_b, label=f'ROC (AUC = {roc_auc_mean:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Micro-average)')
plt.legend(); plt.grid(True, alpha=0.3)
savefig("roc_curve.png")

# ==========================================
# PLOT 7 – Overall Performance Summary
# ==========================================
metrics_values = [precision_val, recall_val, f1_val, roc_auc_mean]
metric_labels  = ["Precision", "Recall", "F1-Score", "ROC-AUC"]

plt.figure(figsize=(10, 6))
bars = plt.bar(metric_labels, metrics_values,
               alpha=0.7, color='gold', edgecolor='orange', linewidth=2)
plt.ylim(0, 1)
plt.title(""); plt.ylabel("Score")
plt.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
savefig("overall_performance.png")

# ==========================================
# SAVE METRICS JSON (used by Flask /plots page)
# ==========================================
metrics_json = {
    "test_accuracy":  round(float(test_results[1]), 4),
    "precision":      round(float(precision_val), 4),
    "recall":         round(float(recall_val), 4),
    "f1_score":       round(float(f1_val), 4),
    "roc_auc":        round(float(roc_auc_mean), 4),
    "class_names":    class_names,
    "confusion_matrix": cm.tolist(),
    "precision_per_class": [round(v, 4) for v in precision_per_class],
    "recall_per_class":    [round(v, 4) for v in recall_per_class],
    "f1_per_class":        [round(v, 4) for v in f1_per_class],
    "support":             support.tolist(),
    "plots": [
        "training_history.png",
        "confusion_matrix.png",
        "per_class_metrics.png",
        "support_distribution.png",
        "prf1_over_epochs.png",
        "roc_curve.png",
        "overall_performance.png"
    ]
}

with open(os.path.join(PLOTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=2)

print(f"\nAll plots saved to: {PLOTS_DIR}")

# Final console summary
print("\n" + "="*70)
print("FINAL PERFORMANCE SUMMARY")
print("="*70)
print(f"Test Accuracy:        {test_results[1]:.4f}")
print(f"Weighted Precision:   {precision_val:.4f}")
print(f"Weighted Recall:      {recall_val:.4f}")
print(f"Weighted F1-Score:    {f1_val:.4f}")
print(f"ROC-AUC (micro):      {roc_auc_mean:.4f}")
print("="*70)


# ==========================================
# PER-CLASS PERFORMANCE SUMMARY (Table)
# ==========================================
print("\n" + "="*70)
print("PER-CLASS PERFORMANCE SUMMARY")
print("="*70)
print(f"{'Classes':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print("-"*70)
for i, cls in enumerate(class_names):
    print(f"{cls:<25} {precision_per_class[i]:>10.2f} {recall_per_class[i]:>10.2f} {f1_per_class[i]:>10.2f} {support[i]:>10}")
print("-"*70)

# Macro and weighted averages
macro_precision = precision_score(y_true, y_pred, average='macro',    zero_division=0)
macro_recall    = recall_score(y_true, y_pred,    average='macro',    zero_division=0)
macro_f1        = f1_score(y_true, y_pred,        average='macro',    zero_division=0)
weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
weighted_recall    = recall_score(y_true, y_pred,    average='weighted', zero_division=0)
weighted_f1        = f1_score(y_true, y_pred,        average='weighted', zero_division=0)
total_support = len(y_true)

print(f"\n{'Accuracy':<25} {'':>10} {'':>10} {test_results[1]:>10.2f} {total_support:>10}")
print(f"{'Macro avg.':<25} {macro_precision:>10.2f} {macro_recall:>10.2f} {macro_f1:>10.2f} {total_support:>10}")
print(f"{'Weighted avg.':<25} {weighted_precision:>10.2f} {weighted_recall:>10.2f} {weighted_f1:>10.2f} {total_support:>10}")
print("="*70)


print("\nTraining complete! Model saved as 'models/best_model.keras'")
print("All evaluation plots saved and ready for Flask app.")

