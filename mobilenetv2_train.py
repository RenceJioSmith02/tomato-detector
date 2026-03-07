"""
Tomato Disease Detector – CPU-Optimized Training Script
Hardware target : CPU only (no GPU)
Dataset         : 3 classes  –  healthy / late_blight / other_diseases
Priority        : Maximize overall accuracy

Key techniques  :
  • EfficientNetV2S backbone (better than MobileNetV2)
  • CPU threading tuned to use all available cores
  • float32 throughout  (float16 is slower on CPU – no tensor cores)
  • Image size 224×224  (smaller = faster on CPU, diminishing returns at 260)
  • Batch size 32       (sweet spot for CPU RAM and gradient quality)
  • Oversampling late_blight 3× to fix class imbalance
  • Label-smoothed cross-entropy (stable + handles majority-class overconfidence)
  • Cosine-decay LR for fine-tuning (prevents the accuracy regression you saw)
  • Test-Time Augmentation  (3 passes – balanced speed vs accuracy on CPU)
  • Per-class ROC curves
"""

import os, json, random, shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
    precision_score, recall_score, f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════════════════════════
# 0.  CPU THREADING SETUP
#     Tells TensorFlow to use ALL available CPU cores.
#     Must be called before any other TF operations.
# ══════════════════════════════════════════════════════════════════
cpu_cores = os.cpu_count() or 4
tf.config.threading.set_intra_op_parallelism_threads(cpu_cores)   # ops within a single op
tf.config.threading.set_inter_op_parallelism_threads(cpu_cores)   # ops across the graph
print(f"CPU cores detected: {cpu_cores}  →  TF threading set to {cpu_cores}")

# Confirm no GPU is being used
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs available: {gpus if gpus else 'None – running on CPU as expected'}")

# ══════════════════════════════════════════════════════════════════
# 1.  CONFIG
# ══════════════════════════════════════════════════════════════════
DATA_DIR         = "dataset"
IMAGE_SIZE       = (224, 224)    # Smaller than GPU version → faster on CPU
BATCH_SIZE       = 32            # CPU sweet spot

INITIAL_EPOCHS   = 40            # Head-only training
FINE_TUNE_EPOCHS = 30            # Fine-tune window
FINE_TUNE_AT     = 100           # Unfreeze base layers from this index onward

LR_HEAD          = 1e-3
LR_FINE          = 5e-5          # Starting LR for cosine decay
TTA_STEPS        = 3             # 3 passes on CPU (7 would be too slow)

PLOTS_DIR = os.path.join("web", "static", "plots")
os.makedirs("models",  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {os.path.join(PLOTS_DIR, name)}")

# ══════════════════════════════════════════════════════════════════
# 2.  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════
random.seed(42); np.random.seed(42); tf.random.set_seed(42)

# ══════════════════════════════════════════════════════════════════
# 3.  LOAD DATASETS
# ══════════════════════════════════════════════════════════════════
AUTOTUNE = tf.data.AUTOTUNE

def load_split(split):
    return tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=(split == "train"),
        seed=123,
    )

train_ds_raw = load_split("train")
val_ds_raw   = load_split("val")
test_ds_raw  = load_split("test")

class_names  = train_ds_raw.class_names
NUM_CLASSES  = len(class_names)
print("Classes:", class_names)

# Cache validation and test sets – they never change so caching saves re-reading disk
val_ds  = val_ds_raw.cache().prefetch(AUTOTUNE)
test_ds = test_ds_raw.cache().prefetch(AUTOTUNE)

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

# ══════════════════════════════════════════════════════════════════
# 4.  CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════
train_labels_raw = np.concatenate(
    [y.numpy() for _, y in train_ds_raw], axis=0
)
train_idx = np.argmax(train_labels_raw, axis=1)

cw = compute_class_weight("balanced", classes=np.unique(train_idx), y=train_idx)
class_weight_dict = dict(enumerate(cw))

# Extra boost for the minority class on top of balanced weights
lb_idx = class_names.index("late_blight")
class_weight_dict[lb_idx] *= 1.5
print("Class weights:", {class_names[k]: round(v, 3)
                          for k, v in class_weight_dict.items()})

# ══════════════════════════════════════════════════════════════════
# 5.  OVERSAMPLE late_blight  (3× in the training stream)
# ══════════════════════════════════════════════════════════════════
def oversample_class(raw_ds, class_names, target="late_blight", multiplier=3):
    """Duplicate the target class `multiplier` times and shuffle into training."""
    idx      = class_names.index(target)
    base     = raw_ds.unbatch()
    minority = base.filter(lambda _, lbl: tf.equal(tf.argmax(lbl), idx))
    oversampled = minority
    for _ in range(multiplier - 1):
        oversampled = oversampled.concatenate(minority)
    return (base.concatenate(oversampled)
                .shuffle(4096, seed=42)
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE))

train_ds = oversample_class(train_ds_raw, class_names, "late_blight", 3)

# Verify new distribution
balanced_idx = np.argmax(
    np.concatenate([y.numpy() for _, y in train_ds], axis=0), axis=1
)
unique, counts = np.unique(balanced_idx, return_counts=True)
print("\nBalanced training distribution:")
for u, c in zip(unique, counts):
    print(f"  {class_names[u]}: {c} samples")

# ══════════════════════════════════════════════════════════════════
# 6.  AUGMENTATION  (inside model → fused with forward pass on CPU)
# ══════════════════════════════════════════════════════════════════
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.35),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.35),
    layers.RandomBrightness(0.35),
    layers.RandomTranslation(0.2, 0.2),
], name="augmentation")

# ══════════════════════════════════════════════════════════════════
# 7.  BUILD MODEL  –  EfficientNetV2S
#     include_preprocessing=True  →  no manual preprocess_input call needed
# ══════════════════════════════════════════════════════════════════
base_model = tf.keras.applications.EfficientNetV2S(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
    include_preprocessing=True,
)
base_model.trainable = False

inputs  = keras.Input(shape=(*IMAGE_SIZE, 3))
x       = augmentation(inputs)
x       = base_model(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.5)(x)
x       = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
x       = layers.Activation("relu")(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.3)(x)
x       = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
x       = layers.Activation("relu")(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)  # float32 throughout

model = keras.Model(inputs, outputs)
model.summary(line_length=90)

# ══════════════════════════════════════════════════════════════════
# 8.  LOSS  –  Label-smoothed cross-entropy
#     Prevents overconfidence on the majority class.
#     Numerically stable on CPU (no float16 underflow risk).
# ══════════════════════════════════════════════════════════════════
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ══════════════════════════════════════════════════════════════════
# 9.  METRICS CALLBACK
# ══════════════════════════════════════════════════════════════════
class MetricsCallback(tf.keras.callbacks.Callback):
    """Logs weighted precision / recall / F1 on val set each epoch."""
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data
        self.precisions, self.recalls, self.f1s = [], [], []

    def on_epoch_end(self, epoch, logs=None):
        yt, yp = [], []
        for imgs, lbls in self.val_data:
            preds = self.model.predict(imgs, verbose=0)
            yt.extend(np.argmax(lbls.numpy(), 1))
            yp.extend(np.argmax(preds, 1))
        p = precision_score(yt, yp, average="weighted", zero_division=0)
        r = recall_score(yt, yp,    average="weighted", zero_division=0)
        f = f1_score(yt, yp,        average="weighted", zero_division=0)
        self.precisions.append(p); self.recalls.append(r); self.f1s.append(f)
        print(f"  val  P={p:.4f}  R={r:.4f}  F1={f:.4f}")

# ══════════════════════════════════════════════════════════════════
# 10.  CALLBACK FACTORY
# ══════════════════════════════════════════════════════════════════
def make_callbacks(ckpt="models/best_model.keras", patience=12):
    return [
        ModelCheckpoint(ckpt, save_best_only=True,
                        monitor="val_accuracy", mode="max", verbose=1),
        EarlyStopping(patience=patience, restore_best_weights=True,
                      monitor="val_accuracy", mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=5, min_lr=1e-8, verbose=1),
        MetricsCallback(val_ds),
    ]

# ══════════════════════════════════════════════════════════════════
# 11.  STAGE 1 – TRAIN HEAD  (base model frozen)
# ══════════════════════════════════════════════════════════════════
model.compile(
    optimizer=keras.optimizers.Adam(LR_HEAD),
    loss=loss_fn,
    metrics=["accuracy"],
)
print("\n═══ Stage 1 : Head training (base frozen) ═══")
cb_head  = make_callbacks("models/best_head.keras", patience=12)
history1 = model.fit(
    train_ds,
    epochs=INITIAL_EPOCHS,
    validation_data=val_ds,
    callbacks=cb_head,
    class_weight=class_weight_dict,
    verbose=1,
)
head_epochs = len(history1.history["accuracy"])
print(f"Head training stopped at epoch {head_epochs}")

# ══════════════════════════════════════════════════════════════════
# 12.  STAGE 2 – FINE-TUNE  (unfreeze top layers of backbone)
# ══════════════════════════════════════════════════════════════════
print("\n═══ Stage 2 : Fine-tuning ═══")
base_model.trainable = True

# Freeze early layers (low-level edge/texture detectors – not useful to retrain)
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# Always keep BatchNorm frozen – preserves pretrained running statistics
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

n_trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"Trainable base layers: {n_trainable} / {len(base_model.layers)}")

# Cosine decay: smoothly reduces LR from LR_FINE → ~0 over fine-tune window
# Prevents the accuracy regression caused by a flat LR that was too small (3e-6)
steps_per_epoch = len(train_ds)
total_ft_steps  = FINE_TUNE_EPOCHS * steps_per_epoch
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR_FINE,
    decay_steps=total_ft_steps,
    alpha=1e-8,
)

model.compile(
    optimizer=keras.optimizers.Adam(lr_schedule),
    loss=loss_fn,
    metrics=["accuracy"],
)
cb_fine  = make_callbacks("models/best_finetune.keras", patience=12)
history2 = model.fit(
    train_ds,
    epochs=head_epochs + FINE_TUNE_EPOCHS,
    initial_epoch=head_epochs,
    validation_data=val_ds,
    callbacks=cb_fine,
    class_weight=class_weight_dict,
    verbose=1,
)

# ══════════════════════════════════════════════════════════════════
# 13.  SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════
model.save("models/final_model.keras")
best_src = "models/best_finetune.keras"
if os.path.exists(best_src):
    shutil.copy(best_src, "models/best_model.keras")
    model.load_weights("models/best_model.keras")
    print("Best fine-tuned weights loaded.")

# ══════════════════════════════════════════════════════════════════
# 14.  TEST-TIME AUGMENTATION  (TTA – 3 passes on CPU)
#      Averages softmax predictions over N augmented copies of each image.
#      No @tf.function decorator here – plain Python loop is fine on CPU
#      and avoids graph-compilation overhead for small batch counts.
# ══════════════════════════════════════════════════════════════════
print(f"\nRunning TTA ({TTA_STEPS} passes) on test set…")

tta_augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name="tta_aug")

y_true_all, y_pred_all, y_score_all = [], [], []

for images, labels in test_ds:
    # Accumulate predictions over TTA_STEPS passes
    batch_preds = np.zeros((images.shape[0], NUM_CLASSES), dtype=np.float32)
    for _ in range(TTA_STEPS):
        aug_imgs    = tta_augment(images, training=True)
        batch_preds += model.predict(aug_imgs, verbose=0)
    batch_preds /= TTA_STEPS

    y_score_all.extend(batch_preds)
    y_true_all.extend(np.argmax(labels.numpy(), 1))
    y_pred_all.extend(np.argmax(batch_preds,    1))

y_true  = np.array(y_true_all)
y_pred  = np.array(y_pred_all)
y_score = np.array(y_score_all)

# Standard (non-TTA) accuracy for comparison
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
tta_acc = float(np.mean(y_true == y_pred))
print(f"\nStandard accuracy : {test_acc:.4f}")
print(f"TTA accuracy      : {tta_acc:.4f}  (Δ {tta_acc - test_acc:+.4f})")

# ══════════════════════════════════════════════════════════════════
# 15.  METRICS
# ══════════════════════════════════════════════════════════════════
prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec_w  = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
f1_w   = f1_score(y_true, y_pred,        average="weighted", zero_division=0)

report  = classification_report(y_true, y_pred,
                                  target_names=class_names, output_dict=True)
prec_pc = [report[c]["precision"] for c in class_names]
rec_pc  = [report[c]["recall"]    for c in class_names]
f1_pc   = [report[c]["f1-score"]  for c in class_names]

# ══════════════════════════════════════════════════════════════════
# 16.  PLOTS
# ══════════════════════════════════════════════════════════════════
train_acc  = history1.history["accuracy"]     + history2.history["accuracy"]
val_acc    = history1.history["val_accuracy"] + history2.history["val_accuracy"]
train_loss = history1.history["loss"]         + history2.history["loss"]
val_loss   = history1.history["val_loss"]     + history2.history["val_loss"]
ep_range   = range(1, len(train_acc) + 1)

# — Plot 1: Training history ————————————————————————————————————
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
for ax, tr, va, ylabel, title in [
    (ax1, train_loss, val_loss, "Loss",     "Loss"),
    (ax2, train_acc,  val_acc,  "Accuracy", "Accuracy"),
]:
    ax.plot(ep_range, tr, label="Train", linewidth=2)
    ax.plot(ep_range, va, label="Val",   linewidth=2)
    ax.axvline(head_epochs, color="red", linestyle="--",
               linewidth=1.2, label="Fine-tune start")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
savefig("training_history.png")

# — Plot 2: Confusion matrix ————————————————————————————————————
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix (TTA)")
plt.xticks(rotation=45); plt.yticks(rotation=0)
savefig("confusion_matrix.png")

# — Plot 3: Per-class metrics ———————————————————————————————————
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ["salmon" if c == "late_blight" else "steelblue" for c in class_names]
for ax, data, name in zip(axes,
                            [prec_pc, rec_pc, f1_pc],
                            ["Precision", "Recall", "F1-Score"]):
    ax.bar(class_names, data, color=colors, edgecolor="navy", alpha=0.85)
    ax.set_title(f"{name} per Class"); ax.set_ylabel(name); ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(data):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center",
                fontsize=9, fontweight="bold")
savefig("per_class_metrics.png")

# — Plot 4: Support distribution ————————————————————————————————
support = np.bincount(y_true)
plt.figure(figsize=(8, 5))
plt.bar(class_names, support, color="lightcoral", edgecolor="darkred", alpha=0.8)
plt.title("Test Samples per Class")
plt.xlabel("Class"); plt.ylabel("Count")
plt.xticks(rotation=30); plt.grid(axis="y", alpha=0.3)
for i, v in enumerate(support):
    plt.text(i, v + 0.4, str(v), ha="center", fontweight="bold")
savefig("support_distribution.png")

# — Plot 5: P / R / F1 over epochs ————————————————————————————
mc_head = cb_head[-1]
mc_fine = cb_fine[-1]
all_p   = mc_head.precisions + mc_fine.precisions
all_r   = mc_head.recalls    + mc_fine.recalls
all_f   = mc_head.f1s        + mc_fine.f1s
if all_p:
    er = range(1, len(all_p) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(er, all_p, label="Precision", linewidth=2, marker="o", markersize=4)
    plt.plot(er, all_r, label="Recall",    linewidth=2, marker="s", markersize=4)
    plt.plot(er, all_f, label="F1-Score",  linewidth=2, marker="^", markersize=4)
    plt.axvline(head_epochs, color="red", linestyle="--", label="Fine-tune start")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.title("Precision / Recall / F1 Over Epochs")
    plt.legend(); plt.grid(alpha=0.3)
    savefig("prf1_over_epochs.png")

# — Plot 6: Per-class ROC curves ————————————————————————————————
plt.figure(figsize=(9, 7))
roc_aucs   = {}
colors_roc = plt.cm.tab10(np.linspace(0, 0.5, NUM_CLASSES))
for i, (cls, col) in enumerate(zip(class_names, colors_roc)):
    fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
    roc_aucs[cls] = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=col, linewidth=2,
             label=f"{cls}  (AUC = {roc_aucs[cls]:.3f})")
# Macro average
all_fpr  = np.unique(np.concatenate(
    [roc_curve(y_true == i, y_score[:, i])[0] for i in range(NUM_CLASSES)]
))
mean_tpr = np.mean(
    [np.interp(all_fpr, *roc_curve(y_true == i, y_score[:, i])[:2])
     for i in range(NUM_CLASSES)], axis=0
)
roc_auc_mean = auc(all_fpr, mean_tpr)
plt.plot(all_fpr, mean_tpr, "k--", linewidth=2.5,
         label=f"Macro avg  (AUC = {roc_auc_mean:.3f})")
plt.plot([0, 1], [0, 1], "grey", linewidth=0.8, alpha=0.5)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves per Class")
plt.legend(loc="lower right"); plt.grid(alpha=0.3)
savefig("roc_curve.png")

# — Plot 7: Overall summary ————————————————————————————————————
metric_vals   = [tta_acc, prec_w, rec_w, f1_w, roc_auc_mean]
metric_labels = ["Accuracy\n(TTA)", "Precision", "Recall", "F1-Score", "ROC-AUC"]
plt.figure(figsize=(10, 5))
bars = plt.bar(metric_labels, metric_vals,
               color="gold", edgecolor="darkorange", linewidth=2, alpha=0.85)
plt.ylim(0, 1.1); plt.ylabel("Score")
plt.title("Overall Performance Summary (TTA)")
plt.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, metric_vals):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
             f"{v:.3f}", ha="center", fontweight="bold")
savefig("overall_performance.png")

# ══════════════════════════════════════════════════════════════════
# 17.  SAVE METRICS JSON
# ══════════════════════════════════════════════════════════════════
metrics_json = {
    "test_accuracy":           round(tta_acc, 4),
    "standard_accuracy":       round(float(test_acc), 4),
    "precision":               round(float(prec_w), 4),
    "recall":                  round(float(rec_w), 4),
    "f1_score":                round(float(f1_w), 4),
    "roc_auc":                 round(float(roc_auc_mean), 4),
    "roc_auc_per_class":       {c: round(v, 4) for c, v in roc_aucs.items()},
    "class_names":             class_names,
    "confusion_matrix":        cm.tolist(),
    "precision_per_class":     [round(v, 4) for v in prec_pc],
    "recall_per_class":        [round(v, 4) for v in rec_pc],
    "f1_per_class":            [round(v, 4) for v in f1_pc],
    "support":                 support.tolist(),
    "tta_steps":               TTA_STEPS,
    "plots": [
        "training_history.png", "confusion_matrix.png",
        "per_class_metrics.png", "support_distribution.png",
        "prf1_over_epochs.png", "roc_curve.png", "overall_performance.png",
    ],
}
with open(os.path.join(PLOTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=2)

# ══════════════════════════════════════════════════════════════════
# 18.  FINAL CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════
macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
macro_r = recall_score(y_true, y_pred,    average="macro", zero_division=0)
macro_f = f1_score(y_true, y_pred,        average="macro", zero_division=0)
total   = len(y_true)

print("\n" + "═"*72)
print("  FINAL PERFORMANCE SUMMARY  (TTA predictions)")
print("═"*72)
print(f"  Standard Accuracy : {test_acc:.4f}")
print(f"  TTA Accuracy      : {tta_acc:.4f}  ← use this figure")
print(f"  Weighted Precision: {prec_w:.4f}")
print(f"  Weighted Recall   : {rec_w:.4f}")
print(f"  Weighted F1-Score : {f1_w:.4f}")
print(f"  ROC-AUC (macro)   : {roc_auc_mean:.4f}")
print("═"*72)

print(f"\n  {'Class':<25} {'Precision':>10} {'Recall':>10}"
      f" {'F1':>10} {'Support':>10}")
print("  " + "─"*65)
for i, cls in enumerate(class_names):
    flag = "  ◄ minority" if cls == "late_blight" else ""
    print(f"  {cls:<25} {prec_pc[i]:>10.4f} {rec_pc[i]:>10.4f}"
          f" {f1_pc[i]:>10.4f} {support[i]:>10}{flag}")
print("  " + "─"*65)
print(f"  {'Accuracy':<25} {'':>10} {'':>10} {tta_acc:>10.4f} {total:>10}")
print(f"  {'Macro avg.':<25} {macro_p:>10.4f} {macro_r:>10.4f}"
      f" {macro_f:>10.4f} {total:>10}")
print(f"  {'Weighted avg.':<25} {prec_w:>10.4f} {rec_w:>10.4f}"
      f" {f1_w:>10.4f} {total:>10}")
print("═"*72)
print("\n  Models → models/best_model.keras  |  models/final_model.keras")
print(f"  Plots  → {PLOTS_DIR}")
print("\n  Tip: training on CPU will take a while – consider running overnight.")

