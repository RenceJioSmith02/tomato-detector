#app.py

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# compile=False: avoids needing to register any custom loss function
# at load time. We only use the model for inference, so this is safe.
model = load_model("../models/best_model.keras", compile=False)

with open("../models/class_names.json", "r") as f:
    CLASSES = json.load(f)

print("Loaded class order:", CLASSES)
# Expected: ['healthy', 'late_blight', 'other_diseases']

# ── Load optimized thresholds saved by training script ───────────────────────
_THRESHOLD_CONFIG_PATH = "../models/threshold_config.json"
if os.path.exists(_THRESHOLD_CONFIG_PATH):
    with open(_THRESHOLD_CONFIG_PATH) as f:
        _thresh = json.load(f)
    LATE_BLIGHT_THRESHOLD = _thresh.get("late_blight_threshold", 0.50)
    OD_THRESHOLD          = _thresh.get("od_threshold",           0.60)
    LATE_BLIGHT_IDX       = _thresh.get("late_blight_class_index", CLASSES.index("late_blight"))
    OD_IDX                = _thresh.get("od_class_index",          CLASSES.index("other_diseases"))
    HEALTHY_IDX           = _thresh.get("healthy_class_index",     CLASSES.index("healthy"))
    print(f"Loaded thresholds — Late Blight: {LATE_BLIGHT_THRESHOLD}, OD: {OD_THRESHOLD}")
else:
    # Fallback defaults if threshold_config.json not yet generated.
    # Run mobilenetv2_train.py first to produce the optimized thresholds.
    LATE_BLIGHT_THRESHOLD = 0.50
    OD_THRESHOLD          = 0.60
    LATE_BLIGHT_IDX       = CLASSES.index("late_blight")
    OD_IDX                = CLASSES.index("other_diseases")
    HEALTHY_IDX           = CLASSES.index("healthy")
    print("WARNING: threshold_config.json not found — using fallback defaults.")
    print("         Run mobilenetv2_train.py to generate optimized thresholds.")

TREATMENTS = {
    "healthy": (
        "TOMATO PLANT IS HEALTHY! "
        "Maintain proper watering, balanced fertilization, staking support, "
        "and good field sanitation."
    ),
    "late_blight": (
        "LATE BLIGHT detected! Immediately remove and destroy infected plant parts "
        "to prevent spread. Apply copper-based or mancozeb fungicides as directed. "
        "Avoid overhead irrigation, improve air circulation between plants, "
        "practice crop rotation, and consider planting blight-resistant tomato varieties."
    ),
    "other_diseases": (
        "OTHER DISEASE detected! Consult your local agricultural extension service "
        "for accurate diagnosis and treatment recommendations."
    ),
}

PLOTS_DIR = os.path.join("static", "plots")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_metrics():
    path = os.path.join(PLOTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def classify_with_thresholds(preds):
    """
    Apply training-optimized thresholds to raw model output probabilities.

    Priority order:
      1. If late_blight score  >= LATE_BLIGHT_THRESHOLD → late_blight
      2. If healthy score      >= OD_THRESHOLD           → healthy
      3. Else                                             → other_diseases  (last resort)
    """
    late_blight_score = float(preds[LATE_BLIGHT_IDX])
    healthy_score     = float(preds[HEALTHY_IDX])

    if late_blight_score >= LATE_BLIGHT_THRESHOLD:
        return "late_blight", late_blight_score
    elif healthy_score >= OD_THRESHOLD:
        return "healthy", healthy_score
    else:
        od_score = float(preds[OD_IDX])
        return "other_diseases", od_score


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/plots')
def plots():
    metrics = load_metrics()
    return render_template('plots.html', metrics=metrics)


@app.route('/api/metrics')
def api_metrics():
    metrics = load_metrics()
    if metrics is None:
        return jsonify({"error": "No metrics found. Run training first."}), 404
    return jsonify(metrics)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    x   = image.img_to_array(img)
    x   = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]

    # Threshold-based classification
    # Priority: late_blight -> other_diseases -> healthy
    predicted_class, decision_score = classify_with_thresholds(preds)

    # Build full probability dict keyed by class name
    probs_dict = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}

    # Chart data in fixed display order for frontend consistency
    chart_classes = ["late_blight", "healthy", "other_diseases"]
    chart_probs   = [
        probs_dict.get("late_blight",      0.0),
        probs_dict.get("healthy",          0.0),
        probs_dict.get("other_diseases",   0.0),
    ]

    # Raw argmax for logging / transparency
    raw_idx   = int(np.argmax(preds))
    raw_class = CLASSES[raw_idx]
    raw_conf  = float(preds[raw_idx])

    treatment = TREATMENTS.get(predicted_class, "Consult an agricultural expert.")

    print(
        f"RAW argmax: {raw_class} ({raw_conf:.1%}) | "
        f"LB={preds[LATE_BLIGHT_IDX]:.1%} (thresh={LATE_BLIGHT_THRESHOLD}) | "
        f"Healthy={preds[HEALTHY_IDX]:.1%} (thresh={OD_THRESHOLD}) | "
        f"OD={preds[OD_IDX]:.1%} (last resort) | "
        f"→ FINAL: {predicted_class}"
    )

    all_probs = {CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASSES))}

    return jsonify({
        "prediction":     predicted_class,
        "confidence":     round(decision_score * 100, 2),
        "raw_class":      raw_class,
        "raw_confidence": round(raw_conf * 100, 2),
        "treatment":      treatment,
        "image_url":      f"/{filepath.replace(os.sep, '/')}",
        "probabilities":  chart_probs,
        "classes":        chart_classes,
        "all_probs":      all_probs,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

    