# # app.py

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model("../models/best_model.keras")

with open("../models/class_names.json", "r") as f:
    CLASSES = json.load(f)

print("Loaded class order:", CLASSES)

TREATMENTS = {
    "healthy": "TOMATO PLANT IS HEALTHY! Maintain proper watering, balanced fertilization, staking support, and good field sanitation.",
    "late_blight": "YELLOW LEAF CURL VIRUS detected! Immediately remove and destroy infected plants to prevent spread. Control whitefly vectors using approved insecticides or neem-based products. Install yellow sticky traps, use reflective mulches if possible, practice crop rotation, and plant virus-resistant tomato varieties.",
    "other_diseases": "Model not confident. Consult local agricultural extension service for accurate tomato disease diagnosis."
}

CONFIDENCE_THRESHOLD = 0.90
SPECIFIC_DISEASES     = ["late_blight", "healthy"]
PLOTS_DIR             = os.path.join("static", "plots")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_metrics():
    path = os.path.join(PLOTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── routes ───────────────────────────────────────────────────────────────────

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

    preds    = model.predict(x, verbose=0)[0]
    top_idx  = int(np.argmax(preds))
    top_prob = float(preds[top_idx])

    if CLASSES[top_idx] in SPECIFIC_DISEASES and top_prob > CONFIDENCE_THRESHOLD:
        predicted_class = CLASSES[top_idx]
    else:
        predicted_class = "other_diseases"

    probs_dict   = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
    late_blight_prob   = probs_dict.get("late_blight", 0.0)
    healthy_prob = probs_dict.get("healthy", 0.0)
    other_prob   = max(0.0, 1.0 - late_blight_prob - healthy_prob)

    chart_probs   = [late_blight_prob, healthy_prob, other_prob]
    chart_classes = ["late_blight", "healthy", "other_diseases"]

    treatment = TREATMENTS.get(predicted_class, "Consult expert.")
    print(f"RAW: {CLASSES[top_idx]} ({top_prob:.1%}) → FINAL: {predicted_class}")

    all_probs = {CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASSES))}

    return jsonify({
        "prediction":   predicted_class,
        "confidence":   round(top_prob * 100, 2),
        "raw_class":    CLASSES[top_idx],
        "treatment":    treatment,
        "image_url":    f"/{filepath.replace(os.sep, '/')}",
        "probabilities": chart_probs,
        "classes":       chart_classes,
        "all_probs":     all_probs,         
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
