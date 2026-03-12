from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import json
import os
from utils.preprocess import preprocess_image
from utils.image_quality import check_image_quality

app = Flask(__name__)
CORS(app)

# 🔥 LOAD FULL MODEL (NO WARNINGS ANYMORE)
model = tf.keras.models.load_model(
    "model/hybrid_skin_cancer_full_model.keras",
    compile=False
)

# Load class labels
with open("model/class_labels.json", "r") as f:
    class_labels = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]
        temp_path = "temp.jpg"
        image.save(temp_path)

        valid, message = check_image_quality(temp_path)
        if not valid:
            os.remove(temp_path)
            return jsonify({"error": message}), 400

        img_array = preprocess_image(temp_path)
        preds = model.predict(img_array)[0]
        index = int(preds.argmax())
        confidence = float(preds[index]) * 100
        label = class_labels[str(index)]

        os.remove(temp_path)

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
