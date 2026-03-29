import tensorflow as tf
import numpy as np
import cv2
import os
from config import *
from utils.gradcam import make_gradcam


class SkinCancerPredictor:
    def __init__(self, model_path):
        print("🔄 Loading model from:", model_path)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully.")

    def preprocess(self, image_path):
        if not os.path.exists(image_path):
            return None, "file_missing"

        img = cv2.imread(image_path)
        if img is None:
            return None, "opencv_failed"

        # Reject fully black images
        if np.mean(img) < 10:
            return None, "black_image"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return img, None

    def predict(self, image_path):
        img, error = self.preprocess(image_path)

        if error == "file_missing":
            return {"error": "File not found"}
        if error == "opencv_failed":
            return {"error": "Invalid image file"}
        if error == "black_image":
            return {"error": "Please upload a proper lesion image"}

        # -----------------------------
        # Prediction
        # -----------------------------
        preds = self.model.predict(img, verbose=0)[0]
        predicted_index = int(np.argmax(preds))
        predicted_class = CLASSES[predicted_index]
        confidence = float(preds[predicted_index])

        # -----------------------------
        # Cancer probability aggregation
        # -----------------------------
        cancer_probability = sum(
            preds[i] for i, cls in enumerate(CLASSES)
            if cls in CANCER_CLASSES
        )

        # -----------------------------
        # Risk logic
        # -----------------------------
        if cancer_probability > CANCER_ALERT_THRESHOLD:
            decision = "Cancer Suspected"
            risk = "High"
        elif confidence < LOW_CONFIDENCE_THRESHOLD:
            decision = "Uncertain - Needs Clinical Review"
            risk = "Medium"
        else:
            decision = "Non-Cancer"
            risk = "Low"

        # -----------------------------
        # Generate Grad-CAM Heatmap
        # -----------------------------
        heatmap_path = os.path.join("static", "outputs", "heatmap.jpg")
        full_heatmap_path = os.path.join(os.getcwd(), heatmap_path)

        try:
            make_gradcam(
                self.model,
                image_path,
                predicted_index,
                full_heatmap_path
            )
        except Exception as e:
            print("GradCAM Error:", e)

        return {
            "decision": decision,
            "class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "cancer_probability": round(float(cancer_probability) * 100, 2),
            "risk": risk,
            "heatmap": "/" + heatmap_path,
            "raw_probs": preds.tolist()
        }
