import tensorflow as tf
import numpy as np
import cv2
from config import IMG_SIZE


def make_gradcam(model, image_path, class_index, output_path):
    try:
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        input_tensor = np.expand_dims(img, axis=0)

        # Use EfficientNet last conv layer explicitly
        last_conv_layer = model.get_layer("top_conv")
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor)

            # FIX: ensure predictions is tensor
            if isinstance(predictions, list):
                predictions = predictions[0]
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8
        heatmap = heatmap.numpy()

        # Resize and colorize heatmap
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay heatmap on original image
        original = cv2.imread(image_path)
        original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
        superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        cv2.imwrite(output_path, superimposed)
        print("🔥 Heatmap saved at:", output_path)

    except Exception as e:
        print("❌ GradCAM error:", e)
