"""
PixelPen ML Pipeline - evaluate.py
Evaluates the trained digit_cnn.keras model on MNIST test data and custom user samples.
Produces:
- Classification report (precision, recall, f1-score per digit 0-9)
- Confusion matrix heatmap (saved to ml/confusion_matrix.png)
- Visual gallery of misclassified digits (saved to ml/misclassified_samples.png)
- Custom user photos evaluation (scans ml/data/ for real-world images)
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def load_trained_model(model_path="backend/model/digit_cnn.keras"):
    if not os.path.exists(model_path):
        alt_path = "ml/digit_cnn.keras"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {alt_path}. Run train.py first!")
    print(f"Loading trained PixelPen CNN model from: {model_path}")
    return keras.models.load_model(model_path)

def plot_confusion_matrix(y_true, y_pred, output_path="ml/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("PixelPen Digit Recognition - Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [str(i) for i in range(10)], fontsize=11)
    plt.yticks(tick_marks, [str(i) for i in range(10)], fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    plt.ylabel("True Digit Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Digit Label", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def plot_misclassified(x_test, y_true, y_pred, probabilities, output_path="ml/misclassified_samples.png", max_samples=16):
    misclassified_indices = np.where(y_true != y_pred)[0]
    total_mis = len(misclassified_indices)
    print(f"Found {total_mis} misclassified test samples (out of {len(y_true)}).")

    if total_mis == 0:
        print("Flawless predictions! No misclassifications to display.")
        return

    n_display = min(total_mis, max_samples)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(16):
        if i < n_display:
            idx = misclassified_indices[i]
            img = x_test[idx].squeeze()
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            confidence = probabilities[idx][pred_label] * 100

            axes[i].imshow(img, cmap="inferno")
            axes[i].set_title(f"True: {true_label} | Pred: {pred_label}\n({confidence:.1f}%)", fontsize=10, color="red")
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    plt.suptitle("PixelPen: Most Confusing Samples Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Misclassified samples plot saved to: {output_path}")

def evaluate_custom_samples(model, data_dir="ml/data"):
    """
    Evaluates real-world samples if user has added any to ml/data.
    Expected naming convention: digit_<label>_<index>.png/jpg
    """
    import cv2
    import sys
    # Add backend/app to path for preprocess
    sys.path.append(os.path.abspath("backend/app"))
    try:
        from preprocess import preprocess_digit
    except ImportError:
        print("Note: Preprocessing module will be tested separately.")
        return

    image_files = glob.glob(os.path.join(data_dir, "**/*.*"), recursive=True)
    valid_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not valid_files:
        print(f"No custom image files found in {data_dir}. You can drop test photos here to benchmark real-world accuracy.")
        return

    print(f"\nEvaluating on {len(valid_files)} custom user samples in {data_dir}...")
    correct = 0
    total = 0

    for file_path in valid_files:
        base_name = os.path.basename(file_path)
        # Try extracting label like digit_7_01.jpg
        parts = base_name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            expected = int(parts[1])
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            processed, _ = preprocess_digit(img_bytes, source="photo")
            probs = model.predict(processed, verbose=0)[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred])

            is_correct = (pred == expected)
            if is_correct:
                correct += 1
            total += 1
            status = "CORRECT" if is_correct else "MISCLASSIFIED"
            print(f"[{status}] {base_name} -> True: {expected}, Pred: {pred} ({conf*100:.1f}%)")

    if total > 0:
        print(f"\nCustom Sample Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

def main():
    model = load_trained_model()
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test_norm = x_test.astype("float32") / 255.0
    x_test_norm = np.expand_dims(x_test_norm, -1)

    print("Running inference on 10,000 MNIST test images...")
    probabilities = model.predict(x_test_norm, batch_size=256, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[f"Digit {i}" for i in range(10)], digits=4))

    plot_confusion_matrix(y_test, y_pred)
    plot_misclassified(x_test_norm, y_test, y_pred, probabilities)
    evaluate_custom_samples(model)

if __name__ == "__main__":
    main()
