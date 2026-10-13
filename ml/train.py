"""
PixelPen ML Pipeline - train.py
Trains a robust Convolutional Neural Network (CNN) on the MNIST dataset with
data augmentation layers designed for both canvas-drawn digits and real-world camera photos.

Saves model to:
- backend/model/digit_cnn.keras
- ml/digit_cnn.keras
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def build_digit_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds the CNN architecture specified in PixelPen specification:
    Input (28, 28, 1) -> Augmentation (Rotation, Translation, Zoom, Contrast) ->
    Conv2D(32, 3, relu) -> MaxPool(2) ->
    Conv2D(64, 3, relu) -> MaxPool(2) ->
    Flatten() -> Dropout(0.4) ->
    Dense(128, relu) -> Dense(10, softmax)
    """
    inputs = keras.Input(shape=input_shape, name="digit_input")

    # Data augmentation layers (active strictly during training)
    x = layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0, name="aug_rotation")(inputs)
    x = layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0, name="aug_translation")(x)
    x = layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0, name="aug_zoom")(x)
    x = layers.RandomContrast(0.1, name="aug_contrast")(x)

    # Feature Extractor
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Classification Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="digit_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PixelPen_Digit_CNN")
    return model

def load_and_preprocess_data():
    """
    Loads MNIST dataset and formats into (N, 28, 28, 1) normalized float32 arrays in [0, 1].
    """
    print("[1/5] Loading MNIST dataset from Keras...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1] range
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Expand dimensions for channel: (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"      Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
    return (x_train, y_train), (x_test, y_test)

def plot_training_history(history, output_path="ml/training_history.png"):
    """
    Generates and saves visual plots for training vs validation accuracy and loss.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(history.history["accuracy"], label="Train Accuracy", color="#00f0ff", lw=2)
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", color="#a855f7", lw=2, linestyle="--")
    ax1.set_title("PixelPen CNN: Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(history.history["loss"], label="Train Loss", color="#00f0ff", lw=2)
    ax2.plot(history.history["val_loss"], label="Val Loss", color="#ff0055", lw=2, linestyle="--")
    ax2.set_title("PixelPen CNN: Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[5/5] Training history plot saved to {output_path}")

def train(epochs=12, batch_size=128, learning_rate=0.001, output_dirs=None):
    if output_dirs is None:
        output_dirs = ["backend/model", "ml"]

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    print("[2/5] Constructing PixelPen CNN architecture with data augmentation...")
    model = build_digit_cnn()
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks: Learning rate reduction & Model checkpointing
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    print(f"[3/5] Commencing training for {epochs} epochs (batch size: {batch_size})...")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    print("[4/5] Evaluating final model on unseen MNIST test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"      ==========================================")
    print(f"      🏁 Test Accuracy: {test_acc * 100:.2f}% (Loss: {test_loss:.4f})")
    print(f"      ==========================================")

    # Save to targets
    for target_dir in output_dirs:
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, "digit_cnn.keras")
        model.save(save_path)
        print(f"      Model saved successfully to: {save_path}")

    plot_training_history(history, "ml/training_history.png")
    return model, test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PixelPen Handwritten Digit Recognition CNN")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs (default: 12)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial Adam learning rate (default: 0.001)")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
