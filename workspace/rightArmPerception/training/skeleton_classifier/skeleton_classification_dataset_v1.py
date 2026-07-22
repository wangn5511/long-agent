#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


RANDOM_SEED = 42
TIME_STEPS = 16
NUM_KEYPOINTS = 21
DIMENSION = 2
FRAME_DIMENSION = NUM_KEYPOINTS * DIMENSION
INPUT_DIMENSION = TIME_STEPS * FRAME_DIMENSION


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a skeleton gesture classifier from train/val CSV files."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("model/skeleton_classifier/skeleton_train.csv"),
        help="Training CSV path.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("model/skeleton_classifier/skeleton_val.csv"),
        help="Validation CSV path.",
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=Path("model/skeleton_classifier/skeleton_classifier_label.csv"),
        help="Label CSV path.",
    )
    parser.add_argument(
        "--model-save-path",
        type=Path,
        default=Path("model/skeleton_classifier/skeleton_classifier_v1.keras"),
        help="Path to save the trained Keras model.",
    )
    parser.add_argument(
        "--tflite-save-path",
        type=Path,
        default=Path("model/skeleton_classifier/skeleton_classifier_v1.tflite"),
        help="Path to save the exported TFLite model.",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        type=Path,
        default=Path("model/skeleton_classifier/skeleton_confusion_matrix_v1.png"),
        help="Path to save the confusion matrix image.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Use the LSTM-based model head.",
    )
    parser.add_argument(
        "--show-confusion-matrix",
        action="store_true",
        help="Display the confusion matrix after evaluation.",
    )
    return parser.parse_args()


def load_dataset(csv_path):
    feature_columns = list(range(1, INPUT_DIMENSION + 1))
    x_dataset = np.loadtxt(
        csv_path,
        delimiter=",",
        dtype="float32",
        usecols=feature_columns,
    )
    y_dataset = np.loadtxt(
        csv_path,
        delimiter=",",
        dtype="int32",
        usecols=(0,),
    )

    x_dataset = np.atleast_2d(x_dataset).astype("float32")
    y_dataset = np.atleast_1d(y_dataset).astype("int32")
    return x_dataset, y_dataset


def load_labels(label_csv):
    with open(label_csv, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f) if row]


def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(num_classes, use_lstm):
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    if use_lstm:
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(INPUT_DIMENSION,)),
            tf.keras.layers.Reshape((TIME_STEPS, FRAME_DIMENSION)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(INPUT_DIMENSION,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])

    return compile_model(model)


def save_confusion_matrix(y_true, y_pred, labels, output_path, show_plot):
    print("Classification Report")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    label_indices = list(range(len(labels)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=label_indices)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_cmx, annot=True, fmt="g", square=False, cmap="Blues")
    ax.set_ylim(len(labels), 0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"saved confusion matrix to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def export_tflite(model, tflite_save_path, use_lstm):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if use_lstm:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

    tflite_quantized_model = converter.convert()
    tflite_save_path.write_bytes(tflite_quantized_model)


def main():
    args = parse_args()

    labels = load_labels(args.label_csv)
    num_classes = len(labels)
    if num_classes == 0:
        raise ValueError(f"No labels found in {args.label_csv}")

    x_train, y_train = load_dataset(args.train_csv)
    x_val, y_val = load_dataset(args.val_csv)

    if x_train.shape[1] != INPUT_DIMENSION:
        raise ValueError(f"Unexpected train feature size: {x_train.shape}")
    if x_val.shape[1] != INPUT_DIMENSION:
        raise ValueError(f"Unexpected val feature size: {x_val.shape}")

    print(f"train samples: {len(x_train)}")
    print(f"val samples: {len(x_val)}")
    print(f"num classes: {num_classes}")
    print(f"labels: {labels}")

    model = build_model(num_classes=num_classes, use_lstm=args.use_lstm)
    model.summary()

    args.model_save_path.parent.mkdir(parents=True, exist_ok=True)
    args.tflite_save_path.parent.mkdir(parents=True, exist_ok=True)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(args.model_save_path),
        verbose=1,
        save_weights_only=False,
    )
    es_callback = tf.keras.callbacks.EarlyStopping(
        patience=20,
        verbose=1,
        restore_best_weights=True,
    )

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        callbacks=[cp_callback, es_callback],
        verbose=1,
    )

    model = tf.keras.models.load_model(str(args.model_save_path))
    model = compile_model(model)

    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"val loss: {loss:.6f}")
    print(f"val accuracy: {accuracy:.6f}")

    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    save_confusion_matrix(
        y_val,
        y_pred,
        labels,
        args.confusion_matrix_path,
        args.show_confusion_matrix,
    )

    export_tflite(model, args.tflite_save_path, args.use_lstm)
    print(f"saved keras model to: {args.model_save_path}")
    print(f"saved tflite model to: {args.tflite_save_path}")


if __name__ == "__main__":
    main()
