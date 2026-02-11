# MLP 
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==========================
# CONFIG: SET YOUR PATH HERE
# ==========================
BASE_DIR = r"/Users/kumar/Desktop/Semester_3 /ML Projetcs/Mov_data"
STABLE_DIR = os.path.join(BASE_DIR, "Stable_Objects")
MOVING_DIR = os.path.join(BASE_DIR, "Moving_Objects")


def find_csvs(folder):
    return glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)


def load_csvs(file_list, label):
    frames = []
    for i, path in enumerate(file_list, 1):
        try:
            # df = pd.read_csv(path)
            df = pd.read_csv(path, header=0) # try with header first
            # df = pd.read_csv(path, header=None) # try no header first
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")

        df["label"] = label
        frames.append(df)

        if i == 1 or i % 10 == 0 or i == len(file_list):
            print(f"Loaded label={label}: {i}/{len(file_list)}")

    return pd.concat(frames, ignore_index=True)


def main():

    stable_files = find_csvs(STABLE_DIR)
    moving_files = find_csvs(MOVING_DIR)

    print("Stable CSV files:", len(stable_files))
    print("Moving CSV files:", len(moving_files))

    stable = load_csvs(stable_files, label=0)
    moving = load_csvs(moving_files, label=1)

    data = pd.concat([stable, moving], ignore_index=True)

    print("\nDataset shape:", data.shape)
    print("Class distribution:\n", data["label"].value_counts())

    # --------------------------
    # FEATURES
    # --------------------------
    y = data["label"].astype(int)

    X_raw = data.drop(columns=["label"], errors="ignore")
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce").fillna(0)

    features = pd.DataFrame({
        "mean": X_raw.mean(axis=1),
        "median": X_raw.median(axis=1),
        "std": X_raw.std(axis=1),
        "var": X_raw.var(axis=1),
        "max": X_raw.max(axis=1),
        "min": X_raw.min(axis=1),
        "energy": (X_raw ** 2).sum(axis=1),
    })

    # --------------------------
    # TRAIN / TEST SPLIT
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --------------------------
    # MLP MODEL
    # --------------------------
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate="adaptive",
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # --------------------------
    # EVALUATION
    # --------------------------
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nAccuracy:", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm)
    plt.title("Movement Detection – MLP Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.xticks([0, 1], ["NO-MOV", "MOV"])
    plt.yticks([0, 1], ["NO-MOV", "MOV"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig("confusion_matrix_mlp.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSaved plot as confusion_matrix_mlp.png")


if __name__ == "__main__":
    main()