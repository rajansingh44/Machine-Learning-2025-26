import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# ==========================
# CONFIG: PATHS
# ==========================
BASE_DIR = "/Users/kumar/Desktop/Semester_3 /ML Projetcs/Mov_data"
STABLE_DIR = os.path.join(BASE_DIR, "Stable_Objects")
MOVING_DIR = os.path.join(BASE_DIR, "Moving_Objects")


# ==========================
# UTILITY FUNCTIONS
# ==========================
def find_csvs(folder):
    return glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)


def load_csvs(file_list, label):
    frames = []
    for i, path in enumerate(file_list, 1):
        try:
            # df = pd.read_csv(path)
            # df = pd.read_csv(path, nrows=2000)
            # df = pd.read_csv(path, nrows=2000, usecols=range(0, 4096))
            df = pd.read_csv(path, header=None, nrows=2000, usecols=range(0, 4096))
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")

        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        df["label"] = label
        frames.append(df)

        if i == 1 or i % 10 == 0 or i == len(file_list):
            print(f"Loaded label={label}: {i}/{len(file_list)}")

    return pd.concat(frames, ignore_index=True)


def extract_features(data):
    X_raw = data.drop(columns=["label"], errors="ignore")

    features = pd.DataFrame({
        "mean": X_raw.mean(axis=1),
        "median": X_raw.median(axis=1),
        "std": X_raw.std(axis=1),
        "var": X_raw.var(axis=1),
        "max": X_raw.max(axis=1),
        "min": X_raw.min(axis=1),
        "energy": (X_raw ** 2).sum(axis=1),
    })

    return features


def plot_confusion(cm, title, filename):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.xticks([0, 1], ["NO-MOV", "MOV"])
    plt.yticks([0, 1], ["NO-MOV", "MOV"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================
# MAIN PIPELINE
# ==========================
def run_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_test, y_pred))

    plot_confusion(cm, f"{name} – Confusion Matrix", f"confusion_matrix_{name.lower()}.png")


def main():
    print("Loading CSV files...")

    stable_files = find_csvs(STABLE_DIR)
    moving_files = find_csvs(MOVING_DIR)

    stable = load_csvs(stable_files, label=0)
    moving = load_csvs(moving_files, label=1)

    data = pd.concat([stable, moving], ignore_index=True)

    print("\nDataset shape:", data.shape)
    print("Class distribution:\n", data["label"].value_counts())

    y = data["label"].astype(int)
    X = extract_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ==========================
    # MODELS
    # ==========================
    models = {
        "Logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }

    for name, model in models.items():
        run_model(name, model, X_train, X_test, y_train, y_test)

    print("\n ALL MODELS FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()