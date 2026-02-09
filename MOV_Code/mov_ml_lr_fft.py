# Movement Detection using Logistic Regression (Time + FFT Features)

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from scipy.fft import rfft


# CONFIG: SET YOUR PATH HERE

BASE_DIR = r"/Users/kumar/Desktop/Semester_3 /ML Projetcs/Mov_data"
STABLE_DIR = os.path.join(BASE_DIR, "Stable_Objects")
MOVING_DIR = os.path.join(BASE_DIR, "Moving_Objects")


def find_csvs(folder):
    pattern = os.path.join(folder, "**", "*.csv")
    return sorted(glob.glob(pattern, recursive=True))


def load_csvs(file_list, label):
    frames = []
    for i, path in enumerate(file_list, 1):
        try:
            # df = pd.read_csv(path, header=None)
            df = pd.read_csv(path, header=0) 
            # df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")

        df["label"] = label
        df["source_file"] = os.path.basename(path)
        frames.append(df)

        if i == 1 or i % 10 == 0 or i == len(file_list):
            print(f"Loaded label={label}: {i}/{len(file_list)}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    if not os.path.exists(STABLE_DIR):
        raise FileNotFoundError(f"Stable folder not found: {STABLE_DIR}")
    if not os.path.exists(MOVING_DIR):
        raise FileNotFoundError(f"Moving folder not found: {MOVING_DIR}")

    stable_files = find_csvs(STABLE_DIR)
    moving_files = find_csvs(MOVING_DIR)

    print("Stable CSV files:", len(stable_files))
    print("Moving CSV files:", len(moving_files))

    stable = load_csvs(stable_files, label=0)
    moving = load_csvs(moving_files, label=1)

    data = pd.concat([stable, moving], ignore_index=True)

    print("\nDataset shape:", data.shape)
    print("Class distribution:\n", data["label"].value_counts())

    
    # PREPROCESS + FEATURES
    
    y = data["label"].astype(int)

    X_raw = data.drop(columns=["label", "source_file"], errors="ignore")
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ===== TIME-DOMAIN FEATURES =====
    time_features = pd.DataFrame({
        "td_mean": X_raw.mean(axis=1),
        "td_std": X_raw.std(axis=1),
        "td_var": X_raw.var(axis=1),
        "td_max": X_raw.max(axis=1),
        "td_min": X_raw.min(axis=1),
        "td_energy": (X_raw ** 2).sum(axis=1),
    })

    # ===== FFT FEATURES =====
    fft_vals = np.abs(rfft(X_raw.values, axis=1))

    fft_features = pd.DataFrame({
        "fft_mean": fft_vals.mean(axis=1),
        "fft_std": fft_vals.std(axis=1),
        "fft_max": fft_vals.max(axis=1),
        "fft_energy": (fft_vals ** 2).sum(axis=1),
        "fft_peak_idx": np.argmax(fft_vals, axis=1),
    })

    # ===== COMBINE FEATURES =====
    features = pd.concat([time_features, fft_features], axis=1)

    
    # TRAIN / TEST / MODEL
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    
    # EVALUATION
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nAccuracy:", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm)
    plt.title("Movement Detection – LR (Time + FFT)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.xticks([0, 1], ["NO-MOV", "MOV"])
    plt.yticks([0, 1], ["NO-MOV", "MOV"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig("confusion_matrix_LR_FFT.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSaved plot as confusion_matrix_LR_FFT.png")


if __name__ == "__main__":
    main()