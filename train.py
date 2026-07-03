"""
train.py
--------
Reproduces 01_data_cleaning_preprocessing.ipynb + 02_ann_model_training.ipynb
as a single script and saves the 3 artifacts the deployed app needs:

    preprocessor.pkl          (fitted ColumnTransformer: scaling + one-hot)
    y_scaler.pkl               (fitted StandardScaler for the target)
    insurance_ann_model.keras  (trained Keras ANN)

Usage:
    python train.py --data insurance.csv
"""

import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].str.strip().str.lower()
    return df


def build_preprocessor():
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )


def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main(data_path):
    df = load_and_clean(data_path)

    X = df.drop(columns=["charges"])
    y = df["charges"].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    model = build_model(X_train_proc.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    model.fit(
        X_train_proc, y_train_scaled,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1,
    )

    # quick sanity check on test set
    from sklearn.metrics import mean_absolute_error, r2_score
    y_pred = y_scaler.inverse_transform(model.predict(X_test_proc))
    print(f"Test MAE : {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"Test R^2 : {r2_score(y_test, y_pred):.4f}")

    # save deployment artifacts
    joblib.dump(preprocessor, "preprocessor.pkl")
    joblib.dump(y_scaler, "y_scaler.pkl")
    model.save("insurance_ann_model.keras")
    print("\nSaved: preprocessor.pkl, y_scaler.pkl, insurance_ann_model.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to insurance.csv")
    args = parser.parse_args()
    main(args.data)
