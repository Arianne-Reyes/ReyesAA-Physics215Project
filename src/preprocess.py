import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def load_and_preprocess_data():
    df = pd.read_csv("data/raw/diabetes_raw.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Save preprocessed data
    os.makedirs("data/preprocessed", exist_ok=True)
    np.save("data/preprocessed/X_train.npy", X_train)
    np.save("data/preprocessed/X_test.npy", X_test)
    np.save("data/preprocessed/y_train.npy", y_train)
    np.save("data/preprocessed/y_test.npy", y_test)

    return X_train, X_test, y_train, y_test
