import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import os

def evaluate_model(model, X_test, y_test, title="Model"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Save predictions
    os.makedirs("data/final", exist_ok=True)
    np.save("data/final/y_pred.npy", y_pred)
    np.save("data/final/y_test.npy", y_test)
    with open("data/final/mse.txt", "w") as f:
        f.write(f"MSE: {mse:.2f}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title} (MSE: {mse:.2f})")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/final/evaluation_plot.png")  # Save the plot
    plt.show()
