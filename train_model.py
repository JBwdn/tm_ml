import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)

from one_hot_dna import one_hot_dna

if __name__ == "__main__":
    # Load training data and separate into features (X) and labels (Y):
    df = pd.read_csv("training_data.csv", header=None)
    dataset = df.values

    X_seq = dataset[:, 0]
    # One hot ecode primer sequences...
    X = np.array([one_hot_dna(seq, 50) for seq in X_seq], dtype=int)
    Y = np.array(dataset[:, 1], dtype="float64")

    X_train = X[:4000]
    X_test = X[4000:]
    X_test_seq = X_seq[4000:]
    Y_train = Y[:4000]
    Y_test = Y[4000:]

    # Define the model (200 input node, 1 fully connected hidden layer, output):
    tm_model = keras.Sequential([
            layers.Dense(200, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ])
    tm_model.compile(loss="mean_absolute_error", optimizer=keras.optimizers.Adam(0.001))

    # Train model:
    tm_model.fit(X_train, Y_train, epochs=100)

    # Validate the model against the training set:
    print("Validating tm model...")
    predictions = tm_model.predict(X_test)

    print("Example model predictions:")
    for i in range(5):
        pred = round(float(predictions[i]), 5)
        real = round(Y_test[i], 5)
        diff = round(abs(float(predictions[i]) - Y_test[i]), 5)
        print(
            f"""{i+1}: Model={pred}, Real={real}, Diff={diff}
                Sequence={X_test_seq[i]}"""
        )

    # Save model to file for use later:
    tm_model.save("models/tm_model")

    pass
