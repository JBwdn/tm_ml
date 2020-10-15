from tensorflow import keras
import numpy as np
import pandas as pd
import primer3
from time import perf_counter

# from generate_training_data import random_dna_sequence
from one_hot_dna import one_hot_dna


# Load model:
tm_model = keras.models.load_model("models/tm_model")


# Load testing data:
df = pd.read_csv("training_data.csv", header=None)
dataset = df.values
X_seq = dataset[:, 0]
X = np.array([one_hot_dna(seq, 50) for seq in X_seq], dtype=int)
X_test = X[4000:]
X_test_seq = X_seq[4000:]


# Time how long each method takes to predict Tm across training data:
a = perf_counter()
nn_preds = tm_model.predict(X_test)
b = perf_counter()
print(f"NN model per prediction = {(b - a)/4000}s")

a = perf_counter()
primer3_preds = [primer3.calcTm(seq) for seq in X_test_seq]
b = perf_counter()
print(f"Primer3 model per prediction = {(b - a)/4000}s")
