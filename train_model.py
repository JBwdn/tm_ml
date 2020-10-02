import pandas
import tensorflow as tf

from one_hot_dna import one_hot_dna


if __name__ == "__main__":
    # Load training data and separate into input (X) and output (Y):
    df = pandas.read_csv("training_data.csv", header=None)
    dataset = df.values

    X_seq = dataset[:,0]
    X = [one_hot_dna(seq) for seq in X_seq]
    Y = dataset[:,1]
    
    pass
