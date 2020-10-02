import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from one_hot_dna import one_hot_dna

df = pandas.read_csv("training_data.csv", header=None)
dataset = df.values

X_seq = dataset[:,0]
X = [one_hot_dna(seq) for seq in X_seq]
Y = dataset[:,1]

def baseline_model():

