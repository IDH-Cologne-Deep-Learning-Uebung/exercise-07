import numpy as np
import bz2

def get_labels_and_texts(file, n=12000):
    labels = []
    texts = []
    i = 0
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        i = i + 1
        if i >= n:
          return np.array(labels), texts
    return np.array(labels), texts


train_labels, train_text = get_labels_and_texts(train.ft.txt.bz2)
test_labels, test_text = get_labels_and_texts(test.ft.txt.bz2)

#Conversion on a bag of words representation (BOW)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    max_features=1000,
    lowercase=True
    )
vectorizer.fit(train_text)

texts_vec_train = vectorizer.transform(train_text)
texts_vec2 = vectorizer.transform(test_text)

#Regularization

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


#First model

model_regu = keras.Sequential()
model_regu.add(layers.Input(shape=1000)))
model_regu.add(layers.Dense(64, activation="sigmoid"))
model_regu.add(layers.Dense(32, activation="relu", activity_regularizer=regularizers.L2(0.2)))
model_regu.add(layers.Dense(1, activation="softmax"))

model_regu.summary()

model_regu.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model_regu.fit(train_test, test_text, epochs=20, batch_size=25)

#Second model

model_dropout = keras.Sequential()
model_dropout.add(layers.Input(shape=1000)))
model_dropout.add(layers.Dense(64, activation="sigmoid"))
model_dropout.add(layers.Dense(32, activation="relu"))
model_dropout.add(layers.Dropout(0.5))
model_dropout.add(layers.Dense(1, ctivation="softmax"))

model_dropout.summary()

model_dropout.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model_dropout.fit(train_test, test_text, epochs=20, batch_size=25)
