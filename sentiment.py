import bz2
import numpy as np

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

y_train, texts_train=get_labels_and_texts("data/train.ft.txt.bz2")
y_test, texts_test=get_labels_and_texts("data/test.ft.txt.bz2")
#print(y_test)

from sklearn.feature_extraction.text import CountVectorizer


vectorizer=CountVectorizer(max_features=1000, lowercase=True)
# ’learn ’ mapping of words to dimensions and apply to text train
x_train=vectorizer.fit_transform(texts_train).toarray()
#print(x_train.shape)
# apply mapping to test texts
x_test =vectorizer.transform(texts_test).toarray()
#print(x_test.shape)
#print(x_test)


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from sklearn.metrics import precision_score, recall_score, f1_score
def evalu(mo):
  y_pred = mo.predict(x_test)
  print(y_test)
  print(y_pred)
  print("precision: "+ str(precision_score(y_test, y_pred)))
  print("recall: "+ str(recall_score(y_test, y_pred)))
  print("f1: "+ str(f1_score(y_test, y_pred)))


# model 1
model_1 = keras.Sequential()
model_1.add(layers.Input(shape=(1000,)))
model_1.add(layers.Dense(10, activation="sigmoid"))
model_1.add(layers.Dense(30, activation="relu", activity_regularizer = regularizers .L2 (0.2)))
model_1.add(layers.Dense(1, activation="softmax"))
model_1.summary()


model_1.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model_1.fit(x_train, y_train, epochs=20, batch_size=25)


# model 2
model_2 = keras.Sequential()
model_2.add(layers.Input(shape=(1000,)))
model_2.add(layers.Dense(10, activation="sigmoid"))
model_2.add(layers.Dense(30, activation="relu"))
model_2.add(layers.Dropout(0.5))
model_2.add(layers.Dense(1, activation="softmax"))
model_2.summary()


model_2.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model_2.fit(x_train, y_train, epochs=20, batch_size=25)

evalu(model_1)
evalu(model_2)