from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
import bz2
import numpy as np
import keras 

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

labelsTrain, textTrain = get_labels_and_texts("./data/train.bz2",10000)
labelsTest, textTest = get_labels_and_texts("./data/test.bz2", 10000)

vectorizer = CountVectorizer(max_features= 1000, lowercase=True)


textsVec = vectorizer.fit_transform(textTrain).toarray()
textTestVec = vectorizer.fit_transform(textTest).toarray()


## First Model
#model = keras.Sequential()
#model.add(keras.layers.Dense(32, activation="relu", input_dim = textsVec.shape[1]))
#model.add(keras.layers.Dense(10,activation="relu", kernel_regularizer= keras.regularizers.L1(0.01), activity_regularizer= keras.regularizers.L2(0.01)))
#model.add(keras.layers.Dense(20, activation="relu"))
#model.add(keras.layers.Dense(1,activation="sigmoid"))

#model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
#model.summary()

#model.fit(textsVec, labelsTrain, epochs=20, batch_size=20, verbose = True)

## Second Model 

model2 = keras.Sequential()
model2.add(keras.layers.Dense(32, activation="relu",input_dim = textsVec.shape[1]))
model2.add(keras.layers.Dense(10, activation="relu"))
model2.add(keras.layers.Dense(30, activation="relu"))
model2.add(keras.layers.Dropout(.9))
model2.add(keras.layers.Dense(1,activation="sigmoid"))

model2.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
model2.summary()

model2.fit(textsVec, labelsTrain, epochs=20, batch_size=20, verbose = True)

#Task3: the model with the regulizers seems to not have so mutch accurazy than normally which is a good indication that it doesnt overfit to mutch, the same can be seen with the dropout model when i set the dropout way higher the model does quit earlyer and its a bit more comfortable

