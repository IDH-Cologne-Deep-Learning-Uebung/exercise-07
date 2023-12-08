import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split


# read the data from a file
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


test_labels, test_text = get_labels_and_texts("exercise-07/data/test.ft.txt.bz2")
train_labels, train_text = get_labels_and_texts("exercise-07/data/train.ft.txt.bz2")

print(test_labels)
#print(test_text)
print(train_labels)


#to bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(
max_features=1000,  
lowercase=True 
)

#vocab on train
#apply cevtorization to train and test

# ’learn’ mapping of words to dimensions  #learn vocab
#X = vectorizer.fit(train_text[1])
vectorizer.fit(train_text)
#print(X.vocabulary_)

# apply mapping to text  #get bag of words
#X = vectorizer.transform(train_text[1], test_text[1])

#print(texts_vec)


test_text_vec = vectorizer.transform(test_text)
train_text_vec = vectorizer.transform(train_text)

#print(train_text_vec)

x_test =test_text_vec 
y_test = test_labels 
x_train =train_text_vec #2d -> 30
y_train = train_labels #1d -> 1

print(x_train.shape)
print(y_train.shape)
#print(x_test.shape)
#print(X)
#print(X.shape)
#print(y_train.shape)

#print(x_train[1])
#print("other")
#print(y_train[0])



#evaluate
from sklearn.metrics import precision_score, recall_score, f1_score

def eval(mo):
    y_pred = mo.predict(x_test)
    #print("yTest: ",y_test)
    #print("yPred: ", y_pred)
    print("precision: "+ str(precision_score(y_test, y_pred)))
    print("recall: "+ str(recall_score(y_test, y_pred)))
    print("f1: "+ str(f1_score(y_test, y_pred)))



#models
import keras
from keras import layers
from keras import regularizers

#regularization
model = keras.Sequential()
model.add(layers.Input(shape=(x_train.shape[1])))
model.add(layers.Dense(20, activation="relu" ,activity_regularizer=regularizers.L2(0.4)))
model.add(layers.Dense(30, activation="softmax", activity_regularizer=regularizers.L2(0.7)))
model.add(layers.Dense(10, activation="sigmoid", activity_regularizer=regularizers.L2(0.1)))
model.add(layers.Dense(1, activation="softmax"))  
model.summary()

#dropout
model2 = keras.Sequential()
model2.add(layers.Input(shape=(None,1000 )))#x_train.shape)))
model2.add(layers.Dense(15, activation="sigmoid"))
model2.add(layers.Dropout(0.3))
model2.add(layers.Dense(20, activation="relu"))
model2.add(layers.Dropout(0.7))
model2.add(layers.Dense(10, activation="sigmoid"))
model2.add(layers.Dropout(0.2))
model2.add(layers.Dense(1, activation="softmax"))  
model2.summary()



#split training and test data.
#x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

def do_model(mo):
    mo.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])
    mo.fit(x_train, y_train, epochs=10, batch_size=20)

model.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])
model.fit(x_train, y_train, epochs=15, batch_size=20)
print()
model2.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])
model2.fit(x_train, y_train, epochs=15, batch_size=20)

# do_model(model)
# print()
# do_model(model2)

#model.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])
#model.compile(loss="mean_squared_error",optimizer="sgd",metrics=["accuracy"])
#model.fit(x_train, y_train, epochs=10, batch_size=10)

print()

print("model1: ")
eval(model)

print("model2: ")
eval(model2)



#warum selbe Ergebnis?
#haben zusammen verlaufenden loss?

#was mit data beim publishen?