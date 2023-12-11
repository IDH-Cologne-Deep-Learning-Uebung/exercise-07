import bz2
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

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

def load_files(filenames):
    all_labels = []
    all_texts = []
    for filename in filenames:
        labels, texts = get_labels_and_texts(filename)
        all_labels.append(labels)
        all_texts.append(texts)
    return all_labels, all_texts

filenames = ['data/train.ft.txt.bz2', 'data/test.ft.txt.bz2']
labels, texts = load_files(filenames)


vectorizer = CountVectorizer()
vectorizer.fit(texts[0])


train_data = vectorizer.transform(texts[0]).toarray()
test_data = vectorizer.transform(texts[1]).toarray()


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(train_data, labels[0], epochs=10, batch_size=32, validation_split=0.2)


model_reg = Sequential()
model_reg.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(train_data.shape[1],)))
model_reg.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model_reg.add(Dense(1, activation='sigmoid'))


model_reg.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model_reg.fit(train_data, labels[0], epochs=10, batch_size=32, validation_split=0.2)


model_dropout = Sequential()
model_dropout.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model_dropout.add(Dropout(0.5))
model_dropout.add(Dense(64, activation='relu'))
model_dropout.add(Dropout(0.5))
model_dropout.add(Dense(1, activation='sigmoid'))


model_dropout.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model_dropout.fit(train_data, labels[0], epochs=10, batch_size=32, validation_split=0.2)
