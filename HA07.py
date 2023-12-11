import bz2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, train_test_splitv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

test_labels, test_texts = get_labels_and_texts("data/test.ft.txt.bz2")
df_test = pd.DataFrame({'label': test_labels, 'text': test_texts})
train_labels, train_texts = get_labels_and_texts("data/train.ft.txt.bz2")
df_train = pd.DataFrame({'label': train_labels, 'text': train_texts})

all_texts = pd.concat([df_train['text'], df_test['text']], ignore_index=True)
vectorizer = CountVectorizer()
X_all = vectorizer.fit_transform(all_texts)

X_train = vectorizer.transform(df_train['text'])
X_test = vectorizer.transform(df_test['text'])
y_train = df_train['label']
y_test = df_test['label']
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def build_model(input_dim, use_dropout=False):
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=input_dim))
if use_dropout:
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
if use_dropout:
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
return model

def train_and_evaluate(model, X_train, y_train, X_val, y_val):
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
return history

train_file = "data/train.ft.txt.bz2"
test_file = "data/test.ft.txt.bz2"
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(train_file, test_file)
model_base = build_model(X_train.shape[1])
history_base = train_and_evaluate(model_base, X_train, y_train, X_val, y_val)

model_dropout = build_model(X_train.shape[1], use_dropout=True)
history_dropout = train_and_evaluate(model_dropout, X_train, y_train, X_val, y_val)

y_pred_base = model_base.predict(X_test)
y_pred_dropout = model_dropout.predict(X_test)