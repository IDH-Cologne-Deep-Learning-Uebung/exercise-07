import bz2
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout  
from tensorflow.keras import regularizers


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

# Dateipfade zu den beiden Dateien
file1 = "data/test.ft.txt.bz2"
file2 = "data/train.ft.txt.bz2"

# Daten aus den beiden Dateien extrahieren
labels1, texts1 = get_labels_and_texts(file1)
labels2, texts2 = get_labels_and_texts(file2)

# Labels und Texte zusammenführen
all_labels = np.concatenate((labels1, labels2))
all_texts = texts1 + texts2

# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.2, random_state=42)

# CountVectorizer nur auf den Trainingsdaten anwenden
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)

# Das Vokabular auf die Testdaten anwenden
X_test_bow = vectorizer.transform(X_test)

# Ausgabe der Bags-of-Words-Repräsentation der Trainingsdaten
#print("Bags of Words - Trainingsdaten:")
for i, row in enumerate(X_train_bow.toarray()):
    non_zero_indices = row.nonzero()[0]
    words_and_counts = [(vectorizer.get_feature_names_out()[index], row[index]) for index in non_zero_indices]
   # print(f"Zeile {i + 1}: {words_and_counts}")

# Ausgabe der Bags-of-Words-Repräsentation der Testdaten
#print("Bags of Words - Testdaten:")
for i, row in enumerate(X_test_bow.toarray()):
    non_zero_indices = row.nonzero()[0]
    words_and_counts = [(vectorizer.get_feature_names_out()[index], row[index]) for index in non_zero_indices]
    #print(f"Zeile {i + 1}: {words_and_counts}")


#print("Vokabular:")
#print(vectorizer.vocabulary_)

## NN
def create_neural_network(hidden_layer_size, regularization=None, dropout_rate=None):
    model = Sequential()
    #model.add(Dense(hidden_layer_size, input_shape=(X_train_bow[1],)activation='relu', kernel_regularizer=regularization))
    model.add(Dense(hidden_layer_size, input_shape=(X_train_bow.shape[1],), activation='relu', kernel_regularizer=regularization))

    if dropout_rate:
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training ohne Reg und ohne DropOut
model_without_regularization_dropout = create_neural_network(hidden_layer_size=64)
history_without_regularization_dropout = model_without_regularization_dropout.fit(X_train_bow.toarray(), np.array(y_train), epochs=10, batch_size=1, validation_data=(X_test_bow.toarray(), np.array(y_test)))


#Training mit L2-Reg
model_with_regularization = create_neural_network(hidden_layer_size=64, regularization=regularizers.l2(0.01))
history_with_regularization = model_with_regularization.fit(X_train_bow.toarray(), np.array(y_train), epochs=10, batch_size=1, validation_data=(X_test_bow.toarray(), np.array(y_test)))

#Training mit dropout
model_with_dropout = create_neural_network(hidden_layer_size=64, dropout_rate=0.5)
history_with_dropout = model_with_dropout.fit(X_train_bow.toarray(), np.array(y_train), epochs=10, batch_size=1, validation_data=(X_test_bow.toarray(), np.array(y_test)))

# Modellauswertung auf Testset
score_without_regularization_dropout = model_without_regularization_dropout.evaluate(X_test_bow.toarray(), np.array(y_test))
score_with_regularization = model_with_regularization.evaluate(X_test_bow.toarray(), np.array(y_test))
score_with_dropout = model_with_dropout.evaluate(X_test_bow.toarray(), np.array(y_test))


print("Performance Score (ohne Regularisierung und Dropout):", score_without_regularization_dropout)
print("Performance Score (mit L2-Regularisierung):", score_with_regularization)
print("Performance Score (mit Dropout):", score_with_dropout)


