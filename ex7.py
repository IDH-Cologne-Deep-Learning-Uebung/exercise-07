import bz2
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import tensorflow as tf


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

test = get_labels_and_texts('test.ft.txt.bz2')
texts = get_labels_and_texts('train.ft.txt.bz2')

labels_array, text_data = texts
test_labels_array, test_text_data = test

vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(text_data)
test_bag_of_words = vectorizer.transform(test_text_data)

print("Input shape:", bag_of_words.shape)

model_dropout = Sequential()
model_dropout.add(Dense(1, activation='relu', input_shape=(12000, 34106)))
model_dropout.add(Dropout(0.9))
model_dropout.add(Dense(9, activation='relu'))
model_dropout.add(Dropout(0.9))
model_dropout.add(Dense(1, activation='sigmoid'))

model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#error ( [Op:SerializeManySparse]) attemted to sort spare tensor with tf.sparse.reorder both labels and bow
#resulting in eror input must be spare vector, models were running with dummy data
#full error:

# raise core._status_to_exception(e) from None  # pylint: disable=protected-access
# tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__SerializeManySparse_device_/job:localhost/replica:0/task:0/device:CPU:0}} indices[1] = [0,10937] is out of order. Many sparse ops require sorted indices.
#     Use `tf.sparse.reorder` to create a correctly ordered copy.
#
#  [Op:SerializeManySparse]

model_dropout.fit(bag_of_words, labels_array, epochs=10, batch_size=32, verbose=1)

# Model with REguler
model_l2 = Sequential()
model_l2.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(10,)))
model_l2.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model_l2.add(Dense(1, activation='sigmoid'))

model_l2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_l2.fit(text_data, labels_array, epochs=10, batch_size=32, verbose=1)

test_text_data = np.random.rand(20, 10)
test_labels_array = np.random.randint(2, size=(20, 1))

loss_dropout, accuracy_dropout = model_dropout.evaluate(test_text_data, test_labels_array)
loss_l2, accuracy_l2 = model_l2.evaluate(test_text_data, test_labels_array)

print("Dropout loss", loss_dropout)
print("Dropout Accuracy:", accuracy_dropout)
print("Regu Loss:", loss_l2)
print("Regu Accuracy:", accuracy_l2)
