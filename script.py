import sklearn
from sklearn . feature_extraction . text import CountVectorizer
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

testDataLabels, textDataText = get_labels_and_texts(test.ft.txt.bz2)
trainDataLabels, trainDataText = get_labels_and_texts(train.ft.txt.bz2)

vectorizer = CountVectorizer (
max_features =1000 , # how many tokens to distinguish ?
lowercase = True # make everything lower - cased ?
)
# ’learn ’ mapping of words to dimensions
vectorizer.fit(trainDataText)

# apply mapping to texts
texts_vec = vectorizer.transform(trainDataTrain)
texts_vec2 = vectorizer.transform(testDataText)

#---------------------------------------------------------------------------------------
# regularization
model.models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(input_size,)))
model.add(layers.Dense(32, activation='sigmoid', activity_regularizer = regularizers . L2 (0.2)))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

predictionsRegulTrain = model.predict(trainDataText)
predictionsRegulTest = model.predict(testDataText)

print(accuracy_score(trainDataLabels, predictionsRegulTrain))
print(accuracy_score(testDataLabels, predictionsRegulTest))

#Dropout
model.models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(input_size,)))
model.add(layers.Dense(32, activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

predictionsRegulTrain = model.predict(trainDataText)
predictionsRegulTest = model.predict(testDataText)

print(accuracy_score(trainDataLabels, predictionsRegulTrain))
print(accuracy_score(testDataLabels, predictionsRegulTest))