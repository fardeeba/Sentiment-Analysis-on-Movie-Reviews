import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
ws = WordNetLemmatizer()
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


'''movie_reviews corpus contains 2000 textfiles containing movie reviews
    Each file is labelled with either pos or neg based on category of reviews
    Now we split each text file into words
    all_words is a list of tuples where first element of tuple is list of tokenized words whereas second 
    element constitutes label of file
'''

all_words = []

#Eliminating stopwords using bulitin set of english language stop words
stop_words = set(stopwords.words('english'))
print(len(stop_words))
for review_label in movie_reviews.categories():
    for files in movie_reviews.fileids(review_label):
        lst = list(movie_reviews.words(files)) #tokenization
        all_words.append(([w for w in lst if w not in stop_words],review_label)) #elimination of stop words

random.shuffle(all_words)

#Lemmatization
for element in all_words:
    lmt = []
    lst = list(element)
    l1 = list(lst[0])
    for e in l1:
        lmt.append(ws.lemmatize(e))
    l1 = lmt
    lst[0] = tuple(l1)
    element = tuple(lst)

#Vocabular
vocab = []
lookup = []
for element in all_words:
    vocab.append(nltk.FreqDist(element[0]))
for x in vocab:
    lookup.append(list(x))
dict = lookup[:15]
print(len(dict[0]))


def find_features(document):
    features = {}
    for w in dict:
        for x in w:
            if x in document:
                features[x] = 1
            else:
                features[x] = 0
    return features

featureset = []
for words in all_words:
    featureset.append([find_features(words[0]),words[1]])
random.shuffle(featureset)

#Decision Tree classifier
def calculate_fitness(rev_feature,label):

    X = rev_feature
    Y = label
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, min_samples_leaf=2)
    clf_entropy.fit(X_train, y_train)
    y_pred = clf_entropy.predict(X_test)
    return accuracy_score(y_test, y_pred) * 100

data = []
label = []

#Extracting features and corresponding labels from featuredet which are stored in data and label respectively

for x in featureset:
    lst = []
    for v in x[0]:
        lst.append(x[0][v])
    data.append(lst)
    label.append(x[1])

print('Accuracy',calculate_fitness(data,label))
