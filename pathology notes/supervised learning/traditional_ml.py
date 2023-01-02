import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import sys
import csv
import matplotlib.pyplot as plt
from utilities.process_text import process_text
from utilities.resample import balance_classes
from utilities.stop_words import stop_words
import os


#get filepath to data
args = (sys.argv)
if len(args) < 2:
    raise Exception("Usage: python traditional_ml.py <filepath txt> [label_start] [label_stop]")
    #if label_start and label_stop are omitted, uses whole label.
    #if only label_stop is omitted, slicing a single character at position label_start to use as label.
filename = args[1]
label_start = None
label_stop = None
if len(args) >= 3:
    label_start = int(args[2])
    if len(args) == 4:
        label_stop = int(args[3])
    else: 
        label_stop = int(label_start)+1

print("loading corpus")
#load corpus
data = pd.read_csv(filename, sep="\t")
labels = [] 
docs = []
for label, doc in zip(list(data[data.columns[-2]]),list(data[data.columns[-1]])):
    docs.append(process_text(doc))
    labels.append(label[label_start:label_stop])

#tfidf vectorization
print('vectorizing documents')
vectorizer = TfidfVectorizer(min_df=10, stop_words=stop_words,ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)

#label encoder
if type(labels[0]) != int:
    le = LabelEncoder()
    y = le.fit_transform(labels)
else:
    y = np.array(labels)

#kfold cross validation
splits = 10
kf = StratifiedKFold(n_splits=splits,shuffle=True,random_state=1234)

#classify - replace clf with desired classifier and settings
def run_classifier(X, y, kf, clf, balance=False):
    print("training classifier", clf)
    scores = []
    n_classes = len(set(y))
    conf_matrix = np.zeros((n_classes, n_classes))
    i = 0
    for train_index, test_index in kf.split(X,y):
        i += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if balance:
            X_train, y_train = balance_classes(X_train, y_train)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        kcm = confusion_matrix(y_test, y_pred)
        if kcm.shape == conf_matrix.shape:
            conf_matrix += confusion_matrix(y_test, y_pred)
        score = f1_score(y_test, y_pred, average='macro')
        scores.append(score)

        print("Classifier", clf, "- kfold %i of %i accuracy: %.4f%%" % (i,splits,score*100))
    
    print("Classifier", clf,  "- overall accuracy: %.4f" % (np.mean(scores)*100))
    cm = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    cm.plot(ax=ax)
    clf_name = str(clf)[:str(clf).find('(')]
    file_alias = filename[filename.rfind('/')+1:filename.find('.')]
    plt.title(clf_name)
    plt.savefig("graphics/%s_%s_cm.eps" % (file_alias, clf_name))

if not os.path.exists('graphics'):
    os.makedirs('graphics')


classifiers = [
    MultinomialNB(),
    LogisticRegression(class_weight='balanced', random_state=0),
    RandomForestClassifier(class_weight='balanced',max_depth=5, random_state=0), 
    SVC(class_weight='balanced', random_state=0)
]
for clf in classifiers:
    try:
        if clf.class_weight == 'balanced':
            run_classifier(X, y, kf, clf)
        else: 
            run_classifier(X, y, kf, clf, balance=True)
    except AttributeError:
        run_classifier(X,y, kf, clf, balance=True)    
