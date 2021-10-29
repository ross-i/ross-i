import pandas as pd
import numpy as np
from pylab import plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

def process_text(text):
    text = text.lower()
    text = text.translate({ord(i): None for i in '()'})
    text = re.sub("dr\.",'dr', text)
    text = re.sub('m\.d\.', 'md', text)
    text = re.sub('a\.m\.','am', text)
    text = re.sub('p\.m\.','pm', text)
    text = re.sub("\d+\.\d+", 'floattoken', text)
    text = re.sub("['=[]*&#]", '', text)
    text = re.sub("\.{2,}", '.', text)
    text = re.sub('[^\w_|\.|\?|!]+', ' ', text)
    text = re.sub('\.', ' . ', text)
    text = re.sub('\?', ' ? ', text)
    text = re.sub('!', ' ! ', text)
    text = re.sub('305 \d{3} \w{0,4}\d{4}', '', text)
    text = re.sub('\w{0,4}(phone)', '', text)
    text = re.sub('fax', '', text)
    text = re.sub('1400 nw \d* ?\w* ?room ?\d*', '', text)
    return text

data = pd.read_csv("data/icd10sitesonly.txt","\t")
labels = list(c50_data['c.icd10_after_spilt'])
docs = []
for doc in list(c50_data['c.path_notes']):
    docs.append(process_text(doc))
    
print("vectorizing docs")
vectorizer = TfidfVectorizer(min_df=10, stop_words='english',ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)

print("encoding labels")
le = LabelEncoder()
y = le.fit_transform(labels)

splits = 10
kf = StratifiedKFold(n_splits=splits,shuffle=True,random_state=0)

print("training XGBoost")
scores = []
i = 0
for train_index, test_index in kf.split(X,y):
    i += 1

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    gbm = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=300)
    gbm.fit(X_train.tocsc(), y_train)
    prediction = gbm.predict(X_test.tocsc())
    score = float(np.sum(y_test==prediction))/y_test.shape[0]
    scores.append(score)

    print("XGBoost - kfold %i of %i accuracy: %.4f%%" % (i,splits,score*100))
    
print("XGBoost - overall accuracy: %.4f" % (np.mean(scores)*100))
