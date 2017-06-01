# !~/anaconda/bin/python

import os
import sys
import re

import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
# import enchant

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# d = enchant.Dict("en_US")
lmtzr = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
stopwords = nltk.corpus.stopwords.words('english')

def processLine(line):
    line = re.sub(r'https?:\/\/.*\s+', ' ', line, flags=re.MULTILINE) # remove URLs
    line = re.sub(r'\W+',' ',line, flags=re.MULTILINE) # remove non-word characters
    
    line = line.decode('utf-8').lower()
    tokens = tokenizer.tokenize(line)
    output = []
    for t in tokens:
        t = lmtzr.lemmatize(t)
        if (len(t) <= 2) or (len(t) >= 8) or (t in stopwords) or (t.isdigit()):
            continue
        output.append(t)
    return ' '.join(output)

def split_training_testing(X, y):
    # split the data into training and testing for hold-out testing    
    n_rows, n_features = np.shape(X)
        
    train_size = int(n_rows*0.8)
       
    X_train = X[0:train_size,:]
    y_train = y[0:train_size,:]
        
    X_test = X[train_size:n_rows,:]
    y_test = y[train_size:n_rows,:]
          
    return (X_train, y_train, X_test, y_test)

def main():  
    fname = sys.argv[1]
    
    fh = open(fname,'r')
    lines = fh.readlines()
    fh.close()

    comments = [] # a list of all comments
    labels = [] # a list of all label lists
    for i in range(1,len(lines)):
        arr = lines[i].split('\t')
        
        label_lst = []
        del arr[0]
        comment = ' '.join(arr[:-10])
        post_comment = processLine(comment)
        if len(post_comment) < 5:
            continue
        
        for j in range(-10,0):
            label_lst.append(arr[j])
        labels.append(label_lst)

        comments.append(post_comment)

    # obtain tf-idf sparse matrix
    # vectorizer = CountVectorizer(min_df=1)
    vectorizer = TfidfVectorizer(max_features=10000, smooth_idf=True, use_idf=True, norm="l2", sublinear_tf=False)
    tfidf = vectorizer.fit_transform(comments)

    # build classifier
    X = tfidf.toarray()
    y = np.array(labels)
    y = y.astype(int)
    
    X_train, y_train, X_test, y_test = split_training_testing(X, y)

    classifier1 = OneVsRestClassifier(SVC(kernel='linear', C=10))
    classifier2 = OneVsRestClassifier(SVC(kernel='rbf', C=10))
    classifier3 = OneVsRestClassifier(SVC(kernel='poly', C=10))
    classifier4 = OneVsRestClassifier(SVC(kernel='sigmoid', C=10))
    # train
    classifier1.fit(X_train, y_train)
    y_predict_1 = classifier1.predict(X_test)

    classifier2.fit(X_train, y_train)
    y_predict_2 = classifier2.predict(X_test)

    classifier3.fit(X_train, y_train)
    y_predict_3 = classifier3.predict(X_test)

    classifier4.fit(X_train, y_train)
    y_predict_4 = classifier4.predict(X_test)

    y_predict = [y_predict_1, y_predict_2, y_predict_3, y_predict_4]

    # measure: accuracy score
    acc = []
    for i in y_predict:
        accuracy1 = accuracy_score(y_test, i)
        accuracy2 = hamming_loss(y_test, i)
        accuracy3 = jaccard_similarity_score(y_test, i)
        acc.append([accuracy1, accuracy2, accuracy3])

    print(np.array(acc))

if __name__ == '__main__':
    main()
