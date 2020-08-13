# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 01:08:00 2020

@author: galla
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import metrics

#Reading data as pandas dataframe
frame = pd.read_csv('F:/Downloads/fake_or_real_news.csv')

#Inspecing Shape
frame.shape

#Inspecting top 5 rows
frame.head()
frame = frame.set_index("Unnamed: 0")
frame.head()
y = frame.label
y.head()
frame.drop("label", axis=1)
frame.head()



# Create a series to store the labels: y
y = frame['label']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(frame["text"],y,test_size=0.33,random_state=53)



#By Using Count Vectorizer
# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words="english")

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)


# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])


#By Using Tf-IDF Vectorizer
# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english",max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.toarray()[:5])

# Create the CountVectorizer DataFrame: count_df
count_df =pd.DataFrame(count_train.toarray(), columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.toarray(),columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))



# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(count_train,y_train)

pred = nb_classifier.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print(score)

cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
print(cm)

nb_classifier = MultinomialNB()
nb_classifier.fit(tfidf_train,y_train)

pred = nb_classifier.predict(tfidf_test)
score = metrics.accuracy_score(y_test,pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test,pred,labels=['FAKE','REAL'])
print(cm)

# Create the list of alphas: alphas
alphas = np.arange(0,1,0.1)

def train_and_predict(alpha):
    nb_classifier = MultinomialNB(alpha=alpha)
    nb_classifier.fit(tfidf_train,y_train)
    pred = nb_classifier.predict(tfidf_test)
    score = metrics.accuracy_score(y_test,pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()

class_labels = nb_classifier.classes_
feature_names = tfidf_vectorizer.get_feature_names()
feat_with_weights = sorted(zip(nb_classifier.coef_[0],feature_names))
print(class_labels[0], feat_with_weights[:20])
print(class_labels[1], feat_with_weights[-20:])
