# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 20:05:00 2020

@author: vssal
"""
import pandas as pd
import pickle

dataset = pd.read_csv(r'C:\Users\vssal\OneDrive\Desktop\ML\kaggle_diabetes_edit.csv')
X = dataset.iloc[:, :6].values
y = dataset.iloc[:, 6].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='mean')
imputer.fit(X[:, 2:5])
X[:, 2:5] = imputer.transform(X[:, 2:5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.26, random_state = 0, shuffle= True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

filename = 'diabPCKL.pkl'
pickle.dump(classifier, open(filename, 'wb'))