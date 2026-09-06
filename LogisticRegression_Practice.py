# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import pickle


# Load the dataset
dataset = pd.read_csv(r"C:\Users\patharapilli\AVSCODE\10.ML\logit classification.csv")
# Check the shape of the dataset
print("Dataset Shape:", dataset.shape)  # (30, 2)

# Feature selection (independent variable x and dependent variable y)
x = dataset.iloc[:,[2,3]]  # Years of experience (Independent variable)
y = dataset.iloc[:, -1]   # Salary (Dependent variable)


# Split the dataset into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train =sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Fit the Logistic Regression model to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()  
classifier.fit(x_train, y_train)

# Predicting the results for the test set
y_pred = classifier.predict(x_test)

classifier.get_params()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ('confusion matrix',cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print('Accuracy',ac)

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print('classification Report',cr)

bias = classifier.score(x_train,y_train)
print('Bias',bias)


var = classifier.score(x_test,y_test)
print('Variance',var)

dataset1 = pd.read_csv(r"C:\Users\patharapilli\AVSCODE\filetest.csv")

d2 = dataset1.copy()
d2
dataset1 = dataset1.iloc[:,[2,3]]

d2
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()

M = sc.fit_transform(dataset1)
M
y_pred1 = pd.DataFrame()

d2 ['y_pred1'] = classifier.predict(dataset1)
d2
d2.to_csv('final1.csv')

import os 
os.getcwd()
