# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import pickle


# Load the dataset
dataset = pd.read_csv(r"C:\Users\DELL\AVSCODE\10.Ml\logit classification.csv")
# Check the shape of the dataset
print("Dataset Shape:", dataset.shape)  # (30, 2)

# Feature selection (independent variable x and dependent variable y)
x = dataset.iloc[:,[2,3]]  # Years of experience (Independent variable)
y = dataset.iloc[:, -1]   # Salary (Dependent variable)


# Split the dataset into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)

# Fit the Logistic Regression model to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()  
classifier.fit(x_train, y_train)

# Predicting the results for the test set
y_pred = classifier.predict(x_test)

classifier.get_params()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr)






