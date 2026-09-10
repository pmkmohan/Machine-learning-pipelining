# XGBoost

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"D:\NIT\1. NIT_Batches\1. MORNING BATCH\N_Batch -- 9AM _ Nov26\4. July\28th, 29th - Ensamble learning\7.XGBOOST\Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2]) 

print(X)
# One Hot Encoding the "Geography" column

from sklearn.compose import ColumnTransformer  
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', 
                                      OneHotEncoder(), [1])], 
                                      remainder='passthrough')
X = np.array(ct.fit_transform(X)) 


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training XGBoost on the Training set
from xgboost import XGBClassifier 
classifier = XGBClassifier() 
classifier.fit(X_train, y_train)  

# Predicting the Test set results
y_pred = classifier.predict(X_test)  

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac) 

bias = classifier.score(X_train,y_train)
bias 
 
# logistic 
# svm
# dt
# rf 

# Applying k-Fold Cross Validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100)) 
#print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))  














