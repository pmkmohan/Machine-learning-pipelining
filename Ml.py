# Importing liberary

import pandas as pd

# Importing the dataset from csv file

dataset = pd.read_csv(r'C:\Users\DELL\AVSCODE\10.ML\Data.csv')
# Identifying the indpendent and dependent variable
# X Independent variable iloc- Index location
X = dataset.iloc[:,:-1].values
# Y Dependent variable
Y = dataset.iloc[:,3].values

# importing SKLEARN Framework for data cleaning. Impute: is Transformers for missing value imputation

from sklearn.impute import SimpleImputer
# SimpleImpute had filled the mean value.
imputer = SimpleImputer()
# for paramater tuning/ Hyper parameter tuning.
# imputer = SimpleImputer(strategy='median')
X
imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

# Imputing categorical value for Independent variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
# labelencoder_X.fit_transform(X[:,0])
X[:,0] = labelencoder_X.fit_transform(X[:,0])

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size =0.3,train_size= 0.7,random_state=0)
