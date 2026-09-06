# Importing the required liberiers 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading or importing the dataset
dataset = pd.read_csv(r'C:\Users\patharapilli\AVSCODE\10.ML\Salary_Data.csv')
dataset

# Split the data set to X and Y, (X-Independent variable and Y-Dependent Variable)
X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1].values

# checking if any data is missing in the dataset
dataset.isnull().sum()

# splitting the data to Xtrain and Xtest and Y train and Ytest in 80:20 ration
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train =sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Building the model using liner regression algorithem
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the results for the test set
Y_Pred =regressor.predict(X_test)

# Visualizing the Training set results
plt.scatter(X_train, Y_train, color = 'red')  # Real salary data (training)
plt.plot(X_train, regressor.predict(X_train), color = 'blue')  # Predicted regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test,Y_test, color = 'Yellow') #Real salar data (testing)
plt.plot(X_test,regressor.predict(X_test),color= 'Green') 
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print (f"Coefficient:{regressor.coef_}")
print(f"intercept:{regressor.intercept_}")

comparision = pd.DataFrame({'Actual':Y_test,'Predected':Y_Pred})
print(comparision)

# Predict salary of 12 and 20 years of experience using the trained model.
# we need y=mx+c
# for that we need m and c values
# 
m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

# Validation data
y_pred_12 = m_slope*12+c_intercept
print(y_pred_12)

y_pred_20 = m_slope*20+c_intercept
print(y_pred_20)

y_12 = regressor.predict([[12]])
y_12

y_20 = regressor.predict([[20]])
y_20

### Static on Machinelearning
dataset.mean()
dataset.median()
dataset['Salary'].mean()
dataset.mode()
dataset['Salary'].mode()
dataset.var()
dataset.std()
dataset.corr()
dataset['Salary'].corr(dataset["YearsExperience"])

#Checking Model performance
bias = regressor.score(X_train, Y_train)
print(bias)
variance =regressor.score(X_test,Y_test)
print(variance)

# calculating MSA, MSE and RMSE on test and traindata
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
#MAE
actual = Y_train
actual
predicted = regressor.predict(X_train)
predicted
mae = mean_absolute_error(actual,predicted)
print('MAE',mae)
mse = mean_squared_error(actual, predicted)
print('MSE',mse)
rmse = np.sqrt(mse)
print('RMSE',rmse)


# Predictions
Y_train_pred = regressor.predict(X_train)
Y_test_pred = regressor.predict(X_test)

# Calculate R-squared
train_r2 = r2_score(Y_train, Y_train_pred)
test_r2 = r2_score(Y_test, Y_test_pred)

# Calculate adjusted R-squared
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Number of predictors
p = X_train.shape[1]

# Number of observations in each dataset
n_train = len(Y_train)
n_test = len(Y_test)

train_adjusted_r2 = adjusted_r2(train_r2,
    n_train,
    p
)

test_adjusted_r2 = adjusted_r2(
    test_r2,
    n_test,
    p
)

# Display results
results = pd.DataFrame({
    "Dataset": ["Training", "Testing"],
    "R_squared": [train_r2, test_r2],
    "Adjusted_R_squared": [
        train_adjusted_r2,
        test_adjusted_r2
    ]
})

print(results)


# ANOVA to get the Model Score

Y_mean = np.mean(Y)
SSR = np.sum((Y_Pred - Y_mean)**2)
print(SSR)

Y =Y[0:6] # as we have only 6 records for testing.
SSE = np.sum((Y-Y_Pred)**2)
print(SSE)


mean_total = np.mean(dataset.values)
# here df.to_numpy() will convert pandas Dataframe to Numpy
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

r_square = 1-(SSR/SST)
r_square

print(r_square)  # simple liner model score
print(bias)  # bias score
print(variance) # variance score

#pickle to move large code from backend to from end like zipping the code and sending.

import pickle
# Save the model to disk
filename = 'linear_regression_model.pkl'
# Open the file in write-binary mode and use pickle to dump the model
with open(filename,'wb') as f:
    pickle.dump(regressor, f)
print(f"Model saved to {filename}")

import os
os.getcwd()  # Get the current working directory