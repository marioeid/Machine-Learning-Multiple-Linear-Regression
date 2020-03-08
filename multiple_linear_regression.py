# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing our data set
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:, :-1].values 
y=dataset.iloc[:,4].values

# Encoding Categorial Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# sklearn new format for one hot encoding check the medium website in the book marks
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X=np.array(columnTransformer.fit_transform(X), dtype = np.float)

# Avoiding the dummy variable trap
X=X[:,1:] # the library will take care of it but we will leave it to remind our selves

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scalling 
# we don't need to apply feature scalling in multiple linear regression cause the library will do it for us
# so we are ready

# Fitting Multiple linear regression to our training set 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results 
y_pred=regressor.predict(X_test)

# Building The optimal model using back elimination
import statsmodels.api as sm
# adding x0=1 cause sm library won't included 
X=np.append(arr=np.ones((X.shape[0],1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

# do back elimnation by removing the feature variable with the highest p-value
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

# do back elimnation by removing the feature variable with the highest p-value
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

# do back elimnation by removing the feature variable with the highest p-value
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

# do back elimnation by removing the feature variable with the highest p-value
X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

# do back elimnation by removing the feature variable with the highest p-value
X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

# doing the backwardElimination generally not with hand 
def backwardElimination(X, sl):
    features=len(X[0])
    for i in range(0,features):
        regressor_OLS=sm.OLS(y,X).fit()
        maxi=max(regressor_OLS.pvalues).astype(float)
        if maxi>sl:
            for j in range(0,features-i):
                if (regressor_OLS.pvalues[j].astype(float)==maxi):
                    X=np.delete(X,j,1)
    regressor_OLS.summary()
    return X

X_opt=backwardElimination(X,0.05)