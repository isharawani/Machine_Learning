# Train a model using:
#X = [1,2,3,4]
#y = [5,10,15,20]
#👉 Predict for x = 5 
import numpy as np 
from sklearn.linear_model import LinearRegression

X =np.array([[1],[2],[3],[4]])
y = np.array([[5],[10],[15],[20]])
model=LinearRegression()
model.fit(X,y)
print(model.predict([[5]]))

#Q2: SALARY PREDICTION
#Experience = [1,2,3,4,5]
#Salary = [15000,25000,35000,45000,55000]
#👉 Predict salary for 6 years
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([[15000],[25000],[35000],[45000],[55000]])
model=LinearRegression()
model.fit(X , y)
print(model.predict([[6]]))

#MULTIPLE FEATURES
#📊 Problem:
#X = [[1,2],[2,3],[3,4],[4,5]]
#y = [6,9,12,15]
#👉 Predict for [5,6]#

import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1,2],[2,3],[3,4],[4,5]])
y = np.array([6,9,12,15])
model = LinearRegression()
model.fit(X,y)
print(model.predict([[5,6]]))

#Q4: MODEL WITH METRICS 🔥
#📊 Problem:
#X = [1,2,3,4,5]
#y = [10,20,30,40,50]#
#👉 Find:
#MAE,MSE,R²

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([10,20,30,40,50])
model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(X)
print("MAE:",mean_absolute_error(y, y_pred))
print("MSE:",mean_squared_error(y, y_pred))
print("R2:",r2_score(y, y_pred))


#Q5: MODIFY & EXPERIMENT (IMPORTANT)
#📊 Problem:
#X = [1,2,3,4,5]
#y = [12,22,32,42,52]
#👉 Tasks:
#Train model
#Predict for 6
#Print slope & intercept
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([12,22,32,42,52])
model = LinearRegression()
model.fit(X,y)
print("prediction:" , model.predict([[6]]))
print("SLOPE :", model.coef_)
print("intercept:" , model.intercept_)