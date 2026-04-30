#Q1: Ridge Basic
#📊 Problem
#X = [1,2,3,4]
#y = [2,4,6,8]
#👉 Train Ridge model and predict for x = 5
import numpy as np
from  sklearn.linear_model import Ridge
X = np.array([[1],[2],[3],[4]])
y = np.array([[2],[4],[6],[8]])
model=Ridge(alpha=1.0)
model.fit(X,y)
print(model.predict([[5]]))


#Q2: Lasso Basic
#📊 Problem
#Same dataset → use Lasso

import numpy as np
from sklearn.linear_model  import Lasso
X = np.array([[1],[2],[3],[4]])
y = np.array([[2],[4],[6],[8]])
model=Lasso(alpha=0.1)
model.fit(X,y)
print(model.predict([[5]]))


#Q3: Compare Ridge vs Lasso
#📊 Problem
#X = [1,2,3,4,5]
#y = [5,10,15,20,25]
#👉 Train both models and compare coefficients

import numpy as np
from sklearn.linear_model import Ridge, Lasso
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([[5],[10],[15],[20],[25]])
ridge=Ridge(alpha=1.0)
lasso=Lasso(alpha=0.1)
ridge.fit(X,y)
lasso.fit(X,y)
print("ridge coeff:",ridge.coef_)
print("lasso coeff:",lasso.coef_)

#Q4: Effect of Alpha
#📊 Problem
#👉 Train Ridge with:
#alpha = 0.01
#alpha = 10
#👉 Compare coefficients

import numpy as np
from sklearn.linear_model import Ridge
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([[5],[10],[15],[20],[25]])
ridge1=Ridge(alpha=1.0)
ridge2=Ridge(alpha=10)
ridge1.fit(X,y)
ridge2.fit(X,y)
print("ridge1:",ridge1.coef_)
print("ridge2:",ridge2.coef_)

#Multi-Feature Ridge
#📊 Problem
#X = [[1,2],[2,3],[3,4],[4,5]]
#y = [3,5,7,9]
#👉 Train Ridge and predict for [5,6]

import numpy as np 
from sklearn.linear_model import Ridge
X = np.array([[1,2],[2,3],[3,4],[4,5]])
y = np.array([3,5,7,9])
model=Ridge(alpha=1.0)
model.fit(X,y)
print(model.predict([[5,6]]))


#Lasso Feature Selection
#📊 Problem
#X = [[1,2,3],
#     [2,4,6],
#     [3,6,9],
#     [4,8,12]]
#y = [6,12,18,24]
#👉 Apply Lasso and check coefficients

import numpy as np
from sklearn.linear_model import Lasso
X = np.array([[1,2,3],
     [2,4,6],
     [3,6,9],
     [4,8,12]])
y = np.array([6,12,18,24])
model=Lasso(alpha=0.1)
model.fit(X,y)
print("lasso cofficient:",model.coef_)
