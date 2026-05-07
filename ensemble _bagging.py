#BAGGING
#✅ Q1: Basic Bagging Classification
#📊 Problem
#Train a Bagging classifier and predict output.#
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
model=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators =10
)
model.fit(X,y)
print("prediction:",model.predict([[3.5]]))

#✅ Q2: Calculate Accuracy
#📊 Problem
#Find training accuracy.
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
model=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators =10
)
model.fit(X,y)
y_pred=model.predict(X)
print("accuracy:",accuracy_score(y,y_pred))

#✅ Q3: Change Number of Estimators
#📊 Problem
#Train Bagging model using:
#5 trees
#50 trees
#Compare accuracy.#
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X= np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])

model1=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=5
)
model2=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50
)
model1.fit(X,y)
model2.fit(X,y)


print("accuracy with 5 trees:",accuracy_score(y,model1.predict(X)))
print("accuracy with 50 trees:",accuracy_score(y,model2.predict(X)))

#✅ Q4: Train-Test Split with Bagging
#📊 Problem
#Evaluate Bagging on test data.#
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X= np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25
)
model=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=5
)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("accuracy on test data:",accuracy_score(y_test,y_pred))

#✅ Q5: Bagging Regressor
#📊 Problem
#Predict continuous values.
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
X= np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
model=BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=5
)
model.fit(X,y)
print("prediction:",model.predict([[2.5]]))

#✅ Q6: Compare Decision Tree vs Bagging
#📊 Problem
#Compare# accuracy.
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X= np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
tree=DecisionTreeClassifier()
bagging=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=5
)
tree.fit(X,y)
bagging.fit(X,y)
pred_tree=tree.predict(X)
pred_bagging=bagging.predict(X)
print("tree accuracy:",accuracy_score(y,pred_tree))
print("bagging accuracy:",accuracy_score(y,pred_bagging))

#✅ Q7: OOB Score (VERY IMPORTANT 🔥)
#📊 Problem
#Use Out-of-Bag evaluation.#
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([0,0,0,0,1,1,1,1])

model=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    oob_score=True  #Enable Out-of-Bag evaluation
)
model.fit(X,y)
print("oobscore:",model.oob_score_)

#✅ Q8: Feature Sampling in Bagging
#📊 Problem
#Use random feature subsets.#
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
X = np.array([
    [1,2],
    [2,3],
    [3,4],
    [4,5],
    [5,6]
])
y = np.array([0,0,0,1,1])

model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=20,
    max_features=1  #Randomly select 1 feature for each tree
)

model.fit(X, y)
print("Prediction:",
      model.predict([[3,3]]))







