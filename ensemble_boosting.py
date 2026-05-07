#✅ Q1: Basic AdaBoost Classification
#📊 Problem
#Train an AdaBoost classifier and predict output.#

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=10
)
model.fit(X, y)
print("Prediction:", model.predict([[3.5]]))