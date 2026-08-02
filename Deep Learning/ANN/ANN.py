import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import pandas as pd

df=pd.DataFrame({
    "soil_moisture": [0.10, 0.15, 0.20, 0.25, 0.40, 0.60, 0.35, 0.18,
                      0.45, 0.05, 0.80, 0.27, 0.55, 0.70, 0.12, 0.30],
    "temperature_c": [34, 30, 26, 22, 28, 30, 19, 22,
                      35, 24, 33, 33, 21, 25, 20, 29],
    "sunlight_hours": [9, 8, 7, 4, 8, 10, 3, 10,
                       12, 5, 9, 11, 2, 6, 1, 9],
    "needs_water": [1, 1, 1, 0, 0, 0, 0, 1,
                    0, 1, 0, 1, 0, 0, 1, 1]
})
     
print(df)
print(df.columns)
X=df[["soil_moisture","temperature_c","sunlight_hours"]]
y=df[["needs_water"]]
#it is important to scale the features before training a neural network, as it can help improve the convergence of the model and prevent issues 
# such as vanishing or exploding gradients. Scaling the features ensures that they are all on a similar scale, which can help the model learn more effectively. In this case, we are using min-max scaling to scale the features to a range between 0 and 1.
X_max=X.max()
X_min=X.min()
X_scaled=(X-X_min) / (X_max- X_min + 1e-8) 
print(X_scaled)

#it is important to stratify the data to ensure that the distribution of the target variable is similar in both the training and testing sets.
#  This is especially important when dealing with imbalanced datasets, where one class may be underrepresented. By stratifying the data, we can ensure that both the training and testing sets have a similar distribution of the target variable, which can help improve the performance of the model.
X_train , X_test , y_train , y_test=train_test_split(
    X_scaled,y, test_size=0.25,random_state=42,stratify=y
)

#this is a simple feedforward neural network with two hidden layers. The first hidden layer has 8 neurons and uses the ReLU activation function, 
# while the second hidden layer has 4 neurons and also uses the ReLU activation function. The input layer takes in the features from the training data, which has a shape of (X_train.shape[1],). The output layer is not defined in this code snippet, but it would typically have a single neuron for regression tasks or multiple neurons for classification tasks, depending on the problem being solved.
model=keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(8,activation='relu'),
    layers.Dense(1,activation="sigmoid")
])

model.compile(optimizer="sgd",loss="binary_crossentropy", metrics=["accuracy"])
history_full = model.fit(X_train, y_train, epochs=100, batch_size=len(X_train), verbose=1)
history =model.fit(
    X_train.values,y_train.values,
    validation_data=(X_test.values,y_test.values),
    epochs=100,batch_size=1,verbose=1

)
history_minibatch = model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1)

from tensorflow.keras import optimizers
     
#for better noise reduction use momentum with sdg
opt = optimizers.SGD(learning_rate = 0.01, momentum = 0.9)
     

     