# Ex-1: Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.


## THEORY

Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
<img width="822" height="674" alt="image" src="https://github.com/user-attachments/assets/64bb95ee-009b-4111-a27f-0d032e296627" />


## DESIGN STEPS

**STEP 1: Generate Dataset**
Create input values from 1 to 50 and add random noise to introduce variations in output values .

**STEP 2: Initialize the Neural Network Model**
Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

**STEP 3: Define Loss Function and Optimizer**
Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

**STEP 4: Train the Model**
Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

**STEP 5: Plot the Loss Curve**
Track the loss function values across epochs to visualize convergence.

**STEP 6: Visualize the Best-Fit Line**
Plot the original dataset along with the learned linear model.

**STEP 7: Make Predictions**
Use the trained model to predict for a new input value .

## PROGRAM

**Name:** Rahul M R

**Register Number:** 2305003005
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df=df.astype({'INPUT':'float'})
df=df.astype({'OUTPUT':'float'})
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df[['INPUT']].values
y = df[['OUTPUT']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

model=Sequential([
    #Hidden ReLU Layers
    Dense(units=5,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    #Linear Output Layer
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=3000)

loss= pd.DataFrame(model.history.history)
loss.plot()

X_test1 =Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1=[[4]]
X_n1_1=Scaler.transform(X_n1)
model.predict(X_n1_1)
```


**Dataset Information**

<img width="538" height="453" alt="image" src="https://github.com/user-attachments/assets/1ac233e8-8c4e-4e77-bad1-ec660cc616ce" />


## OUTPUT

**Training Loss Vs Iteration Plot:**

<img width="560" height="413" alt="image" src="https://github.com/user-attachments/assets/9821da4c-c9c7-42fc-ac31-78e0649353e5" />


**Epoch Training:**

<img width="1032" height="337" alt="image" src="https://github.com/user-attachments/assets/9e105828-173a-4026-a8d0-51d9587adb65" />

**Test Data Root Mean Squared Error:**

<img width="768" height="66" alt="image" src="https://github.com/user-attachments/assets/c02b40d8-1760-4ef5-8898-adc106362c2d" />

**New Sample Data Prediction:**

<img width="641" height="65" alt="image" src="https://github.com/user-attachments/assets/5deb473c-7c52-425e-bab7-64684d0e621e" />


## RESULT

Thus, a neural network regression model was successfully developed and trained using PyTorch.
