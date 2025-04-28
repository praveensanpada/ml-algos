
# Linear Regression
# Example and Use Case:
# Linear Regression is used to predict a continuous outcome based on one or more predictor variables. For instance, predicting house prices based on features like size, number of bedrooms, and location.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Example dataset
data = {
    'size': np.random.randint(500, 4000, 50),
    'bedrooms': np.random.randint(1, 5, 50),
    'location': np.random.randint(1, 5, 50),
    'price': np.random.randint(100000, 500000, 50)
}

df = pd.DataFrame(data)

# Prepare data
X = df[['size', 'bedrooms', 'location']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
