# Lasso Regression
# Example and Use Case:
# Lasso Regression is used for regression tasks with L1 regularization, which can shrink some coefficients to zero, effectively performing feature selection. For instance, predicting property prices based on various features.

from sklearn.linear_model import Lasso

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

# Lasso Regression
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
