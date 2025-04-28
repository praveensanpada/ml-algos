# Elastic Net Regression
# Example and Use Case:
# Elastic Net is a linear regression model with both L1 and L2 regularization. It is useful when there are multiple features that are correlated. For example, predicting stock prices based on various financial indicators.

from sklearn.linear_model import ElasticNet

# Example dataset
data = {
    'feature1': np.random.rand(50),
    'feature2': np.random.rand(50),
    'feature3': np.random.rand(50),
    'price': np.random.rand(50) * 1000
}

df = pd.DataFrame(data)

# Prepare data
X = df[['feature1', 'feature2', 'feature3']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Elastic Net Regression
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
