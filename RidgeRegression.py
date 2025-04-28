# Ridge Regression
# Example and Use Case:
# Ridge Regression is used for regression tasks with regularization to prevent overfitting. For example, predicting sales based on advertising budget in different media channels.

from sklearn.linear_model import Ridge

# Example dataset
data = {
    'tv': np.random.rand(50) * 300,
    'radio': np.random.rand(50) * 50,
    'newspaper': np.random.rand(50) * 100,
    'sales': np.random.rand(50) * 30
}

df = pd.DataFrame(data)

# Prepare data
X = df[['tv', 'radio', 'newspaper']]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
