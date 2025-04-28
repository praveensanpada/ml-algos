# Logistic Regression
# Example and Use Case:
# Logistic Regression is used for binary classification problems. For example, predicting whether a customer will buy a product (yes/no) based on features like age, income, and past purchases.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example dataset
data = {
    'age': np.random.randint(18, 70, 50),
    'income': np.random.randint(20000, 120000, 50),
    'past_purchases': np.random.randint(0, 20, 50),
    'buy': np.random.randint(0, 2, 50)  # 0 or 1
}

df = pd.DataFrame(data)

# Prepare data
X = df[['age', 'income', 'past_purchases']]
y = df['buy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
