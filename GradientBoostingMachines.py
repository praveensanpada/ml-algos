# Gradient Boosting Machines (GBM)
# Example and Use Case:
# GBM is used for both regression and classification tasks. For instance, predicting customer churn based on usage patterns and demographics.

from sklearn.ensemble import GradientBoostingClassifier

# Example dataset
data = {
    'usage': np.random.rand(50) * 100,
    'age': np.random.randint(18, 70, 50),
    'account_age': np.random.randint(1, 10, 50),
    'churn': np.random.randint(0, 2, 50)  # 0: No, 1: Yes
}

df = pd.DataFrame(data)

# Prepare data
X = df[['usage', 'age', 'account_age']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting Classifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
