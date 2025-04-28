# AdaBoost
# Example and Use Case:
# AdaBoost is used for classification tasks such as detecting fraud in financial transactions.

from sklearn.ensemble import AdaBoostClassifier

# Example dataset
data = {
    'transaction_amount': np.random.rand(50) * 1000,
    'transaction_count': np.random.randint(1, 50, 50),
    'account_age': np.random.randint(1, 10, 50),
    'fraud': np.random.randint(0, 2, 50)  # 0: No, 1: Yes
}

df = pd.DataFrame(data)

# Prepare data
X = df[['transaction_amount', 'transaction_count', 'account_age']]
y = df['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoost Classifier
model = AdaBoostClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
