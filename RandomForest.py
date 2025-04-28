# Random Forest
# Example and Use Case:
# Random Forest is an ensemble method used for classification and regression. It can be used to predict credit risk based on various customer features.

from sklearn.ensemble import RandomForestClassifier

# Example dataset
data = {
    'age': np.random.randint(18, 70, 50),
    'income': np.random.randint(20000, 120000, 50),
    'credit_score': np.random.randint(300, 850, 50),
    'default': np.random.randint(0, 2, 50)  # 0: No, 1: Yes
}

df = pd.DataFrame(data)

# Prepare data
X = df[['age', 'income', 'credit_score']]
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
