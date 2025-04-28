# Quadratic Discriminant Analysis (QDA)
# Example and Use Case:
# QDA is similar to LDA but assumes each class has its own covariance matrix. It can be used for tasks where the classes have different distributions.

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Example dataset
data = {
    'feature1': np.random.rand(50),
    'feature2': np.random.rand(50),
    'feature3': np.random.rand(50),
    'class': np.random.randint(0, 3, 50)  # 3 classes
}

df = pd.DataFrame(data)

# Prepare data
X = df[['feature1', 'feature2', 'feature3']]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# QDA
model = QuadraticDiscriminantAnalysis()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
