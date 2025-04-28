# Neural Networks (Multilayer Perceptrons)
# Example and Use Case:
# Neural Networks can be used for image recognition tasks like recognizing digits in handwritten images.

from sklearn.neural_network import MLPClassifier

# Example dataset
data = {
    'feature1': np.random.rand(50),
    'feature2': np.random.rand(50),
    'feature3': np.random.rand(50),
    'label': np.random.randint(0, 2, 50)  # Binary classification
}

df = pd.DataFrame(data)

# Prepare data
X = df[['feature1', 'feature2', 'feature3']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Classifier
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
