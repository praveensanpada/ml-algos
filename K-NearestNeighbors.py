# K-Nearest Neighbors (KNN)
# Example and Use Case:
# KNN is used for classification tasks like predicting the species of a plant based on its features such as petal length and width.

from sklearn.neighbors import KNeighborsClassifier

# Example dataset
data = {
    'petal_length': np.random.rand(50) * 5,
    'petal_width': np.random.rand(50) * 2,
    'species': np.random.randint(0, 3, 50)  # 0: Setosa, 1: Versicolor, 2: Virginica
}

df = pd.DataFrame(data)

# Prepare data
X = df[['petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
