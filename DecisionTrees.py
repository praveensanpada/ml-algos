# Decision Trees
# Example and Use Case:
# Decision Trees can be used for both regression and classification tasks. For example, predicting the type of fruit based on features like color, weight, and texture.

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Example dataset
data = {
    'color': np.random.randint(1, 4, 50),  # 1: Red, 2: Green, 3: Yellow
    'weight': np.random.randint(100, 500, 50),
    'texture': np.random.randint(1, 3, 50),  # 1: Smooth, 2: Rough
    'type': np.random.randint(0, 2, 50)  # 0: Apple, 1: Orange
}

df = pd.DataFrame(data)

# Prepare data
X = df[['color', 'weight', 'texture']]
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
