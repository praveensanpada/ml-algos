# Support Vector Machines (SVM)
# Example and Use Case:
# SVM is used for classification tasks such as detecting spam emails based on features like email content and metadata.

from sklearn.svm import SVC

# Example dataset
data = {
    'word_count': np.random.randint(50, 500, 50),
    'special_chars': np.random.randint(0, 50, 50),
    'uppercase_words': np.random.randint(0, 50, 50),
    'spam': np.random.randint(0, 2, 50)  # 0: Not Spam, 1: Spam
}

df = pd.DataFrame(data)

# Prepare data
X = df[['word_count', 'special_chars', 'uppercase_words']]
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Classifier
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
