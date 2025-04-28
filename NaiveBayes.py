# Naive Bayes
# Example and Use Case:
# Naive Bayes is used for text classification tasks such as classifying news articles into different categories like sports, politics, and technology.

from sklearn.naive_bayes import MultinomialNB

# Example dataset
data = {
    'word_freq': np.random.rand(50, 10),  # 10 features representing word frequencies
    'category': np.random.randint(0, 3, 50)  # 0: Sports, 1: Politics, 2: Technology
}

df = pd.DataFrame(data)

# Prepare data
X = df.drop('category', axis=1)
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
