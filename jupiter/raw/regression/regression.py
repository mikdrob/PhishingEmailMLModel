import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Load dataset
df = pd.read_csv('dataset/emails_set_batch.csv')

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['is_phishing'], test_size=0.4, random_state=42)

# Step 3: Convert email text into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train a Logistic Regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train_tfidf, y_train)

# Step 5: Evaluate the model on test data
accuracy = clf.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy)
