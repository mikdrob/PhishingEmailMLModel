import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Load dataset
df = pd.read_csv('dataset/emails_set.csv')

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

# Step 6: Print misclassified emails with probabilities
y_pred_proba = clf.predict_proba(X_test_tfidf)
y_pred = clf.predict(X_test_tfidf)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
misclassified = []
for i in range(len(y_test)):

    if y_test[i] != y_pred[i]:
        misclassified.append(i)


misclassified_prob_diff = []
correctly_classified_prob_diff = []
proba_all = clf.predict_proba(X_test_tfidf)
for i in range(len(proba_all)):
    pred_prob = y_pred_proba[i][1]
    true_prob = y_pred_proba[i][0]
    prob_diff = abs(pred_prob - true_prob)
    if y_test[i] != y_pred[i]:
        misclassified_prob_diff.append(prob_diff)
    else:
        correctly_classified_prob_diff.append(prob_diff)



print("Average difference in probability for misclassified emails:", sum(misclassified_prob_diff)/len(misclassified_prob_diff))
print("Average difference in probability for correctly classified emails:", sum(correctly_classified_prob_diff)/len(correctly_classified_prob_diff))
