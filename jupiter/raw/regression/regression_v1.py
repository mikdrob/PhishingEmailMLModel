import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('../../dataset/input/emails_set.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['is_phishing'], test_size=0.2, random_state=42)

# Tokenize text into words
df['text'] = df['text'].apply(lambda x: word_tokenize(x.lower()))

# Remove stop words
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Stem words
stemmer = PorterStemmer()
df['text'] = df['text'].apply(lambda x: [stemmer.stem(word) for word in x])

# Join words back into a string
df['text'] = df['text'].apply(lambda x: " ".join(x))

# Add feature for email length
df['email_length'] = df['text'].apply(lambda x: len(x))

from sklearn.ensemble import RandomForestClassifier

# Convert email text into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Random Forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_tfidf, y_train)

# Evaluate the Random Forest classifier model on test data
random_forest_accuracy = rf.score(X_test_tfidf, y_test)
print("Accuracy of Random Forest classifier:", random_forest_accuracy)


# Train a Logistic Regression
clf = LogisticRegression(random_state=42)
clf.fit(X_train_tfidf, y_train)

# Evaluate the Logistic Regression model on test data
logistic_regression = clf.score(X_test_tfidf, y_test)
print("Accuracy of Logistic Regression:", logistic_regression)
