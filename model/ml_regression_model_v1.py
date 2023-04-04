import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from open_ai_api.davinci_email_classifier import check_email_by_level_legacy


class EmailClassifier:
    def __init__(self):
        self.df = None
        self.X_test_tfidf = None
        self.X_train_tfidf = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.vectorizer = TfidfVectorizer()
        self.clf = LogisticRegression(random_state=42)

    def load_dataset(self, path):
        self.df = pd.read_csv(path)

    def split_dataset(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df['text'], self.df['is_phishing'], test_size=test_size, random_state=random_state)

    def extract_features(self):
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test)

    def train_model(self):
        self.clf.fit(self.X_train_tfidf, self.y_train)

    def evaluate_model(self):
        accuracy = self.clf.score(self.X_test_tfidf, self.y_test)
        print("Accuracy:", accuracy)

    def print_misclassified_emails(self):
        y_pred_proba = self.clf.predict_proba(self.X_test_tfidf)
        y_pred = self.clf.predict(self.X_test_tfidf)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)
        misclassified = []
        for i in range(len(self.y_test)):
            if self.y_test[i] != y_pred[i]:
                misclassified.append(i)
        misclassified_prob_diff = []
        correctly_classified_prob_diff = []
        proba_all = self.clf.predict_proba(self.X_test_tfidf)
        for i in range(len(proba_all)):
            pred_prob = y_pred_proba[i][1]
            true_prob = y_pred_proba[i][0]
            prob_diff = abs(pred_prob - true_prob)
            if self.y_test[i] != y_pred[i]:
                misclassified_prob_diff.append(prob_diff)
            else:
                correctly_classified_prob_diff.append(prob_diff)
        diff_misclassified = sum(misclassified_prob_diff) / len(misclassified_prob_diff)
        diff_correct = sum(correctly_classified_prob_diff) / len(correctly_classified_prob_diff)
        print("Average difference in probability for misclassified emails:", diff_misclassified)
        print("Average difference in probability for correctly classified emails:", diff_correct)

    def evaluate_with_api(self):
        correct = 0
        total = 0
        for i, record in self.X_test.iteritems():
            X_test_tfidf = self.vectorizer.transform([record])
            proba = self.clf.predict_proba(X_test_tfidf)[0][1]
            if 0.46 < proba < 0.54:
                pred = check_email_by_level_legacy(record)
                if pred.status == "success":
                    print(self.clf.predict(X_test_tfidf), pred.value, self.y_test[i])
                    if pred.value == self.y_test[i]:
                        correct += 1
                else:
                    if self.clf.predict(X_test_tfidf) == self.y_test[i]:
                        correct += 1
            else:
                if self.clf.predict(X_test_tfidf) == self.y_test[i]:
                    correct += 1
            total += 1
        accuracy = correct / total
        print("Accuracy:", accuracy)

