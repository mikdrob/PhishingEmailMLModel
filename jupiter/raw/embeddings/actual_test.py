import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve, \
    precision_recall_curve, confusion_matrix

# Define the path to the input file
INPUT_FILE_PATH = "../dataset/input/embeddings.csv"

# Load the data from the CSV file
df = pd.read_csv(INPUT_FILE_PATH)

# Convert the string embeddings to arrays
df["embedding"] = df.embedding.apply(eval).apply(np.array)
embedding_values = list(df.embedding.values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    embedding_values, df.is_phishing, test_size=0.3, random_state=42
)

# Train a random forest classifier on the training set
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Make predictions on the test set
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)

# Evaluate the performance of the classifier using various metrics
report = classification_report(y_test, preds, digits=4)
accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probas[:, 1])

print("Classification Report:\n", report)
print("Accuracy: {:.5f}".format(accuracy))
print("F1 Score: {:.5f}".format(f1))
print("ROC AUC Score: {:.5f}".format(roc_auc))

# Compute precision, recall and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])

# Plot the precision-recall curve
plt.plot(recall, precision, color='blue', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Compute the false positive rate and true positive rate for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
roc_auc = roc_auc_score(y_test, probas[:, 1])

# Plot the ROC curve
plt.plot(fpr, tpr, linestyle='-', label='ROC curve (AUC = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, preds)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not phishing', 'Phishing'], rotation=45)
plt.yticks(tick_marks, ['Not phishing', 'Phishing'])
plt.tight_layout()
plt.xlabel('Predicted label')
plt.ylabel('True label')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.show()