import time

import numpy as np
import openai
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score

import config

openai.api_key = config.API_KEY

test = pd.read_json('../dataset/input/emails_set_modified.jsonl', lines=True)
test = test.head(60)

# Check the first email in the dataset
ft_model = "ada:ft-personal:binary-email-classification-v2-2023-04-22-16-44-05"
correct_guess = 0
failed = 0
attempt = 0
start = False
responses = []

for index, row in test.iterrows():
    response = {}
    try:
        if attempt % 60 == 0 and start:
            time.sleep(60)

        start = True
        res = openai.Completion.create(model=ft_model,
                                       prompt=row.prompt + '\n\n###\n\n',
                                       max_tokens=1,
                                       temperature=0,
                                       logprobs=10)

        response["value"] = int(res['choices'][0]['text'])
        response["actual"] = row.completion
        response["probability_of_true"] = res['choices'][0]['logprobs']['top_logprobs'][0][" 1"]
        response["probability_of_false"] = res['choices'][0]['logprobs']['top_logprobs'][0][" 0"]
        responses.append(response)

        print(response)
        attempt += 1

    except Exception as e:
        print(f"Error: {e}")
        pass

result = pd.DataFrame(responses)

actual = np.array(result['actual'])
predicted = np.array(result['value'])
probs = np.array(result['probability_of_true'])

# Print accuracy
accuracy = accuracy_score(actual, predicted)
print(f"Accuracy: {accuracy:.2f}")

# Plot confusion matrix
conf_matrix = confusion_matrix(actual, predicted)

plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i][j]), ha='center', va='center')
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(actual, probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=0.7,
         label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Random', alpha=.8)
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(actual, probs)
average_precision = average_precision_score(actual, probs)
plt.plot(recall, precision, lw=1, alpha=0.7,
         label='Precision-recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.show()

# Histogram of probabilities
plt.hist(probs, bins=10)
plt.xlabel('Probability of true')
plt.ylabel('Frequency')
plt.show()
