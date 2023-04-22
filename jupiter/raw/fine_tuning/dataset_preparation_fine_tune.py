from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import openai
#
# input_datapath = "../dataset/emails_set.csv"  # to save space, we provide a pre-filtered dataset
# df = pd.read_csv(input_datapath, index_col=0)
# df = df[["text", "is_phishing"]]
# df = df.rename(columns={"text": "prompt", "is_phishing": "completion"})
# df = df.dropna()
#
# # Calculate the split index
# split_idx = int(len(df) * 0.7)
#
# # Split into training and testing sets
# train_data = df.iloc[:split_idx]
# test_data = df.iloc[split_idx:]
#
# train_data.to_json("../dataset/input/emails_set_train.jsonl", orient='records', lines=True)
# test_data.to_json("../dataset/input/emails_set_test.jsonl", orient='records', lines=True)


# read in the JSON file and convert it to a DataFrame
df = pd.read_json('../dataset/input/emails_set_test.jsonl', lines=True)

# create a new column that contains the length of the desired string property
df['property_length'] = df['prompt'].apply(lambda x: len(x))

# sort the DataFrame based on the length of the property
sorted_df = df.sort_values(by='property_length')

sorted_df.to_json('../dataset/input/emails_set_test_sorted.jsonl', orient='records', lines=True)