import os

import pandas as pd

# Load the CSV file into a DataFrame
current_dir = os.path.dirname(__file__)

# Load the CSV file using the relative path
df = pd.read_csv(os.path.join(current_dir, "generated_emails.csv"))

# Remove rows where the 'is_phishing' column is empty or not an integer
df = df[pd.to_numeric(df['is_phishing'], errors='coerce').notnull()]
df['is_phishing'] = df['is_phishing'].astype(int)

# Remove rows where the 'example_what_would_non_legitimate_email_look_like' column is shorter than 100 characters
df = df[df['example_what_would_non_legitimate_email_look_like'].str.len() >= 100]

# Melt the third column and create a new DataFrame with the inserted rows
df = pd.concat([df, df['example_what_would_non_legitimate_email_look_like'].apply(lambda x: pd.Series({'text': x, 'is_phishing': 1}))])

# Drop the third column
df = df.drop(columns='example_what_would_non_legitimate_email_look_like')

# Reset the index
df = df.reset_index(drop=True)

# Remove duplicates
df = df.drop_duplicates()

# df = df.sample(frac=1)

# Save the modified DataFrame object to a new CSV file
df.to_csv(os.path.join(current_dir, "generated_emails_preprocessed.csv"), index=False)

# Rename columns
df = df.rename(columns={'text': 'prompt', 'is_phishing': 'completion'})

df.to_json(os.path.join(current_dir, "generated_emails_preprocessed.jsonl"), orient='records', lines=True)
