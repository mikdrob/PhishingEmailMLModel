import config
import openai

import pandas as pd

openai.api_key = config.API_KEY


def check_email(content):
    instruction = "Please provide a binary response of 1 or 0 to indicate whether the given email content is a " \
                  "phishing attempt or not."

    message = [{"role": "user", "content": content}, {"role": "assistant", "content": instruction}]

    try:
        completions = openai.ChatCompletion.create(
            model="gpt-3.5-turbo1",
            messages=message
        )

        response = completions.choices[0].message.content

        print(response)

        return parse_response(response)
    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
        return {"status": "failed"}


def parse_response(input_str):
    try:
        bool_val = bool(int(input_str))
        return {"status": "success", "value": bool_val}
    except ValueError:
        return {"status": "failed"}


# Load the CSV file into a DataFrame
df = pd.read_csv('dataset/emails_set.csv')

# Access the first row of the 'text' column and convert it to a string
first_row_text = str(df.loc[0, 'text'])

check_email(first_row_text)
