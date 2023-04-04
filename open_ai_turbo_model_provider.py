import config
import openai

import pandas as pd

from EmailCheck import EmailCheckResponse
from util import parse_response_boolean

openai.api_key = config.API_KEY


def check_email(content):
    instruction = "Please provide a binary response of 1 or 0 to indicate whether the given email content is a " \
                  "phishing attempt or not."

    message = [{"role": "user", "content": content}, {"role": "assistant", "content": instruction}]

    try:
        completions = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message
        )

        response = completions.choices[0].message.content

        print(response)

        return parse_response_boolean(response)
    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
        return EmailCheckResponse(status="failed")


# Load the CSV file into a DataFrame
df = pd.read_csv('dataset/emails_set.csv')

# Access the first row of the 'text' column and convert it to a string
first_row_text = str(df.loc[0, 'text'])

check_email(first_row_text)
