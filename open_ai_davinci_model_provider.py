import config
import openai

import pandas as pd

from EmailCheck import EmailCheckResponse
from util import parse_response_boolean, evaluate_email_content_davinci, parse_response_integer


def check_email_by_level_legacy(content):
    instruction_alert_level = "Please provide a response as single digit from 0 to 10 of how likely the given email " \
                              "content is a phishing email. \n \n"

    prompt = instruction_alert_level + content

    try:
        response = evaluate_email_content_davinci(prompt)

        return parse_response_integer(response)
    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
        return EmailCheckResponse(status="failed")


def check_email_legacy(content):
    instruction = "Please provide a binary response of 1 or 0 to indicate whether the given email content is " \
                  "more likely to be a phishing email. Where 1 - more likely to be a phishing email and 0 - more " \
                  "likely to be a legitimate email \n \n"

    prompt = instruction + content

    try:
        response = evaluate_email_content_davinci(prompt)

        return parse_response_boolean(response)
    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
        return EmailCheckResponse(status="failed")


# Load the CSV file into a DataFrame
df = pd.read_csv('dataset/emails_set.csv')

# Access the first row of the 'text' column and convert it to a string
first_row_text = str(df.loc[2, 'text'])

# print(check_email_by_level_legacy(first_row_text).value)
