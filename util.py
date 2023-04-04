import openai

import config
from EmailCheck import EmailCheckResponse

openai.api_key = config.API_KEY


def parse_response_boolean(input_str):
    try:
        bool_val = bool(int(input_str))
        return EmailCheckResponse(status="success", value=bool_val)
    except ValueError:
        return EmailCheckResponse(status="failed")

def parse_response_integer(input_str):
    try:
        alert_level = int(input_str)
        is_phishing = alert_level > 5
        return EmailCheckResponse(status="success", value=is_phishing)
    except ValueError:
        return EmailCheckResponse(status="failed")


def evaluate_email_content_davinci(prompt):
    completions = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    response = completions.choices[0].text.strip()
    return response
