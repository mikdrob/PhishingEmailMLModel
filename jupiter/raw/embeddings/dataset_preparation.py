import openai
import pandas as pd
import tiktoken

from openai.embeddings_utils import get_embedding

import config

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# load & inspect dataset
input_datapath = "../dataset/input/emails_set.csv"  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, index_col=0)
df = df[["text", "is_phishing"]]
df = df.dropna()

# subsample to 1k most recent reviews and remove samples that are too long

encoding = tiktoken.get_encoding(embedding_encoding)

# omit reviews that are too long to embed
df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens]
len(df)

# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage
openai.api_key = config.API_KEY

# This may take a few minutes
df["embedding"] = df.text.apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("../dataset/embeddings.csv")
