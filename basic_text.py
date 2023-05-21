import os
from secret import OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN , PGDATABASE, PGHOST, PGPASSWORD, PGPORT, PGUSER
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN 
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
from datetime import datetime

dt = datetime.now()

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])


local_path = './models/ggml-gpt4all-l13b-snoozy.bin'  # replace with your desired local file path

# import requests

from pathlib import Path
from tqdm import tqdm

Path(local_path).parent.mkdir(parents=True, exist_ok=True)

# # Example model. Check https://github.com/nomic-ai/gpt4all for the latest models.
url = 'http://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin'

# send a GET request to the URL to download the file. Stream since it's large
response = requests.get(url, stream=True)

# open the file in binary mode and write the contents of the response to it in chunks
# This is a large file, so be prepared to wait.
with open(local_path, 'wb') as f:
    for chunk in tqdm(response.iter_content(chunk_size=8192)):
        if chunk:
            f.write(chunk)


# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
# If you want to use a custom model add the backend parameter
# Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
llm = GPT4All(model=local_path, backend='gptj', callbacks=callbacks, verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)


