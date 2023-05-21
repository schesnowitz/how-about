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
from datetime import datetime

dt = datetime.now()
repo_id = "nomic-ai/gpt4all-13b-snoozy" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
# llm = OpenAI(temperature=0.9, verbose=True)
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994? "

print(llm_chain.run(question))


loader = WebBaseLoader(
    "https://www.nytimes.com/2023/05/14/world/europe/turkey-erdogan-presidential-election.html"
)
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=10, length_function=len
)

docs = text_splitter.split_documents(data)


embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
db.add_documents(docs)

"""
-------------------------------------------------------------
Reporter Name
-------------------------------------------------------------
"""

prompt_template = """Use the context below to create a ficticious name for a news reporter.
    name: {name}
    context: {query}
    Reporter Name:"""

story_reporter = PromptTemplate(
    template=prompt_template, input_variables=["name", "query"]
)


chain = LLMChain(llm=llm, prompt=story_reporter, verbose=False)


query = "write a title for the story"
docs = db.similarity_search(query, k=1)
story_reporter_name = chain.run({"name": docs, "query": query})
print(story_reporter_name)
