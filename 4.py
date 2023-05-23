import os
from secret import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime

dt = datetime.now()

llm = OpenAI(temperature=0.9, verbose=True)


loader = WebBaseLoader(
    "https://www.nytimes.com/2023/05/14/world/europe/turkey-erdogan-presidential-election.html"
)
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=250, length_function=len
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
the name should be less than 25 characters, only generate a full name:
    name: {name}
    context: {query}
    Reporter Name:"""

story_reporter = PromptTemplate(
    template=prompt_template, input_variables=["name", "query"]
)


chain = LLMChain(llm=llm, prompt=story_reporter, verbose=False)


query = "write a title for the story"
docs = db.similarity_search(query, k=3)
story_reporter_name = chain.run({"name": docs, "query": query})
print(story_reporter_name)


"""
-------------------------------------------------------------
Story Title
-------------------------------------------------------------
"""

prompt_template = """Use the context below to write  a title.:
    Context: {title}
    Topic: {query}
    Blog title:"""

title_content = PromptTemplate(
    template=prompt_template, input_variables=["title", "query"]
)


chain = LLMChain(llm=llm, prompt=title_content, verbose=False)


query = "write a title for the story"
docs = db.similarity_search(query, k=3)
story_title = chain.run({"title": docs, "query": query})
print(story_title)

"""
-------------------------------------------------------------
Story Content
-------------------------------------------------------------
"""

prompt_template = """Use the context below to write a 700 word blog post about the context below:
    Context: {context}
    Topic: {query}
    Blog post:"""

prompt_content = PromptTemplate(
    template=prompt_template, input_variables=["context", "query"]
)


chain = LLMChain(llm=llm, prompt=prompt_content, verbose=False)


query = "write a detailed synopsis of this story"
docs = db.similarity_search(query, k=3)
story_content = chain.run({"context": docs, "query": query})
print((story_content))