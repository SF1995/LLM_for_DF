#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain openai weaviate-client')


# In[2]:


pip install python-dotenv


# In[3]:


import os

os.environ['OPENAI_API_KEY'] = 'your open ai key'


# In[4]:


import dotenv
dotenv.load_dotenv()


# In[5]:


import requests
from langchain.document_loaders import TextLoader

url = "https://raw.githubusercontent.com/SF1995/LLM_for_DF/main/e-mails.txt"
res = requests.get(url)
with open("e-mails.txt", "w") as f:
    f.write(res.text)

loader = TextLoader('./e-mails.txt')
documents = loader.load()


# In[6]:


from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)


# In[7]:


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

client = weaviate.Client(
  embedded_options = EmbeddedOptions()
)

vectorstore = Weaviate.from_documents(
    client = client,    
    documents = chunks,
    embedding = OpenAIEmbeddings(),
    by_text = False
)


# In[8]:


retriever = vectorstore.as_retriever()


# In[9]:


from langchain.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)


# In[10]:


from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

query = "Keep it short. What where the sequence of events that led to the incident at SecureTech and when did they happen?"
rag_chain.invoke(query)
