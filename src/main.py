from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Tea")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

# Specify the model directly when creating an instance of OpenAI class
openai_instance = OpenAI(model="gpt-3.5-turbo-instruct")

# Pass the OpenAI instance to the RetrievalQA constructor
qa = RetrievalQA.from_chain_type(llm=openai_instance, chain_type="stuff", retriever=docsearch.as_retriever())

while True:
    query = input("Ask a question about tea\n")
    print(qa.run(query))
