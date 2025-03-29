from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from lxml.html.clean import clean_html
import requests
import os

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=40)
all_splits = text_splitter.split_documents(docs_list)
all_splits[0]

# Cleanup documents as PostgreSQL won't accept the NUL character, '\x00', in TEXT fields.
for doc in all_splits:
    doc.page_content = doc.page_content.replace("\x00", "")

# Create the index and ingest the documents
embeddings = HuggingFaceEmbeddings()

CONNECTION_STRING = "postgresql+psycopg://postgres:password@localhost:5432/vectordb"
COLLECTION_NAME = "documents_test"

db = PGVector.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    use_jsonb=True,
    # pre_delete_collection=True # This deletes existing collection and its data, use carefully!
)