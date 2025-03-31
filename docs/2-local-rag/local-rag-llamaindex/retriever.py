from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
import logging
import sys
import requests
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# the html_to_text=True option requires html2text to be installed
docs_list = SimpleWebPageReader(html_to_text=True).load_data(urls)

# We use some sensible defaults for text chunking and splitting
text_splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=40)

# Maintain relationship with source doc index, to help inject doc metadata
text_chunks = []
doc_idxs = []
for doc_idx, doc in enumerate(docs_list):
    cur_text_chunks = text_splitter.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

# Construct Nodes from Text Chunks
nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = docs_list[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

# Create the index and ingest the documents
embeddings = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# Generate Embeddings for each Node
for node in nodes:
    node_embedding = embeddings.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

DB_NAME="vectordb"
CONNECTION_STRING = "postgresql+psycopg://postgres:password@localhost:5432/vectordb"
COLLECTION_NAME = "documents_test"
url = make_url(CONNECTION_STRING)

db = PGVectorStore.from_params(
    database=DB_NAME,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name=COLLECTION_NAME,
    embed_dim=768,
)

db.add(nodes)
