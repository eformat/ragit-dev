# Local RAG with LangChain

[LangChain](https://python.langchain.com/docs/tutorials/rag/) is a very popular open source toolkit and framework for developing applications powered by LLM's.

We are going to use a number of popular tools to build our local RAG including

- [LangChain](https://python.langchain.com/docs/introduction/) framework.
- [Chainlit](https://docs.chainlit.io/get-started/overview) to build a conversational user interface.
- [Postgres + Vector Extentsion](https://github.com/pgvector/pgvector) as the vector store.
- [vLLM](https://docs.vllm.ai/en/latest/) a library to server our LLM for inference and serving.

## Pre-requsites

- Laptop with a Nvidia GPU is preferable
- Python3.11+ installed
- Podman
- Fast internet

?> **Tip** if you do not have a GPU, it is possible to run this example on a CPU only. You can find the code and instructions here https://github.com/eformat/rag-local-intel-cpu. It is a lot (10x) slower though that on a GPU!

## Setup a workspace and project

```bash
mkdir ~/rag-langchain && cd ~/rag-langchain
```

Setup a venv

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies.

```bash
pip install --upgrade --quiet \
  langchain \
  langchain_community \
  langchain_huggingface \
  lxml[html_clean] \
  bs4 \
  psycopg \
  pgvector
```

## Run a local postgres database

Pull the image.

```bash
podman pull quay.io/rh-aiservices-bu/postgresql-15-pgvector-c9s:latest
```

Run the postgres database locally.

```bash
podman run -d --name postgres \
-e POSTGRESQL_USER=user \
-e POSTGRESQL_PASSWORD=password \
-e POSTGRESQL_ADMIN_PASSWORD=password \
-e POSTGRESQL_DATABASE=vectordb \
-p 5432:5432 \
quay.io/rh-aiservices-bu/postgresql-15-pgvector-c9s:latest
```

Create the vector extension to hold our embeddings.

```bash
podman exec -it postgres psql -d vectordb -c "CREATE EXTENSION vector;"
```

## Retrieve some documents

Lilian Weng is a Researcher with interests ni Deep Learning, Machine Learning, Network Science. We are going to use some of her web based posts to RAG with.

Lets create a python class to retrieve the posts, chunk them and embed them into postgres.

Paste this into a python file called `retriever.py`

One of the first main steps in RAG is to split and convert each document into chunks. We use a recursive character test splitter with some sane default settings for that task. 

```python
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

# We use some sensible defaults for text chunking and splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=40)
all_splits = text_splitter.split_documents(docs_list)
all_splits[0]
```

You should be able to run this without error - `python retriever.py`

Next, we use the default `HuggingFaceEmbeddings` class to store our document splits in the vector store - postgres.

The model that this class uses is an open source [embedding model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) called **all-mpnet-base-v2** which you can go read and about.

Next append to `retriever.py` the code to load our document embeddings into postgres.

```python
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
```

You should be able to run this without error.

```bash
python retriever.py
```

Check the vector database size.

```bash
podman exec -it postgres psql -d vectordb -c "SELECT pg_size_pretty (pg_database_size ('vectordb')) size;"

  size   
---------
 8885 kB
(1 row)
```

You can also go checkout the database layout - its nice and simple - one table for the collection name (we called ours `documents_test`) and one table for the embeddings.

```bash
podman exec -it postgres psql -d vectordb -c "\dt+"

                                             List of relations
 Schema |          Name           | Type  |  Owner   | Persistence | Access method |  Size   | De
scription 
--------+-------------------------+-------+----------+-------------+---------------+---------+---
----------
 public | langchain_pg_collection | table | postgres | permanent   | heap          | 16 kB   | 
 public | langchain_pg_embedding  | table | postgres | permanent   | heap          | 1128 kB | 
(2 rows)
```

## Deploy an LLM locally

Install vLLM in your venv - this may take some time as the packages are large.

```bash
pip install vllm --upgrade
```

We are going to use vLLM to run locally. You can choose any model suitable for chat really (aka they have been Instruction fine tuned). The main limitations are based on your GPU vRAM size (i have a NVIDIA GeForce RTX 4070 - 8GB VRAM on this laptop), the contenxt length we use and the model supports - and we of course want an open source model!

We are going to use a quantized model [lmstudio-community/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF). Download it locally - this may take some time as it is an 4GB model.

// FIXME - Granite GGUF fails to load in latest vllm?
// ValueError: GGUF model with architecture granite is not supported yet.
// curl -LO https://huggingface.co/ibm-research/granite-3.2-2b-instruct-GGUF/resolve/main/granite-3.2-2b-instruct-f16.gguf?download=true
// We are going to use a quantized model [ibm-research/granite-3.2-2b-instruct-GGUF](https://huggingface.co/ibm-research/granite-3.2-2b-instruct-GGUF). Download it locally - this may take some time as it is an 4GB model.

```bash
curl -LO https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf?download=true
```

OK, lets spin it up using vLLM!

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ~/rag-langchain/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --served-model-name Llama-3.2-3B-Instruct-Q8_0.gguf \
  --quantization gguf \
  --enforce-eager \
  --port 8080 \
  --max-model-len=10000
```

![images/2-vllm-started.png](images/2-vllm-started.png)

I have [**nvtop**](https://rpmfind.net/linux/epel/9/Everything/x86_64/Packages/n/nvtop-3.1.0-2.el9.x86_64.rpm) installed - its like top but gor GPU's - the LLM is loaded into the GPU on the left - NVIDIA GeForce RTX 4070

![images/2-nvtop.png](images/2-nvtop.png)

You can see that its taking up about 80% of the NVRAM with 10k max token length (the model supports 128k tokens).

```bash
INFO 03-29 15:52:45 [model_runner.py:1146] Model loading took 3.2637 GB and 18.052839 seconds
INFO 03-29 15:53:07 [worker.py:267] Memory profiling takes 21.11 seconds
INFO 03-29 15:53:07 [worker.py:267] the current vLLM instance can use total_gpu_memory (7.65GiB) x gpu_memory_utilization (0.90) = 6.89GiB
INFO 03-29 15:53:07 [worker.py:267] model weights take 3.26GiB; non_torch_memory takes 0.02GiB; PyTorch activation peak memory takes 1.22GiB; the rest of the memory reserved for KV Cache is 2.38GiB.
INFO 03-29 15:53:07 [executor_base.py:111] # cuda blocks: 1395, # CPU blocks: 2340
INFO 03-29 15:53:07 [executor_base.py:116] Maximum concurrency for 10000 tokens per request: 2.23x
```

We can check the OpenAPI is available now using curl:

```json
curl -s -X 'GET' http://localhost:8080/v1/models -H 'accept: application/json' | jq .
{
  "object": "list",
  "data": [
    {
      "id": "Llama-3.2-3B-Instruct-Q8_0.gguf",
      "object": "model",
      "created": 1743227707,
      "owned_by": "vllm",
      "root": "/home/mike/rag-langchain/Llama-3.2-3B-Instruct-Q8_0.gguf",
      "parent": null,
      "max_model_len": 10000,
      "permission": [
        {
          "id": "modelperm-1cfb721726214c0086226dc442a47487",
          "object": "model_permission",
          "created": 1743227707,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

## Write our Chat User Interface

Install our extra dependencies

```bash
pip install --upgrade --quiet \
  chainlit \
  httpx \
  langchain_openai
```

Create a `app.py` file for our Chat UI.

We will break down the `app.py` so we can explain what is going on.

Import our dependencies and setup our connection strings to the database.

```python
import os
import re
import httpx
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import PGVector

MODEL_NAME = os.getenv("MODEL_NAME", "Llama-3.2-3B-Instruct-Q8_0.gguf")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8080/v1")

DB_CONNECTION_STRING = os.getenv(
    "DB_CONNECTION_STRING",
    "postgresql+psycopg://postgres:password@localhost:5432/vectordb",
)
DB_COLLECTION_NAME = os.getenv("DB_COLLECTION_NAME", "documents_test")
```

Next, we append our system prompt. In RAG the prompt is modified by adding the relevant documents from the vector store along with the question. This is denoted by the `{context}` and `{question}` template variables. 

The begin, end and assistant part of the prompt has to match the model architecture (Llama-3). You can read about the [LLama-3.2 prompt format here.](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/). We setup a fairly standard prompt asking for the model to act nicely.

The last line loads this into a Lang Chain Prompt template object.

```python
template = "Q: {question} A:"

if re.search(r"LLama-3", MODEL_NAME, flags=re.IGNORECASE):
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    You are a helpful, respectful and honest assistant answering questions named HatBot.
    You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
    Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Context:
    {context}

    Question: {question}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question"], template=template)
```

The next piece of code is out startup handler. It loads the Chat Setting we expose as well as setting up our vector store connection to postgres.

```python
@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    settings = await cl.ChatSettings(
        [
            Select(
                id="model_name",
                label="OpenAI - Model",
                values=["Llama-3.2-3B-Instruct-Q8_0.gguf"],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="OpenAI - Temperature",
                initial=0.2,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="top_p",
                label="Top P",
                initial=0.3,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="max_tokens",
                label="Max output tokens",
                initial=4096,
                min=0,
                max=32768,
                step=256,
            ),
            Slider(
                id="presence_penalty",
                label="Presence Penalty",
                initial=1.05,
                min=-2,
                max=2,
                step=0.05,
            ),
        ],
    ).send()
    cl.user_session.set("settings", settings)

    # Document store: pgvector vector store
    embeddings = HuggingFaceEmbeddings()
    store = PGVector(
        connection_string=DB_CONNECTION_STRING,
        collection_name=DB_COLLECTION_NAME,
        embedding_function=embeddings,
        use_jsonb=True,
    )
    cl.user_session.set("store", store)

class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")

def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata['source'] not in unique_list:
            unique_list.append(item.metadata['source'])
    return unique_list

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
```

The last piece of code is the actual Chat interface (`ChatOpenAI` class). We are using a local model without an APIKEY, and we use a langchain `RetrievalQA.from_chain_type` chain for the question and answers.

The chain does the heavy lifting of retrieving the embeddings using a similarity search ([cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)) and feeding that with the question to the LLM.

The code is async, and the final handler prints out our response from the model along with document references.

```python
@cl.on_message
async def query_llm(message: cl.Message):
    settings = cl.user_session.get("settings")

    llm = ChatOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=INFERENCE_SERVER_URL,
        model_name=settings["model_name"],
        top_p=settings["top_p"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        presence_penalty=settings["presence_penalty"],
        streaming=True,
        verbose=False,
        http_async_client=httpx.AsyncClient(verify=False),
        http_client=httpx.Client(verify=False),
    )

    store = cl.user_session.get("store")

    llm_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
        verbose=True,
    )

    resp = await llm_chain.acall(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler(), StreamHandler()]
    )

    msg = cl.Message(content="")
    sources = remove_source_duplicates(resp['source_documents'])
    if len(sources) != 0:
        await msg.stream_token("\n*Sources:* \n")
        for source in sources:
            await msg.stream_token("* " + str(source) + "\n")
    await msg.send()
```

We can run our UI using the chainlit command.

```bash
chainlit run app.py -w --port 8081 --host 0.0.0.0
```

![images/2-chainlit.png](images/2-chainlit.png)

Next we ask our RAG a question that is relevant to Lilian's blog posts.

> Q. What are the types of agent memory based on Lilian Weng's blog post?

> A. Based on Lilian Weng's blog post, the types of agent memory are:
    Short-term memory (STM): This type of memory stores information that is currently being used and needed to carry out complex cognitive tasks such as learning and reasoning. It has a limited capacity of about 7 items and lasts for 20-30 seconds.
    Long-term memory (LTM): This type of memory can store information for a remarkably long time, ranging from a few days to decades, with an essentially unlimited storage capacity. There are two subtypes of LTM:
    Explicit / declarative memory: This refers to memories of facts and events that can be consciously recalled, including episodic memory (events and experiences) and semantic memory (facts and concepts).
    Implicit / procedural memory: This type of memory is unconscious and involves skills and routines that are performed automatically, like riding a bike or typing on a keyboard.
    Sources:
    https://lilianweng.github.io/posts/2023-06-23-agent/

![images/2-chainlit-rag.png](images/2-chainlit-rag.png)

Adjust the temperature and other settings on each call to experiment !

![images/2-chainlit-settings.png](images/2-chainlit-settings.png)

You can also investigate the embeddings and text chunks that were retrieved from the vector store and passed into the LLM by opening the `Used RetrievalQA` drop down. You will see `VectorStoreRretriever` and a `StuffDocumentChain` feeding into a `query` then the LLM `Output`.

![images/2-retrieval-chain.png](images/2-retrieval-chain.png)

The source code for this excercise can be found here:

- [retriever.py](https://raw.githubusercontent.com/eformat/ragit-dev/refs/heads/main/docs/2-local-rag/local-rag-langchain/retriever.py)
- [app.py](https://raw.githubusercontent.com/eformat/ragit-dev/refs/heads/main/docs/2-local-rag/local-rag-langchain/app.py)
- [requirements.txt](https://raw.githubusercontent.com/eformat/ragit-dev/refs/heads/main/docs/2-local-rag/local-rag-langchain/requirements.txt)

🥳🥳 Well done. You have completed the local RAG with langchain example!