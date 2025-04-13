# Local RAG with LlamaIndex

[LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/) is a very popular open source toolkit and framework for developing applications powered by LLM's.

We are going to use a number of popular tools to build our local RAG including

- [LlamaIndex](https://docs.llamaindex.ai/en/stable/) framework.
- [Chainlit](https://docs.chainlit.io/get-started/overview) to build a conversational user interface.
- [Postgres + Vector Extentsion](https://github.com/pgvector/pgvector) as the vector store.
- [vLLM](https://docs.vllm.ai/en/latest/) a library to server our LLM for inference and serving.

## Pre-requsites

- Laptop with a Nvidia GPU is preferable
- Python3.11+ installed
- Podman
- Fast internet

## Setup a workspace and project

```bash
mkdir ~/rag-llamaindex && cd ~/rag-llamaindex
```

Setup a venv

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies.

```bash
pip install --upgrade --quiet \
  llama_index \
  llama_index.readers.web \
  llama_index.embeddings.huggingface \
  llama_index.vector_stores.postgres \
  sqlalchemy \
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
```

You should be able to run this without error - `python retriever.py`

Next, we use the default `HuggingFaceEmbeddings` class to store our document splits in the vector store - postgres.

The model that this class uses is an open source [embedding model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) called **all-mpnet-base-v2** which you can go read and about.

Next append to `retriever.py` the code to load our document embeddings into postgres.

```python
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
 7957 kB
(1 row)
```

You can also go checkout the database layout - its nice and simple - one table for the embeddings.

```bash
podman exec -it postgres psql -d vectordb -c "\dt+"

 Schema |        Name         | Type  |  Owner   | Persistence | Access method |  Size  | Description 
--------+---------------------+-------+----------+-------------+---------------+--------+-------------
 public | data_documents_test | table | postgres | permanent   | heap          | 328 kB | 
(1 row)
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
      "created": 1743405102,
      "owned_by": "vllm",
      "root": "Llama-3.2-3B-Instruct-Q8_0.gguf",
      "parent": null,
      "max_model_len": 10000,
      "permission": [
        {
          "id": "modelperm-180b949cdb7b4ceb96e5f1946c0c9f86",
          "object": "model_permission",
          "created": 1743405102,
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
  openai \
  llama_index.llms.openai_like
```

Create a `app.py` file for our Chat UI.

We will break down the `app.py` so we can explain what is going on.

Import our dependencies and setup our connection strings to the database.

```python
import os
import re
import httpx
import openai
import textwrap
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

MODEL_NAME = os.getenv("MODEL_NAME", "Llama-3.2-3B-Instruct-Q8_0.gguf")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8080/v1")

DB_NAME="vectordb"
DB_CONNECTION_STRING = os.getenv(
    "DB_CONNECTION_STRING",
    "postgresql+psycopg://postgres:password@localhost:5432/vectordb",
)
DB_COLLECTION_NAME = os.getenv("DB_COLLECTION_NAME", "documents_test")
url = make_url(DB_CONNECTION_STRING)
```

Next, we append our system prompt. In RAG the prompt is modified by adding the relevant documents from the vector store along with the question. This is denoted by the `{context_str}` and `{question_str}` template variables.

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
    {context_str}

    Question: {query_str}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """

DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    template, prompt_type=PromptType.QUESTION_ANSWER
)
```

The next piece of code is our startup handler. It loads the Chat Setting we expose as well as setting up our vector store connection to postgres.

```python
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

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
                id="top_k",
                label="Top K",
                initial=2,
                min=0,
                max=4,
                step=1,
            ),
            Slider(
                id="max_tokens",
                label="Max output tokens",
                initial=4096,
                min=0,
                max=32768,
                step=256,
            ),
        ],
    ).send()
    cl.user_session.set("settings", settings)

    store = PGVectorStore.from_params(
        database=DB_NAME,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=DB_COLLECTION_NAME,
        embed_dim=768,
    )
    index = VectorStoreIndex.from_vector_store(vector_store=store, embed_model=Settings.embed_model)
    cl.user_session.set("index", index)

async def set_sources(response, response_message):
    label_list = []
    count = 1
    for sr in response.source_nodes:
        elements = [
            cl.Text(
                name="S" + str(count),
                content=f"{sr.node.text}",
                display="side",
                size="small",
            )
        ]
        response_message.elements = elements
        label_list.append("S" + str(count))
        await response_message.update()
        count += 1
    response_message.content += "\n\nSources: " + ", ".join(label_list)
    await response_message.update()
```

The last piece of code is the actual Chat interface (`OpenAILike` class). We are using a local model without an APIKEY, and we query the index engine `query_engine.query` with the question and wait async for the answers.

The index does the heavy lifting of retrieving the embeddings using a similarity search ([cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)) and feeding that with the question to the LLM.

The code is async, and the final handler prints out our response from the model along with document references.

```python
@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")
    index = cl.user_session.get("index")

    Settings.llm = OpenAILike(
        api_key="EMPTY",
        api_base=INFERENCE_SERVER_URL,
        model=settings["model_name"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        verbose=False,
        async_http_client=httpx.AsyncClient(verify=False),
        http_client=httpx.Client(verify=False),
        context_window=10000,
    )

    Settings.callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()])

    msg = cl.Message(content="", author="Assistant")

    query_engine = index.as_query_engine(streaming=True, similarity_top_k=settings["top_k"])
    query_engine.update_prompts({"response_synthesizer:text_qa_template": DEFAULT_TEXT_QA_PROMPT})

    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
      await msg.stream_token(token)
    await msg.send()

    if res.source_nodes:
      await set_sources(res, msg)
```

We can run our UI using the chainlit command.

```bash
chainlit run app.py -w --port 8081 --host 0.0.0.0
```

![images/2-chainlit.png](images/2-chainlit.png)

Next we ask our RAG a question that is relevant to Lilian's blog posts.

> Q. What are the types of agent memory based on Lilian Weng's blog post?

> A. I'll provide a detailed answer to your question based on the context you provided.
    The question is about the types of memory in human brains, specifically in relation to Lilian Weng's blog post.
    According to Lilian Weng's blog post, there are three main types of memory in human brains:
    Sensory Memory: This is the earliest stage of memory, providing the ability to retain impressions of sensory information (visual, auditory, etc.) after the original stimuli have ended. Sensory memory typically only lasts for up to a few seconds. Subcategories include iconic memory (visual), echoic memory (auditory), and haptic memory (touch).
    Short-Term Memory (STM) or Working Memory: It stores information that we are currently aware of and needed to carry out complex cognitive tasks such as learning and reasoning. Short-term memory is believed to have the capacity of about 7 items (Miller 1956) and lasts for 20-30 seconds.
    Long-Term Memory (LTM): Long-term memory can store information for a remarkably long time, ranging from a few days to decades, with an essentially unlimited storage capacity. There are two subtypes of LTM:
    Explicit / Declarative Memory: This is memory of facts and events, and refers to those memories that can be consciously recalled, including episodic memory (events and experiences) and semantic memory (facts and concepts).
    Implicit / Procedural Memory: This type of memory is unconscious and involves skills and routines that are performed automatically, like riding a bike or typing on a keyboard.
    These types of memory are not mutually exclusive, and they often overlap and interact with each other.
    Sources: S1, S2

![images/2-chainlit-rag-llamaindex.png](images/2-chainlit-rag-llamaindex.png)

Adjust the temperature and other settings on each call to experiment !

![images/2-chainlit-settings-llamaindex.png](images/2-chainlit-settings-llamaindex.png)

You can also investigate the embeddings and text chunks that were retrieved from the vector store and passed into the LLM by opening the `Used Query > Used retrieve` drop down. You will see input and output that is then fed into the `User llm > input`, followed by the LLM `Output`.

![images/2-retrieval-llamaindex.png](images/2-retrieval-llamaindex.png)

The source code for this excercise can be found here:

- [retriever.py](https://raw.githubusercontent.com/eformat/ragit-dev/refs/heads/main/docs/2-local-rag/local-rag-llamaindex/retriever.py)
- [app.py](https://raw.githubusercontent.com/eformat/ragit-dev/refs/heads/main/docs/2-local-rag/local-rag-llamaindex/app.py)
- [requirements.txt](https://raw.githubusercontent.com/eformat/ragit-dev/refs/heads/main/docs/2-local-rag/local-rag-llamaindex/requirements.txt)

🥳🥳 Well done. You have completed the local RAG with langchain example!
