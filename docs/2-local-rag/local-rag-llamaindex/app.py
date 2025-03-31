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
    cl.user_session.set("store", store)

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

@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")
    store = cl.user_session.get("store")

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

    index = VectorStoreIndex.from_vector_store(vector_store=store, embed_model=Settings.embed_model)
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=settings["top_k"])
    query_engine.update_prompts({"response_synthesizer:text_qa_template": DEFAULT_TEXT_QA_PROMPT})

    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
      await msg.stream_token(token)
    await msg.send()

    if res.source_nodes:
      await set_sources(res, msg)
