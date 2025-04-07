# Simple RAG

Demo Simple RAG on RHOAI.

1. Get yourself a [Composer.ai](https://redhat-composer-ai.github.io/documentation/demo/walkthrough) LAB in demo.redhat.com

![3-composer-ai.png](images/3-composer-ai.png)

## Exercises

We extend the [local exercises](2-local-rag/README.md) by deploying them to RHOAI. The high level steps will be as follows:

Choose one of the frameworks from the local rag examples:

- [RamaLama](2-local-rag/2-local-rag-ramalama)
- [LangChain](2-local-rag/2-local-rag-langchain)
- [LlamaIndex](2-local-rag/2-local-rag-llamaindex)
- [quarkus/langchain4j](https://docs.quarkiverse.io/quarkus-langchain4j/dev/easy-rag.html)

We will use it to connect to a [vector store](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/):

- postgres
- chromadb
- quadrant
- milvus

Then serve an LLM model:

- using MaaS
- using RHOAI

And Show how to do a simple rag architecture with RHOAI.

🤑 !! PROFIT !! 🤑

In the next section [Complex Rag](4-complex-rag/README.md) - we will learn some techniques that will add quality control to our RAG so we can get it to production.

🚗 Let's go ! 🚗
