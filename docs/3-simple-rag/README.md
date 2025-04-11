# Simple RAG

Demo Simple RAG on RHOAI.

## Exercises

**Exercises**

If you have access to Red Hat Demo Platform you can try the push button deployment of RAG using Composer AI.

- [Composer AI](3-simple-rag/3-simple-rag-composer-ai)

We extend the [local exercises](2-local-rag/README.md) by deploying them to RHOAI.

- [RamaLama](3-simple-rag/3-simple-rag-ramalama.md)

The high level steps are ...

Choose one of the frameworks from the local rag examples:

- [RamaLama](2-local-rag/2-local-rag-ramalama)
- [LangChain](2-local-rag/2-local-rag-langchain)
- [LlamaIndex](2-local-rag/2-local-rag-llamaindex)
- [quarkus/langchain4j](https://docs.quarkiverse.io/quarkus-langchain4j/dev/easy-rag.html)

Use it to connect to a [vector store](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/):

- postgres
- chromadb
- quadrant
- milvus
- ...

Then serve an LLM model:

- using RHOAI
- using MaaS

And Show how to do a deploy a simple rag architecture with RHOAI.

🤑 !! PROFIT !! 🤑

In the next section [Complex Rag](4-complex-rag/README.md) - we will learn some techniques that will add quality control to our RAG so we can get it to production.

🚗 Let's go ! 🚗
