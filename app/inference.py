import logging
from app.retriever import retrieve_chunks
from app.reranker import rerank_results
from app.rag.ollama import call_ollama
from app.rag.openai import call_openai
from app.config import settings

logger = logging.getLogger(__name__)


def generate_answer(query: str) -> str:
    """
    Generate a response for the user query by retrieving relevant context and querying the LLM.

    Steps:
    1. Embed the query locally
    2. Retrieve top-K chunks from ChromaDB
    3. Rerank using CrossEncoder if enabled
    4. Call appropriate LLM (Ollama or OpenAI)
    """
    logger.info("Received query: %s", query)

    top_k_results = retrieve_chunks(query)
    logger.info("Retrieved %d candidate chunks", len(top_k_results))

    if settings.enable_rerank and top_k_results:
        logger.info("Reranking is enabled, reranking retrieved chunks")
        top_k_results = rerank_results(query, top_k_results)
        logger.info("Reranking completed")

    context_text = "\n\n".join([doc['text'] for doc in top_k_results])
    prompt = f"Use the following context to answer the question:\n\n{context_text}\n\nQuestion: {query}\nAnswer:"

    try:
        if settings.use_ollama:
            logger.info("Calling Ollama model: %s", settings.ollama_model)
            return call_ollama(prompt)
        else:
            logger.info("Calling OpenAI model: %s", settings.openai_model)
            return call_openai(prompt)
    except Exception as e:
        logger.exception("Error during LLM call: %s", str(e))
        return "❌ An error occurred while generating a response. Please try again."
