import openai
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from app.config import OPENAI_CONFIG, DB_CONFIG, EMBEDDING_CONFIG
from app.utils import extract_chunks_from_dir
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_DOC_TOKENS = 10000  # approximate limit per document
MAX_DOCS = 3           # number of docs to include in context

class RAGChatbot:
    def __init__(self):
        openai.api_key = OPENAI_CONFIG["api_key"]
        self.data_dir = DB_CONFIG["data_dir"]

        # Initialize embedding function
        embedder = SentenceTransformerEmbeddingFunction(
            EMBEDDING_CONFIG["model_name"]
        )
        # Initialize database
        self.chroma = PersistentClient(path=DB_CONFIG["store_path"])
        self.collection = self.chroma.get_or_create_collection(
            name=DB_CONFIG["collection_name"], embedding_function=embedder
        )

    def build_index(self) -> None:
        """Build the search index from documents."""
        logger.info(f"Building index from directory: {self.data_dir}")
        try:
            """
            Build the index of documents.
            """
            texts = extract_chunks_from_dir(self.data_dir)
            logger.info(f"Found {len(texts)} text chunks to index")
            for i, chunk in enumerate(texts):
                self.collection.add(documents=[chunk], ids=[f"doc_{i}"])
            logger.info("Successfully built search index")
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise

    def query(self, question: str) -> str:
        try:
            # Step 1: Retrieve relevant documents
            results = self.collection.query(query_texts=[question], n_results=MAX_DOCS)
            raw_documents = results.get('documents', [[]])[0]

            # Step 2: Limit each document length to avoid exceeding token budget
            documents = [doc[:MAX_DOC_TOKENS] for doc in raw_documents]
            context = "\n---\n".join(documents)

            # Step 3: Build prompt
            if context.strip():
                prompt = (
                    "You are an expert assistant. Use the context below if relevant "
                    "to answer the user's question.\n"
                    "\n"
                    "[Context]\n"
                    f"{context}\n"
                    "\n"
                    "[Question]\n"
                    f"{question}\n"
                    "\n"
                    "[Answer]"
                )
            else:
                prompt = (
                    "You are a helpful assistant. Answer the question below "
                    "concisely and clearly.\n"
                    "\n"
                    "[Question]\n"
                    f"{question}\n"
                    "\n"
                    "[Answer]"
                )

            # Step 4: Call OpenAI with safety caps
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500,  # adjust based on your output length needs
            )
            return response["choices"][0]["message"]["content"]

        except openai.error.RateLimitError as e:
            return "[Rate Limit Error] Too many requests — try again later."
        except openai.error.OpenAIError as e:
            return f"[OpenAI API Error] {str(e)}"
        except Exception as e:
            return f"[Unhandled Error] {str(e)}"

    def reset_db(self):
        self.client.delete_collection(DB_CONFIG["collection_name"])
        self.collection = self.chroma.get_or_create_collection(
            name=DB_CONFIG["collection_name"], embedding_function=embedder
        )