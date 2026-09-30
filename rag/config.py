"""
Configuración central, leída una sola vez de variables de entorno (.env).

Todos los scripts (setup_index.py, ingest.py, evaluate.py, main.py)
importan de acá, así no hay nombres de índice o namespaces repetidos y
distintos entre archivos.
"""
import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = os.getenv("INDEX_NAME", "volkswagen-rag-hibrido")

# Namespace: separa estos datos de cualquier otro uso del mismo índice
# (ej: "dev" vs "prod", o distintos datasets). Ver "Ignorar el Namespace"
# en los errores comunes de la consigna.
NAMESPACE = os.getenv("NAMESPACE", "vw-modelos-dev")

PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
METRIC = "cosine"  # all-MiniLM-L6-v2 está entrenado para similitud coseno

DATA_DIR = os.getenv("DATA_DIR", "data")

# La consigna sugiere 500-800 tokens: punto medio entre contexto y precisión.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Batch upsert: la clase marca 100-200 vectores como el "punto dulce".
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

TOP_K = int(os.getenv("TOP_K", "5"))

# Pesos del EnsembleRetriever: [BM25, vectorial]
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.5"))


def requerir_api_key() -> str:
    """Falla de forma clara (no silenciosa) si falta la key de Pinecone."""
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "Falta PINECONE_API_KEY en el .env (se consigue gratis en app.pinecone.io)"
        )
    return PINECONE_API_KEY
