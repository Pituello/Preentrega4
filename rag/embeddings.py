"""
Modelo de embeddings: UNO solo, fijo, para indexar y para consultar
(mismo criterio que la Pre-Entrega 3: mezclar modelos da resultados
aleatorios).

La dimensión NO se hardcodea: se mide generando un embedding de prueba.
Así, si algún día se cambia EMBEDDINGS_MODEL por uno de otra dimensión,
setup_index.py lo detecta en vez de subir vectores de 384 a un índice de
1536 (el "mismatch de dimensiones" que marca la consigna). El 1536 que
menciona la consigna es específico de text-embedding-3-small de OpenAI;
acá usamos un modelo local y gratuito, de 384.
"""
import os

from langchain_core.embeddings import Embeddings

EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_embeddings_cache: Embeddings | None = None
_dimension_cache: int | None = None


def get_embeddings() -> Embeddings:
    global _embeddings_cache
    if _embeddings_cache is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings_cache = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    return _embeddings_cache


def get_dimension() -> int:
    """Dimensión real de los vectores que genera el modelo configurado."""
    global _dimension_cache
    if _dimension_cache is None:
        _dimension_cache = len(get_embeddings().embed_query("prueba de dimensión"))
    return _dimension_cache
