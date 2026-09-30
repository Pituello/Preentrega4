"""
Punto único para armar un RAGSystem listo para usar (lo usan main.py y
evaluate.py).

BM25 necesita el corpus completo EN MEMORIA (no hay "BM25 en Pinecone"
en este esquema): se reconstruye desde /data con el mismo
DocumentProcessor que usó la ingesta, así los textos son idénticos a los
que están en Pinecone. Como eso solo es cierto si la última ingesta se
corrió después del último cambio en /data, se chequea que la cantidad de
vectores del namespace coincida con la de chunks locales.
"""
from loguru import logger

from . import config
from .chunking import DocumentProcessor
from .pinecone_store import get_client, get_vectorstore
from .retriever import RAGSystem


def crear_rag_system(**kwargs) -> RAGSystem:
    chunks = DocumentProcessor().load_and_process(config.DATA_DIR)
    documentos = [doc for _, doc in chunks]

    index = get_client().Index(config.INDEX_NAME)
    stats = index.describe_index_stats()
    en_pinecone = stats.namespaces.get(config.NAMESPACE)
    cantidad = en_pinecone.vector_count if en_pinecone else 0

    if cantidad == 0:
        raise RuntimeError(
            f"El namespace '{config.NAMESPACE}' está vacío. Corré primero: python ingest.py"
        )
    if cantidad != len(documentos):
        logger.warning(
            f"Pinecone tiene {cantidad} vectores pero /data genera {len(documentos)} chunks: "
            f"BM25 y la búsqueda vectorial podrían no estar viendo lo mismo. Corré python ingest.py"
        )

    return RAGSystem(get_vectorstore(index), documentos, **kwargs)
