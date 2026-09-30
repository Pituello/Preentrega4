"""
Verifica si el índice existe y lo crea si hace
falta (modo Serverless). Si ya existe, valida que su dimensión coincida
con el modelo de embeddings.

Uso:
    python setup_index.py
"""
from rag import config
from rag.pinecone_store import setup_index


def main():
    index = setup_index()
    stats = index.describe_index_stats()
    print(f"\n✅ Índice '{config.INDEX_NAME}' listo")
    print(f"   Dimensión: {stats.dimension} | Vectores totales: {stats.total_vector_count}")
    print(f"   Namespaces: {list(stats.namespaces.keys()) or '(ninguno todavía)'}")


if __name__ == "__main__":
    main()
