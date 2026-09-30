"""
Pipeline de ingesta: /data (.md, .txt, .pdf) -> limpieza -> chunking por tokens -> metadata
validada -> embeddings -> batch upsert a Pinecone.

Es idempotente: correrlo dos veces seguidas no duplica nada, y si cambiás,
agregás o borrás un archivo de /data, deja el namespace sincronizado.

Uso:
    python ingest.py            # sincroniza (solo sube lo nuevo/modificado)
    python ingest.py --forzar   # vuelve a vectorizar y subir todo
"""
import asyncio
import sys
from collections import Counter

from rag import config
from rag.chunking import DocumentProcessor
from rag.pinecone_store import setup_index, sincronizar_namespace


async def main():
    forzar = "--forzar" in sys.argv

    index = setup_index()

    chunks = DocumentProcessor().load_and_process(config.DATA_DIR)
    por_archivo = Counter(doc.metadata["source"] for _, doc in chunks)
    print(f"\n📄 {len(por_archivo)} documentos -> {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    for archivo, cantidad in sorted(por_archivo.items()):
        print(f"   - {archivo}: {cantidad}")

    resumen = await sincronizar_namespace(index, chunks, forzar=forzar)
    print(f"\n📦 Namespace '{config.NAMESPACE}': {resumen['subidos']} subidos "
          f"(en {resumen['lotes']} lotes), {resumen['borrados']} borrados, "
          f"{resumen['sin_cambios']} sin cambios")

    # Ejemplo de la metadata que viaja con cada vector (uno por formato)
    vistos = set()
    for _, doc in chunks:
        formato = doc.metadata["formato"]
        if formato not in vistos:
            vistos.add(formato)
            meta = {k: v for k, v in doc.metadata.items() if k != "text"}
            print(f"\n🏷️  Metadata de ejemplo, {formato} (sin el texto): {meta}")


if __name__ == "__main__":
    asyncio.run(main())
