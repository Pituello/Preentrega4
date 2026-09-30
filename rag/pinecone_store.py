"""
Todo lo que habla con Pinecone "por abajo" (SDK nativo):

- setup_index(): crea el índice Serverless si no existe y, si ya existe,
  verifica que su dimensión y métrica coincidan con el modelo de
  embeddings (el "mismatch de dimensiones" falla ACÁ, con un mensaje
  claro, no a mitad de la ingesta).
- sincronizar_namespace(): ingesta por lotes (batch upsert) que además
  deja el namespace idéntico a /data: sube solo los chunks nuevos o
  modificados y borra los que ya no existen.
- get_vectorstore(): el PineconeVectorStore de LangChain, que usa el
  retriever vectorial para las consultas.
"""
import asyncio

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from loguru import logger
from pinecone import Pinecone, ServerlessSpec

from . import config
from .embeddings import get_dimension, get_embeddings

MAX_UPSERTS_CONCURRENTES = 4


def get_client() -> Pinecone:
    return Pinecone(api_key=config.requerir_api_key())


def setup_index(pc: Pinecone | None = None):
    """Crea el índice si no existe y valida dimensión/métrica si ya existe."""
    pc = pc or get_client()
    dimension = get_dimension()

    if config.INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"creando índice '{config.INDEX_NAME}' (dim={dimension}, métrica={config.METRIC})")
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=dimension,
            metric=config.METRIC,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    else:
        descripcion = pc.describe_index(config.INDEX_NAME)
        if descripcion.dimension != dimension:
            raise RuntimeError(
                f"Mismatch de dimensiones: el índice '{config.INDEX_NAME}' es de "
                f"{descripcion.dimension} dims, pero el modelo de embeddings genera {dimension}. "
                f"Usá otro INDEX_NAME o borrá el índice desde la consola de Pinecone."
            )
        if descripcion.metric != config.METRIC:
            raise RuntimeError(
                f"El índice '{config.INDEX_NAME}' usa métrica '{descripcion.metric}', "
                f"pero este sistema espera '{config.METRIC}'."
            )
        logger.info(f"el índice '{config.INDEX_NAME}' ya existe (dim={dimension}) — no se recrea")

    return pc.Index(config.INDEX_NAME)


def _ids_en_namespace(index) -> set[str]:
    """Todos los IDs guardados hoy en el namespace (index.list pagina solo)."""
    ids = set()
    for pagina in index.list(namespace=config.NAMESPACE):
        ids.update(pagina)
    return ids


async def _subir_lote(index, lote: list[tuple[str, Document]], numero: int, semaforo: asyncio.Semaphore):
    async with semaforo:
        textos = [doc.page_content for _, doc in lote]
        # embed_documents y upsert son síncronos (CPU y red respectivamente):
        # se corren en threads para no bloquear el event loop y poder
        # procesar varios lotes a la vez.
        vectores = await asyncio.to_thread(get_embeddings().embed_documents, textos)
        payload = [
            {"id": id_, "values": vector, "metadata": doc.metadata}
            for (id_, doc), vector in zip(lote, vectores)
        ]
        await asyncio.to_thread(index.upsert, vectors=payload, namespace=config.NAMESPACE, show_progress=False)
        logger.info(f"lote {numero}: {len(payload)} vectores subidos")


async def sincronizar_namespace(index, chunks: list[tuple[str, Document]], forzar: bool = False) -> dict:
    """
    Deja el namespace igual a lo que hay en /data:
    - IDs nuevos (archivo nuevo o chunk modificado -> hash distinto) -> se suben.
    - IDs que ya no existen en /data (archivo borrado o versión vieja de un
      chunk modificado) -> se borran.
    - IDs que no cambiaron -> no se vuelven a vectorizar ni subir (salvo forzar=True).
    """
    nuevos = dict(chunks)
    existentes = await asyncio.to_thread(_ids_en_namespace, index)

    a_subir = [(i, d) for i, d in chunks if forzar or i not in existentes]
    a_borrar = sorted(existentes - nuevos.keys())

    lotes = [a_subir[i : i + config.BATCH_SIZE] for i in range(0, len(a_subir), config.BATCH_SIZE)]
    semaforo = asyncio.Semaphore(MAX_UPSERTS_CONCURRENTES)
    await asyncio.gather(*(_subir_lote(index, lote, n, semaforo) for n, lote in enumerate(lotes, 1)))

    for i in range(0, len(a_borrar), 1000):  # límite de IDs por delete de la API
        await asyncio.to_thread(index.delete, ids=a_borrar[i : i + 1000], namespace=config.NAMESPACE)
    if a_borrar:
        logger.info(f"borrados {len(a_borrar)} vectores obsoletos del namespace")

    return {
        "subidos": len(a_subir),
        "borrados": len(a_borrar),
        "sin_cambios": len(nuevos) - len(a_subir),
        "lotes": len(lotes),
    }


class PineconeVectorStoreConcurrente(PineconeVectorStore):
    """
    PineconeVectorStore con las búsquedas ASÍNCRONAS resueltas en un thread.

    Por qué: en langchain-pinecone 0.2.x, todas las búsquedas asíncronas
    comparten UNA sesión HTTP (self._async_index), y cada búsqueda la cierra
    al terminar (async with idx). Con varias consultas en paralelo
    (asyncio.gather en evaluate.py), la primera en terminar cierra la sesión
    mientras las demás la siguen usando -> "RuntimeError: Session is closed".
    Una consulta por vez (main.py) no lo dispara, por eso tarda en aparecer.

    El camino síncrono usa el cliente normal de Pinecone, con un pool de
    conexiones que sí soporta consultas concurrentes: correrlo con
    asyncio.to_thread mantiene la concurrencia sin compartir esa sesión.
    Es el mismo patrón que usa la ingesta para el upsert.
    """

    async def asimilarity_search(self, query, k=4, **kwargs):
        return await asyncio.to_thread(self.similarity_search, query, k=k, **kwargs)

    async def asimilarity_search_with_score(self, query, k=4, **kwargs):
        return await asyncio.to_thread(self.similarity_search_with_score, query, k=k, **kwargs)


def get_vectorstore(index=None) -> PineconeVectorStore:
    """
    PineconeVectorStore apuntando al namespace configurado. text_key='text'
    porque el texto original vive en metadata['text'] (ver ChunkMetadata).
    """
    index = index or get_client().Index(config.INDEX_NAME)
    return PineconeVectorStoreConcurrente(
        index=index,
        embedding=get_embeddings(),
        text_key="text",
        namespace=config.NAMESPACE,
    )
