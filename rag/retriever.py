"""
Recuperador híbrido: BM25 (léxico) + Pinecone (semántico), combinados con
EnsembleRetriever.

Por qué EnsembleRetriever y no "sumar scores": BM25 devuelve puntajes sin
techo fijo y la similitud coseno va de -1 a 1 — no comparten escala, y
sumarlos directo es uno de los errores que marca la consigna.
EnsembleRetriever fusiona con Reciprocal Rank Fusion (RRF): usa la
POSICIÓN de cada documento en cada ranking, no su puntaje.
"""
import re
import unicodedata

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from . import config

MODOS = ("vectorial", "bm25", "hibrido")

# Palabras que aparecen en casi todas las preguntas y no ayudan a
# distinguir documentos (BM25 igual las pondera poco, pero sacarlas
# evita ruido en textos cortos).
_STOPWORDS = set(
    "el la los las un una unos unas de del al a en y o que con por para es son se su sus "
    "como cual cuales cuanto cuantos cuanta cuantas que qué tiene tienen viene vienen hay "
    "le lo me mi te tu ya mas pero sin sobre entre desde hasta este esta estos estas ese esa".split()
)


def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def tokenizar(texto: str) -> list[str]:
    """
    Preprocesamiento para BM25. El default de BM25Retriever es texto.split(),
    que en español falla feo: "Tera?" no matchea con "Tera", "Garantía" no
    matchea con "garantia". Acá: minúsculas, sin tildes, solo palabras/números,
    sin stopwords.
    """
    tokens = re.findall(r"\w+", _sin_tildes(texto.lower()))
    return [t for t in tokens if t not in _STOPWORDS]


class RAGSystem:
    """
    Encapsula el EnsembleRetriever: recibe una consulta y devuelve los
    top-k documentos combinando resultados léxicos y semánticos.

    modo: "hibrido" (default), "vectorial" o "bm25" -> los dos últimos
          existen para que evaluate.py pueda comparar contra el híbrido.
    filtrar_por_modelo: si la pregunta nombra UN solo modelo ("Tera",
          "Polo"...), acota la búsqueda a ese modelo con un filtro de
          metadata de Pinecone ({"modelo": {"$eq": "tera"}}) y también el
          corpus de BM25. Es la mejora que quedó pendiente en la Pre-Entrega 3.
    """

    def __init__(
        self,
        vectorstore,
        documentos: list[Document],
        k: int = config.TOP_K,
        modo: str = "hibrido",
        filtrar_por_modelo: bool = False,
        pesos: tuple[float, float] = (config.BM25_WEIGHT, config.VECTOR_WEIGHT),
    ):
        if modo not in MODOS:
            raise ValueError(f"modo debe ser uno de {MODOS}")
        self.vectorstore = vectorstore
        self.k = k
        self.modo = modo
        self.filtrar_por_modelo = filtrar_por_modelo
        self.pesos = list(pesos)

        self.modelos = sorted({d.metadata["modelo"] for d in documentos})
        self._bm25_global = self._crear_bm25(documentos)
        self._bm25_por_modelo = {
            m: self._crear_bm25([d for d in documentos if d.metadata["modelo"] == m]) for m in self.modelos
        }

    def _crear_bm25(self, documentos: list[Document]) -> BM25Retriever:
        bm25 = BM25Retriever.from_documents(documentos, preprocess_func=tokenizar)
        bm25.k = self.k
        return bm25

    def detectar_modelo(self, consulta: str) -> str | None:
        """Devuelve el modelo si la consulta nombra exactamente uno; si nombra 0 o varios, None."""
        tokens = set(tokenizar(consulta))
        encontrados = [m for m in self.modelos if m in tokens]
        return encontrados[0] if len(encontrados) == 1 else None

    def _filtro_vectorial(self, modelo: str) -> dict:
        return {"modelo": {"$eq": modelo}}

    def _armar_retriever(self, consulta: str):
        modelo = self.detectar_modelo(consulta) if self.filtrar_por_modelo else None

        search_kwargs = {"k": self.k}
        if modelo:
            search_kwargs["filter"] = self._filtro_vectorial(modelo)
        vectorial = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        bm25 = self._bm25_por_modelo[modelo] if modelo else self._bm25_global

        if self.modo == "vectorial":
            return vectorial
        if self.modo == "bm25":
            return bm25
        return EnsembleRetriever(retrievers=[bm25, vectorial], weights=self.pesos)

    async def aretrieve(self, consulta: str) -> list[Document]:
        """Top-k documentos para la consulta (el ensemble puede traer hasta 2k: se recorta)."""
        docs = await self._armar_retriever(consulta).ainvoke(consulta)
        return docs[: self.k]

    def retrieve(self, consulta: str) -> list[Document]:
        return self._armar_retriever(consulta).invoke(consulta)[: self.k]
